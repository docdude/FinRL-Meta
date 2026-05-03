"""
Intensity Timing Wrapper for Vectorized Trading Environments.

Implements the Cox-process intensity timing mechanism from
Zhao, Tse & Zheng (2026), arXiv:2604.02035v1, as a composable wrapper
around any vectorized trading environment (MACEVecEnv, MarginVecEnv, etc.).

The wrapper intercepts the agent's continuous actions and gates them through
a per-stock Bernoulli stopping process:
  - J=0 (flat): agent's positive action → entry intensity λ_α
    → Bernoulli draw q = 1 - exp(-λ_α * dt) → buy if triggered
  - J=1 (holding): agent's negative action → exit intensity λ_β
    → Bernoulli draw q = 1 - exp(-λ_β * dt) → sell if triggered

The agent learns *when* to trade (timing) jointly with *how much* (sizing).
Trades only execute when the intensity mechanism fires, preventing
over-trading and enforcing patience.

Usage:
    base_env = MACEVecEnv(config, ...)
    wrapped = IntensityTimingWrapper(base_env)
    # wrapped has same interface as base_env — plug into ElegantRL/SB3
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch as th


class IntensityTimingWrapper:
    """Wraps a vectorized trading env with Cox-process intensity timing.

    The wrapper augments the observation with per-stock regime state (J),
    hold age, and entry price reference, then gates the agent's actions
    through Bernoulli stopping draws.

    Parameters
    ----------
    base_env
        The underlying vectorized trading environment. Must expose:
        ``step(actions)``, ``reset()``, ``stocks``, ``num_envs``,
        ``stock_dim``, ``state_dim``, ``action_dim``, ``max_step``,
        ``device``.
    M : float
        Intensity cap. Higher M allows more decisive entry/exit.
    eta : float
        Entropy temperature. Lower η → sharper (more deterministic) timing.
    rho : float
        Subjective discount rate. Controls patience (entry/exit tradeoff).
    dt : float
        Time step size for Bernoulli conversion: q = 1 - exp(-λ·dt).
    Psi : float
        Fixed transaction cost parameter in the utility function.
    varpi : float
        Risk aversion exponent in the Prospect Theory utility.
    k_loss : float
        Loss aversion multiplier.
    gamma : float
        Sale scaling parameter (1.0 for no scaling).
    iota : float
        Purchase scaling parameter (1.0 for no scaling).
    R : float
        Reference point for utility evaluation.
    entropy_reward_scale : float
        How much of the entropy cost to subtract from reward.
        Set to 0 to disable entropy reward shaping.
    augment_state : bool
        If True, augment observations with J, hold_age, entry_price_ratio.
    """

    def __init__(
        self,
        base_env,
        M: float = 5.0,
        eta: float = 0.005,
        rho: float = 0.002,
        dt: float = 0.25,
        Psi: float = 0.20,
        varpi: float = 0.5,
        k_loss: float = 2.0,
        gamma: float = 1.0,
        iota: float = 1.0,
        R: float = 0.0,
        entropy_reward_scale: float = 0.0,
        augment_state: bool = True,
    ):
        self.base_env = base_env
        self.M = M
        self.eta = eta
        self.rho = rho
        self.dt = dt
        self.Psi = Psi
        self.varpi = varpi
        self.k_loss = k_loss
        self.gamma_param = gamma
        self.iota = iota
        self.R = R
        self.entropy_reward_scale = entropy_reward_scale
        self.augment_state = augment_state

        # Proxy base env attributes
        self.num_envs = base_env.num_envs
        self.stock_dim = base_env.stock_dim
        self.device = base_env.device
        self.max_step = base_env.max_step
        self.action_dim = base_env.action_dim  # same action space
        self.if_discrete = False
        self.target_return = getattr(base_env, "target_return", float("inf"))
        self.env_name = f"IntensityTiming-{getattr(base_env, 'env_name', 'VecEnv')}"

        # State augmentation: +3 features per stock (J, hold_age_norm, entry_price_ratio)
        self._aug_per_stock = 3 if augment_state else 0
        self._aug_dim = self._aug_per_stock * self.stock_dim
        self.state_dim = base_env.state_dim + self._aug_dim

        # Per-stock regime tracking tensors
        self._J = th.zeros(
            (self.num_envs, self.stock_dim), dtype=th.float32, device=self.device
        )  # 0=flat, 1=holding
        self._hold_age = th.zeros(
            (self.num_envs, self.stock_dim), dtype=th.float32, device=self.device
        )
        self._entry_price = th.zeros(
            (self.num_envs, self.stock_dim), dtype=th.float32, device=self.device
        )

        # Forward obs normalizer if present
        self.obs_normalizer = None

    # ------------------------------------------------------------------
    # Gibbs mean intensity: λ̄ = M/(1-e^{-z}) - η/Δ, z = MΔ/η
    # ------------------------------------------------------------------
    def _mean_lam(self, delta: th.Tensor) -> th.Tensor:
        z = self.M * delta / self.eta
        az = th.abs(z)
        safe = th.where(az > 0.1, z, th.ones_like(z))
        exact = th.clamp(
            self.M / (1 - th.exp(-safe) + 1e-30) - self.M / safe, 0, self.M
        )
        taylor = th.clamp(self.M / 2 + self.M * z / 12, 0, self.M)
        return th.where(az > 0.1, exact, taylor)

    # ------------------------------------------------------------------
    # Utility: U(x) = x^ϖ if x≥0, else -k|x|^ϖ
    # ------------------------------------------------------------------
    def _U(self, x: th.Tensor) -> th.Tensor:
        a = th.abs(x) + 1e-8
        return th.where(x >= 0, a.pow(self.varpi), -self.k_loss * a.pow(self.varpi))

    # ------------------------------------------------------------------
    # Trade edge: (γ·p_exit - ι·p_entry - Ψ - R) / scale
    # ------------------------------------------------------------------
    def _trade_edge(
        self, current_price: th.Tensor, entry_price: th.Tensor
    ) -> th.Tensor:
        return self.gamma_param * current_price - self.iota * entry_price - self.Psi - self.R

    # ------------------------------------------------------------------
    # HJB source term for entropy cost
    # ------------------------------------------------------------------
    def _hjb_src(self, delta: th.Tensor) -> th.Tensor:
        z = self.M * delta / self.eta
        az = th.abs(z)
        sa = az + 1e-6
        exact = self.eta * (th.relu(z) + th.log1p(-th.exp(-sa)) - th.log(sa))
        return th.where(az < 0.1, self.M * delta / 2.0, exact)

    def _ent_cost(self, delta: th.Tensor) -> th.Tensor:
        return th.clamp(self._mean_lam(delta) * delta - self._hjb_src(delta), min=0)

    # ------------------------------------------------------------------
    # Sync J from base env positions (for robustness after reset/auto-reset)
    # ------------------------------------------------------------------
    def _sync_J_from_positions(self):
        """Derive regime J from actual holdings in the base env."""
        self._J = (self.base_env.stocks > 0).float()

    # ------------------------------------------------------------------
    # State augmentation
    # ------------------------------------------------------------------
    def _augment_obs(self, base_obs: th.Tensor) -> th.Tensor:
        if not self.augment_state:
            return base_obs

        # Normalize hold age: clamp to [0, 1] over ~250 trading days
        hold_age_norm = th.clamp(self._hold_age / 250.0, 0.0, 1.0)

        # Entry price ratio: current_price / entry_price - 1 (0 when flat)
        current_prices = self.base_env.price_array[self.base_env.time]  # (stock_dim,)
        entry_ratio = th.zeros_like(self._entry_price)
        has_position = self._J > 0.5
        safe_entry = th.where(
            has_position, th.clamp(self._entry_price, min=1e-6), th.ones_like(self._entry_price)
        )
        entry_ratio = th.where(
            has_position,
            current_prices.unsqueeze(0) / safe_entry - 1.0,
            th.zeros_like(self._entry_price),
        )
        entry_ratio = th.clamp(entry_ratio, -2.0, 2.0)

        # Concatenate: [base_obs, J, hold_age_norm, entry_ratio]
        aug = th.cat([self._J, hold_age_norm, entry_ratio], dim=-1)  # (num_envs, 3*stock_dim)
        return th.cat([base_obs, aug], dim=-1)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[th.Tensor, dict]:
        state, info = self.base_env.reset(**kwargs)
        self._J.zero_()
        self._hold_age.zero_()
        self._entry_price.zero_()
        self._sync_J_from_positions()
        return self._augment_obs(state), info

    # ------------------------------------------------------------------
    # step — the core timing gate
    # ------------------------------------------------------------------
    def step(
        self, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Gate actions through intensity timing, then forward to base env.

        The agent's action ∈ [-1, 1] per stock is interpreted as:
          - action > 0 when J=0: entry signal. Magnitude → entry intensity.
          - action < 0 when J=1: exit signal. Magnitude → exit intensity.
          - Mismatched signs (sell when flat, buy when holding) are zeroed.

        The intensity λ = mean_lam(|action| * M_scale) converts the agent's
        continuous signal into a stopping rate, then a Bernoulli draw
        determines whether the trade actually executes.
        """
        # actions shape: (num_envs, stock_dim)
        actions = actions.to(self.device)

        # Current prices for edge computation
        current_prices = self.base_env.price_array[
            min(self.base_env.time, self.base_env.max_step - 1)
        ]  # (stock_dim,)

        flat_mask = self._J < 0.5       # (num_envs, stock_dim), True where flat
        hold_mask = ~flat_mask           # True where holding

        # --- Entry intensity (flat stocks with positive action) ---
        entry_signal = th.clamp(actions, min=0.0) * flat_mask.float()  # only positive, only flat
        # Map action magnitude [0,1] to advantage-like delta for intensity
        # Higher action → higher delta → higher intensity → more likely to enter
        entry_delta = entry_signal * 2.0  # scale to reasonable delta range
        entry_lam = self._mean_lam(entry_delta) * flat_mask.float()
        entry_q = 1.0 - th.exp(-entry_lam * self.dt)
        entry_draw = th.rand_like(entry_q) < entry_q
        entry_fire = entry_draw & flat_mask  # (num_envs, stock_dim) bool

        # --- Exit intensity (holding stocks with negative action) ---
        # Compute trade edge as the exit advantage signal
        trade_edge = self._trade_edge(current_prices.unsqueeze(0), self._entry_price)
        # Use utility advantage: G(p,b) as the exit delta
        exit_delta = th.clamp(trade_edge, min=-5.0, max=5.0) * hold_mask.float()
        exit_lam = self._mean_lam(exit_delta) * hold_mask.float()
        exit_q = 1.0 - th.exp(-exit_lam * self.dt)
        exit_draw = th.rand_like(exit_q) < exit_q
        exit_fire = exit_draw & hold_mask  # (num_envs, stock_dim) bool

        # --- Build gated action ---
        # Start with zeros (hold everything)
        gated_action = th.zeros_like(actions)

        # Entry fires: forward the agent's positive action (buy signal)
        gated_action = th.where(entry_fire.unsqueeze(-1).expand_as(gated_action.unsqueeze(-1)).squeeze(-1)
                                if actions.dim() != entry_fire.dim() else entry_fire,
                                actions.clamp(min=0.01),  # ensure positive buy
                                gated_action)

        # Exit fires: force full sell (-1)
        gated_action = th.where(exit_fire, th.full_like(gated_action, -1.0), gated_action)

        # --- Forward to base env ---
        next_state, reward, done, truncated, info = self.base_env.step(gated_action)

        # --- Update regime tracking ---
        # New entries: J transitions 0→1
        newly_entered = entry_fire & (self.base_env.stocks > 0)  # confirm base env accepted
        self._J = th.where(newly_entered, th.ones_like(self._J), self._J)
        self._hold_age = th.where(newly_entered, th.ones_like(self._hold_age), self._hold_age)
        self._entry_price = th.where(
            newly_entered,
            current_prices.unsqueeze(0).expand_as(self._entry_price),
            self._entry_price,
        )

        # Exits: J transitions 1→0
        newly_exited = exit_fire & (self.base_env.stocks <= 0)  # confirm base env sold
        self._J = th.where(newly_exited, th.zeros_like(self._J), self._J)
        self._hold_age = th.where(newly_exited, th.zeros_like(self._hold_age), self._hold_age)
        self._entry_price = th.where(newly_exited, th.zeros_like(self._entry_price), self._entry_price)

        # Age all held positions
        self._hold_age = th.where(self._J > 0.5, self._hold_age + 1.0, self._hold_age)

        # --- Entropy reward shaping (optional) ---
        if self.entropy_reward_scale > 0:
            entry_ent = self._ent_cost(entry_delta).sum(dim=-1) * self.dt
            exit_ent = self._ent_cost(exit_delta).sum(dim=-1) * self.dt
            reward = reward - self.entropy_reward_scale * (entry_ent + exit_ent)

        # --- Handle auto-reset: sync J for envs that reset ---
        if done.any():
            reset_mask = done.bool()
            self._J[reset_mask] = 0.0
            self._hold_age[reset_mask] = 0.0
            self._entry_price[reset_mask] = 0.0
            # Re-sync from base env positions after auto-reset
            if reset_mask.any():
                self._J[reset_mask] = (self.base_env.stocks[reset_mask] > 0).float()

        # Record gating stats in info
        info["entry_fires"] = entry_fire.sum().item()
        info["exit_fires"] = exit_fire.sum().item()
        info["gated_zeros"] = (gated_action.abs() < 1e-6).sum().item()
        info["total_actions"] = actions.numel()

        return self._augment_obs(next_state), reward, done, truncated, info

    # ------------------------------------------------------------------
    # Forward everything else to base env
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        """Proxy unknown attributes to the base env."""
        if name.startswith("_") or name in (
            "base_env", "M", "eta", "rho", "dt", "Psi", "varpi", "k_loss",
            "gamma_param", "iota", "R", "entropy_reward_scale", "augment_state",
            "num_envs", "stock_dim", "device", "max_step", "action_dim",
            "if_discrete", "target_return", "env_name", "state_dim",
            "obs_normalizer",
        ):
            raise AttributeError(name)
        return getattr(self.base_env, name)

    def close(self):
        return self.base_env.close()

    def save_normalizer_state(self, path):
        if hasattr(self.base_env, "save_normalizer_state"):
            return self.base_env.save_normalizer_state(path)

    def load_normalizer_state(self, path, freeze=False):
        if hasattr(self.base_env, "load_normalizer_state"):
            return self.base_env.load_normalizer_state(path, freeze=freeze)
