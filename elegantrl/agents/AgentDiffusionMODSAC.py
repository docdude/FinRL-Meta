"""
AgentDiffusionModSAC — Drop-in ElegantRL agent that replaces ModSAC's
ActorFixSAC with a Generative Diffusion Model (GDM) policy.

Based on:
  - "Generative AI for Deep Reinforcement Learning" (arXiv:2405.20568)
  - ElegantRL's AgentModSAC (reliable_lambda + Two Time-scale Update Rule)

Usage:
    from agent_diffusion_modsac import AgentDiffusionModSAC
    # Use exactly like AgentModSAC — same Config, same training loop.
    agent = AgentDiffusionModSAC(net_dims, state_dim, action_dim, gpu_id, args)
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple, List, Optional

from .AgentSAC import AgentModSAC, CriticEnsemble
from .AgentBase import AgentBase, ActorBase, build_mlp, layer_init_with_orthogonal
from ..train import Config, ReplayBuffer


TEN = th.Tensor

# ============================================================================
#  Diffusion Components
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: TEN) -> TEN:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return th.cat((emb.sin(), emb.cos()), dim=-1)


class DenoisingMLP(nn.Module):
    """
    MLP that predicts noise (or x_0) given noisy action, timestep, and state.
    Architecture matches the repo's model1.py but with configurable dims.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, t_dim: int = 16):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_noisy: TEN, t: TEN, state: TEN) -> TEN:
        s = self.state_mlp(state.float())
        t_emb = self.time_mlp(t)
        h = th.cat([x_noisy, t_emb, s], dim=1)
        return self.mid_layer(h)


def vp_beta_schedule(n_timesteps: int) -> TEN:
    """Variance-preserving beta schedule (default in the paper)."""
    t = np.arange(1, n_timesteps + 1)
    T = n_timesteps
    b_max, b_min = 10.0, 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    return th.tensor(1 - alpha, dtype=th.float32)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> TEN:
    """Cosine beta schedule."""
    steps = n_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return th.tensor(np.clip(betas, 0, 0.999), dtype=th.float32)


def _extract(a: TEN, t: TEN, x_shape: tuple) -> TEN:
    """Gather coefficients at timestep t and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionProcess(nn.Module):
    """
    DDPM-style diffusion for action generation.
    Replaces the MLP actor in SAC/ModSAC.

    The reverse process (denoising) generates actions from Gaussian noise
    conditioned on the current state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_timesteps: int = 3,
        beta_schedule: str = "vp",
        max_action: float = 1.0,
        predict_noise: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.n_timesteps = n_timesteps
        self.predict_noise = predict_noise

        # Denoising network
        self.model = DenoisingMLP(state_dim, action_dim, hidden_dim)

        # Beta schedule
        if beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            betas = th.linspace(1e-4, 2e-2, n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        alphas_cumprod_prev = th.cat([th.ones(1), alphas_cumprod[:-1]])

        # Register all diffusion constants as buffers (auto device transfer)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", th.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", th.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", th.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", th.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", th.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * th.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ---- Forward diffusion (training) ----

    def q_sample(self, x_start: TEN, t: TEN, noise: Optional[TEN] = None) -> TEN:
        """Add noise to x_start at timestep t."""
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def diffusion_loss(self, x_start: TEN, state: TEN) -> TEN:
        """Compute denoising loss for training the diffusion actor."""
        batch_size = x_start.shape[0]
        t = th.randint(0, self.n_timesteps, (batch_size,), device=x_start.device).long()
        noise = th.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x_recon = self.model(x_noisy, t, state)

        if self.predict_noise:
            return F.mse_loss(x_recon, noise)
        else:
            return F.mse_loss(x_recon, x_start)

    # ---- Reverse diffusion (sampling) ----

    def _predict_x0(self, x_t: TEN, t: TEN, noise_pred: TEN) -> TEN:
        if self.predict_noise:
            return (
                _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
            )
        else:
            return noise_pred

    def _p_mean_variance(self, x: TEN, t: TEN, state: TEN):
        x_recon = self._predict_x0(x, t, self.model(x, t, state))
        x_recon = x_recon.clamp(-self.max_action, self.max_action)

        model_mean = (
            _extract(self.posterior_mean_coef1, t, x.shape) * x_recon
            + _extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_log_var = _extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, posterior_log_var

    def _p_sample(self, x: TEN, t: TEN, state: TEN) -> TEN:
        b = x.shape[0]
        model_mean, model_log_var = self._p_mean_variance(x, t, state)
        noise = th.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise

    def sample(self, state: TEN) -> TEN:
        """Generate actions via full reverse diffusion chain."""
        batch_size = state.shape[0]
        x = th.randn(batch_size, self.action_dim, device=state.device)

        for i in reversed(range(self.n_timesteps)):
            t = th.full((batch_size,), i, device=state.device, dtype=th.long)
            x = self._p_sample(x, t, state)

        return x.clamp(-self.max_action, self.max_action)

    def forward(self, state: TEN) -> TEN:
        """Forward pass = sample actions."""
        return self.sample(state)


# ============================================================================
#  Diffusion Actor (ElegantRL-compatible)
# ============================================================================

class ActorDiffusion(ActorBase):
    """
    Diffusion-based actor that is interface-compatible with ElegantRL's
    ActorFixSAC. Uses a hybrid approach:
      - Diffusion model generates high-quality multi-modal actions
      - Lightweight parametric head provides log_prob estimates for SAC's
        entropy regularization

    The log_prob head is trained to track the diffusion policy via a
    distillation loss, updated jointly during actor updates.
    """

    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        n_timesteps: int = 3,
        beta_schedule: str = "vp",
        diffusion_hidden_dim: int = 256,
        lambda_diffusion: float = 1.0,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.lambda_diffusion = lambda_diffusion

        # ---- Diffusion model (action generator) ----
        self.diffusion = DiffusionProcess(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=diffusion_hidden_dim,
            n_timesteps=n_timesteps,
            beta_schedule=beta_schedule,
        )

        # ---- Parametric log_prob head (for SAC entropy) ----
        # Mirrors ActorFixSAC's structure for the log_prob computation
        self.encoder_s = build_mlp(dims=[state_dim, *net_dims])
        self.decoder_a_avg = build_mlp(dims=[net_dims[-1], action_dim])
        self.decoder_a_std = build_mlp(dims=[net_dims[-1], action_dim])
        self.soft_plus = nn.Softplus()

        layer_init_with_orthogonal(self.decoder_a_avg[-1], std=0.1)
        layer_init_with_orthogonal(self.decoder_a_std[-1], std=0.1)

    def forward(self, state: TEN) -> TEN:
        """Deterministic action for evaluation (diffusion sample)."""
        return self.diffusion.sample(state)

    def get_action(self, state: TEN, **_kwargs) -> TEN:
        """Stochastic action for exploration (diffusion sample)."""
        return self.diffusion.sample(state)

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Generate action via diffusion, estimate log_prob via parametric head.

        The parametric head learns to approximate the diffusion policy's
        density, providing the log_prob signal that SAC needs for:
          - Entropy-regularized critic targets
          - Alpha (temperature) adjustment
          - Actor objective (max Q - alpha * log_prob)
        """
        # Generate action from diffusion (high quality, multi-modal)
        action = self.diffusion.sample(state)

        # Estimate log_prob from parametric head
        state_tmp = self.encoder_s(state)
        action_log_std = self.decoder_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.decoder_a_avg(state_tmp)

        # Compute log_prob of the diffusion-generated action under the
        # parametric Gaussian. We use atanh to map back from [-1,1] to
        # unbounded space for proper density evaluation.
        action_unbounded = th.atanh(action.clamp(-0.999, 0.999))
        noise = (action_unbounded - action_avg) / (action_std + 1e-8)
        noise = noise.clamp(-5.0, 5.0)  # Prevent extreme log_prob when head is uncalibrated

        # Gaussian log_prob
        logprob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))

        # Tanh squashing correction (numerically stable form from ActorFixSAC)
        logprob -= (np.log(2.0) - action_unbounded - self.soft_plus(-2.0 * action_unbounded)) * 2.0

        # Clamp total log_prob to prevent Q-target explosion from poorly
        # calibrated head early in training
        logprob_sum = logprob.sum(1).clamp(-action_avg.shape[-1] * 2.0, 2.0)
        return action, logprob_sum

    def get_logprob_head_loss(self, state: TEN) -> TEN:
        """
        Distillation loss: train the parametric head to match the diffusion
        policy's output distribution.

        Generate actions from diffusion (detached), then maximize the
        parametric head's log_prob of those actions.
        """
        with th.no_grad():
            target_actions = self.diffusion.sample(state)

        state_tmp = self.encoder_s(state)
        action_log_std = self.decoder_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.decoder_a_avg(state_tmp)

        action_unbounded = th.atanh(target_actions.clamp(-0.999, 0.999))
        noise = (action_unbounded - action_avg) / (action_std + 1e-8)
        logprob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))
        logprob -= (np.log(2.0) - action_unbounded - self.soft_plus(-2.0 * action_unbounded)) * 2.0

        return -logprob.sum(1).mean()  # negative log-likelihood


# ============================================================================
#  AgentDiffusionModSAC
# ============================================================================

class AgentDiffusionModSAC(AgentBase):
    """
    ModSAC with a Diffusion actor replacing ActorFixSAC.

    Key features:
      - GDM (Generative Diffusion Model) policy for multi-modal action generation
      - CriticEnsemble with 8 heads (from ModSAC)
      - reliable_lambda + Two Time-scale Update Rule (from ModSAC)
      - Hybrid log_prob: diffusion generates actions, parametric head estimates density
      - Additional diffusion loss term for actor training

    Compatible with ElegantRL's training loop, vec envs, and Config system.
    All standard Config parameters are respected. Additional diffusion-specific
    params can be set via Config attributes:
      - args.n_timesteps (int, default 3): diffusion denoising steps
      - args.beta_schedule (str, default 'vp'): noise schedule
      - args.diffusion_hidden_dim (int, default 256): denoising MLP width
      - args.lambda_diffusion (float, default 1.0): weight of diffusion loss
      - args.lambda_logprob_head (float, default 0.1): weight of head distillation loss
    """

    def __init__(
        self,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: "Config" = None,
    ):
        if args is None:
            args = Config()

        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        # ---- Diffusion-specific hyperparameters ----
        self.n_timesteps = getattr(args, "n_timesteps", 3)
        self.beta_schedule = getattr(args, "beta_schedule", "vp")
        self.diffusion_hidden_dim = getattr(args, "diffusion_hidden_dim", 256)
        self.lambda_diffusion = getattr(args, "lambda_diffusion", 1.0)
        self.lambda_logprob_head = getattr(args, "lambda_logprob_head", 0.1)
        self.num_ensembles = getattr(args, "num_ensembles", 8)

        # ---- Networks ----
        self.act = ActorDiffusion(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=action_dim,
            n_timesteps=self.n_timesteps,
            beta_schedule=self.beta_schedule,
            diffusion_hidden_dim=self.diffusion_hidden_dim,
            lambda_diffusion=self.lambda_diffusion,
        ).to(self.device)

        self.cri = CriticEnsemble(
            net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles
        ).to(self.device)

        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        # ---- Optimizers ----
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        # ---- SAC entropy ----
        self.alpha_log = th.tensor(
            (-1,), dtype=th.float32, requires_grad=True, device=self.device
        )
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        self.target_entropy = getattr(args, "target_entropy", -np.log(action_dim))

        # ---- ModSAC reliable_lambda ----
        self.critic_tau = getattr(args, "critic_tau", 0.995)
        self.critic_value = 1.0
        self.update_a = 0

    def explore_action(self, state: TEN) -> TEN:
        """Generate exploration action via diffusion sampling.
        
        For VecEnv: state.shape == (num_envs, state_dim)
        For single env: state.shape == (1, state_dim)
        """
        return self.act.get_action(state)

    def update_objectives(
        self, buffer: "ReplayBuffer", update_t: int
    ) -> Tuple[float, float]:
        """
        Single update step. Called by update_net() in the training loop.

        Flow:
          1. Sample batch from replay buffer
          2. Compute critic target with entropy-regularized Bellman backup
          3. Update critic (all ensemble heads)
          4. Update alpha (temperature)
          5. Update actor: Q-value maximization + diffusion loss + head distillation
             (gated by reliable_lambda / Two Time-scale Update Rule)
        """
        with th.no_grad():
            if self.if_use_per:
                (
                    state, action, reward, undone, unmask, next_state,
                    is_weight, is_index,
                ) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(
                    self.batch_size
                )
                is_weight, is_index = None, None

            # Target actions from the *target* actor (diffusion sample)
            next_action, next_logprob = self.act.get_action_logprob(next_state)
            next_q = th.min(
                self.cri_target.get_q_values(next_state, next_action), dim=1
            )[0]
            alpha = self.alpha_log.exp()
            # Defense-in-depth: clamp logprob to prevent Q-target explosion
            next_logprob = next_logprob.clamp(-self.action_dim * 2.0, 2.0)
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        # ================================================================
        #  Critic update
        # ================================================================
        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        td_error = self.criterion(q_values, q_labels).mean(dim=1) * unmask

        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()

        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = (
                buffer.cum_rewards[buffer.ids0, buffer.ids1]
                .detach_()
                .mean()
                .repeat(q_values.shape[1])
            )
            obj_critic += (
                self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean()
                * self.lambda_fit_cum_r
            )

        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        # critic_value stays fixed at 1.0 to match vanilla ModSAC.
        # reliable_lambda = exp(-1) ≈ 0.368 → actor updates ~61% of the time.

        # ================================================================
        #  Alpha (temperature) update
        # ================================================================
        action_pg, logprob = self.act.get_action_logprob(state)
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
        self.optimizer_backward(self.alpha_optim, obj_alpha)

        # ================================================================
        #  Actor update (gated by reliable_lambda)
        # ================================================================
        alpha = self.alpha_log.exp().detach()
        with th.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

        # ModSAC's adaptive Two Time-scale Update Rule
        reliable_lambda = math.exp(-(self.critic_value ** 2))
        self.update_a = 0 if update_t == 0 else self.update_a

        if (self.update_a / (update_t + 1)) < (1 / (2 - reliable_lambda)):
            self.update_a += 1

            # --- SAC actor objective: maximize Q - alpha * entropy ---
            q_value_pg = self.cri_target(state, action_pg).mean()
            obj_actor_sac = (q_value_pg - logprob * alpha).mean()

            # --- Diffusion loss: train denoiser on replay actions ---
            obj_diffusion = self.act.diffusion.diffusion_loss(action, state)

            # --- Log_prob head distillation loss ---
            obj_head = self.act.get_logprob_head_loss(state)

            # --- Combined actor loss ---
            obj_actor_total = (
                -obj_actor_sac
                + self.lambda_diffusion * obj_diffusion
                + self.lambda_logprob_head * obj_head
            )

            self.optimizer_backward(self.act_optimizer, obj_actor_total)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)

            obj_actor = obj_actor_sac
        else:
            obj_actor = th.tensor(th.nan)

        return obj_critic.item(), obj_actor.item()


# ============================================================================
#  Convenience: quick experiment helper
# ============================================================================

def make_config(
    env_class,
    env_args: dict,
    net_dims: List[int] = None,
    n_timesteps: int = 3,
    beta_schedule: str = "vp",
    diffusion_hidden_dim: int = 256,
    lambda_diffusion: float = 1.0,
    lambda_logprob_head: float = 0.1,
    num_ensembles: int = 8,
    **kwargs,
) -> "Config":
    """
    Create a Config pre-filled with diffusion-specific parameters.

    Example:
        args = make_config(
            env_class=StockTradingEnv,
            env_args={"if_train": True, ...},
            net_dims=[256, 256],
            n_timesteps=3,
            learning_rate=3e-4,
        )
        train_agent(AgentDiffusionModSAC, args)
    """
    if net_dims is None:
        net_dims = [256, 256]

    args = Config(agent_class=AgentDiffusionModSAC, env_class=env_class, env_args=env_args)
    args.net_dims = net_dims

    # Diffusion params
    args.n_timesteps = n_timesteps
    args.beta_schedule = beta_schedule
    args.diffusion_hidden_dim = diffusion_hidden_dim
    args.lambda_diffusion = lambda_diffusion
    args.lambda_logprob_head = lambda_logprob_head
    args.num_ensembles = num_ensembles

    # Apply any extra kwargs (learning_rate, gamma, batch_size, etc.)
    for k, v in kwargs.items():
        setattr(args, k, v)

    return args
