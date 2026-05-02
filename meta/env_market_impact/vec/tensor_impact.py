from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch as th

from .common import EPS


@dataclass
class TensorImpactConfig:
    """Config for the square-root impact model (AC Thum-Hauptmann calibration)."""
    Y: float = 0.6
    perm_fraction: float = 0.25
    perm_half_life_days: float = 5.0


@dataclass
class TensorACImpactConfig:
    """Config for the classic Almgren-Chriss impact model."""
    alpha: float = 1.0
    beta: float = 1.0
    epsilon: float = 0.0005
    perm_half_life_days: float = 5.0


@dataclass
class TensorBaselineImpactConfig:
    """Config for the fixed-fee baseline impact model."""
    basis_points: float = 10.0
    perm_half_life_days: float = 5.0


@dataclass
class TensorOWImpactConfig:
    """Config for the Obizhaeva-Wang impact model."""
    Y: float = 0.6
    perm_fraction: float = 0.25
    half_life_days: float = 0.08
    perm_half_life_days: float = 5.0


class TensorImpactBase:
    """Common permanent-state bookkeeping for tensor impact models.

    Subclasses override :meth:`_impact_core` to compute cost and price shift
    for one stock across all envs.  Subclasses may also call
    :meth:`apply_trades_batched` to compute impact for the full
    ``(num_envs, stock_dim)`` trade matrix in a single broadcast pass (the
    "vmap-equivalent" path).
    """

    perm_decay_rate: float

    def __init__(self, num_envs: int, stock_dim: int, device: th.device) -> None:
        self.device = device
        self.num_envs = num_envs
        self.stock_dim = stock_dim
        self.perm_state = th.zeros(
            (num_envs, stock_dim), dtype=th.float32, device=device
        )
        self._impact_records: list[dict[str, object]] = []

    def reset(self, env_mask: Optional[th.Tensor] = None) -> None:
        if env_mask is None:
            self.perm_state.zero_()
            self._impact_records = []
            return
        self.perm_state[env_mask] = 0.0

    def get_perm_state_array(self) -> th.Tensor:
        return self.perm_state

    def get_perm_state_for_price(
        self, env_indices: Optional[th.Tensor], stock_index: int
    ) -> th.Tensor:
        if env_indices is None:
            return self.perm_state[:, stock_index]
        return self.perm_state[env_indices, stock_index]

    def end_day(
        self,
        env_mask: Optional[th.Tensor] = None,
        date_str: Optional[str] = None,
        stock_symbols: Optional[list[str]] = None,
    ) -> None:
        if self.perm_decay_rate <= 0:
            self._record_impact_history(date_str=date_str, stock_symbols=stock_symbols)
            return
        decay_factor = 1.0 - self.perm_decay_rate
        if env_mask is None:
            self.perm_state.mul_(decay_factor)
        else:
            self.perm_state[env_mask] *= decay_factor
        self._record_impact_history(date_str=date_str, stock_symbols=stock_symbols)

    def _record_impact_history(
        self,
        *,
        date_str: Optional[str],
        stock_symbols: Optional[list[str]],
    ) -> None:
        if date_str is None or stock_symbols is None or self.num_envs != 1:
            return
        for stock_idx, symbol in enumerate(stock_symbols):
            self._impact_records.append(
                {
                    "date": date_str,
                    "symbol": symbol,
                    "permanent_impact": float(self.perm_state[0, stock_idx].item()),
                }
            )

    def get_impact_history(self) -> pd.DataFrame:
        if not self._impact_records:
            return pd.DataFrame(columns=["date", "symbol", "permanent_impact"])
        return pd.DataFrame(self._impact_records)

    def _expand_batched_inputs(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        ts = trade_size.to(dtype=th.float32, device=self.device)
        px = price.to(dtype=th.float32, device=self.device)
        vol = volatility.to(dtype=th.float32, device=self.device)
        vm = volume.to(dtype=th.float32, device=self.device)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0).expand(self.num_envs, -1)
        if px.ndim == 1:
            px = px.unsqueeze(0).expand(self.num_envs, -1)
        if vol.ndim == 1:
            vol = vol.unsqueeze(0).expand(self.num_envs, -1)
        if vm.ndim == 1:
            vm = vm.unsqueeze(0).expand(self.num_envs, -1)
        return ts, px, vol, vm

    def preview_trades_batched(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        ts, px, vol, vm = self._expand_batched_inputs(
            trade_size,
            price,
            volatility,
            volume,
        )

        valid = (vm > 0) & (ts != 0)
        cost = th.zeros_like(ts)
        price_shift = th.zeros_like(ts)
        if valid.any():
            safe_ts = th.where(valid, ts, th.zeros_like(ts))
            safe_px = th.where(valid, px, th.ones_like(px))
            safe_vol = th.where(valid, vol, th.zeros_like(vol))
            safe_vm = th.where(valid, vm, th.ones_like(vm))
            c, s = self._impact_core(safe_ts, safe_px, safe_vol, safe_vm)
            cost = th.where(valid, c, cost)
            price_shift = th.where(valid, s, price_shift)
        return cost, price_shift

    # ------------------------------------------------------------------
    # Core per-stock kernel: subclasses implement this.
    # ------------------------------------------------------------------

    def _impact_core(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-stock env-batched entry point used by the vec envs.
    # ------------------------------------------------------------------

    def apply_trade(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        stock_index: int,
        env_indices: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        trade_size = trade_size.to(dtype=th.float32, device=self.device)
        price = price.to(dtype=th.float32, device=self.device)
        volatility = volatility.to(dtype=th.float32, device=self.device)
        volume = volume.to(dtype=th.float32, device=self.device)

        valid = (volume > 0) & (trade_size != 0)
        cost = th.zeros_like(trade_size, dtype=th.float32)
        price_shift = th.zeros_like(trade_size, dtype=th.float32)
        if valid.any():
            cost_valid, shift_valid = self._impact_core(
                trade_size[valid], price[valid], volatility[valid], volume[valid]
            )
            cost[valid] = cost_valid
            price_shift[valid] = shift_valid
            if env_indices is None:
                self.perm_state[valid, stock_index] += shift_valid
            else:
                self.perm_state[env_indices[valid], stock_index] += shift_valid
        return cost, price_shift

    # ------------------------------------------------------------------
    # All-stocks batched entry point: processes (num_envs, stock_dim)
    # matrices in a single broadcast pass.  Equivalent to vmapping
    # apply_trade across the stock dimension, without the Python loop.
    # ------------------------------------------------------------------

    def apply_trades_batched(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Apply trades for all stocks in a single tensor op.

        All inputs are broadcast to ``(num_envs, stock_dim)``.  Permanent
        state is updated in place with the resulting price shift.
        """
        cost, price_shift = self.preview_trades_batched(
            trade_size,
            price,
            volatility,
            volume,
        )
        if price_shift.any():
            self.perm_state += price_shift
        return cost, price_shift


class TensorOWImpactModel(TensorImpactBase):
    """Obizhaeva-Wang impact model with transient state."""

    def __init__(
        self,
        num_envs: int,
        stock_dim: int,
        device: th.device,
        config: Optional[TensorOWImpactConfig] = None,
    ) -> None:
        super().__init__(num_envs=num_envs, stock_dim=stock_dim, device=device)
        config = config or TensorOWImpactConfig()
        self.config = config
        self.Y = float(config.Y)
        self.perm_fraction = float(config.perm_fraction)
        self.half_life_days = float(config.half_life_days)
        self.perm_half_life_days = float(config.perm_half_life_days)
        if self.perm_half_life_days > 0:
            self.perm_decay_rate = 1 - 0.5 ** (1.0 / self.perm_half_life_days)
        else:
            self.perm_decay_rate = 0.0
        if self.half_life_days > 0:
            self.transient_decay_factor = math.exp(
                -math.log(2.0) / self.half_life_days
            )
        else:
            self.transient_decay_factor = 0.0
        self.transient_state = th.zeros(
            (num_envs, stock_dim), dtype=th.float32, device=device
        )

    def reset(self, env_mask: Optional[th.Tensor] = None) -> None:
        super().reset(env_mask=env_mask)
        if env_mask is None:
            self.transient_state.zero_()
            return
        self.transient_state[env_mask] = 0.0

    def end_day(
        self,
        env_mask: Optional[th.Tensor] = None,
        date_str: Optional[str] = None,
        stock_symbols: Optional[list[str]] = None,
    ) -> None:
        super().end_day(
            env_mask=env_mask,
            date_str=date_str,
            stock_symbols=stock_symbols,
        )
        if env_mask is None:
            self.transient_state.mul_(self.transient_decay_factor)
        else:
            self.transient_state[env_mask] *= self.transient_decay_factor

    def apply_trade(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        stock_index: int,
        env_indices: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        trade_size = trade_size.to(dtype=th.float32, device=self.device)
        price = price.to(dtype=th.float32, device=self.device)
        volatility = volatility.to(dtype=th.float32, device=self.device)
        volume = volume.to(dtype=th.float32, device=self.device)

        valid = (volume > 0) & (trade_size != 0)
        cost = th.zeros_like(trade_size, dtype=th.float32)
        price_shift = th.zeros_like(trade_size, dtype=th.float32)
        if not valid.any():
            return cost, price_shift

        if env_indices is None:
            target_env = th.arange(
                trade_size.shape[0], dtype=th.long, device=self.device
            )[valid]
        else:
            target_env = env_indices[valid]

        participation = trade_size[valid].abs() / volume[valid].clamp_min(EPS)
        instant_impact_frac = self.Y * volatility[valid] * participation.sqrt()
        perm_frac = self.perm_fraction * instant_impact_frac
        shift_valid = trade_size[valid].sign() * perm_frac * price[valid]
        perm_cost = 0.5 * shift_valid.abs() * trade_size[valid].abs()
        temp_frac = (1.0 - self.perm_fraction) * instant_impact_frac
        temp_cost = temp_frac * trade_size[valid].abs() * price[valid]
        prev_transient = self.transient_state[target_env, stock_index]
        trans_cost = prev_transient * trade_size[valid].abs() * price[valid]

        cost_valid = perm_cost + temp_cost + trans_cost
        cost[valid] = cost_valid
        price_shift[valid] = shift_valid
        self.perm_state[target_env, stock_index] += shift_valid
        self.transient_state[target_env, stock_index] += instant_impact_frac
        return cost, price_shift

    def _impact_core(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        participation = trade_size.abs() / volume.clamp_min(EPS)
        instant_impact_frac = self.Y * volatility * participation.sqrt()
        perm_frac = self.perm_fraction * instant_impact_frac
        price_shift = trade_size.sign() * perm_frac * price
        perm_cost = 0.5 * price_shift.abs() * trade_size.abs()
        temp_frac = (1.0 - self.perm_fraction) * instant_impact_frac
        temp_cost = temp_frac * trade_size.abs() * price
        return perm_cost + temp_cost, price_shift

    def apply_trades_batched(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        ts, px, vol, vm = self._expand_batched_inputs(
            trade_size,
            price,
            volatility,
            volume,
        )
        cost, price_shift = self.preview_trades_batched(ts, px, vol, vm)
        if price_shift.any():
            self.perm_state += price_shift
        valid = (vm > 0) & (ts != 0)
        participation = ts.abs() / vm.clamp_min(EPS)
        instant_impact_frac = self.Y * vol * participation.sqrt()
        self.transient_state += th.where(
            valid,
            instant_impact_frac,
            th.zeros_like(instant_impact_frac),
        )
        return cost, price_shift

    def preview_trades_batched(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        ts, px, vol, vm = self._expand_batched_inputs(
            trade_size,
            price,
            volatility,
            volume,
        )
        valid = (vm > 0) & (ts != 0)
        participation = ts.abs() / vm.clamp_min(EPS)
        instant_impact_frac = self.Y * vol * participation.sqrt()
        perm_frac = self.perm_fraction * instant_impact_frac
        price_shift = ts.sign() * perm_frac * px
        perm_cost = 0.5 * price_shift.abs() * ts.abs()
        temp_frac = (1.0 - self.perm_fraction) * instant_impact_frac
        temp_cost = temp_frac * ts.abs() * px
        trans_cost = self.transient_state * ts.abs() * px
        cost = perm_cost + temp_cost + trans_cost
        cost = th.where(valid, cost, th.zeros_like(cost))
        price_shift = th.where(valid, price_shift, th.zeros_like(price_shift))
        return cost, price_shift


class TensorSqrtImpactModel(TensorImpactBase):
    """Square-root impact: ``I = Y * sigma * sqrt(|x|/V)``.

    Matches :class:`meta.env_market_impact.envs.impact_models.SqrtImpactModel`.
    """

    def __init__(
        self,
        num_envs: int,
        stock_dim: int,
        device: th.device,
        config: Optional[TensorImpactConfig] = None,
    ) -> None:
        super().__init__(num_envs=num_envs, stock_dim=stock_dim, device=device)
        config = config or TensorImpactConfig()
        self.config = config
        self.Y = float(config.Y)
        self.perm_fraction = float(config.perm_fraction)
        self.perm_half_life_days = float(config.perm_half_life_days)
        self._k = 2.0 / 3.0
        if self.perm_half_life_days > 0:
            self.perm_decay_rate = 1 - 0.5 ** (1.0 / self.perm_half_life_days)
        else:
            self.perm_decay_rate = 0.0

    def _impact_core(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        participation = trade_size.abs() / volume.clamp_min(EPS)
        peak_frac = self.Y * volatility * participation.sqrt()
        perm_frac = self.perm_fraction * peak_frac
        price_shift = trade_size.sign() * perm_frac * price
        cost = self._k * peak_frac * trade_size.abs() * price
        return cost, price_shift


class TensorACImpactModel(TensorImpactBase):
    """Almgren-Chriss impact model.

    Matches :class:`meta.env_market_impact.envs.impact_models.ACImpactModel`.

    Permanent price shift: ``dP = alpha * sigma * (x/V) * P``.
    Total cost: ``0.5*|dP|*|x| + eps*|x|*P + beta*sigma*(|x|/V)*|x|*P``.
    """

    def __init__(
        self,
        num_envs: int,
        stock_dim: int,
        device: th.device,
        config: Optional[TensorACImpactConfig] = None,
    ) -> None:
        super().__init__(num_envs=num_envs, stock_dim=stock_dim, device=device)
        config = config or TensorACImpactConfig()
        self.config = config
        self.alpha = float(config.alpha)
        self.beta = float(config.beta)
        self.epsilon = float(config.epsilon)
        self.perm_half_life_days = float(config.perm_half_life_days)
        if self.perm_half_life_days > 0:
            self.perm_decay_rate = 1 - 0.5 ** (1.0 / self.perm_half_life_days)
        else:
            self.perm_decay_rate = 0.0

    def _impact_core(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        participation = trade_size.abs() / volume.clamp_min(EPS)
        eta = self.alpha * volatility
        price_shift = eta * (trade_size / volume.clamp_min(EPS)) * price
        perm_cost = 0.5 * price_shift.abs() * trade_size.abs()
        spread_cost = self.epsilon * trade_size.abs() * price
        gamma = self.beta * volatility
        depth_cost = gamma * participation * trade_size.abs() * price
        cost = perm_cost + spread_cost + depth_cost
        return cost, price_shift


class TensorBaselineImpactModel(TensorImpactBase):
    """Fixed-fee baseline impact model.

    Matches :class:`meta.env_market_impact.envs.impact_models.BaselineImpactModel`.
    """

    def __init__(
        self,
        num_envs: int,
        stock_dim: int,
        device: th.device,
        config: Optional[TensorBaselineImpactConfig] = None,
    ) -> None:
        super().__init__(num_envs=num_envs, stock_dim=stock_dim, device=device)
        config = config or TensorBaselineImpactConfig()
        self.config = config
        self.basis_points = float(config.basis_points)
        self.fee_rate = self.basis_points / 10000.0
        self.perm_half_life_days = float(config.perm_half_life_days)
        self.perm_decay_rate = 0.0

    def _impact_core(
        self,
        trade_size: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        del volatility, volume
        cost = self.fee_rate * trade_size.abs() * price
        price_shift = th.zeros_like(cost)
        return cost, price_shift
