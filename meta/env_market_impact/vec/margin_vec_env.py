from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Tuple

import torch as th

from meta.env_market_impact.backtest_vec_config import VecMarginEnvParams

from .common import EPS
from .common import resolve_device
from .tensor_impact import TensorImpactBase
from .tensor_impact import TensorImpactConfig
from .tensor_impact import TensorSqrtImpactModel


class MarginTraderVecEnv:
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        params: Optional[VecMarginEnvParams] = None,
        initial_capital: float = 1e9,
        max_stock_pct: Optional[float] = None,
        initial_stocks: Optional[th.Tensor] = None,
        margin_rate: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
        maintenance_margin: Optional[float] = None,
        maintenance_warning: Optional[float] = None,
        max_trade_volume_pct: Optional[float] = None,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        sharpe_window: Optional[int] = None,
        margin_adjust_period: Optional[int] = None,
        num_envs: int = 128,
        gpu_id: int = -1,
        device: Optional[str] = None,
        impact_config: Optional[TensorImpactConfig] = None,
        impact_model: Optional[TensorImpactBase] = None,
        initial_margin_state: Optional[Dict] = None,
        if_random_reset: bool = False,
        auto_reset: bool = True,
    ) -> None:
        params = params or VecMarginEnvParams()
        self.params = params
        self.device = resolve_device(gpu_id=gpu_id, device=device)
        self.num_envs = int(num_envs)
        self.auto_reset = auto_reset
        self.if_random_reset = if_random_reset

        self.date_list = list(config["date_list"])
        self.price_array = th.as_tensor(config["price_array"], dtype=th.float32, device=self.device)
        self.tech_array = th.as_tensor(config["tech_array"], dtype=th.float32, device=self.device) * (2**-7)
        default_vol = th.ones_like(self.price_array) * 0.02
        default_volume = th.ones_like(self.price_array) * 1e6
        self.volatility_array = th.as_tensor(config.get("volatility_array", default_vol), dtype=th.float32, device=self.device)
        self.volume_array = th.as_tensor(config.get("volume_array", default_volume), dtype=th.float32, device=self.device)
        self.stock_symbols = list(config["tic_list"])

        self.stock_dim = self.price_array.shape[1]
        self.initial_capital = float(initial_capital)
        self.max_stock_pct = (
            params.max_stock_pct if max_stock_pct is None else max_stock_pct
        )
        self.margin_rate = params.margin_rate if margin_rate is None else margin_rate
        self.long_short_ratio = (
            params.long_short_ratio
            if long_short_ratio is None
            else long_short_ratio
        )
        self.maintenance_margin = (
            params.maintenance_margin
            if maintenance_margin is None
            else maintenance_margin
        )
        self.maintenance_warning = (
            params.maintenance_warning
            if maintenance_warning is None
            else maintenance_warning
        )
        self.max_trade_volume_pct = (
            params.max_trade_volume_pct
            if max_trade_volume_pct is None
            else max_trade_volume_pct
        )
        self.lambda_1 = params.lambda_1 if lambda_1 is None else lambda_1
        self.lambda_2 = params.lambda_2 if lambda_2 is None else lambda_2
        self.sharpe_window = (
            params.sharpe_window if sharpe_window is None else sharpe_window
        )
        self.margin_adjust_period = (
            params.margin_adjust_period
            if margin_adjust_period is None
            else margin_adjust_period
        )
        self.initial_margin_state = initial_margin_state

        base_stocks = th.zeros(self.stock_dim, dtype=th.float32, device=self.device)
        self.initial_stocks = base_stocks if initial_stocks is None else initial_stocks.to(self.device, dtype=th.float32)

        self.state_dim = 6 + (4 * self.stock_dim) + self.tech_array.shape[1]
        self.action_dim = self.stock_dim
        self.env_name = "MarginTraderVecEnv-v1"
        self.if_discrete = False
        self.max_step = int(self.price_array.shape[0] - 1)
        self.target_return = float("inf")

        self.impact_model = impact_model if impact_model is not None else TensorSqrtImpactModel(
            num_envs=self.num_envs,
            stock_dim=self.stock_dim,
            device=self.device,
            config=impact_config,
        )

        self.time = 0
        self.long_cash = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.loan = th.zeros_like(self.long_cash)
        self.long_equity = th.zeros_like(self.long_cash)
        self.short_limit = th.zeros_like(self.long_cash)
        self.short_credit = th.zeros_like(self.long_cash)
        self.short_equity = th.zeros_like(self.long_cash)
        self.total_asset = th.zeros_like(self.long_cash)
        self.initial_total_asset = th.zeros_like(self.long_cash)
        self.cash = th.zeros_like(self.long_cash)
        self.stocks = th.zeros((self.num_envs, self.stock_dim), dtype=th.float32, device=self.device)
        self.stocks_cool_down = th.zeros_like(self.stocks)
        self.episode_return = th.ones(self.num_envs, dtype=th.float32, device=self.device)
        self.last_episode_return = self.episode_return.clone()
        self._equity_history = th.zeros(
            (self.num_envs, self.sharpe_window + 1), dtype=th.float32, device=self.device
        )
        self._equity_count = 0

    def close(self) -> None:
        return None

    def get_impact_history(self):
        return self.impact_model.get_impact_history()

    def _build_aggregate_trade_entry(
        self,
        *,
        stock_idx: int,
        side: str,
        executed: th.Tensor,
        price: th.Tensor,
        volume: th.Tensor,
        turnover_percentile: th.Tensor,
    ) -> Optional[dict[str, float | int | str]]:
        total_shares = int(executed.to(th.int64).sum().item())
        if total_shares <= 0:
            return None
        active = executed > 0
        stock_volume = volume.clamp_min(EPS)
        return {
            "stock_idx": stock_idx,
            "side": side,
            "shares": total_shares,
            "notional": float(price.item()) * total_shares,
            "pov": float(
                (executed[active].to(th.float32) / stock_volume).mean().item()
            )
            if active.any()
            else 0.0,
            "turnover_percentile": float(turnover_percentile.item()),
        }

    def _get_perm_impact(self) -> th.Tensor:
        return self.impact_model.get_perm_state_array()

    def get_normalizer_state(self) -> Optional[Dict]:
        return None

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        return None

    def _expand_scalar_state(self, value: object) -> th.Tensor:
        tensor = th.as_tensor(value, dtype=th.float32, device=self.device)
        if tensor.ndim == 0:
            return tensor.repeat(self.num_envs)
        if tensor.ndim == 1:
            if tensor.shape[0] == self.num_envs:
                return tensor.clone()
            if tensor.shape[0] == 1:
                return tensor.repeat(self.num_envs)
        raise ValueError("initial_margin_state scalar fields must be scalar, shape (1,), or shape (num_envs,)")

    def _expand_stock_state(self, value: object) -> th.Tensor:
        tensor = th.as_tensor(value, dtype=th.float32, device=self.device)
        if tensor.ndim == 1:
            if tensor.shape[0] != self.stock_dim:
                raise ValueError("initial_margin_state['stocks'] must have shape (stock_dim,) or (num_envs, stock_dim)")
            return tensor.unsqueeze(0).repeat(self.num_envs, 1)
        if tensor.ndim == 2:
            if tensor.shape == (self.num_envs, self.stock_dim):
                return tensor.clone()
            if tensor.shape == (1, self.stock_dim):
                return tensor.repeat(self.num_envs, 1)
        raise ValueError("initial_margin_state['stocks'] must have shape (stock_dim,), (1, stock_dim), or (num_envs, stock_dim)")

    def get_margin_state(self, env_index: Optional[int] = None) -> Dict:
        if env_index is None and self.num_envs == 1:
            env_index = 0

        if env_index is None:
            return {
                "long_cash": self.long_cash.clone(),
                "loan": self.loan.clone(),
                "long_equity": self.long_equity.clone(),
                "short_limit": self.short_limit.clone(),
                "short_credit": self.short_credit.clone(),
                "short_equity": self.short_equity.clone(),
                "stocks": self.stocks.clone(),
            }

        return {
            "long_cash": self.long_cash[env_index].clone(),
            "loan": self.loan[env_index].clone(),
            "long_equity": self.long_equity[env_index].clone(),
            "short_limit": self.short_limit[env_index].clone(),
            "short_credit": self.short_credit[env_index].clone(),
            "short_equity": self.short_equity[env_index].clone(),
            "stocks": self.stocks[env_index].clone(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[th.Tensor, Dict]:
        if seed is not None:
            th.manual_seed(seed)

        self.time = 0
        price = self.price_array[self.time]
        if self.initial_margin_state is not None:
            state = self.initial_margin_state
            self.long_cash = self._expand_scalar_state(state["long_cash"])
            self.loan = self._expand_scalar_state(state["loan"])
            self.long_equity = self._expand_scalar_state(state["long_equity"])
            self.short_limit = self._expand_scalar_state(state["short_limit"])
            self.short_credit = self._expand_scalar_state(state["short_credit"])
            self.short_equity = self._expand_scalar_state(state["short_equity"])
            self.stocks = self._expand_stock_state(state["stocks"])
        else:
            r = self.long_short_ratio / (self.long_short_ratio + 1)
            equity_long = th.full((self.num_envs,), r * self.initial_capital, dtype=th.float32, device=self.device)
            equity_short = th.full((self.num_envs,), self.initial_capital - r * self.initial_capital, dtype=th.float32, device=self.device)

            self.long_cash = equity_long * self.margin_rate
            self.loan = equity_long * (self.margin_rate - 1)
            self.long_equity = equity_long.clone()

            self.short_limit = equity_short * self.margin_rate
            self.short_credit = (self.margin_rate + 1) * equity_short
            self.short_equity = equity_short.clone()

            self.stocks = self.initial_stocks.repeat(self.num_envs, 1).clone()
        self.stocks_cool_down.zero_()
        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()

        if self.if_random_reset and self.initial_margin_state is None:
            cash_scale = th.rand(self.num_envs, dtype=th.float32, device=self.device) * 0.10 + 0.95
            self.long_cash *= cash_scale
            self.short_credit *= cash_scale

        self.total_asset = self.long_equity + self.short_equity
        self.initial_total_asset = self.total_asset.clone()
        self.cash = (self.long_cash - self.loan).clamp_min(0.0)
        self.episode_return.fill_(1.0)
        self.last_episode_return.fill_(1.0)
        self._equity_count = 1
        self._equity_history.zero_()
        self._equity_history[:, 0] = self.total_asset

        return self.get_state(price), {}

    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict]:
        actions = th.as_tensor(actions, dtype=th.float32, device=self.device).view(self.num_envs, self.action_dim)

        self.time += 1
        self.stocks_cool_down += 1
        done_flag = self.time >= self.max_step

        trade_price = self.price_array[self.time - 1]
        volatility = self.volatility_array[self.time - 1]
        volume = self.volume_array[self.time - 1]
        begin_total_asset = self.long_equity + self.short_equity
        trade_shares = self._calculate_trade_shares(actions, trade_price, volume, begin_total_asset)
        turnover_percentiles = self._calculate_turnover_percentiles(
            self.price_array[min(self.time, self.max_step)],
            self.volume_array[min(self.time, self.max_step)],
        )

        total_traded_value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        total_trade_cost = th.zeros_like(total_traded_value)
        total_buy_value = th.zeros_like(total_traded_value)
        total_sell_value = th.zeros_like(total_traded_value)
        trades: list[dict[str, float | int | str]] = []
        if trade_shares.any():
            stock_order = (
                trade_shares.abs().sum(dim=0).argsort(descending=True).tolist()
            )
        else:
            stock_order = []

        for stock_idx in stock_order:
            if (trade_shares[:, stock_idx] < 0).any():
                trade_value, trade_cost, trade_sides, aggregate_rows = self._execute_sell_direction(
                    stock_idx,
                    -trade_shares[:, stock_idx],
                    trade_price,
                    volatility,
                    volume,
                    turnover_percentiles[stock_idx],
                )
                total_sell_value += trade_value
                total_traded_value += trade_value
                total_trade_cost += trade_cost
                if self.num_envs == 1 and trade_sides:
                    order_shares = int(abs(trade_shares[0, stock_idx].item()))
                    stock_volume = float(volume[stock_idx].item())
                    order_pov = order_shares / stock_volume if stock_volume > 0 else 0.0
                    stock_price = float(trade_price[stock_idx].item())
                    turnover_pct = float(turnover_percentiles[stock_idx].item())
                    for trade_side in trade_sides:
                        trades.append(
                            {
                                "stock_idx": stock_idx,
                                "side": trade_side["side"],
                                "shares": trade_side["shares"],
                                "notional": stock_price * trade_side["shares"],
                                "pov": order_pov,
                                "turnover_percentile": turnover_pct,
                            }
                        )
                elif aggregate_rows:
                    trades.extend(aggregate_rows)

        for stock_idx in reversed(stock_order):
            if (trade_shares[:, stock_idx] > 0).any():
                trade_value, trade_cost, trade_sides, aggregate_rows = self._execute_buy_direction(
                    stock_idx,
                    trade_shares[:, stock_idx],
                    trade_price,
                    volatility,
                    volume,
                    turnover_percentiles[stock_idx],
                )
                total_buy_value += trade_value
                total_traded_value += trade_value
                total_trade_cost += trade_cost
                if self.num_envs == 1 and trade_sides:
                    order_shares = int(trade_shares[0, stock_idx].item())
                    stock_volume = float(volume[stock_idx].item())
                    order_pov = order_shares / stock_volume if stock_volume > 0 else 0.0
                    stock_price = float(trade_price[stock_idx].item())
                    turnover_pct = float(turnover_percentiles[stock_idx].item())
                    for trade_side in trade_sides:
                        trades.append(
                            {
                                "stock_idx": stock_idx,
                                "side": trade_side["side"],
                                "shares": trade_side["shares"],
                                "notional": stock_price * trade_side["shares"],
                                "pov": order_pov,
                                "turnover_percentile": turnover_pct,
                            }
                        )
                elif aggregate_rows:
                    trades.extend(aggregate_rows)

        if self.time % self.margin_adjust_period == 0:
            self._margin_adjust_long(trade_price, volatility, volume)
            self._margin_adjust_short(trade_price, volatility, volume)
        else:
            long_breach = self._check_long_maintenance(trade_price) < self.maintenance_margin
            short_breach = self._check_short_maintenance(trade_price) < self.maintenance_margin
            if long_breach.any():
                self._margin_adjust_long(trade_price, volatility, volume, long_breach)
            if short_breach.any():
                self._margin_adjust_short(trade_price, volatility, volume, short_breach)

        self.impact_model.end_day(
            date_str=self.date_list[self.time],
            stock_symbols=self.stock_symbols,
        )
        new_price = self.price_array[min(self.time, self.max_step)]
        self._update_equities(new_price)
        current_total_equity = self.long_equity + self.short_equity

        profit = current_total_equity - begin_total_asset
        self._append_equity_history(current_total_equity)
        risk = self._rolling_sharpe()
        reward = self.lambda_1 * profit + self.lambda_2 * risk

        self.total_asset = current_total_equity
        self.cash = (self.long_cash - self.loan).clamp_min(0.0)
        self.last_episode_return = self.total_asset / self.initial_total_asset.clamp_min(EPS)
        self.episode_return = self.last_episode_return.clone()

        done = th.full((self.num_envs,), done_flag, dtype=th.bool, device=self.device)
        truncated = th.zeros_like(done)
        turnover = total_traded_value / self.total_asset.clamp_min(EPS)

        if done_flag and self.auto_reset:
            next_state, _ = self.reset(options={"reset_impact_model": True})
        else:
            next_state = self.get_state(new_price)

        info = {
            "turnover": turnover,
            "cost": total_trade_cost,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "cash": self.cash.clone(),
            "episode_return": self.last_episode_return.clone(),
            "trades": trades,
        }
        return next_state, reward, done, truncated, info

    def _calculate_turnover_percentiles(
        self,
        prices: th.Tensor,
        volumes: th.Tensor,
    ) -> th.Tensor:
        gross_notional = prices * volumes
        n = int(gross_notional.shape[0])
        if n <= 1:
            return th.full((n,), 50.0, dtype=th.float32, device=self.device)

        sorted_indices = th.argsort(gross_notional)
        ranks = th.empty(n, dtype=th.float32, device=self.device)
        ranks[sorted_indices] = th.arange(
            1,
            n + 1,
            dtype=th.float32,
            device=self.device,
        )
        return (ranks - 1.0) / float(n - 1) * 100.0

    def _append_equity_history(self, equity: th.Tensor) -> None:
        if self._equity_count < self.sharpe_window + 1:
            self._equity_history[:, self._equity_count] = equity
            self._equity_count += 1
            return
        self._equity_history = th.roll(self._equity_history, shifts=-1, dims=1)
        self._equity_history[:, -1] = equity

    def _rolling_sharpe(self) -> th.Tensor:
        if self._equity_count < 2:
            return th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        equities = self._equity_history[:, : self._equity_count]
        returns = (equities[:, 1:] - equities[:, :-1]) / equities[:, :-1].clamp_min(EPS)
        std = returns.std(dim=1, unbiased=False)
        mean = returns.mean(dim=1)
        sharpe = th.zeros_like(mean)
        valid = std > EPS
        sharpe[valid] = (252**0.5) * mean[valid] / std[valid]
        return sharpe

    def _check_long_maintenance(self, price: th.Tensor) -> th.Tensor:
        adjusted = price.unsqueeze(0) + self._get_perm_impact()
        long_mask = self.stocks > 0
        long_mv = th.where(long_mask, self.stocks * adjusted, th.zeros_like(self.stocks)).sum(dim=1)
        ratio = th.ones(self.num_envs, dtype=th.float32, device=self.device)
        valid = long_mv > 0
        ratio[valid] = self.long_equity[valid] / long_mv[valid].clamp_min(EPS)
        return ratio

    def _check_short_maintenance(self, price: th.Tensor) -> th.Tensor:
        adjusted = price.unsqueeze(0) + self._get_perm_impact()
        short_mask = self.stocks < 0
        short_mv = th.where(short_mask, (-self.stocks) * adjusted, th.zeros_like(self.stocks)).sum(dim=1)
        ratio = th.ones(self.num_envs, dtype=th.float32, device=self.device)
        valid = short_mv > 0
        ratio[valid] = self.short_equity[valid] / short_mv[valid].clamp_min(EPS)
        return ratio

    def _calculate_trade_shares(
        self,
        actions: th.Tensor,
        prices: th.Tensor,
        volume: th.Tensor,
        total_equity: th.Tensor,
    ) -> th.Tensor:
        max_position_value = total_equity.unsqueeze(1).clamp_min(0.0) * self.max_stock_pct
        price_grid = prices.unsqueeze(0).expand(self.num_envs, -1)
        max_shares = th.where(
            price_grid > 0,
            th.div(max_position_value, price_grid, rounding_mode="floor"),
            th.zeros_like(price_grid),
        ).to(th.int32)

        desired_trade = (actions * max_shares.to(dtype=th.float32)).to(th.int32)
        current = self.stocks.to(th.int32)
        target_pos = current + desired_trade
        clamped_pos = th.maximum(-max_shares, th.minimum(target_pos, max_shares))
        trade_shares = clamped_pos - current
        volume_limit = (volume.unsqueeze(0) * self.max_trade_volume_pct).to(th.int32)
        return trade_shares.clamp(min=-volume_limit, max=volume_limit)

    def _execute_sell_direction(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        turnover_percentile: th.Tensor,
    ) -> Tuple[
        th.Tensor,
        th.Tensor,
        list[dict[str, int | str]],
        list[dict[str, float | int | str]],
    ]:
        total_value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        total_cost = th.zeros_like(total_value)
        trade_sides: list[dict[str, int | str]] = []
        aggregate_trades: list[dict[str, float | int | str]] = []
        long_holdings = self.stocks[:, stock_idx].clamp_min(0).to(th.int32)
        sell_long_shares = th.minimum(shares.to(th.int32), long_holdings)
        if sell_long_shares.gt(0).any():
            value, cost, executed = self._sell_long(
                stock_idx,
                sell_long_shares,
                price,
                volatility,
                volume,
            )
            total_value += value
            total_cost += cost
            if self.num_envs == 1 and int(executed[0].item()) > 0:
                trade_sides.append({"side": "sell", "shares": int(executed[0].item())})
            elif self.num_envs > 1:
                aggregate_trade = self._build_aggregate_trade_entry(
                    stock_idx=stock_idx,
                    side="sell",
                    executed=executed,
                    price=price[stock_idx],
                    volume=volume[stock_idx],
                    turnover_percentile=turnover_percentile,
                )
                if aggregate_trade is not None:
                    aggregate_trades.append(aggregate_trade)
        remaining = shares.to(th.int32) - sell_long_shares
        if remaining.gt(0).any():
            value, cost, executed = self._sell_short(
                stock_idx,
                remaining,
                price,
                volatility,
                volume,
            )
            total_value += value
            total_cost += cost
            if self.num_envs == 1 and int(executed[0].item()) > 0:
                trade_sides.append({"side": "short", "shares": int(executed[0].item())})
            elif self.num_envs > 1:
                aggregate_trade = self._build_aggregate_trade_entry(
                    stock_idx=stock_idx,
                    side="short",
                    executed=executed,
                    price=price[stock_idx],
                    volume=volume[stock_idx],
                    turnover_percentile=turnover_percentile,
                )
                if aggregate_trade is not None:
                    aggregate_trades.append(aggregate_trade)
        traded = total_value > 0
        self.stocks_cool_down[traded, stock_idx] = 0
        return total_value, total_cost, trade_sides, aggregate_trades

    def _execute_buy_direction(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        turnover_percentile: th.Tensor,
    ) -> Tuple[
        th.Tensor,
        th.Tensor,
        list[dict[str, int | str]],
        list[dict[str, float | int | str]],
    ]:
        total_value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        total_cost = th.zeros_like(total_value)
        trade_sides: list[dict[str, int | str]] = []
        aggregate_trades: list[dict[str, float | int | str]] = []
        short_holdings = (-self.stocks[:, stock_idx].clamp_max(0)).to(th.int32)
        cover_shares = th.minimum(shares.to(th.int32), short_holdings)
        if cover_shares.gt(0).any():
            value, cost, executed = self._cover_short(
                stock_idx,
                cover_shares,
                price,
                volatility,
                volume,
            )
            total_value += value
            total_cost += cost
            if self.num_envs == 1 and int(executed[0].item()) > 0:
                trade_sides.append({"side": "cover", "shares": int(executed[0].item())})
            elif self.num_envs > 1:
                aggregate_trade = self._build_aggregate_trade_entry(
                    stock_idx=stock_idx,
                    side="cover",
                    executed=executed,
                    price=price[stock_idx],
                    volume=volume[stock_idx],
                    turnover_percentile=turnover_percentile,
                )
                if aggregate_trade is not None:
                    aggregate_trades.append(aggregate_trade)
        remaining = shares.to(th.int32) - cover_shares
        if remaining.gt(0).any():
            value, cost, executed = self._buy_long(
                stock_idx,
                remaining,
                price,
                volatility,
                volume,
            )
            total_value += value
            total_cost += cost
            if self.num_envs == 1 and int(executed[0].item()) > 0:
                trade_sides.append({"side": "buy", "shares": int(executed[0].item())})
            elif self.num_envs > 1:
                aggregate_trade = self._build_aggregate_trade_entry(
                    stock_idx=stock_idx,
                    side="buy",
                    executed=executed,
                    price=price[stock_idx],
                    volume=volume[stock_idx],
                    turnover_percentile=turnover_percentile,
                )
                if aggregate_trade is not None:
                    aggregate_trades.append(aggregate_trade)
        traded = total_value > 0
        self.stocks_cool_down[traded, stock_idx] = 0
        return total_value, total_cost, trade_sides, aggregate_trades

    def _buy_long(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        cost_out = th.zeros_like(value)
        executed = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        px = price[stock_idx].expand(self.num_envs)
        vol = volatility[stock_idx].expand(self.num_envs)
        volm = volume[stock_idx].expand(self.num_envs)
        maintenance_ok = self._check_long_maintenance(price) > self.maintenance_warning
        valid = maintenance_ok & (px > 0) & (shares > 0)
        if valid.any():
            max_affordable = th.div(self.long_cash[valid], px[valid], rounding_mode="floor").to(th.int32)
            actual_shares = th.minimum(shares[valid], max_affordable)
            nonzero = actual_shares > 0
            if nonzero.any():
                valid_idx = th.where(valid)[0][nonzero]
                actual_shares = actual_shares[nonzero]
                impact_cost, _ = self.impact_model.apply_trade(
                    trade_size=actual_shares,
                    price=px[valid_idx],
                    volatility=vol[valid_idx],
                    volume=volm[valid_idx],
                    stock_index=stock_idx,
                    env_indices=valid_idx,
                )
                total_cost = actual_shares.to(th.float32) * px[valid_idx] + impact_cost
                affordable = total_cost <= self.long_cash[valid_idx]
                if affordable.any():
                    idx = valid_idx[affordable]
                    qty = actual_shares[affordable].to(th.float32)
                    self.long_cash[idx] -= total_cost[affordable]
                    self.long_equity[idx] -= impact_cost[affordable]
                    self.stocks[idx, stock_idx] += qty
                    value[idx] = qty * px[idx]
                    cost_out[idx] = impact_cost[affordable]
                    executed[idx] = actual_shares[affordable]
        return value, cost_out, executed

    def _sell_long(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        cost_out = th.zeros_like(value)
        executed = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        px = price[stock_idx].expand(self.num_envs)
        vol = volatility[stock_idx].expand(self.num_envs)
        volm = volume[stock_idx].expand(self.num_envs)
        held = self.stocks[:, stock_idx].clamp_min(0).to(th.int32)
        actual_shares = th.minimum(shares, held)
        valid = (actual_shares > 0) & (px > 0)
        if not valid.any():
            return value, cost_out, executed
        idx = th.where(valid)[0]
        impact_cost, _ = self.impact_model.apply_trade(
            trade_size=-actual_shares[idx],
            price=px[idx],
            volatility=vol[idx],
            volume=volm[idx],
            stock_index=stock_idx,
            env_indices=idx,
        )
        proceeds = actual_shares[idx].to(th.float32) * px[idx] - impact_cost
        positive = proceeds > 0
        if positive.any():
            use_idx = idx[positive]
            qty = actual_shares[use_idx].to(th.float32)
            self.long_cash[use_idx] += proceeds[positive]
            self.long_equity[use_idx] -= impact_cost[positive]
            self.stocks[use_idx, stock_idx] -= qty
            value[use_idx] = qty * px[use_idx]
            cost_out[use_idx] = impact_cost[positive]
            executed[use_idx] = actual_shares[use_idx]
        return value, cost_out, executed

    def _sell_short(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        cost_out = th.zeros_like(value)
        executed = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        px = price[stock_idx].expand(self.num_envs)
        vol = volatility[stock_idx].expand(self.num_envs)
        volm = volume[stock_idx].expand(self.num_envs)
        maintenance_ok = self._check_short_maintenance(price) > self.maintenance_warning
        valid = maintenance_ok & (px > 0) & (shares > 0)
        if not valid.any():
            return value, cost_out, executed
        idx = th.where(valid)[0]
        max_by_limit = th.div(
            self.short_limit[idx],
            px[idx],
            rounding_mode="floor",
        ).to(th.int32)
        actual_shares = th.minimum(shares[idx], max_by_limit)
        nonzero = actual_shares > 0
        if nonzero.any():
            idx = idx[nonzero]
            actual_shares = actual_shares[nonzero]
            market_value = actual_shares.to(th.float32) * px[idx]
            impact_cost, _ = self.impact_model.apply_trade(
                trade_size=-actual_shares,
                price=px[idx],
                volatility=vol[idx],
                volume=volm[idx],
                stock_index=stock_idx,
                env_indices=idx,
            )
            self.short_limit[idx] -= market_value
            self.short_credit[idx] -= impact_cost
            self.short_equity[idx] -= impact_cost
            self.stocks[idx, stock_idx] -= actual_shares.to(th.float32)
            value[idx] = market_value
            cost_out[idx] = impact_cost
            executed[idx] = actual_shares
        return value, cost_out, executed

    def _cover_short(
        self,
        stock_idx: int,
        shares: th.Tensor,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        cost_out = th.zeros_like(value)
        executed = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        px = price[stock_idx].expand(self.num_envs)
        vol = volatility[stock_idx].expand(self.num_envs)
        volm = volume[stock_idx].expand(self.num_envs)
        held = (-self.stocks[:, stock_idx].clamp_max(0)).to(th.int32)
        actual_shares = th.minimum(shares, held)
        valid = (actual_shares > 0) & (px > 0)
        if valid.any():
            idx = th.where(valid)[0]
            impact_cost, _ = self.impact_model.apply_trade(
                trade_size=actual_shares[idx],
                price=px[idx],
                volatility=vol[idx],
                volume=volm[idx],
                stock_index=stock_idx,
                env_indices=idx,
            )
            market_value = actual_shares[idx].to(th.float32) * px[idx]
            self.short_limit[idx] += market_value
            self.short_credit[idx] -= impact_cost
            self.short_equity[idx] -= impact_cost
            self.stocks[idx, stock_idx] += actual_shares[idx].to(th.float32)
            value[idx] = market_value
            cost_out[idx] = impact_cost
            executed[idx] = actual_shares[idx]
        return value, cost_out, executed

    def _update_equities(self, price: th.Tensor) -> None:
        adjusted = price.unsqueeze(0) + self._get_perm_impact()
        long_mv = th.where(self.stocks > 0, self.stocks * adjusted, th.zeros_like(self.stocks)).sum(dim=1)
        self.long_equity = self.long_cash + long_mv - self.loan

        short_mv = th.where(self.stocks < 0, (-self.stocks) * adjusted, th.zeros_like(self.stocks)).sum(dim=1)
        self.short_equity = self.short_credit - self.short_limit - short_mv

    def _margin_adjust_long(
        self,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        env_mask: Optional[th.Tensor] = None,
    ) -> None:
        self._update_equities(price)
        if env_mask is None:
            env_mask = th.ones(self.num_envs, dtype=th.bool, device=self.device)
        loan_diff = self.long_equity - self.loan
        profit_mask = env_mask & (loan_diff > 0)
        if profit_mask.any():
            self.long_cash[profit_mask] += loan_diff[profit_mask]
            self.loan[profit_mask] = self.long_equity[profit_mask]

        loss_mask = env_mask & (loan_diff < 0)
        if loss_mask.any():
            shortfall = (-loan_diff[loss_mask]).clone()
            idx = th.where(loss_mask)[0]
            can_pay = self.long_cash[idx] >= shortfall
            if can_pay.any():
                pay_idx = idx[can_pay]
                self.long_cash[pay_idx] -= shortfall[can_pay]
                self.loan[pay_idx] -= shortfall[can_pay]

            need_sales = ~can_pay
            if need_sales.any():
                # Faithful rank-based liquidation: iterate stocks sorted by
                # value per env and sell until ``remaining`` is covered.
                # The ``active`` mask below gives each env its own early-break
                # equivalent — envs with remaining == 0 are skipped at the
                # next rank, matching the single-env ``if cash >= remaining:
                # break`` semantics.
                sale_idx = idx[need_sales]
                remaining = shortfall[need_sales] - self.long_cash[sale_idx]
                self.long_cash[sale_idx] = 0.0
                adjusted = price.unsqueeze(0) + self._get_perm_impact()
                long_values = th.where(
                    self.stocks[sale_idx] > 0,
                    self.stocks[sale_idx] * adjusted[sale_idx],
                    th.full_like(self.stocks[sale_idx], float("inf")),
                )
                order = long_values.argsort(dim=1)
                for rank in range(self.stock_dim):
                    target_stock = order[:, rank]
                    qty = self.stocks[sale_idx, target_stock].to(th.int32)
                    active = remaining > 0
                    if not active.any():
                        break
                    active_env = sale_idx[active]
                    active_stock = target_stock[active]
                    shares = qty[active]
                    for unique_stock in active_stock.unique():
                        stock_mask = active_stock == unique_stock
                        if stock_mask.any():
                            self._sell_long(
                                int(unique_stock.item()),
                                self._scatter_qty(active_env[stock_mask], shares[stock_mask]),
                                price,
                                volatility,
                                volume,
                            )
                    settled = th.minimum(self.long_cash[sale_idx], remaining)
                    self.long_cash[sale_idx] -= settled
                    remaining -= settled
                self.loan[sale_idx] -= shortfall[need_sales]

        self._update_equities(price)

    def _margin_adjust_short(
        self,
        price: th.Tensor,
        volatility: th.Tensor,
        volume: th.Tensor,
        env_mask: Optional[th.Tensor] = None,
    ) -> None:
        self._update_equities(price)
        if env_mask is None:
            env_mask = th.ones(self.num_envs, dtype=th.bool, device=self.device)
        adjusted = price.unsqueeze(0) + self._get_perm_impact()
        short_mv = th.where(self.stocks < 0, (-self.stocks) * adjusted, th.zeros_like(self.stocks)).sum(dim=1)
        borrow_used = self.short_limit + short_mv
        max_borrow = self.margin_rate * self.short_equity
        borrow_diff = max_borrow - borrow_used

        profit_mask = env_mask & (borrow_diff > 0)
        if profit_mask.any():
            self.short_limit[profit_mask] += borrow_diff[profit_mask]
            self.short_credit[profit_mask] += borrow_diff[profit_mask]

        loss_mask = env_mask & (borrow_diff < 0)
        if loss_mask.any():
            shortfall = (-borrow_diff[loss_mask]).clone()
            idx = th.where(loss_mask)[0]
            can_pay = self.short_limit[idx] >= shortfall
            if can_pay.any():
                pay_idx = idx[can_pay]
                self.short_limit[pay_idx] -= shortfall[can_pay]
                self.short_credit[pay_idx] -= shortfall[can_pay]

            need_cover = ~can_pay
            if need_cover.any():
                cover_idx = idx[need_cover]
                remaining = shortfall[need_cover] - self.short_limit[cover_idx]
                self.short_limit[cover_idx] = 0.0
                short_values = th.where(
                    self.stocks[cover_idx] < 0,
                    (-self.stocks[cover_idx]) * adjusted[cover_idx],
                    th.full_like(self.stocks[cover_idx], float("inf")),
                )
                order = short_values.argsort(dim=1)
                for rank in range(self.stock_dim):
                    target_stock = order[:, rank]
                    qty = (-self.stocks[cover_idx, target_stock]).to(th.int32)
                    active = remaining > 0
                    if not active.any():
                        break
                    active_env = cover_idx[active]
                    active_stock = target_stock[active]
                    shares = qty[active]
                    for unique_stock in active_stock.unique():
                        stock_mask = active_stock == unique_stock
                        if stock_mask.any():
                            self._cover_short(
                                int(unique_stock.item()),
                                self._scatter_qty(active_env[stock_mask], shares[stock_mask]),
                                price,
                                volatility,
                                volume,
                            )
                    settled = th.minimum(self.short_limit[cover_idx], remaining)
                    self.short_limit[cover_idx] -= settled
                    remaining -= settled
                self.short_credit[cover_idx] -= shortfall[need_cover]

        self._update_equities(price)

    def _scatter_qty(self, env_idx: th.Tensor, shares: th.Tensor) -> th.Tensor:
        qty = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        qty[env_idx] = shares.to(th.int32)
        return qty

    def get_state(self, price: th.Tensor) -> th.Tensor:
        scale = 2**-12
        perm_impact = self._get_perm_impact()
        head = th.stack(
            [
                self.long_cash,
                self.loan,
                self.long_equity,
                self.short_limit,
                self.short_credit,
                self.short_equity,
            ],
            dim=1,
        ) * scale
        state = th.cat(
            [
                head,
                price.unsqueeze(0).expand(self.num_envs, -1) * scale,
                self.stocks * scale,
                perm_impact * scale,
                self.stocks_cool_down,
                self.tech_array[min(self.time, self.max_step)].unsqueeze(0).expand(self.num_envs, -1),
            ],
            dim=1,
        )
        return state.to(dtype=th.float32)

    def render(self, mode: str = "human") -> th.Tensor:
        return self.total_asset