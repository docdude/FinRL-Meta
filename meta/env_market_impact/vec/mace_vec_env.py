from __future__ import annotations

import os
from typing import Dict
from typing import Optional
from typing import Tuple

import torch as th

from meta.env_market_impact.backtest_vec_config import VecMACEEnvParams

from .common import EPS
from .common import TorchRunningMeanStd
from .common import resolve_device
from .tensor_impact import TensorImpactBase
from .tensor_impact import TensorImpactConfig
from .tensor_impact import TensorSqrtImpactModel


class MACEVecEnv:
    _LEGACY_VEC_NORMALIZE_FILENAME = "vec_normalize.pt"

    def __init__(
        self,
        config: Dict,
        params: Optional[VecMACEEnvParams] = None,
        num_envs: int = 128,
        gpu_id: int = -1,
        device: Optional[str] = None,
        impact_config: Optional[TensorImpactConfig] = None,
        impact_model: Optional[TensorImpactBase] = None,
        initial_capital: float = 1e6,
        initial_stocks: Optional[th.Tensor] = None,
        normalizer_state_path: Optional[str] = None,
        freeze_loaded_normalizer: bool = False,
        if_random_reset: bool = False,
        auto_reset: bool = True,
    ) -> None:
        params = params or VecMACEEnvParams()
        self.params = params
        self.device = resolve_device(gpu_id=gpu_id, device=device)
        self.num_envs = int(num_envs)
        self.auto_reset = auto_reset
        self.if_random_reset = if_random_reset

        self.date_list = list(config["date_list"])
        self.price_array = th.as_tensor(config["price_array"], dtype=th.float32, device=self.device)
        self.tech_array = th.as_tensor(config["tech_array"], dtype=th.float32, device=self.device) * (2**-7)
        self.volatility_array = th.as_tensor(config["volatility_array"], dtype=th.float32, device=self.device)
        self.volume_array = th.as_tensor(config["volume_array"], dtype=th.float32, device=self.device)
        self.adv20_array = th.as_tensor(config["adv20_array"], dtype=th.float32, device=self.device)
        self.tbill_rates = th.as_tensor(config["tbill_rates"], dtype=th.float32, device=self.device)
        self.stock_symbols = list(config["tic_list"])

        self.stock_dim = self.price_array.shape[1]
        self.max_stock_pct = params.max_stock_pct
        self.max_trade_volume_pct = params.max_trade_volume_pct
        self.reward_scaling = params.reward_scaling
        self.initial_capital = float(initial_capital)
        base_stocks = th.zeros(self.stock_dim, dtype=th.float32, device=self.device)
        self.initial_stocks = base_stocks if initial_stocks is None else initial_stocks.to(self.device, dtype=th.float32)
        self.normalizer_state_path = normalizer_state_path
        self.freeze_loaded_normalizer = bool(freeze_loaded_normalizer)
        self.sharpe_window = params.sharpe_window
        self.alpha = 1.0 / params.horizon
        self.eta_dd = params.eta_dd

        self.include_permanent_impact_in_state = params.include_permanent_impact_in_state
        self.include_cooldown_in_state = params.include_cooldown_in_state
        self.include_tbill_in_state = params.include_tbill_in_state

        self.state_dim = 1 + (2 * self.stock_dim) + self.tech_array.shape[1] + self.stock_dim
        if self.include_permanent_impact_in_state:
            self.state_dim += self.stock_dim
        if self.include_cooldown_in_state:
            self.state_dim += self.stock_dim
        if self.include_tbill_in_state:
            self.state_dim += 1

        self.env_name = "MACEVecEnv-v1"
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.max_step = int(self.price_array.shape[0] - 1)
        self.target_return = float("inf")

        self.obs_clip = params.obs_clip
        self._obs_norm_update = params.obs_norm_update
        if params.use_obs_normalizer:
            self.obs_normalizer = TorchRunningMeanStd((self.state_dim,), device=self.device)
        else:
            self.obs_normalizer = None
        if self.obs_normalizer is not None and self.normalizer_state_path and os.path.isfile(self.normalizer_state_path):
            self.load_normalizer_state(
                self.normalizer_state_path,
                freeze=self.freeze_loaded_normalizer,
            )

        self.impact_model = impact_model if impact_model is not None else TensorSqrtImpactModel(
            num_envs=self.num_envs,
            stock_dim=self.stock_dim,
            device=self.device,
            config=impact_config,
        )

        self.time = 0
        self.cash = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.stocks = th.zeros((self.num_envs, self.stock_dim), dtype=th.float32, device=self.device)
        self.stocks_cool_down = th.zeros_like(self.stocks)
        self.mu_prev = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.m2_prev = th.full((self.num_envs,), 1e-6, dtype=th.float32, device=self.device)
        self.dd = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        self.total_asset = th.full((self.num_envs,), self.initial_capital, dtype=th.float32, device=self.device)
        self.peak = self.total_asset.clone()
        self.episode_return = th.ones(self.num_envs, dtype=th.float32, device=self.device)
        self.last_episode_return = self.episode_return.clone()
        self._obs_buffer = th.empty(
            (self.num_envs, self.state_dim),
            dtype=th.float32,
            device=self.device,
        )

    def close(self) -> None:
        if self.normalizer_state_path:
            self.save_normalizer_state(self.normalizer_state_path)
        return None

    def get_impact_history(self):
        return self.impact_model.get_impact_history()

    def _build_aggregate_trade_entries(
        self,
        shares: th.Tensor,
        prices: th.Tensor,
        volume: th.Tensor,
        turnover_percentiles: th.Tensor,
        side: str,
    ) -> list[dict[str, float | int | str]]:
        trades: list[dict[str, float | int | str]] = []
        active_stocks = th.where(shares.sum(dim=0) > 0)[0].tolist()
        for stock_idx in active_stocks:
            executed = shares[:, stock_idx]
            active = executed > 0
            total_shares = int(executed.sum().item())
            if total_shares <= 0:
                continue
            stock_volume = volume[stock_idx].clamp_min(EPS)
            trades.append(
                {
                    "stock_idx": stock_idx,
                    "side": side,
                    "shares": total_shares,
                    "notional": float(
                        (executed * prices[:, stock_idx]).sum().item()
                    ),
                    "pov": float(
                        (executed[active].to(th.float32) / stock_volume).mean().item()
                    )
                    if active.any()
                    else 0.0,
                    "turnover_percentile": float(
                        turnover_percentiles[stock_idx].item()
                    ),
                }
            )
        return trades

    def _get_perm_impact(self) -> th.Tensor:
        return self.impact_model.get_perm_state_array()

    def get_normalizer_state(self) -> Optional[Dict]:
        if self.obs_normalizer is None:
            return None
        state = self.obs_normalizer.get_state()
        return {
            "mean": state.mean.detach().cpu().numpy().copy(),
            "var": state.var.detach().cpu().numpy().copy(),
            "count": state.count,
        }

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        if self.obs_normalizer is None:
            return
        tensor_state = {
            "mean": th.as_tensor(state["mean"], dtype=th.float32, device=self.device),
            "var": th.as_tensor(state["var"], dtype=th.float32, device=self.device),
            "count": float(state["count"]),
        }
        self.obs_normalizer.set_state(tensor_state)
        if freeze:
            self._obs_norm_update = False

    def save_normalizer_state(self, path: str) -> None:
        state = self.get_normalizer_state()
        if state is None:
            return
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        th.save(state, path)

    def load_normalizer_state(self, path: str, freeze: bool = True) -> None:
        if self.obs_normalizer is None or not os.path.isfile(path):
            return
        state = th.load(path, map_location="cpu", weights_only=False)
        self.set_normalizer_state(state, freeze=freeze)

    def _resolve_normalizer_io_path(self, path: str) -> str:
        if (
            os.path.basename(path) == self._LEGACY_VEC_NORMALIZE_FILENAME
            and self.normalizer_state_path
        ):
            return self.normalizer_state_path
        return path

    def save(self, path: str) -> None:
        self.save_normalizer_state(self._resolve_normalizer_io_path(path))

    def load(self, path: str, verbose: bool = False) -> None:
        resolved_path = self._resolve_normalizer_io_path(path)
        if not os.path.isfile(resolved_path):
            return
        self.load_normalizer_state(
            resolved_path,
            freeze=self.freeze_loaded_normalizer,
        )
        if verbose:
            print(
                f"| Loaded MACE normalizer state from {resolved_path}",
                flush=True,
            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[th.Tensor, Dict]:
        if seed is not None:
            th.manual_seed(seed)

        self.time = 0
        self.cash.fill_(self.initial_capital)
        self.stocks = self.initial_stocks.repeat(self.num_envs, 1).clone()
        self.stocks_cool_down.zero_()
        self.mu_prev.zero_()
        self.m2_prev.fill_(1e-6)
        self.dd.zero_()
        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()

        if self.if_random_reset:
            self.cash *= th.rand(self.num_envs, dtype=th.float32, device=self.device) * 0.10 + 0.95

        price = self.price_array[self.time].unsqueeze(0) + self._get_perm_impact()
        self.total_asset = self._calculate_total_asset(price)
        self.peak = self.total_asset.clone()
        self.episode_return.fill_(1.0)
        state = self.get_state(price, price, self.total_asset)
        return state, {}

    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict]:
        actions = th.as_tensor(actions, dtype=th.float32, device=self.device).view(self.num_envs, self.action_dim)
        prev_base_price = self.price_array[self.time]
        prev_adjusted_prices = prev_base_price.unsqueeze(0) + self._get_perm_impact()

        self.time += 1
        self.stocks_cool_down += 1
        terminated = self.time >= self.max_step

        base_price = self.price_array[self.time]
        adjusted_prices = base_price.unsqueeze(0) + self._get_perm_impact()
        volatility = self.volatility_array[self.time]
        volume = self.volume_array[self.time]
        trade_shares = self._calculate_trade_shares(actions, adjusted_prices, volume)
        turnover_percentiles = self._calculate_turnover_percentiles(
            base_price,
            self.volume_array[self.time],
        )

        total_traded_value = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        total_trade_cost = th.zeros_like(total_traded_value)
        total_buy_value = th.zeros_like(total_traded_value)
        total_sell_value = th.zeros_like(total_traded_value)
        trades: list[dict[str, float | int | str]] = []

        sell_shares = (-trade_shares.clamp(max=0)).to(dtype=th.float32)
        if sell_shares.gt(0).any():
            sell_cost, _ = self.impact_model.apply_trades_batched(
                -sell_shares,
                adjusted_prices,
                volatility,
                volume,
            )
            sell_proceeds = sell_shares * adjusted_prices - sell_cost
            self.stocks -= sell_shares
            self.cash += sell_proceeds.sum(dim=1)
            self.stocks_cool_down = th.where(
                sell_shares > 0,
                th.zeros_like(self.stocks_cool_down),
                self.stocks_cool_down,
            )
            total_sell_value += sell_proceeds.sum(dim=1)
            total_traded_value += sell_proceeds.sum(dim=1)
            total_trade_cost += sell_cost.sum(dim=1)

            if self.num_envs == 1:
                for stock_idx in th.where(sell_shares[0] > 0)[0].tolist():
                    executed = int(sell_shares[0, stock_idx].item())
                    stock_volume = float(volume[stock_idx].item())
                    trades.append(
                        {
                            "stock_idx": stock_idx,
                            "side": "sell",
                            "shares": executed,
                            "notional": float(adjusted_prices[0, stock_idx].item())
                            * executed,
                            "pov": executed / stock_volume if stock_volume > 0 else 0.0,
                            "turnover_percentile": float(
                                turnover_percentiles[stock_idx].item()
                            ),
                        }
                    )
            else:
                trades.extend(
                    self._build_aggregate_trade_entries(
                        sell_shares,
                        adjusted_prices,
                        volume,
                        turnover_percentiles,
                        "sell",
                    )
                )

        buy_shares = trade_shares.clamp(min=0).to(dtype=th.float32)
        if buy_shares.gt(0).any():
            preview_buy_cost, _ = self.impact_model.preview_trades_batched(
                buy_shares,
                adjusted_prices,
                volatility,
                volume,
            )
            requested_buy_total = buy_shares * adjusted_prices + preview_buy_cost
            available_cash = self.cash.clone()
            accepted_buy_shares = th.zeros_like(buy_shares)

            for stock_idx in range(self.stock_dim):
                stock_shares = buy_shares[:, stock_idx]
                stock_total = requested_buy_total[:, stock_idx]
                can_buy = (stock_shares > 0) & (stock_total <= available_cash)
                accepted_buy_shares[can_buy, stock_idx] = stock_shares[can_buy]
                available_cash[can_buy] -= stock_total[can_buy]

            if accepted_buy_shares.gt(0).any():
                buy_cost, _ = self.impact_model.apply_trades_batched(
                    accepted_buy_shares,
                    adjusted_prices,
                    volatility,
                    volume,
                )
                buy_total = accepted_buy_shares * adjusted_prices + buy_cost
                self.stocks += accepted_buy_shares
                self.cash -= buy_total.sum(dim=1)
                self.stocks_cool_down = th.where(
                    accepted_buy_shares > 0,
                    th.zeros_like(self.stocks_cool_down),
                    self.stocks_cool_down,
                )
                total_buy_value += buy_total.sum(dim=1)
                total_traded_value += buy_total.sum(dim=1)
                total_trade_cost += buy_cost.sum(dim=1)

                if self.num_envs == 1:
                    for stock_idx in th.where(accepted_buy_shares[0] > 0)[0].tolist():
                        executed = int(accepted_buy_shares[0, stock_idx].item())
                        stock_volume = float(volume[stock_idx].item())
                        trades.append(
                            {
                                "stock_idx": stock_idx,
                                "side": "buy",
                                "shares": executed,
                                "notional": float(adjusted_prices[0, stock_idx].item())
                                * executed,
                                "pov": executed / stock_volume if stock_volume > 0 else 0.0,
                                "turnover_percentile": float(
                                    turnover_percentiles[stock_idx].item()
                                ),
                            }
                        )
                else:
                    trades.extend(
                        self._build_aggregate_trade_entries(
                            accepted_buy_shares,
                            adjusted_prices,
                            volume,
                            turnover_percentiles,
                            "buy",
                        )
                    )

        self.cash += self.cash * self._calc_rf_rate(self.time)
        self.impact_model.end_day(
            date_str=self.date_list[self.time],
            stock_symbols=self.stock_symbols,
        )
        adjusted_prices_post = base_price.unsqueeze(0) + self._get_perm_impact()
        current_total_asset = self._calculate_total_asset(adjusted_prices_post)
        self.total_asset = current_total_asset

        if terminated:
            reward_asset = current_total_asset
        else:
            tomorrow_adjusted_prices = (
                self.price_array[self.time + 1].unsqueeze(0) + self._get_perm_impact()
            )
            reward_asset = self._calculate_total_asset(tomorrow_adjusted_prices)

        reward = self._calculate_reward(reward_asset)
        self.last_episode_return = current_total_asset / self.initial_capital
        self.episode_return = self.last_episode_return.clone()
        turnover = total_traded_value / self.total_asset.clamp_min(EPS)
        done = th.full((self.num_envs,), terminated, dtype=th.bool, device=self.device)
        truncated = th.zeros_like(done)

        if terminated and self.auto_reset:
            next_state, _ = self.reset(options={"reset_impact_model": True})
        elif terminated:
            next_state = th.zeros((self.num_envs, self.state_dim), dtype=th.float32, device=self.device)
        else:
            next_state = self.get_state(adjusted_prices_post, prev_adjusted_prices, self.total_asset)

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

    def _calculate_trade_shares(
        self,
        actions: th.Tensor,
        adjusted_prices: th.Tensor,
        volume: th.Tensor,
    ) -> th.Tensor:
        max_stocks_per_position = self._calculate_max_stock_per_position(adjusted_prices)
        desired_shares = (actions * max_stocks_per_position.to(dtype=th.float32)).to(th.int32)
        current = self.stocks.to(th.int32)
        limit = max_stocks_per_position.to(th.int32)

        # Mirror the original env's if/elif/else: the forced-sell branch only
        # runs when desired_shares >= 0. Otherwise a negative action on a
        # position already over the limit would be silently overwritten.
        trade_shares = th.zeros_like(desired_shares)
        sell_amount = th.minimum((-desired_shares).clamp_min(0), current.clamp_min(0))
        buy_cap = (limit - current).clamp_min(0)
        buy_amount = th.minimum(desired_shares.clamp_min(0), buy_cap)
        forced_sell = limit - current

        sell_branch = desired_shares < 0
        forced_branch = (~sell_branch) & (current > limit)
        buy_branch = (~sell_branch) & (~forced_branch)

        trade_shares = th.where(sell_branch, -sell_amount, trade_shares)
        trade_shares = th.where(forced_branch, forced_sell, trade_shares)
        trade_shares = th.where(buy_branch, buy_amount, trade_shares)

        volume_limit = (volume.unsqueeze(0) * self.max_trade_volume_pct).to(th.int32)
        return trade_shares.clamp(min=-volume_limit, max=volume_limit)

    def _calc_rf_rate(self, time_index: int) -> th.Tensor:
        return (1 + self.tbill_rates[time_index] / 100.0).pow(1.0 / 252.0) - 1.0

    def _differential_sharpe(self, returns_t: th.Tensor) -> th.Tensor:
        mu_next = (1 - self.alpha) * self.mu_prev + self.alpha * returns_t
        m2_next = (1 - self.alpha) * self.m2_prev + self.alpha * returns_t.square()

        var_next = (m2_next - mu_next.square()).clamp_min(EPS)
        sigma_next = var_next.sqrt()

        var_prev = (self.m2_prev - self.mu_prev.square()).clamp_min(EPS)
        sigma_prev = var_prev.sqrt()
        sr_prev = self.mu_prev / (sigma_prev + EPS)

        x = (returns_t - self.mu_prev) / sigma_next
        dsr_t = x - 0.5 * sr_prev * x.square()

        self.mu_prev = mu_next
        self.m2_prev = m2_next
        return dsr_t

    def _calculate_reward(self, reward_asset: th.Tensor) -> th.Tensor:
        returns_t = reward_asset / self.total_asset.clamp_min(EPS) - 1.0
        dsr_reward = self._differential_sharpe(returns_t)
        self.peak = th.maximum(self.peak, reward_asset)
        dd_new = (self.peak - reward_asset) / self.peak.clamp_min(EPS)
        delta_dd = (dd_new - self.dd).clamp_min(0.0)
        self.dd = dd_new
        return (dsr_reward - self.eta_dd * delta_dd.square()) * self.reward_scaling

    def _calculate_total_asset(self, adjusted_prices: th.Tensor) -> th.Tensor:
        total = self.cash + (self.stocks * adjusted_prices).sum(dim=1)
        return total.clamp_min(EPS)

    def _calculate_max_stock_per_position(self, current_prices: th.Tensor) -> th.Tensor:
        portfolio_value = self.cash + (self.stocks * current_prices).sum(dim=1)
        max_position_value = portfolio_value.unsqueeze(1) * self.max_stock_pct
        return th.where(
            current_prices > 0,
            th.div(max_position_value, current_prices, rounding_mode="floor"),
            th.zeros_like(current_prices),
        ).to(th.int32)

    def get_state(
        self,
        adjusted_prices: th.Tensor,
        prev_adjusted_prices: th.Tensor,
        end_total_asset: th.Tensor,
    ) -> th.Tensor:
        price_ret_1d = th.log((adjusted_prices + EPS) / (prev_adjusted_prices + EPS))
        position_value_pct = (self.stocks * adjusted_prices) / end_total_asset.unsqueeze(1).clamp_min(EPS)
        shares_over_adv = self.stocks / self.adv20_array[self.time].unsqueeze(0).clamp_min(EPS)
        perm_impact = self._get_perm_impact()
        impact_bps = perm_impact / (adjusted_prices + EPS) * 1e4
        cash_pct = (self.cash / end_total_asset.clamp_min(EPS)).unsqueeze(1)
        obs = self._obs_buffer
        offset = 0
        obs[:, offset : offset + 1] = cash_pct
        offset += 1
        obs[:, offset : offset + self.stock_dim] = price_ret_1d
        offset += self.stock_dim
        obs[:, offset : offset + self.stock_dim] = position_value_pct
        offset += self.stock_dim

        tech = self.tech_array[self.time].unsqueeze(0).expand(self.num_envs, -1)
        obs[:, offset : offset + tech.shape[1]] = tech
        offset += tech.shape[1]
        obs[:, offset : offset + self.stock_dim] = shares_over_adv
        offset += self.stock_dim

        if self.include_permanent_impact_in_state:
            obs[:, offset : offset + self.stock_dim] = impact_bps
            offset += self.stock_dim
        if self.include_cooldown_in_state:
            obs[:, offset : offset + self.stock_dim] = (
                self.stocks_cool_down.clamp(0, 10.0) / 10.0
            )
            offset += self.stock_dim
        if self.include_tbill_in_state:
            obs[:, offset : offset + 1] = (
                self._calc_rf_rate(self.time).expand(self.num_envs).unsqueeze(1)
            )

        if self.obs_normalizer is not None:
            if self._obs_norm_update:
                self.obs_normalizer.update(obs)
            obs = (obs - self.obs_normalizer.mean) / (self.obs_normalizer.var + EPS).sqrt()
            obs = obs.clamp(-self.obs_clip, self.obs_clip)
            return obs
        return obs.clone()