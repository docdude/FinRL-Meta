import os
import torch as th
import numpy as np
import numpy.random as rd
from typing import Tuple

ARY = np.ndarray


EPS = 1e-8


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113, stock_cd_steps=0, min_stock_rate=0.0,
                 npz_path=None, use_position_cap=False, max_stock_pct=0.02,
                 use_drawdown_penalty=False, eta_dd=0.5,
                 reward_scale=2**-12,
                 use_dsr=False, dsr_horizon=20,
                 debug_dsr=False):
        self.df_pwd = './China_A_shares.pandas.dataframe'
        self.npz_pwd = npz_path if npz_path is not None else './China_A_shares.numpy.npz'

        self.close_ary, self.tech_ary = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        # print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        # print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")

        self.max_stock = max_stock
        self.cost_pct = cost_pct
        self.reward_scale = reward_scale
        self.initial_amount = initial_amount
        self.gamma = gamma

        # A1 (capital-invariant action scaling): when use_position_cap is True,
        # per-step per-stock max shares = floor(max_stock_pct * V / price), so
        # actions scale with current portfolio value rather than a fixed constant.
        # Over-cap positions are force-sold at the start of step() to prevent
        # un-sellable inventory from accumulating.
        self.use_position_cap = use_position_cap
        self.max_stock_pct = max_stock_pct

        # A3 (drawdown penalty)
        self.use_drawdown_penalty = use_drawdown_penalty
        self.eta_dd = eta_dd

        # A2 (Differential Sharpe Ratio): when use_dsr is True, the per-step
        # reward is DSR (scale-invariant) instead of raw ΔV * reward_scale.
        # DSR uses EMA moments with alpha = 1/dsr_horizon.
        self.use_dsr = use_dsr
        self.dsr_alpha = 1.0 / dsr_horizon
        self.debug_dsr = debug_dsr

        # stock_cd: cooling down period (simulates T+1 settlement)
        # After buying a stock, must wait stock_cd_steps before selling
        self.stock_cd_steps = stock_cd_steps
        # min_stock_rate: minimum stock holding ratio before selling is allowed
        # Prevents selling small positions (reduces overtrading)
        self.min_stock_rate = min_stock_rate

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.if_random_reset = True

        self.amount = None
        self.shares = None
        self.stock_cd = None  # cooling down counter per stock
        self.shares_num = self.close_ary.shape[1]
        amount_dim = 1

        # environment information
        self.env_name = 'StockTradingEnv-v2'
        self.state_dim = self.shares_num + self.close_ary.shape[1] + self.tech_ary.shape[1] + amount_dim
        self.action_dim = self.shares_num
        self.if_discrete = False
        self.max_step = self.close_ary.shape[0] - 1
        self.target_return = +np.inf

    def reset(self) -> Tuple[ARY, dict]:
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        # Initialize stock cooling down counters (0 = can trade)
        self.stock_cd = np.zeros(self.shares_num, dtype=np.int32)

        self.rewards = []
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        self.peak = self.total_asset
        self.dd = 0.0
        # DSR EMA state
        self.dsr_mu = 0.0
        self.dsr_m2 = 1e-6
        return self.get_state(), {}

    def get_state(self) -> ARY:
        state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                           self.shares * 2 ** -9,
                           self.close_ary[self.day] * 2 ** -7,
                           self.tech_ary[self.day] * 2 ** -6,))
        return state

    def _differential_sharpe(self, r_t: float) -> float:
        """Differential Sharpe Ratio (Moody & Saffell, NeurIPS 1998)."""
        a = self.dsr_alpha
        mu_next = (1 - a) * self.dsr_mu + a * r_t
        m2_next = (1 - a) * self.dsr_m2 + a * (r_t ** 2)

        var_next = max(m2_next - mu_next ** 2, EPS)
        sigma_next = np.sqrt(var_next)

        var_prev = max(self.dsr_m2 - self.dsr_mu ** 2, EPS)
        sigma_prev = np.sqrt(var_prev)
        sr_prev = self.dsr_mu / (sigma_prev + EPS)

        x = (r_t - self.dsr_mu) / sigma_next
        dsr_t = x - 0.5 * sr_prev * x ** 2

        self.dsr_mu = mu_next
        self.dsr_m2 = m2_next
        return dsr_t

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        self.day += 1

        # Decrement cooling down counters
        self.stock_cd = np.maximum(self.stock_cd - 1, 0)

        action = action.copy()
        action[(-0.1 < action) & (action < 0.1)] = 0
        if self.use_position_cap:
            # Per-stock max shares at current portfolio value
            V = (self.close_ary[self.day] * self.shares).sum() + self.amount
            prices = self.close_ary[self.day]
            n_max = np.floor(self.max_stock_pct * V / np.maximum(prices, 1e-8)).astype(np.int64)
            action_int = (action * n_max).astype(np.int64)
            # Clamp buys to per-stock headroom so positions can't exceed n_max.
            shares_i = self.shares.astype(np.int64)
            headroom = np.maximum(n_max - shares_i, 0)
            buy_mask = action_int > 0
            action_int = np.where(buy_mask, np.minimum(action_int, headroom), action_int)
            # Force-sell over-cap positions (price drift / random init).
            excess = np.maximum(shares_i - n_max, 0)
            if excess.any():
                action_int = action_int - excess
        else:
            action_int = (action * self.max_stock).astype(int)
        # actions initially is scaled between -1 and 1
        # convert into integer because we can't buy fraction of shares

        # Calculate minimum shares needed before selling is allowed
        total_shares_value = (self.close_ary[self.day] * self.shares).sum()
        min_shares_threshold = self.min_stock_rate * self.initial_amount

        # --- Sells first (frees cash for subsequent buys) ---
        for index in range(self.action_dim):
            stock_action = action_int[index]
            if stock_action < 0 and self.shares[index] > 0:
                if self.stock_cd[index] > 0:
                    continue
                if total_shares_value < min_shares_threshold:
                    continue
                adj_close_price = self.close_ary[self.day, index]
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * (1 - self.cost_pct)
                self.shares[index] -= delta_stock

        # --- Then buys (using cash freed from sells) ---
        for index in range(self.action_dim):
            stock_action = action_int[index]
            if stock_action > 0:
                adj_close_price = self.close_ary[self.day, index]
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * (1 + self.cost_pct)
                self.shares[index] += delta_stock
                if delta_stock > 0 and self.stock_cd_steps > 0:
                    self.stock_cd[index] = self.stock_cd_steps

        total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount

        if self.use_dsr:
            r_t = (total_asset / self.total_asset) - 1.0 if self.total_asset != 0 else 0.0
            reward = self._differential_sharpe(r_t) * self.reward_scale
        else:
            reward = (total_asset - self.total_asset) * self.reward_scale

        # A3: drawdown penalty
        if self.use_drawdown_penalty:
            self.peak = max(self.peak, total_asset)
            dd_new = (self.peak - total_asset) / self.peak if self.peak > 0 else 0.0
            delta_dd = max(0.0, dd_new - self.dd)
            reward -= self.eta_dd * delta_dd ** 2
            self.dd = dd_new

        self.rewards.append(reward)
        self.total_asset = total_asset

        terminal = self.day == self.max_step
        if terminal:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = total_asset / self.initial_amount * 100

        state = self.get_state()
        truncated = False
        return state, reward, terminal, truncated, {}

    def load_data_from_disk(self, tech_id_list=None) -> Tuple[ARY, ARY]:
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            import pandas as pd

            df = pd.read_pickle(self.df_pwd)

            tech_ary = []
            close_ary = []
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)

                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.numpy.npz" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


'''function for vmap'''


def _inplace_amount_shares_when_buy(amount, shares, stock_action, close, cost_pct):
    stock_delta = th.min(stock_action, th.div(amount, close, rounding_mode='floor'))
    amount -= close * stock_delta * (1 + cost_pct)
    shares += stock_delta
    return th.zeros(1)


def _inplace_amount_shares_when_sell(amount, shares, stock_action, close, cost_rate):
    stock_delta = th.min(-stock_action, shares)
    amount += close * stock_delta * (1 - cost_rate)
    shares -= stock_delta
    return th.zeros(1)


class StockTradingVecEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113, num_envs=4, gpu_id=0,
                 stock_cd_steps=0, min_stock_rate=0.0, npz_path=None,
                 use_position_cap=False, max_stock_pct=0.02,
                 use_drawdown_penalty=False, eta_dd=0.5,
                 reward_scale=2**-12,
                 use_dsr=False, dsr_horizon=20,
                 debug_dsr=False):
        self.df_pwd = './elegantrl/envs/China_A_shares.pandas.dataframe'
        self.npz_pwd = npz_path if npz_path is not None else './elegantrl/envs/China_A_shares.numpy.npz'
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''load data'''
        close_ary, tech_ary = self.load_data_from_disk()
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]
        self.close_price = th.tensor(close_ary, dtype=th.float32, device=self.device)
        self.tech_factor = th.tensor(tech_ary, dtype=th.float32, device=self.device)
        # print(f"| StockTradingEnv: close_ary.shape {close_ary.shape}")
        # print(f"| StockTradingEnv: tech_ary.shape {tech_ary.shape}")

        '''init'''
        self.gamma = gamma
        self.cost_pct = cost_pct
        self.max_stock = max_stock
        self.reward_scale = reward_scale
        self.initial_amount = initial_amount
        self.if_random_reset = True

        # A1 (capital-invariant action scaling): see scalar StockTradingEnv above.
        self.use_position_cap = use_position_cap
        self.max_stock_pct = max_stock_pct

        # A3 (drawdown penalty): see scalar StockTradingEnv above.
        self.use_drawdown_penalty = use_drawdown_penalty
        self.eta_dd = eta_dd
        self.peak = None
        self.dd = None

        # A2 (Differential Sharpe Ratio): see scalar StockTradingEnv above.
        self.use_dsr = use_dsr
        self.dsr_alpha = 1.0 / dsr_horizon
        self.debug_dsr = debug_dsr
        self.dsr_mu = None
        self.dsr_m2 = None

        # stock_cd: cooling down period (simulates T+1 settlement)
        # After buying a stock, must wait stock_cd_steps before selling
        self.stock_cd_steps = stock_cd_steps
        # min_stock_rate: minimum stock holding ratio before selling is allowed
        # Prevents selling small positions (reduces overtrading)
        self.min_stock_rate = min_stock_rate

        '''init (reset)'''
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = None

        self.amount = None
        self.shares = None
        self.stock_cd = None  # cooling down counter per stock per env
        self.clears = None
        self.num_shares = self.close_price.shape[1]
        amount_dim = 1

        '''environment information'''
        self.env_name = 'StockTradingEnv-v2'
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = self.num_shares + self.close_price.shape[1] + self.tech_factor.shape[1] + amount_dim
        self.action_dim = self.num_shares
        self.if_discrete = False

        '''vmap function'''
        self.vmap_get_state = th.vmap(
            func=lambda amount, shares, close, techs: th.cat((amount, shares, close, techs)),
            in_dims=(0, 0, None, None), out_dims=0)

        self.vmap_get_total_asset = th.vmap(
            func=lambda close, shares, amount: (close * shares).sum() + amount,
            in_dims=(None, 0, 0), out_dims=0)

        self.vmap_inplace_amount_shares_when_buy = th.vmap(
            func=_inplace_amount_shares_when_buy, in_dims=(0, 0, 0, None, None), out_dims=0)

        self.vmap_inplace_amount_shares_when_sell = th.vmap(
            func=_inplace_amount_shares_when_sell, in_dims=(0, 0, 0, None, None), out_dims=0)

    def reset(self):
        self.day = 0

        self.amount = th.zeros((self.num_envs, 1), dtype=th.float32, device=self.device) + self.initial_amount
        self.shares = th.zeros((self.num_envs, self.num_shares), dtype=th.float32, device=self.device)

        if self.if_random_reset:
            rand_amount = th.rand((self.num_envs, 1), dtype=th.float32, device=self.device) * 0.5 + 0.75
            self.amount = self.amount * rand_amount

            rand_shares = th.randn((self.num_envs, self.num_shares), dtype=th.float32, device=self.device)
            rand_shares = rand_shares.clip(-2, +2) * 2 ** 7
            self.shares = self.shares + th.abs(rand_shares).type(th.int32)

        # Initialize stock cooling down counters (0 = can trade)
        self.stock_cd = th.zeros((self.num_envs, self.num_shares), dtype=th.int32, device=self.device)

        self.rewards = list()
        self.total_asset = self.vmap_get_total_asset(self.close_price[self.day], self.shares, self.amount)
        self.peak = self.total_asset.clone()
        self.dd = th.zeros_like(self.total_asset)
        # DSR EMA state: shape (num_envs, 1)
        self.dsr_mu = th.zeros((self.num_envs, 1), dtype=th.float32, device=self.device)
        self.dsr_m2 = th.full((self.num_envs, 1), 1e-6, dtype=th.float32, device=self.device)
        return self.get_state(), {}

    def get_state(self):
        return self.vmap_get_state((self.amount * 2 ** -18).tanh(),
                                   (self.shares * 2 ** -10).tanh(),
                                   self.close_price[self.day] * 2 ** -7,
                                   self.tech_factor[self.day] * 2 ** -6)  # state

    def _differential_sharpe_vec(self, r_t):
        """Vectorized Differential Sharpe Ratio. r_t: (num_envs, 1)."""
        a = self.dsr_alpha
        mu_next = (1 - a) * self.dsr_mu + a * r_t
        m2_next = (1 - a) * self.dsr_m2 + a * (r_t ** 2)

        var_next = (m2_next - mu_next ** 2).clamp(min=EPS)
        sigma_next = var_next.sqrt()

        var_prev = (self.dsr_m2 - self.dsr_mu ** 2).clamp(min=EPS)
        sigma_prev = var_prev.sqrt()
        sr_prev = self.dsr_mu / (sigma_prev + EPS)

        x = (r_t - self.dsr_mu) / sigma_next
        dsr_t = x - 0.5 * sr_prev * x ** 2

        self.dsr_mu = mu_next
        self.dsr_m2 = m2_next
        return dsr_t

    def step(self, action):
        self.day += 1
        if self.day == 1:
            self.cumulative_returns = 0.

        # Decrement cooling down counters
        self.stock_cd = th.clamp(self.stock_cd - 1, min=0)

        action = action.clone()
        # action = th.ones_like(action)  # DEBUG: removed - was overriding agent actions
        action[(-0.1 < action) & (action < 0.1)] = 0
        if self.use_position_cap:
            # Per-env, per-stock max shares at current portfolio value V.
            #   V.shape      == (num_envs, 1)
            #   prices.shape == (num_shares,)
            #   n_max.shape  == (num_envs, num_shares)  int32
            V = (self.close_price[self.day] * self.shares).sum(dim=1, keepdim=True) + self.amount
            prices = self.close_price[self.day].clamp(min=1e-8)
            n_max = (self.max_stock_pct * V / prices).floor().to(th.int32)
            action_int = (action * n_max.to(action.dtype)).to(th.int32)
            # Clamp buys to per-stock headroom so positions can't exceed n_max.
            shares_i = self.shares.to(th.int32)
            headroom = (n_max - shares_i).clamp(min=0)
            action_int = th.where(action_int > 0,
                                  th.minimum(action_int, headroom),
                                  action_int)
            # Force-sell positions that already exceed the cap (e.g. from
            # price drift or initial random shares above cap).
            excess = (shares_i - n_max).clamp(min=0)
            if excess.any():
                action_int = action_int - excess
        else:
            action_int = (action * self.max_stock).to(th.int32)
        # actions initially is scaled between -1 and 1
        # convert `action` into integer as `stock_action`, because we can't buy fraction of shares

        # Calculate total shares value for min_stock_rate check
        total_shares_value = (self.close_price[self.day] * self.shares).sum(dim=1, keepdim=True)
        min_shares_threshold = self.min_stock_rate * self.initial_amount

        # --- Sells first (frees cash for subsequent buys) ---
        for i in range(self.num_shares):
            sell_mask = (action_int[:, i] < 0) & (self.shares[:, i] > 0)
            if self.stock_cd_steps > 0:
                sell_mask = sell_mask & (self.stock_cd[:, i] == 0)
            if self.min_stock_rate > 0:
                sell_mask = sell_mask & (total_shares_value.squeeze(1) >= min_shares_threshold)
            sell_idx = th.where(sell_mask)[0]
            if sell_idx.shape[0] > 0:
                part_amount = self.amount[sell_idx]
                part_shares = self.shares[sell_idx, i]
                self.vmap_inplace_amount_shares_when_sell(
                    part_amount, part_shares, action_int[sell_idx, i],
                    self.close_price[self.day, i], self.cost_pct)
                self.amount[sell_idx] = part_amount
                self.shares[sell_idx, i] = part_shares

        # --- Then buys (using cash freed from sells) ---
        for i in range(self.num_shares):
            buy_idx = th.where(action_int[:, i] > 0)[0]
            if buy_idx.shape[0] > 0:
                part_amount = self.amount[buy_idx]
                part_shares = self.shares[buy_idx, i]
                old_shares = part_shares.clone()
                self.vmap_inplace_amount_shares_when_buy(
                    part_amount, part_shares, action_int[buy_idx, i],
                    self.close_price[self.day, i], self.cost_pct)
                self.amount[buy_idx] = part_amount
                self.shares[buy_idx, i] = part_shares
                if self.stock_cd_steps > 0:
                    bought_mask = part_shares > old_shares
                    if bought_mask.any():
                        self.stock_cd[buy_idx[bought_mask], i] = self.stock_cd_steps
        # for index in range(self.action_dim):
        #     stock_actions = action_int[:, index]
        #     close_price = self.close_price[self.day, index]
        #
        #     # delta_stock.shape == ()
        #     for i in range(self.num_envs):
        #         if stock_actions[i] > 0:  # buy_stock
        #             delta_stock = th.div(self.amount[i], close_price, rounding_mode='floor')
        #             delta_stock = th.min(delta_stock, stock_actions[0])
        #             self.amount[i] -= close_price * delta_stock * (1 + self.cost_pct)
        #             self.shares[i, index] = self.shares[i, index] + delta_stock
        #         elif self.shares[i, index] > 0:  # sell_stock
        #             delta_stock = th.min(-stock_actions[i], self.shares[i, index])
        #             self.amount[i] += close_price * delta_stock * (1 - self.cost_pct)
        #             self.shares[i, index] = self.shares[i, index] + delta_stock

        '''random clear the inventory'''
        # reset_rate = 1e-2 * self.num_shares / self.max_step
        # if self.if_random_reset and (rd.rand() < reset_rate):
        #     env_i = rd.randint(self.num_envs)
        #     shares_i = rd.randint(self.num_shares)
        #
        #     self.amount[env_i] = (self.amount[env_i] +
        #                           self.shares[env_i, shares_i] * self.close_price[self.day, shares_i])  # not cost_pct
        #     self.shares[env_i, shares_i] = 0

        '''get reward'''
        total_asset = self.vmap_get_total_asset(self.close_price[self.day], self.shares, self.amount)

        if self.use_dsr:
            r_t = (total_asset / self.total_asset.clamp(min=EPS)) - 1.0  # shape (num_envs, 1)
            dsr_reward = self._differential_sharpe_vec(r_t)  # shape (num_envs, 1)
            reward = dsr_reward.squeeze(1) * self.reward_scale  # shape (num_envs,)
            # Optional diagnostics: keep off for normal training to avoid log spam.
            if self.debug_dsr and self.day % 50 == 0 and self.day > 0:
                r_t_mean, r_t_std = r_t.mean().item(), r_t.std().item()
                dsr_mean, dsr_std = dsr_reward.mean().item(), dsr_reward.std().item()
                reward_mean, reward_std = reward.mean().item(), reward.std().item()
                print(f"  [DEBUG DSR day={self.day}] r_t: μ={r_t_mean:.6f} σ={r_t_std:.6f} | dsr: μ={dsr_mean:.6f} σ={dsr_std:.6f} | reward: μ={reward_mean:.6f} σ={reward_std:.6f}")
        else:
            reward = (total_asset - self.total_asset).squeeze(1) * self.reward_scale  # shape == (num_envs, )

        # A3: drawdown penalty (vectorized)
        if self.use_drawdown_penalty:
            self.peak = th.maximum(self.peak, total_asset)
            dd_new = (self.peak - total_asset) / self.peak.clamp(min=1e-8)
            delta_dd = (dd_new - self.dd).clamp(min=0).squeeze(1)
            reward = reward - self.eta_dd * delta_dd ** 2
            self.dd = dd_new

        self.rewards.append(reward)
        self.total_asset = total_asset

        '''get done and state'''
        done = self.day == self.max_step
        if done:
            reward += th.stack(self.rewards).mean(dim=0) * (1. / (1. - self.gamma))
            self.cumulative_returns = (total_asset / self.initial_amount) * 100
            self.cumulative_returns = self.cumulative_returns.squeeze(1).cpu().data.tolist()

        state = self.reset()[0] if done else self.get_state()  # automatically reset in vectorized env
        done = th.tensor(done, dtype=th.bool, device=self.device).expand(self.num_envs)
        truncate = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, done, truncate, {}

    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            import pandas as pd

            df = pd.read_pickle(self.df_pwd)

            tech_ary = []
            close_ary = []
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)

                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.numpy.npz" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


def check_stock_trading_env():
    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    env.if_random_reset = False
    evaluate_time = 4

    print()
    policy_name = 'random action (if_random_reset = False)'
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = rd.uniform(-1, +1, env.action_dim)
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state, info_dict = env.reset()
    print(state.shape)

    print()
    policy_name = 'buy all share (if_random_reset = True)'
    env.if_random_reset = True
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.ones(env.action_dim, dtype=np.float32)
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state, info_dict = env.reset()
    print(state.shape)
    print()


if __name__ == '__main__':
    check_stock_trading_env()
