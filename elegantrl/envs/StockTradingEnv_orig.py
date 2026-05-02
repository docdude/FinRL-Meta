import os
import torch as th
import numpy as np
import numpy.random as rd
from typing import Tuple

ARY = np.ndarray


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113, stock_cd_steps=0, min_stock_rate=0.0,
                 npz_path=None, invalid_action_penalty=0.1, cold_start_rate=0.15,
                 idle_penalty=0.01):
        self.df_pwd = './China_A_shares.pandas.dataframe'
        self.npz_pwd = npz_path if npz_path is not None else './China_A_shares.numpy.npz'

        self.close_ary, self.tech_ary = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        # print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        # print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")

        self.max_stock = max_stock
        self.cost_pct = cost_pct
        self.reward_scale = 2 ** -12
        self.initial_amount = initial_amount
        self.gamma = gamma

        # stock_cd: cooling down period (simulates T+1 settlement)
        # After buying a stock, must wait stock_cd_steps before selling
        self.stock_cd_steps = stock_cd_steps
        # min_stock_rate: minimum stock holding ratio before selling is allowed
        # Prevents selling small positions (reduces overtrading)
        self.min_stock_rate = min_stock_rate

        # invalid_action_penalty: penalty for attempting to sell stocks not held
        self.invalid_action_penalty = invalid_action_penalty
        # cold_start_rate: fraction of resets that start with zero shares
        self.cold_start_rate = cold_start_rate
        # idle_penalty: penalty when no shares change in a step (un-scaled)
        self.idle_penalty = idle_penalty
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
            if rd.random() < self.cold_start_rate:
                # Cold start: zero shares, teaches model to build positions from scratch
                self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
                self.shares = np.zeros(self.shares_num, dtype=np.float32)
            else:
                # Warm start: random portfolio
                self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
                self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        # Initialize stock cooling down counters (0 = can trade)
        self.stock_cd = np.zeros(self.shares_num, dtype=np.int32)

        self.rewards = []
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        return self.get_state(), {}

    def get_state(self) -> ARY:
        state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                           self.shares * 2 ** -9,
                           self.close_ary[self.day] * 2 ** -7,
                           self.tech_ary[self.day] * 2 ** -6,))
        return state

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        self.day += 1

        # Decrement cooling down counters
        self.stock_cd = np.maximum(self.stock_cd - 1, 0)

        action = action.copy()
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).astype(int)
        # actions initially is scaled between -1 and 1
        # convert into integer because we can't buy fraction of shares

        # Calculate minimum shares needed before selling is allowed
        total_shares_value = (self.close_ary[self.day] * self.shares).sum()
        min_shares_threshold = self.min_stock_rate * self.initial_amount

        # Hard mask: zero out invalid sells and accumulate penalty
        invalid_sells = (action_int < 0) & (self.shares <= 0)
        n_masked = invalid_sells.sum()
        if n_masked > 0:
            action_int[invalid_sells] = 0

        shares_before = self.shares.copy()
        penalty = self.invalid_action_penalty * n_masked

        for index in range(self.action_dim):
            stock_action = action_int[index]
            adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price
            if stock_action > 0:  # buy_stock
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * (1 + self.cost_pct)
                self.shares[index] += delta_stock
                # Set cooling down period after buying
                if delta_stock > 0 and self.stock_cd_steps > 0:
                    self.stock_cd[index] = self.stock_cd_steps
            elif stock_action < 0 and self.shares[index] > 0:  # sell_stock (has shares)
                # Check cooling down period (T+1 simulation)
                if self.stock_cd[index] > 0:
                    continue  # Cannot sell during cooling down period
                # Check minimum stock rate threshold
                if total_shares_value < min_shares_threshold:
                    continue  # Cannot sell if below minimum holding threshold
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * (1 - self.cost_pct)
                self.shares[index] -= delta_stock

        # Idle penalty: no shares changed this step
        turnover = np.abs(self.shares - shares_before).sum()
        if turnover == 0:
            penalty += self.idle_penalty

        total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        reward = (total_asset - self.total_asset) * self.reward_scale - penalty
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
                 invalid_action_penalty=0.1, cold_start_rate=0.15,
                 idle_penalty=0.01):
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
        self.reward_scale = 2 ** -12
        self.initial_amount = initial_amount
        self.if_random_reset = True

        # stock_cd: cooling down period (simulates T+1 settlement)
        # After buying a stock, must wait stock_cd_steps before selling
        self.stock_cd_steps = stock_cd_steps
        # min_stock_rate: minimum stock holding ratio before selling is allowed
        # Prevents selling small positions (reduces overtrading)
        self.min_stock_rate = min_stock_rate

        # invalid_action_penalty: penalty for attempting to sell stocks not held
        self.invalid_action_penalty = invalid_action_penalty
        # cold_start_rate: fraction of resets that start with zero shares
        self.cold_start_rate = cold_start_rate
        # idle_penalty: penalty when no shares change in a step (un-scaled)
        self.idle_penalty = idle_penalty

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

            # Cold start: zero out shares for a fraction of envs
            cold_mask = th.rand(self.num_envs, device=self.device) < self.cold_start_rate
            if cold_mask.any():
                self.shares[cold_mask] = 0
                self.amount[cold_mask] = self.initial_amount * rand_amount[cold_mask]

        # Initialize stock cooling down counters (0 = can trade)
        self.stock_cd = th.zeros((self.num_envs, self.num_shares), dtype=th.int32, device=self.device)

        self.rewards = list()
        self.total_asset = self.vmap_get_total_asset(self.close_price[self.day], self.shares, self.amount)
        return self.get_state(), {}

    def get_state(self):
        return self.vmap_get_state((self.amount * 2 ** -18).tanh(),
                                   (self.shares * 2 ** -10).tanh(),
                                   self.close_price[self.day] * 2 ** -7,
                                   self.tech_factor[self.day] * 2 ** -6)  # state

    def step(self, action):
        self.day += 1
        if self.day == 1:
            self.cumulative_returns = 0.

        # Decrement cooling down counters
        self.stock_cd = th.clamp(self.stock_cd - 1, min=0)

        action = action.clone()
        # action = th.ones_like(action)  # DEBUG: removed - was overriding agent actions
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).to(th.int32)
        # actions initially is scaled between -1 and 1
        # convert `action` into integer as `stock_action`, because we can't buy fraction of shares

        # Calculate total shares value for min_stock_rate check
        total_shares_value = (self.close_price[self.day] * self.shares).sum(dim=1, keepdim=True)
        min_shares_threshold = self.min_stock_rate * self.initial_amount

        # Hard mask: zero out invalid sells and count per env
        invalid_sells = (action_int < 0) & (self.shares <= 0)
        n_masked = invalid_sells.sum(dim=1).float()  # (num_envs,)
        action_int[invalid_sells] = 0

        shares_before = self.shares.clone()
        penalty = self.invalid_action_penalty * n_masked

        for i in range(self.num_shares):
            buy_idx = th.where(action_int[:, i] > 0)[0]
            if buy_idx.shape[0] > 0:
                part_amount = self.amount[buy_idx]
                part_shares = self.shares[buy_idx, i]
                old_shares = part_shares.clone()
                self.vmap_inplace_amount_shares_when_buy(part_amount, part_shares, action_int[buy_idx, i],
                                                         self.close_price[self.day, i], self.cost_pct)
                self.amount[buy_idx] = part_amount
                self.shares[buy_idx, i] = part_shares
                # Set cooling down period for stocks that were actually bought
                if self.stock_cd_steps > 0:
                    bought_mask = part_shares > old_shares
                    if bought_mask.any():
                        self.stock_cd[buy_idx[bought_mask], i] = self.stock_cd_steps

            # Build sell mask: action < 0, shares > 0, not in cooling down, above min threshold
            sell_mask = (action_int[:, i] < 0) & (self.shares[:, i] > 0)
            if self.stock_cd_steps > 0:
                sell_mask = sell_mask & (self.stock_cd[:, i] == 0)  # Not in cooling down
            if self.min_stock_rate > 0:
                sell_mask = sell_mask & (total_shares_value.squeeze(1) >= min_shares_threshold)
            sell_idx = th.where(sell_mask)[0]

            if sell_idx.shape[0] > 0:
                part_amount = self.amount[sell_idx]
                part_shares = self.shares[sell_idx, i]
                self.vmap_inplace_amount_shares_when_sell(part_amount, part_shares, action_int[sell_idx, i],
                                                          self.close_price[self.day, i], self.cost_pct)
                self.amount[sell_idx] = part_amount
                self.shares[sell_idx, i] = part_shares

        # Idle penalty: no shares changed this step (per env)
        turnover = (self.shares - shares_before).abs().sum(dim=1)
        idle_mask = turnover == 0
        if idle_mask.any():
            penalty[idle_mask] += self.idle_penalty
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
        reward = (total_asset - self.total_asset).squeeze(1) * self.reward_scale - penalty  # shape == (num_envs, )
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
