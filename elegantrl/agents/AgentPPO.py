import os

import numpy as np
import torch as th
from torch import nn

from .AgentBase import AgentBase
from .AgentBase import build_mlp, layer_init_with_orthogonal
from ..train import Config

TEN = th.Tensor


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _debug_scalar_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(value)
        return f"{float(value):.6g}"
    return str(value)


def _debug_tensor_stats_limited(prefix: str, tensor: TEN, max_stats_elems: int) -> dict[str, float | int]:
    values = tensor.detach().reshape(-1)
    sample_size = int(values.numel())
    stride = 1
    if sample_size > max_stats_elems:
        stride = max(1, int(np.ceil(sample_size / max_stats_elems)))
        values = values[::stride]

    values = values.to(device='cpu')
    if values.dtype == th.bool:
        values = values.float()

    nonfinite = 0
    if values.is_floating_point():
        values = values.float()
        finite = th.isfinite(values)
        nonfinite = int((~finite).sum().item())
        values = values[finite]
    else:
        values = values.float()

    if values.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_nonfinite": nonfinite,
            f"{prefix}_sample_size": sample_size,
            f"{prefix}_sample_stride": stride,
        }

    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_abs_max": float(values.abs().max().item()),
        f"{prefix}_nonfinite": nonfinite,
        f"{prefix}_sample_size": sample_size,
        f"{prefix}_sample_stride": stride,
    }


def _emit_a2c_debug_line(agent_name: str, stage: str, **metrics) -> None:
    payload = [
        "[ERL_A2C_DEBUG]",
        f"agent={agent_name}",
        f"stage={stage}",
    ]
    payload.extend(
        f"{key}={_debug_scalar_value(value)}"
        for key, value in metrics.items()
    )
    print(" ".join(payload), flush=True)


class AgentPPO(AgentBase):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)
        self._a2c_debug_enabled = _env_flag("ERL_A2C_DEBUG")
        self._a2c_debug_update_net_limit = max(1, _env_int("ERL_A2C_DEBUG_NETS", 8))
        self._a2c_debug_update_limit = max(1, _env_int("ERL_A2C_DEBUG_UPDATES", 4))
        self._a2c_debug_max_stats_elems = max(1, _env_int("ERL_A2C_DEBUG_MAX_ELEMS", 65536))
        self._a2c_debug_update_net_calls = 0
        self._a2c_debug_current_update_net = 0

    def _a2c_debug_active(self) -> bool:
        return self._a2c_debug_enabled and self.__class__.__name__.endswith("A2C")

    @staticmethod
    def _debug_scalar(value) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return str(value)
            return f"{float(value):.6g}"
        return str(value)

    def _debug_tensor_stats(self, prefix: str, tensor: TEN) -> dict[str, float | int]:
        values = tensor.detach().reshape(-1)
        sample_size = int(values.numel())
        stride = 1
        if sample_size > self._a2c_debug_max_stats_elems:
            stride = max(1, int(np.ceil(sample_size / self._a2c_debug_max_stats_elems)))
            values = values[::stride]

        values = values.to(device='cpu')
        if values.dtype == th.bool:
            values = values.float()

        nonfinite = 0
        if values.is_floating_point():
            values = values.float()
            finite = th.isfinite(values)
            nonfinite = int((~finite).sum().item())
            values = values[finite]
        else:
            values = values.float()

        if values.numel() == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_abs_max": float("nan"),
                f"{prefix}_nonfinite": nonfinite,
                f"{prefix}_sample_size": sample_size,
                f"{prefix}_sample_stride": stride,
            }

        return {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_abs_max": float(values.abs().max().item()),
            f"{prefix}_nonfinite": nonfinite,
            f"{prefix}_sample_size": sample_size,
            f"{prefix}_sample_stride": stride,
        }

    @staticmethod
    def _debug_grad_stats(prefix: str, optimizer: th.optim.Optimizer) -> dict[str, float | int]:
        total_sq = 0.0
        abs_max = 0.0
        param_count = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                grad = grad.detach().float()
                total_sq += float(grad.pow(2).sum().item())
                abs_max = max(abs_max, float(grad.abs().max().item()))
                param_count += 1
        return {
            f"{prefix}_grad_l2": float(total_sq ** 0.5),
            f"{prefix}_grad_abs_max": abs_max,
            f"{prefix}_grad_params": param_count,
        }

    def _emit_a2c_debug(self, stage: str, **metrics) -> None:
        if not self._a2c_debug_active():
            return
        payload = [
            "[ERL_A2C_DEBUG]",
            f"agent={self.__class__.__name__}",
            f"stage={stage}",
            f"update_net={self._a2c_debug_current_update_net}",
        ]
        payload.extend(
            f"{key}={self._debug_scalar(value)}"
            for key, value in metrics.items()
        )
        print(" ".join(payload), flush=True)

    def _explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            num_envs == 1
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros(horizon_len, dtype=th.int32).to(self.device)
        logprobs = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        state = self.last_state  # shape == (1, state_dim) for a single env.
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = [t[0] for t in self.explore_action(state)]

            states[t] = state
            actions[t] = action
            logprobs[t] = logprob

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device).unsqueeze(0)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state  # state.shape == (1, state_dim) for a single env.
        '''add dim1=1 below for workers buffer_items concat'''
        states = states.view((horizon_len, 1, self.state_dim))
        actions = actions.view((horizon_len, 1, self.action_dim)) \
            if not self.if_discrete else actions.view((horizon_len, 1))
        logprobs = logprobs.view((horizon_len, 1))
        rewards = (rewards * self.reward_scale).view((horizon_len, 1))
        undones = th.logical_not(terminals).view((horizon_len, 1))
        unmasks = th.logical_not(truncates).view((horizon_len, 1))
        return states, actions, logprobs, rewards, undones, unmasks

    def _explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
        logprobs = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = self.last_state  # shape == (num_envs, state_dim) for a vectorized env.

        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = self.explore_action(state)

            states[t] = state
            actions[t] = action
            logprobs[t] = logprob

            state, reward, terminal, truncate, _ = env.step(convert(action))  # next_state

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state
        rewards *= self.reward_scale
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, logprobs, rewards, undones, unmasks

    def explore_action(self, state: TEN) -> tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state)
        return actions, logprobs

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2 ** 10 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = th.cat(values, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
        obj_entropies = []
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor, obj_entropy = self.update_objectives(buffer, update_t)
            obj_entropies.append(obj_entropy)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_entropy_avg = np.array(obj_entropies).mean() if len(obj_entropies) else 0.0
        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg, obj_entropy_avg

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        self.act.set_a2c_debug_context(
            enabled=self._a2c_debug_active(),
            agent_name=self.__class__.__name__,
            update_net=self._a2c_debug_current_update_net,
            update_t=update_t,
        )
        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()

        # surrogate1 = advantage * ratio
        # surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        # surrogate = th.min(surrogate1, surrogate2)  # save as below
        surrogate = advantage * ratio * th.where(advantage.gt(0), 1 - self.ratio_clip, 1 + self.ratio_clip)

        obj_surrogate = (surrogate * unmask).mean()  # major actor objective
        obj_entropy = (entropy * unmask).mean()  # minor actor objective
        obj_actor_full = obj_surrogate + obj_entropy * self.lambda_entropy
        self.optimizer_backward(self.act_optimizer, -obj_actor_full)
        return obj_critic.item(), obj_surrogate.item(), obj_entropy.item()

    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += self.cri(states[truncated]).squeeze(1).detach()
            undones[truncated] = False

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = self.last_state.clone()
        next_value = self.cri(next_state).detach().squeeze(-1)

        advantage = th.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        if self.if_use_v_trace:  # get advantage value in reverse time series (V-trace)
            for t in range(horizon_len - 1, -1, -1):
                next_value = rewards[t] + masks[t] * next_value
                advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
                next_value = values[t]
        else:  # get advantage value using the estimated value of critic network
            for t in range(horizon_len - 1, -1, -1):
                advantages[t] = rewards[t] - values[t] + masks[t] * advantage
                advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages

    def update_avg_std_for_normalization(self, states: TEN):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.act.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        self.act_target.state_avg[:] = self.act.state_avg
        self.act_target.state_std[:] = self.act.state_std
        self.cri_target.state_avg[:] = self.cri.state_avg
        self.cri_target.state_std[:] = self.cri.state_std


class AgentA2C(AgentPPO):
    """A2C algorithm.
    “Asynchronous Methods for Deep Reinforcement Learning”. 2016.
    """

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]
        self._a2c_debug_update_net_calls += 1
        self._a2c_debug_current_update_net = self._a2c_debug_update_net_calls

        should_log_summary = (
            self._a2c_debug_active()
            and self._a2c_debug_current_update_net <= self._a2c_debug_update_net_limit
        )
        if should_log_summary:
            self._emit_a2c_debug(
                "update_net_start",
                buffer_size=buffer_size,
                num_envs=self.num_envs,
                batch_size=self.batch_size,
                repeat_times=self.repeat_times,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                lambda_gae_adv=self.lambda_gae_adv,
                lambda_entropy=float(self.lambda_entropy.item()),
                clip_grad_norm=self.clip_grad_norm,
            )

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            bs = max(1, 2 ** 10 // self.num_envs)  # set a smaller 'batch_size' to avoid CUDA OOM
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = th.cat(values, dim=0).squeeze(-1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            if should_log_summary:
                self._emit_a2c_debug(
                    "update_net_raw",
                    **self._debug_tensor_stats("state", states),
                    **self._debug_tensor_stats("logprob", logprobs),
                    **self._debug_tensor_stats("reward_sum", reward_sums),
                    **self._debug_tensor_stats("adv_raw", advantages),
                    **self._debug_tensor_stats("unmask", unmasks.float()),
                )

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
            if should_log_summary:
                self._emit_a2c_debug(
                    "update_net_norm",
                    **self._debug_tensor_stats("adv_norm", advantages),
                )
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, update_t)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        if should_log_summary:
            self._emit_a2c_debug(
                "update_net_done",
                update_times=update_times,
                obj_critic_avg=float(obj_critic_avg),
                obj_actor_avg=float(obj_actor_avg),
            )
        return obj_critic_avg, obj_actor_avg, 0

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        # Use 2D sampling like PPO for VecEnv compatibility
        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        # logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        value = self.cri(state).squeeze(-1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        critic_grad_stats = None
        should_log_update = (
            self._a2c_debug_active()
            and self._a2c_debug_current_update_net <= self._a2c_debug_update_net_limit
            and update_t < self._a2c_debug_update_limit
        )
        if should_log_update:
            critic_grad_stats = self._debug_grad_stats("critic", self.cri_optimizer)

        self.act.set_a2c_debug_context(
            enabled=self._a2c_debug_active(),
            agent_name=self.__class__.__name__,
            update_net=self._a2c_debug_current_update_net,
            update_t=update_t,
        )
        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        obj_actor = (advantage * new_logprob * unmask).mean()  # obj_actor without policy gradient clip
        obj_entropy = (entropy * unmask).mean()
        obj_actor_full = obj_actor + obj_entropy * self.lambda_entropy
        self.optimizer_backward(self.act_optimizer, -obj_actor_full)
        if should_log_update:
            self._emit_a2c_debug(
                "objective_step",
                update_t=update_t,
                obj_critic=float(obj_critic.item()),
                obj_actor=float(obj_actor.item()),
                obj_entropy=float(obj_entropy.item()),
                **self._debug_tensor_stats("value", value),
                **self._debug_tensor_stats("reward_sum", reward_sum),
                **self._debug_tensor_stats("adv", advantage),
                **self._debug_tensor_stats("new_logprob", new_logprob),
                **self._debug_tensor_stats("entropy", entropy),
                **self._debug_tensor_stats("state", state),
                **self._debug_tensor_stats("unmask", unmask.float()),
                **(critic_grad_stats or {}),
                **self._debug_grad_stats("actor", self.act_optimizer),
            )
        return obj_critic.item(), obj_actor.item()


class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentPPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)


class AgentDiscreteA2C(AgentDiscretePPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentDiscretePPO.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)


'''network'''


class ActorPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)
        self._a2c_debug_enabled = _env_flag("ERL_A2C_DEBUG")
        self._a2c_debug_update_net_limit = max(1, _env_int("ERL_A2C_DEBUG_NETS", 8))
        self._a2c_debug_update_limit = max(1, _env_int("ERL_A2C_DEBUG_UPDATES", 4))
        self._a2c_debug_max_stats_elems = max(1, _env_int("ERL_A2C_DEBUG_MAX_ELEMS", 65536))
        self._a2c_debug_context_enabled = False
        self._a2c_debug_agent_name = self.__class__.__name__
        self._a2c_debug_current_update_net = 0
        self._a2c_debug_current_update_t = -1

    def set_a2c_debug_context(self, enabled: bool, agent_name: str, update_net: int, update_t: int) -> None:
        self._a2c_debug_context_enabled = enabled
        self._a2c_debug_agent_name = agent_name
        self._a2c_debug_current_update_net = update_net
        self._a2c_debug_current_update_t = update_t

    def _should_log_a2c_distribution(self) -> bool:
        return (
            self._a2c_debug_enabled
            and self._a2c_debug_context_enabled
            and self._a2c_debug_current_update_net <= self._a2c_debug_update_net_limit
            and self._a2c_debug_current_update_t < self._a2c_debug_update_limit
        )

    def _emit_a2c_distribution_debug(self, action_avg: TEN, action_std_log: TEN, action_std: TEN) -> None:
        if not self._should_log_a2c_distribution():
            return
        _emit_a2c_debug_line(
            self._a2c_debug_agent_name,
            "distribution_step",
            update_net=self._a2c_debug_current_update_net,
            update_t=self._a2c_debug_current_update_t,
            **_debug_tensor_stats_limited("action_avg", action_avg, self._a2c_debug_max_stats_elems),
            **_debug_tensor_stats_limited("action_std_log", action_std_log, self._a2c_debug_max_stats_elems),
            **_debug_tensor_stats_limited("action_std", action_std, self._a2c_debug_max_stats_elems),
        )

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> tuple[TEN, TEN]:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> tuple[TEN, TEN]:
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()
        self._emit_a2c_distribution_debug(action_avg, self.action_std_log, action_std)

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.tanh()


class ActorDiscretePPO(ActorPPO):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim)
        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        a_prob = self.net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_action(self, state: TEN) -> (TEN, TEN):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> (TEN, TEN):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))  # action.shape == (batch_size, 1), action.dtype = th.int
        dist = self.ActionDist(a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.long()


class CriticPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int):
        super().__init__()
        assert isinstance(action_dim, int)
        self.net = build_mlp(dims=[state_dim, *net_dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        value = self.net(state)
        return value  # advantage value

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)
