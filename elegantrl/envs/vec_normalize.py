"""
VecNormalize: Running normalization wrapper for VecEnvs

Inspired by Stable-Baselines3's VecNormalize, adapted for ElegantRL's
tensor-based vectorized environments.

Usage:
    from elegantrl.envs.vec_normalize import VecNormalize
    
    env = YourVecEnv(...)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Training - stats update automatically
    state, _ = env.reset()
    action = agent(state)
    next_state, reward, done, truncated, info = env.step(action)
    
    # Save stats for later evaluation
    env.save("vec_normalize_stats.pth")
    
    # Evaluation - load stats and freeze
    env.load("vec_normalize_stats.pth")
    env.training = False
"""

import torch as th
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's online algorithm.
    Numerically stable for large numbers of updates.
    
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    
    def __init__(self, shape: Tuple[int, ...], device: th.device, epsilon: float = 1e-4):
        self.mean = th.zeros(shape, dtype=th.float32, device=device)
        self.var = th.ones(shape, dtype=th.float32, device=device)
        self.count = epsilon  # Small value to avoid division by zero initially
        self.device = device
        self.epsilon = epsilon
    
    def update(self, x: th.Tensor) -> None:
        """Update running statistics with a batch of data."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: th.Tensor, batch_var: th.Tensor, batch_count: int) -> None:
        """Update from batch statistics using parallel algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self) -> th.Tensor:
        """Return standard deviation, clamped to avoid division by zero."""
        return th.sqrt(self.var + self.epsilon)
    
    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize input using running statistics."""
        return (x - self.mean) / self.std
    
    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalize input (inverse of normalize)."""
        return x * self.std + self.mean
    
    def state_dict(self) -> dict:
        """Return state for saving."""
        return {
            'mean': self.mean.cpu(),
            'var': self.var.cpu(),
            'count': self.count,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dict."""
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count']


class VecNormalize:
    """
    Wraps a VecEnv to normalize observations and rewards using running statistics.
    
    Args:
        env: The VecEnv to wrap (must have reset() and step() methods)
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        clip_obs: Clipping value for normalized observations (None = no clipping)
        clip_reward: Clipping value for normalized rewards (None = no clipping)
        gamma: Discount factor for reward normalization (returns-based)
        epsilon: Small constant for numerical stability
        training: Whether to update running statistics (set False for evaluation)
    """

    # Attributes that live on the wrapper, NOT forwarded to the inner env.
    # Everything else is forwarded automatically — new env attrs (e.g.
    # cold_start_rate, invalid_action_penalty) work without updating this set.
    _WRAPPER_ATTRS = frozenset({
        'env', 'norm_obs', 'norm_reward', 'clip_obs', 'clip_reward',
        'gamma', 'epsilon', 'training',
        'device', 'num_envs', 'state_dim', 'action_dim', 'max_step',
        'if_discrete', 'env_name',
        'obs_rms', 'ret_rms', 'returns',
        '_update_count', '_debug_mode',
    })
    
    def __init__(
        self,
        env,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: Optional[float] = 10.0,
        clip_reward: Optional[float] = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        training: bool = True,
    ):
        self.env = env
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        
        # Copy env attributes
        self.device = env.device
        self.num_envs = env.num_envs
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.max_step = env.max_step
        self.if_discrete = env.if_discrete
        self.env_name = f"VecNormalize({env.env_name})"
        
        # Initialize running statistics
        self.obs_rms = RunningMeanStd(shape=(self.state_dim,), device=self.device, epsilon=epsilon)
        self.ret_rms = RunningMeanStd(shape=(), device=self.device, epsilon=epsilon)
        
        # For return-based reward normalization
        self.returns = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        
        # Debug tracking
        self._update_count = 0
        self._debug_mode = False
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)
    
    def __setattr__(self, name, value):
        """Forward attribute writes to inner env by default.

        Only attributes in _WRAPPER_ATTRS are stored on the wrapper itself.
        Everything else (if_random_reset, cold_start_rate, invalid_action_penalty,
        and any future env attrs) is forwarded to self.env automatically.
        """
        # During __init__, self.env doesn't exist yet — always write locally
        if 'env' not in self.__dict__:
            super().__setattr__(name, value)
            return

        if name in self._WRAPPER_ATTRS:
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)
    
    def reset(self) -> Tuple[th.Tensor, dict]:
        """Reset environment and normalize initial observation."""
        obs, info = self.env.reset()
        self.returns = th.zeros(self.num_envs, dtype=th.float32, device=self.device)
        
        if self.training and self.norm_obs:
            self.obs_rms.update(obs)
        
        obs = self._normalize_obs(obs)
        return obs, info
    
    def step(self, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict]:
        """Step environment and normalize observation/reward."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update observation statistics
        if self.training and self.norm_obs:
            self.obs_rms.update(obs)
        
        # Update return statistics (for reward normalization)
        # Following SB3: update returns first, then ret_rms, then normalize, then reset
        if self.training and self.norm_reward:
            # Track discounted returns for variance estimation
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(self.returns)
        
        # Normalize
        obs = self._normalize_obs(obs)
        reward = self._normalize_reward(reward)
        
        # Reset returns for done environments AFTER normalization (matching SB3)
        if self.norm_reward:
            if isinstance(done, th.Tensor):
                self.returns = self.returns * (~done).float()
            else:
                self.returns = self.returns * (1 - done)
        
        self._update_count += 1
        if self._debug_mode and self._update_count % 1000 == 0:
            self._print_debug_stats()
        
        return obs, reward, done, truncated, info
    
    def _normalize_obs(self, obs: th.Tensor) -> th.Tensor:
        """Normalize observations."""
        if not self.norm_obs:
            return obs
        
        obs = self.obs_rms.normalize(obs)
        
        if self.clip_obs is not None:
            obs = th.clamp(obs, -self.clip_obs, self.clip_obs)
        
        return obs
    
    def _normalize_reward(self, reward: th.Tensor) -> th.Tensor:
        """Normalize rewards using return-based scaling.
        
        Following SB3: scale by sqrt(var + epsilon), NOT by std + epsilon.
        This preserves the reward signal direction while normalizing magnitude.
        """
        if not self.norm_reward:
            return reward
        
        # Scale by return std (sqrt(var + epsilon)), matching SB3 exactly
        # Note: self.ret_rms.std already includes epsilon inside sqrt, so use it directly
        reward = reward / self.ret_rms.std
        
        if self.clip_reward is not None:
            reward = th.clamp(reward, -self.clip_reward, self.clip_reward)
        
        return reward
    
    def get_original_obs(self) -> th.Tensor:
        """Get unnormalized observation (for debugging)."""
        return self.env.get_state() if hasattr(self.env, 'get_state') else None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalization statistics."""
        path = Path(path)
        state = {
            'obs_rms': self.obs_rms.state_dict(),
            'ret_rms': self.ret_rms.state_dict(),
            'norm_obs': self.norm_obs,
            'norm_reward': self.norm_reward,
            'clip_obs': self.clip_obs,
            'clip_reward': self.clip_reward,
            'gamma': self.gamma,
        }
        th.save(state, path)
    
    def load(self, path: Union[str, Path], verbose: bool = False) -> None:
        """Load normalization statistics and configuration flags."""
        path = Path(path)
        state = th.load(path, map_location=self.device, weights_only=False)
        self.obs_rms.load_state_dict(state['obs_rms'])
        self.ret_rms.load_state_dict(state['ret_rms'])
        # Restore normalization flags so eval matches training config
        if 'norm_obs' in state:
            self.norm_obs = state['norm_obs']
        if 'norm_reward' in state:
            self.norm_reward = state['norm_reward']
        if 'clip_obs' in state:
            self.clip_obs = state['clip_obs']
        if 'clip_reward' in state:
            self.clip_reward = state['clip_reward']
        if 'gamma' in state:
            self.gamma = state['gamma']
        if verbose:
            print(f"VecNormalize stats loaded from {path}")
            print(f"  norm_obs={self.norm_obs}, norm_reward={self.norm_reward}")
            print(f"  Obs stats: mean range [{self.obs_rms.mean.min():.3f}, {self.obs_rms.mean.max():.3f}], "
                  f"std range [{self.obs_rms.std.min():.3f}, {self.obs_rms.std.max():.3f}]")
    
    def set_training(self, mode: bool) -> None:
        """Set training mode (whether to update statistics)."""
        self.training = mode
    
    def enable_debug(self, enabled: bool = True) -> None:
        """Enable/disable debug printing."""
        self._debug_mode = enabled
    
    def _print_debug_stats(self) -> None:
        """Print current normalization statistics for debugging."""
        print(f"\n{'='*60}")
        print(f"VecNormalize Debug Stats (update #{self._update_count})")
        print(f"{'='*60}")
        print(f"Observation RMS (count={self.obs_rms.count:.0f}):")
        print(f"  Mean: min={self.obs_rms.mean.min():.4f}, max={self.obs_rms.mean.max():.4f}, "
              f"absmax={self.obs_rms.mean.abs().max():.4f}")
        print(f"  Std:  min={self.obs_rms.std.min():.4f}, max={self.obs_rms.std.max():.4f}, "
              f"mean={self.obs_rms.std.mean():.4f}")
        if self.norm_reward:
            print(f"Return RMS (count={self.ret_rms.count:.0f}):")
            print(f"  Mean: {self.ret_rms.mean:.4f}, Std: {self.ret_rms.std:.4f}")
        print(f"{'='*60}\n")
    
    def get_stats_summary(self) -> dict:
        """Get summary of normalization statistics (for testing/debugging)."""
        return {
            'obs_mean_min': self.obs_rms.mean.min().item(),
            'obs_mean_max': self.obs_rms.mean.max().item(),
            'obs_std_min': self.obs_rms.std.min().item(),
            'obs_std_max': self.obs_rms.std.max().item(),
            'obs_count': self.obs_rms.count,
            'ret_mean': self.ret_rms.mean.item() if self.norm_reward else None,
            'ret_std': self.ret_rms.std.item() if self.norm_reward else None,
            'ret_count': self.ret_rms.count if self.norm_reward else None,
        }


def test_running_mean_std():
    """Unit test for RunningMeanStd."""
    print("Testing RunningMeanStd...")
    
    device = th.device('cpu')
    rms = RunningMeanStd(shape=(4,), device=device)
    
    # Generate known data
    th.manual_seed(42)
    all_data = []
    for _ in range(100):
        batch = th.randn(32, 4) * 2 + 5  # mean=5, std=2
        all_data.append(batch)
        rms.update(batch)
    
    all_data = th.cat(all_data, dim=0)
    true_mean = all_data.mean(dim=0)
    true_std = all_data.std(dim=0, unbiased=False)
    
    print(f"  True mean: {true_mean}")
    print(f"  RMS mean:  {rms.mean}")
    print(f"  Mean error: {(true_mean - rms.mean).abs().max():.6f}")
    
    print(f"  True std:  {true_std}")
    print(f"  RMS std:   {rms.std}")
    print(f"  Std error: {(true_std - rms.std).abs().max():.6f}")
    
    assert (true_mean - rms.mean).abs().max() < 0.01, "Mean error too large"
    assert (true_std - rms.std).abs().max() < 0.01, "Std error too large"
    print("  ✓ RunningMeanStd test passed!\n")


def test_vec_normalize_obs():
    """Test observation normalization."""
    print("Testing VecNormalize observation normalization...")
    
    # Create mock env
    class MockVecEnv:
        def __init__(self):
            self.device = th.device('cpu')
            self.num_envs = 16
            self.state_dim = 100  # Similar to stock env
            self.action_dim = 30
            self.max_step = 200
            self.if_discrete = False
            self.env_name = "MockEnv"
            self.step_count = 0
            
            # Simulate diverse feature scales (like stock data)
            # Features: prices (~100-500), technical indicators (~0-100), etc.
            self.feature_means = th.tensor([300.0] * 30 +  # prices
                                           [50.0] * 30 +   # RSI-like
                                           [0.0] * 20 +    # MACD-like
                                           [20.0] * 20,    # VIX-like
                                           device=self.device)
            self.feature_stds = th.tensor([100.0] * 30 +
                                          [30.0] * 30 +
                                          [5.0] * 20 +
                                          [10.0] * 20,
                                          device=self.device)
        
        def reset(self):
            return self._get_obs(), {}
        
        def step(self, action):
            self.step_count += 1
            obs = self._get_obs()
            reward = th.randn(self.num_envs, device=self.device) * 0.01
            done = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
            return obs, reward, done, done, {}
        
        def _get_obs(self):
            return th.randn(self.num_envs, self.state_dim, device=self.device) * self.feature_stds + self.feature_means
    
    env = MockVecEnv()
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)
    
    # Collect some raw observations for comparison
    raw_obs_list = []
    norm_obs_list = []
    
    obs, _ = norm_env.reset()
    raw_obs_list.append(env._get_obs())
    norm_obs_list.append(obs)
    
    for _ in range(500):
        action = th.randn(env.num_envs, env.action_dim)
        obs, reward, done, trunc, info = norm_env.step(action)
        if _ % 50 == 0:
            raw_obs_list.append(env._get_obs())
            norm_obs_list.append(obs)
    
    raw_obs = th.stack(raw_obs_list)
    norm_obs = th.stack(norm_obs_list)
    
    print(f"  Raw obs range: [{raw_obs.min():.2f}, {raw_obs.max():.2f}]")
    print(f"  Raw obs mean: {raw_obs.mean():.2f}, std: {raw_obs.std():.2f}")
    print(f"  Normalized obs range: [{norm_obs.min():.2f}, {norm_obs.max():.2f}]")
    print(f"  Normalized obs mean: {norm_obs.mean():.4f}, std: {norm_obs.std():.4f}")
    
    # Normalized observations should be roughly ~N(0,1)
    assert abs(norm_obs.mean()) < 0.5, f"Normalized mean too far from 0: {norm_obs.mean()}"
    assert 0.5 < norm_obs.std() < 2.0, f"Normalized std unexpected: {norm_obs.std()}"
    
    print(f"  Stats: {norm_env.get_stats_summary()}")
    print("  ✓ VecNormalize observation test passed!\n")


def test_save_load():
    """Test saving and loading normalization stats."""
    print("Testing VecNormalize save/load...")
    
    import tempfile
    
    class MockVecEnv:
        def __init__(self):
            self.device = th.device('cpu')
            self.num_envs = 4
            self.state_dim = 10
            self.action_dim = 2
            self.max_step = 100
            self.if_discrete = False
            self.env_name = "MockEnv"
        
        def reset(self):
            return th.randn(self.num_envs, self.state_dim) * 10 + 5, {}
        
        def step(self, action):
            obs = th.randn(self.num_envs, self.state_dim) * 10 + 5
            reward = th.randn(self.num_envs)
            done = th.zeros(self.num_envs, dtype=th.bool)
            return obs, reward, done, done, {}
    
    # Train and save
    env1 = VecNormalize(MockVecEnv(), training=True)
    env1.reset()
    for _ in range(100):
        env1.step(th.randn(4, 2))
    
    stats_before = env1.get_stats_summary()
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        env1.save(f.name)
        
        # Load into new env
        env2 = VecNormalize(MockVecEnv(), training=False)
        env2.load(f.name)
        
        stats_after = env2.get_stats_summary()
    
    print(f"  Before save: obs_mean_max={stats_before['obs_mean_max']:.4f}")
    print(f"  After load:  obs_mean_max={stats_after['obs_mean_max']:.4f}")
    
    assert abs(stats_before['obs_mean_max'] - stats_after['obs_mean_max']) < 1e-5
    assert abs(stats_before['obs_std_max'] - stats_after['obs_std_max']) < 1e-5
    print("  ✓ Save/load test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VecNormalize Unit Tests")
    print("="*60 + "\n")
    
    test_running_mean_std()
    test_vec_normalize_obs()
    test_save_load()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)
