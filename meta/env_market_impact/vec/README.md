# Vectorized Market Impact Environments

This module adds separate torch-native vectorized environments for the market-impact research code without changing the original single-env scripts.

Files:

- `mace_vec_env.py`: batched MACE environment on torch tensors.
- `margin_vec_env.py`: batched margin trader environment on torch tensors.
- `tensor_impact.py`: tensorized impact models (`TensorSqrtImpactModel`,
  `TensorACImpactModel`) sharing a batched permanent-state tensor.
- `runner_utils.py`: direct ElegantRL training helpers and vec backtest utilities.

Design goals:

- keep the original env implementations intact for debugging and backtests
- move rollout-time state to torch tensors on CPU or GPU
- expose `reset()` and `step()` with `(num_envs, action_dim)` actions
- support synchronized auto-reset, which matches the fixed-horizon trading datasets

Current constraints:

- single-env eval/backtest runs emit per-trade logs; multi-env runs emit per-stock aggregate trade summaries so training-time monitoring still has POV and turnover-percentile context
- MACE uses the batched tensor-impact kernel for both sell and buy phases; Margin remains env-batched within each stock pass because maintenance checks and position flipping are order-sensitive
- MACE matches the scalar stock-index order. Margin now sorts stocks by aggregate trade magnitude before the sell and buy passes, which aligns the vec path with the scalar env's biggest-sells-first / biggest-buys-last behaviour
- the permanent-state tensor is `(num_envs, stock_dim)`; per-date impact history is recorded when `num_envs == 1`, while multi-env training runs intentionally skip per-date history rows
- Tensor OW transient state decays at `end_day()`, not before every intra-step trade. That matches MACE's one-net-trade-per-stock flow; Margin can differ slightly from the scalar OW path when the same stock is touched multiple times within a day
- MACE position sizing uses `floor` on positive prices, which matches the scalar env's `astype(int)` behaviour for the current long-only price path; if negative-price synthetic assets are introduced, revisit this assumption explicitly
- observation-normalizer statistics are not shared across multi-worker training processes; vec training now rejects `num_workers > 1` when normalization is enabled, and single-process training remains the parity/debug reference path
- this module is designed for torch-native collectors such as ElegantRL; SB3 can wrap it only with additional adapters, and that usually removes most of the GPU-throughput benefit

Custom impact models:

```python
from meta.env_market_impact.vec import (
    MACEVecEnv,
    TensorACImpactConfig,
    TensorACImpactModel,
)

impact = TensorACImpactModel(
    num_envs=512, stock_dim=stock_dim, device=th.device("cuda:0"),
    config=TensorACImpactConfig(alpha=1.0, beta=1.0, epsilon=5e-4),
)
env = MACEVecEnv(config=train_config, num_envs=512, gpu_id=0, impact_model=impact)
```

Batched ("vmap-equivalent") impact path:

`TensorImpactBase.apply_trades_batched(trade_size, price, volatility, volume)`
processes all stocks and all envs in a single broadcast operation (no Python
loop over stock index).  MACE wires this path directly into its step hot path.
Margin still uses per-stock helper calls because affordability and
maintenance-gate semantics depend on stock order.

Minimal usage:

```python
from meta.env_market_impact.backtest_vec_config import VecMACEEnvParams
from meta.env_market_impact.envs.market_data import MarketDataPreparator
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.vec import MACEVecEnv

prep = MarketDataPreparator(...)
train_config = prep.create_env_config(Split.TRAIN)

env = MACEVecEnv(
    config=train_config,
    params=VecMACEEnvParams(),
    num_envs=512,
    gpu_id=0,
)
state, _ = env.reset()
```

Direct ElegantRL usage:

```python
from elegantrl.train import train_agent
from meta.env_market_impact.vec import MACEVecEnv
from meta.env_market_impact.vec.runner_utils import build_training_args

args = build_training_args(
    env_class=MACEVecEnv,
    train_env_kwargs={"config": train_config, "num_envs": 512, "gpu_id": 0},
    eval_env_kwargs={"config": trade_config, "num_envs": 512, "gpu_id": 0},
    agent_name="ppo",
    model_kwargs=None,
    policy_kwargs=None,
    steps_per_epoch=len(train_config["date_list"]) - 1,
    epoch_index=0,
    run_dir="tmp_vec_run",
    gpu_id=0,
    num_workers=1,
    random_seed=42,
)
train_agent(args, if_single_process=True)
```