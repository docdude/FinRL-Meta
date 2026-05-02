import sys, os, time, shutil
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import numpy as np
from meta.env_market_impact.envs.env_mace_stock_trading import EnvParams
from meta.env_market_impact.vec import MACEVecEnv, MarginTraderVecEnv
from meta.env_market_impact.vec.elegantrl_bridge import train_elegantrl_agent


def make_cfg(n_steps=60, stock_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(50, 200, stock_dim).astype(np.float32)
    rets = rng.normal(0, 0.01, (n_steps, stock_dim)).astype(np.float32)
    price = base * np.cumprod(1 + rets, axis=0)
    return {
        "date_list": [f"d{i}" for i in range(n_steps)],
        "price_array": price,
        "tech_array": rng.normal(0, 1, (n_steps, stock_dim * 2)).astype(np.float32),
        "volatility_array": np.full((n_steps, stock_dim), 0.02, np.float32),
        "volume_array": np.full((n_steps, stock_dim), 1e6, np.float32),
        "adv20_array": np.full((n_steps, stock_dim), 1e6, np.float32),
        "tbill_rates": np.full(n_steps, 2.0, np.float32),
        "tic_list": [f"S{i}" for i in range(stock_dim)],
    }


def clean(*paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "margin"
    cfg = make_cfg()

    if mode == "margin":
        clean("./MarginTraderVecEnv-v1_PPO_0")
        kw = dict(config=cfg, num_envs=16, gpu_id=0, auto_reset=True)
        ekw = dict(config=cfg, num_envs=4, gpu_id=0, auto_reset=False)
        t0 = time.perf_counter()
        train_elegantrl_agent(
            env_class=MarginTraderVecEnv, env_kwargs=kw, eval_env_kwargs=ekw,
            agent_name="ppo", gpu_id=0, break_step=1000, horizon_len=32,
            net_dims=[64, 64], eval_per_step=500, eval_times=2,
            num_workers=1, random_seed=0, if_single_process=True,
        )
        print(f"MARGIN_PPO_OK {time.perf_counter()-t0:.1f}s")

    elif mode == "mp":
        clean("./MACEVecEnv-v1_PPO_0")
        kw = dict(config=cfg, params=EnvParams(), num_envs=16, gpu_id=0, auto_reset=True)
        ekw = dict(config=cfg, params=EnvParams(), num_envs=4, gpu_id=0, auto_reset=False)
        t0 = time.perf_counter()
        train_elegantrl_agent(
            env_class=MACEVecEnv, env_kwargs=kw, eval_env_kwargs=ekw,
            agent_name="ppo", gpu_id=0, break_step=1000, horizon_len=32,
            net_dims=[64, 64], eval_per_step=500, eval_times=2,
            num_workers=2, random_seed=0, if_single_process=False,
        )
        print(f"MULTIPROC_OK {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
