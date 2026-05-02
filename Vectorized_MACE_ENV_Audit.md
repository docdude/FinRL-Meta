# Vectorized MACE & Margin Trader — Audit Report

Audit target: `meta/env_market_impact/vec/*.py` ported from the scalar envs
in `meta/env_market_impact/envs/env_mace_stock_trading.py` and
`envs/env_margin_trader_impact.py`.

Scope: correctness of the port (is the scalar semantics preserved?),
GPU-oriented efficiency, ElegantRL bridge soundness, and code quality.

---

## 1. Critical Correctness Bugs

### 1.1 `MACEVecEnv._buy_stock` — missing return

`vec/mace_vec_env.py`

The `return traded_value, trade_cost, executed_shares` statement is
**indented inside** the `if valid.any():` block:

```python
def _buy_stock(self, env_idx, stock_idx, buy_shares, price, volatility, volume):
    valid = (price > 0) & (buy_shares > 0)
    traded_value  = th.zeros_like(buy_shares, dtype=th.float32)
    trade_cost    = th.zeros_like(buy_shares, dtype=th.float32)
    executed_shares = th.zeros_like(buy_shares, dtype=th.float32)
    if valid.any():
        cost, _ = self.impact_model.apply_trade(...)
        total_cost = price[valid] * buy_shares[valid].to(th.float32) + cost
        affordable = total_cost <= self.cash[env_idx[valid]]
        if affordable.any():
            ...
        return traded_value, trade_cost, executed_shares   # ← still inside if
```

If `valid.any()` is False (all chosen prices ≤ 0 after permanent-impact decay
or a pathological perm-state), the function returns `None`, and the caller
does `trade_value, trade_cost, executed_shares = self._buy_stock(...)` →
`TypeError: cannot unpack non-iterable NoneType object`.

**Fix** — dedent the return:

```python
    if valid.any():
        cost, _ = self.impact_model.apply_trade(...)
        total_cost = price[valid] * buy_shares[valid].to(th.float32) + cost
        affordable = total_cost <= self.cash[env_idx[valid]]
        if affordable.any():
            ...
    return traded_value, trade_cost, executed_shares
```

`_sell_stock` already uses the correct (outside) indentation — use it as
the reference for all other impact helpers.

### 1.2 `MarginTraderVecEnv._sell_long` — missing return

`vec/margin_vec_env.py`

Same latent pattern:

```python
def _sell_long(self, stock_idx, shares, ...):
    value = th.zeros(...)
    cost_out = th.zeros_like(value)
    executed = th.zeros(..., dtype=th.int32)
    ...
    valid = (actual_shares > 0) & (px > 0)
    if valid.any():
        ...
        return value, cost_out, executed   # ← inside if
```

Trigger: any time the caller (`_execute_sell_direction`, `_margin_adjust_long`)
hits a path where all requested `actual_shares` become 0 (possible when
holdings have drifted to zero during a liquidation cascade).

**Fix**: dedent the return. `_buy_long` and `_cover_short` already do this.

### 1.3 `MarginTraderVecEnv._sell_short` — missing return (manifest crash)

Same pattern. **Unlike 1.1/1.2, this one has a straightforward manifest path:**

```python
def _sell_short(self, stock_idx, shares, ...):
    ...
    maintenance_ok = self._check_short_maintenance(price) > self.maintenance_warning
    valid = maintenance_ok & (px > 0) & (shares > 0)
    if valid.any():
        ...
        return value, cost_out, executed   # ← inside if
```

If every env that wants to open a new short has already breached the 40 %
maintenance warning, `valid` is all-False → function returns `None` → caller
crash:

```python
# _execute_sell_direction
if remaining.gt(0).any():
    value, cost, executed = self._sell_short(stock_idx, remaining, ...)  # CRASH
```

This is reachable in any run that uses aggressive short exposure combined
with the paper's `maintenance_warning=0.4`. Fix as above.

---

### 1.4 `build_tensor_impact_model` silently discards scalar parameters

`vec/runner_utils.py`

```python
def build_tensor_impact_model(impact_model_class, *, num_envs, stock_dim, gpu_id=-1, device=None):
    ...
    if issubclass(impact_model_class, ACImpactModel):
        return TensorACImpactModel(
            num_envs=num_envs, stock_dim=stock_dim, device=resolved_device,
            config=TensorACImpactConfig(),                 # ← hard-coded defaults!
        )
    ...
```

`BacktestParams` stores a **class** (e.g. `ACImpactModel`), and the scalar
runner constructs the instance with default kwargs — that part matches.
The problem is forward compatibility and explicit-instance paths
(e.g. example configs that pre-construct `ACImpactModel(alpha=2.0)` and
pass the instance): the tensor-side code has no mechanism to mirror those
kwargs. Any grid that tunes `alpha`, `Y`, `perm_fraction`,
`perm_half_life_days`, or `epsilon` will silently train against the
default model.

**Fix** — accept either a class or an instance and copy its parameters:

```python
def build_tensor_impact_model(
    impact_model, *, num_envs, stock_dim, gpu_id=-1, device=None,
):
    resolved_device = resolve_device(gpu_id=gpu_id, device=device)
    cls = impact_model if isinstance(impact_model, type) else type(impact_model)

    # If an instance was passed, pull its params; otherwise instantiate to read defaults.
    src = impact_model if not isinstance(impact_model, type) else impact_model()

    if issubclass(cls, ACImpactModel):
        cfg = TensorACImpactConfig(
            alpha=float(src.alpha),
            beta=float(src.beta),
            epsilon=float(src.epsilon),
            perm_half_life_days=float(src.perm_half_life_days),
        )
        return TensorACImpactModel(num_envs=num_envs, stock_dim=stock_dim,
                                   device=resolved_device, config=cfg)
    if issubclass(cls, BaselineImpactModel):
        cfg = TensorBaselineImpactConfig(
            basis_points=float(src.basis_points),
            perm_half_life_days=float(src.perm_half_life_days),
        )
        return TensorBaselineImpactModel(num_envs=num_envs, stock_dim=stock_dim,
                                         device=resolved_device, config=cfg)
    if issubclass(cls, SqrtImpactModel):
        cfg = TensorImpactConfig(
            Y=float(src.Y),
            perm_fraction=float(src.perm_fraction),
            perm_half_life_days=float(src.perm_half_life_days),
        )
        return TensorSqrtImpactModel(num_envs=num_envs, stock_dim=stock_dim,
                                     device=resolved_device, config=cfg)
    raise ValueError(f"Unsupported impact model: {cls!r}")
```

---

### 1.5 Trade ordering diverges from scalar Margin Trader

`envs/env_margin_trader_impact.py::step`:

```python
order = np.argsort(trade_shares)     # biggest sells first, biggest buys last
for i_idx in order:
    ...
    self._execute_trade(i, trade_shares[i], ...)
```

`vec/margin_vec_env.py::step`:

```python
for stock_idx in range(self.stock_dim):
    # all sells for this stock
for stock_idx in range(self.stock_dim):
    # all buys for this stock
```

Both preserve "sells before buys", but scalar further sorts the **sells by
magnitude** so that the largest liquidity draw happens first. Maintenance-
warning checks inside `_buy_long`/`_sell_short` depend on equity at the time
of trade, so ordering changes which envs get blocked by the 40 % warning.

For MACE the two implementations match (scalar also iterates in stock
index order). This is a Margin-specific divergence.

**Fix options**

1. **Match scalar**: pre-sort stocks by total gross trade magnitude, use
   that list twice (sells first, buys second). Stock order is the same
   across all envs since they share a price array.

   ```python
   abs_order = trade_shares.abs().sum(dim=0).argsort(descending=True).tolist()
   for stock_idx in abs_order:
       # sells
   for stock_idx in reversed(abs_order):
       # buys
   ```

2. **Accept and document** the discrepancy (the README already calls out a
   <10 bps parity drift; Margin has not been benchmarked — please add).

I recommend option 1 (cheap, one pass) plus a scalar↔vec parity test at
`num_envs=1` for sanity.

---

### 1.6 `os.makedirs("", exist_ok=True)` crashes

`vec/mace_vec_env.py::save_normalizer_state`

```python
def save_normalizer_state(self, path: str) -> None:
    state = self.get_normalizer_state()
    if state is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)   # fails when dir == ""
    th.save(state, path)
```

If `path="normalizer.pt"` (no directory), `os.path.dirname(path)` returns
`""` and `os.makedirs("", exist_ok=True)` raises `FileNotFoundError`.

**Fix**:

```python
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    th.save(state, path)
```

---

## 2. Incomplete / Missing Features

### 2.1 `TensorOWImpactModel` is missing

The scalar `impact_models.py` exposes `OWImpactModel` (Obizhaeva–Wang with
transient-impact decay), but `vec/tensor_impact.py` only ships
`TensorSqrtImpactModel`, `TensorACImpactModel`, and
`TensorBaselineImpactModel`. Any backtest grid that references
`OWImpactModel` will crash with
`ValueError: Unsupported impact model class` in
`build_tensor_impact_model`.

**Fix**: add a batched OW that maintains both `perm_state` and a
`transient_state` tensor of shape `(num_envs, stock_dim)`, decayed at
`end_day()` by `exp(-κ)`.

### 2.2 `end_day` no longer receives a date string

Scalar path:

```python
self.impact_model.end_day(self.date_list[self.time])      # audit log
```

Vec path:

```python
self.impact_model.end_day()                                # no audit log
```

The permanent-state decay is correct, but the per-date
`_impact_records` dataframe is never populated, so
`get_impact_history()` returns an empty frame. This is documented in
the README, but it breaks any downstream analysis code that expects a
non-empty history.

**Fix**: accept `date_str` as an optional argument on
`TensorImpactBase.end_day` and, when `num_envs == 1`, append one record
per stock to `_impact_records`. For `num_envs > 1` still skip logging
(or append an aggregated mean) and document the behaviour.

### 2.3 Trades list only populated when `num_envs == 1`

The `trades` list in `step()` is gated behind `if self.num_envs == 1:`
blocks. That is fine for evaluation envs (`num_envs=1`) but means every
POV / turnover-percentile histogram is empty for training-time logging.
Consider emitting **aggregated** trade stats (sum notional, mean POV)
for the training path so HPO / monitoring code has something to look
at.

### 2.4 `MACEVecEnv` uses `EnvParams` but `MarginTraderVecEnv` does not

Two directly-comparable envs with wildly different constructor shapes.
The Margin Trader scalar env also doesn't use `EnvParams`, so the
divergence is inherited, but since you are already porting, it's a good
time to introduce a `MarginEnvParams` dataclass in
`envs/env_margin_trader_impact.py` and adopt it in both the scalar and
vec constructors. This will also let the HPO scripts (`optuna_*`) share
a single sampling helper.

### 2.5 Normaliser stats are not shared across Workers

When `num_workers > 1`, ElegantRL spins up independent Workers, each
with its own `MACEVecEnv` → its own `TorchRunningMeanStd`. Each Worker
writes to the **same** `normalizer_state_path` on close; whichever
finishes last wins. The Learner runs the Evaluator, which reloads this
file before the evaluation run, so the effective stats on the eval env
are whatever the last Worker shipped.

Two acceptable solutions:

1. **Single-process mode only** (current path: `if_single_process=True`
   in `train_with_epoch_evaluation`). Document the restriction and
   raise if `num_workers > 1`.
2. **Shared-memory normaliser**: allocate `mean/var/count` in a
   `multiprocessing.shared_memory.SharedMemory` block and have all
   Workers CAS-update it. This is the proper solution for truly
   vectorised multi-worker training.

At minimum, print a warning when `num_workers > 1` with
`use_obs_normalizer=True`.

---

## 3. Performance Improvements

### 3.1 Replace the per-stock Python loop with a single batched kernel

The hot path of `MACEVecEnv.step()` is:

```python
for stock_idx in range(self.stock_dim):   # ≈ 100 iterations
    sell_mask = trade_shares[:, stock_idx] < 0
    if sell_mask.any():
        ... self._sell_stock(env_idx, stock_idx, shares, ...)

for stock_idx in range(self.stock_dim):   # ≈ 100 iterations
    buy_mask = ...
```

Each iteration spawns multiple CUDA kernels (mask, gather, impact-apply,
scatter-update). For `num_envs=2048, stock_dim=100` you get roughly
200 × (small masked ops) = **poor GPU utilisation**. The README already
flags this — `TensorImpactBase.apply_trades_batched` exists but isn't
wired up.

**Refactor sketch** (MACE sell phase):

```python
# 1. Compute all impacts in one shot.
sell_shares = (-trade_shares).clamp_min(0)                  # (num_envs, stock_dim)
prices      = adjusted_prices                                # (num_envs, stock_dim)
vol_mat     = volatility.expand(self.num_envs, -1)
volm_mat    = volume.expand(self.num_envs, -1)

cost_mat, _ = self.impact_model.apply_trades_batched(
    -sell_shares, prices, vol_mat, volm_mat,
)                                                            # both (num_envs, stock_dim)

# 2. Update holdings & cash without a Python loop.
proceeds   = sell_shares * prices - cost_mat
self.stocks -= sell_shares
self.cash  += proceeds.sum(dim=1)
self.stocks_cool_down = th.where(sell_shares > 0, 0, self.stocks_cool_down)

# 3. Aggregate for reward / info.
total_sell_value += proceeds.sum(dim=1)
total_trade_cost += cost_mat.sum(dim=1)
```

For the buy phase, cash affordability becomes a per-env problem: either
allow the entire batched buy (optimistic) and clip afterwards, or do a
**one-shot budget allocation** with a closed-form scaling:

```python
total_buy_cost = (buy_shares * prices + cost_mat).sum(dim=1)
scale          = (self.cash / total_buy_cost).clamp(max=1.0)
buy_shares     = (buy_shares * scale.unsqueeze(1)).floor().to(th.int32)
```

A pure vectorised path is >10× faster on a 4090 for NASDAQ-100 sized
universes.

### 3.2 Avoid recomputing `adjusted_prices` twice per step

```python
adjusted_prices       = self.price_array[self.time].unsqueeze(0) + self._get_perm_impact()
...  # impact updates perm_state
self.impact_model.end_day()
adjusted_prices_post  = self.price_array[self.time].unsqueeze(0) + self._get_perm_impact()
```

This is unavoidable (perm_state changes), but hoist `self.price_array[self.time]`
into a local to reduce indexing:

```python
base_price = self.price_array[self.time]
adjusted_prices      = base_price + self._get_perm_impact()
...
adjusted_prices_post = base_price + self._get_perm_impact()
```

Minor, but adds up at 10× increased step throughput.

### 3.3 Skip `end_day` decay multiply when rate is zero

`TensorImpactBase.end_day` already short-circuits this correctly — good.
But `TensorBaselineImpactModel.perm_decay_rate = 0.0` is hard-coded, so
for baseline runs the decay multiply is entirely skipped. Verify the
same for `TensorACImpactModel` with `perm_half_life_days=0`.

### 3.4 `_calculate_max_stock_per_position` allocates large intermediates

```python
portfolio_value    = self.cash + (self.stocks * current_prices).sum(dim=1)
max_position_value = portfolio_value.unsqueeze(1) * self.max_stock_pct
return th.where(
    current_prices > 0,
    th.div(max_position_value, current_prices, rounding_mode="floor"),
    th.zeros_like(current_prices),
).to(th.int32)
```

Fine at `num_envs=2048, stock_dim=100` (820 KB), but `th.div` with
`rounding_mode='floor'` on floats is not the same as scalar's
`astype(int)` (truncate toward zero). For positive arguments they match;
for negative they don't. Stocks are non-negative in MACE, so this is a
latent issue only in paper-trade / short extensions. Worth documenting.

### 3.5 State assembly allocates 7+ intermediate tensors

```python
state_components = [
    cash_pct, price_ret_1d, position_value_pct,
    self.tech_array[self.time].unsqueeze(0).expand(self.num_envs, -1),
    shares_over_adv, ...
]
obs = th.cat(state_components, dim=1).to(dtype=th.float32)
```

Preallocate `obs = th.empty((num_envs, state_dim), ...)` once in
`__init__` and fill slices on each call. Saves 5–10 % at large
`num_envs`.

---

## 4. API & Code Quality

### 4.1 Consistent `Optional` return types

Bugs 1.1–1.3 would be caught immediately by strict type checkers if the
return type were `Tuple[...]` rather than
`Tuple[Tensor, Tensor, Tensor]`. Consider adding a `# type: ignore[return-value]`
fail-fast assertion at the end of each helper:

```python
assert isinstance(value, th.Tensor), "unreachable"
return value, cost_out, executed
```

Or just restructure so each helper has a single exit point.

### 4.2 Inconsistent trade-ordering comments

The README says *"single-env sorts by trade size"* — this is **only true
for Margin Trader**, not for MACE. Update the README to be specific.

### 4.3 `run_vec_simulation` rejects `num_envs > 1`

```python
if getattr(env, "num_envs", 1) != 1:
    raise ValueError(...)
```

Fair limitation, but you're leaving performance on the table. For final
backtests you could run `num_envs=256` and pick the first env's
trajectory as canonical — the others just warm the normaliser.
Alternatively drop the restriction and let the caller select an index.

### 4.4 `probe_env.close()` in `build_elegantrl_config`

`build_elegantrl_config` instantiates a probe env just to read `state_dim`
etc., then closes it. For `MACEVecEnv` that currently writes the
normaliser state. `runner_utils.build_training_args` pops
`normalizer_state_path` before building the probe env — good — but the
coupling is fragile. A cleaner design: make `close()` a no-op and write
an explicit `save()` method users must call.

### 4.5 `get_margin_state` returns mixed Python/Tensor fields

```python
if env_index is not None:
    return {
        "long_cash": float(self.long_cash[env_index].item()),   # python float
        ...
        "stocks":    self.stocks[env_index].clone(),            # tensor
    }
```

Either make everything scalar-Python or everything tensor — consumers
have to know which is which.

### 4.6 Duplicate scaffolding between the two runners

`train_and_backtest_vec_mace.py` and `train_and_backtest_vec_margin.py`
share ~80 % of the code (build impact model, build training args, loop
over configs, write csvs, emit metadata dict). Extract a shared helper
in `runner_utils.py`:

```python
def run_vec_backtest_suite(
    *, env_class, data_prep, grid_configs, num_epochs,
    build_env_kwargs_fn, build_eval_env_kwargs_fn,
    make_continued_env_fn, make_blank_env_fn,
    make_metadata_fn,
    num_envs=None, gpu_id=None, num_workers=1, random_seed=42,
) -> str: ...
```

---

## 5. Recommended Tests (to go in `tests/vec/`)

1. **Scalar ↔ Vec(num_envs=1) parity**. For a fixed action sequence of
   length 20, assert that every info field (`turnover`, `cost`,
   `total_buy_value`, `total_sell_value`, `cash`, `total_asset`) matches
   the scalar env to within float32 rounding. Run for
   MACE × {Baseline, AC, Sqrt} and Margin × {Baseline, AC}.

2. **Empty-trade return test**. Directly call `_buy_stock`,
   `_sell_long`, `_sell_short` with inputs that force `valid.any() = False`;
   assert the return is a 3-tuple of zero tensors, not `None`.

3. **Maintenance-warning trigger**. Construct a Margin env at the 40 %
   boundary, issue a `sell_short` action → assert no exception.

4. **Impact-model parameter propagation**. Build a
   `MACEVecEnv(impact_model=ACImpactModel(alpha=2.0, beta=0.5))` and
   assert the tensor model has `alpha==2.0, beta==0.5`.

5. **Normaliser transfer**. Train for a handful of steps,
   `get_normalizer_state()` → serialise → new env `set_normalizer_state()`
   → assert first observation matches.

6. **Run-simulation determinism**. Fixed seed + fixed actor yields
   byte-identical trajectories across runs.

---

## 6. Priority Patch Order

1. **Fix 1.1 / 1.2 / 1.3** (missing returns) — one-line dedents, ship today.
2. **Fix 1.4** (impact-parameter propagation) — unblocks any HPO that tunes impact.
3. **Fix 1.6** (`os.makedirs("")`) — user-facing crash.
4. **Add test 1** (scalar↔vec parity) — pins behaviour before further refactors.
5. **Fix 1.5** (Margin trade ordering) — pick option 1 and update README.
6. **Implement 3.1** (batched step path) — the reason vec envs exist.
7. **Add `TensorOWImpactModel`** — parity with scalar.
8. **Extract shared runner helper (4.6)** and unify `EnvParams` across Margin (2.4).
