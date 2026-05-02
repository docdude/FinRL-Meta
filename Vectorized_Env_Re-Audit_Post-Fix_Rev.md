# Vectorized Env Re-Audit — Post-Fix Review

All 10 items from the original audit have been addressed. This re-audit focuses
on (a) remaining smaller issues surfaced by reading the new code, (b) gaps in
the regression-test suite, and (c) dead code that should be removed now that
the batched path has replaced the per-stock one.

---

## 1. Remaining Issues

### 1.1 Dead code: `MACEVecEnv._buy_stock` and `_sell_stock`  **[Minor — Cleanup]**

After the batched refactor, `MACEVecEnv.step()` calls
`self.impact_model.apply_trades_batched(...)` directly and never invokes
`_buy_stock` / `_sell_stock`. Both helper methods are now unreachable:

```python
# mace_vec_env.py
def _sell_stock(self, env_idx, stock_idx, sell_shares, price, volatility, volume):
    ...  # dead code: no caller

def _buy_stock(self, env_idx, stock_idx, buy_shares, price, volatility, volume):
    ...  # dead code: no caller
```

Worse, `_buy_stock` still has a dead `if valid.any():` branch after the
early-return guard:

```python
if not valid.any():
    return traded_value, trade_cost, executed_shares
if valid.any():              # ← always True here; remove
    cost, _ = self.impact_model.apply_trade(...)
```

**Action**

1. Delete both methods outright. Any future non-batched path should be added
   only with a dedicated test.
2. If the maintainers want to keep them as a fallback, at minimum remove the
   dead `if valid.any():` and update the docstring saying "unused, retained
   for debugging".

**Tests to add if kept**: a regression test that calls each helper with an
all-zero `sell_shares` / `buy_shares` tensor and asserts the return shape
and dtype. (The existing
`test_mace_buy_stock_invalid_returns_zero_tuple` covers only the
zero-price case.)

---

### 1.2 `MarginTraderVecEnv._execute_*` recomputes turnover percentiles per call  **[Perf — High-ish]**

In `_execute_sell_direction` and `_execute_buy_direction`, each of the four
inner branches (sell-long, sell-short, cover-short, buy-long) re-evaluates
the turnover percentiles by calling:

```python
turnover_percentile=self._calculate_turnover_percentiles(
    self.price_array[min(self.time, self.max_step)],
    self.volume_array[min(self.time, self.max_step)],
)[stock_idx],
```

`_calculate_turnover_percentiles` internally sorts a length-`stock_dim`
tensor, so this is O(N log N) per helper call × 4 branches × N stocks =
**O(N² log N) per step** just for logging. At NAS-100, ~40 k ops per step
for something that should be computed once per step.

**Fix**

`step()` already computes `turnover_percentiles` once at the top. Thread it
through the call chain:

```python
def step(self, actions):
    ...
    turnover_percentiles = self._calculate_turnover_percentiles(...)
    ...
    for stock_idx in stock_order:
        if (trade_shares[:, stock_idx] < 0).any():
            trade_value, trade_cost, trade_sides, agg_rows = self._execute_sell_direction(
                stock_idx, -trade_shares[:, stock_idx], trade_price,
                volatility, volume,
                turnover_percentile=turnover_percentiles[stock_idx],   # <── new
            )
```

Then `_execute_sell_direction(..., turnover_percentile)` just passes the
scalar through to `_build_aggregate_trade_entry`.

---

### 1.3 Obizhaeva-Wang intra-step decay divergence  **[Minor — Document]**

Scalar `OWImpactModel.apply_trade` decays the transient state at **every
call** by `exp(-κ)`. Tensor `TensorOWImpactModel` only decays at
`end_day()`. For Margin (which can invoke `apply_trade` 2× per step per
stock — e.g. `sell_long` then `sell_short`), the two paths diverge on the
second call:

* **Scalar**: second call sees transient × exp(-κ), i.e. essentially zero.
* **Tensor**: second call sees transient without decay.

For MACE (one net trade per stock per step) the two match exactly. For
daily trading the divergence is real but small — within a single day the
second trade pays a slightly larger transient cost in the tensor version.

**Action**

Either:
1. Apply decay inside `TensorOWImpactModel.apply_trade` *before* reading
   `prev_transient` (matches scalar, breaks end-of-day invariance), or
2. Document the divergence in the README and keep the current semantics
   (which arguably are more physically correct — the order-book "resets"
   on a daily basis, not per-trade).

The regression test should call `apply_trade` twice on the same
(env, stock) within one step and assert the ratio of the second
`trans_cost` to the first matches the expected decay.

---

### 1.4 Normalizer stats not *shared* across Workers (only warned)  **[Medium — Scope-Gated]**

`train_with_epoch_evaluation` now emits:

```python
log.warning(
    "Observation normalizer stats are not shared across workers; "
    "with num_workers=%s the last worker to save wins.",
    num_workers,
)
```

This is a good short-term mitigation, but in the most common HPO path
(`gpu_id >= 0`, `num_workers > 1`) every ElegantRL Worker still instantiates
its own `TorchRunningMeanStd`, and `MACEVecEnv.close()` writes to the
shared `normalizer_state_path` in an unsynchronised race. The Evaluator
reads that file at the start of eval, so the eval statistics are
whichever Worker finished last.

**Two viable fixes**

1. **Cheap**: raise (not warn) when `num_workers > 1 and use_obs_normalizer`.
   Document that vec MACE only supports `num_workers=1` for now.
2. **Proper**: put `mean/var/count` in a
   `multiprocessing.shared_memory.SharedMemoryManager`-backed block and
   have every Worker apply `update_from_moments` with a lock. The
   ElegantRL bridge is single-process by default (`if_single_process=True`
   in `train_with_epoch_evaluation`), so until users explicitly opt into
   multi-worker, option 1 is sufficient.

**Recommendation**: option 1 for now, with a TODO pointing at option 2.

---

### 1.5 Buy-affordability loop still Python-level  **[Perf — Low]**

In `MACEVecEnv.step()`:

```python
for stock_idx in range(self.stock_dim):
    stock_shares = buy_shares[:, stock_idx]
    stock_total = requested_buy_total[:, stock_idx]
    can_buy = (stock_shares > 0) & (stock_total <= available_cash)
    accepted_buy_shares[can_buy, stock_idx] = stock_shares[can_buy]
    available_cash[can_buy] -= stock_total[can_buy]
```

100 kernel launches (one per stock) for the affordability step. For
`num_envs=2048, stock_dim=100` this is still a meaningful fraction of the
per-step time.

The scalar semantics are *order-dependent* (stock 0 is tested first, then
1, ...), which prevents a trivial `cumsum` vectorisation — a later small
stock can be rejected because an earlier large stock consumed the budget.

**Two alternatives**

1. **Greedy by cost** (different from scalar): sort stocks by
   `requested_buy_total`, cumulative-sum, accept while `cumsum <= cash`.
   This is fully vectorised but has different acceptance semantics.
2. **Two-pass optimistic**: take all buys at face value, scale them down
   by `min(1, cash / total_requested)` per env. Different semantics again
   (partial fills for all), but trivially vectorised and arguably closer
   to how a real execution algo would allocate a budget.

Either would break scalar↔vec parity (#2 and #3 tests pin the current
order). Flag this as a future optimisation that needs a separate parity
spec.

---

### 1.6 `get_state` returns the raw `_obs_buffer` when normalizer is off  **[Low — Correctness Hazard]**

In `MACEVecEnv.get_state()`:

```python
obs = self._obs_buffer
offset = 0
obs[:, offset : offset + 1] = cash_pct
...
if self.obs_normalizer is not None:
    ...
    obs = (obs - self.obs_normalizer.mean) / ...    # new tensor
    obs = obs.clamp(...)
return obs
```

When the normaliser is **off** we return `self._obs_buffer` directly.
Any downstream consumer that keeps a reference (e.g. a replay buffer
that stores observations by reference rather than by value) will see
its stored observation silently mutate on the next `step()` call.

The ElegantRL `ReplayBuffer` does `states[t] = state`, which makes a
tensor copy into a pre-allocated buffer, so it happens to be safe today.
But this is a latent foot-gun.

**Fix** — return `obs.clone()` in the no-normaliser branch:

```python
if self.obs_normalizer is not None:
    ...
    obs = obs.clamp(-self.obs_clip, self.obs_clip)
    return obs
return obs.clone()
```

---

### 1.7 `stock_order` computation when nothing trades  **[Micro — Low]**

`MarginTraderVecEnv.step()` always computes:

```python
stock_order = trade_shares.abs().sum(dim=0).argsort(descending=True).tolist()
```

Even on a pure-HOLD step (`actions == 0` for all envs), we still do a
sort + `.tolist()` (which forces a device→host sync). Guard it:

```python
if trade_shares.any():
    stock_order = trade_shares.abs().sum(dim=0).argsort(descending=True).tolist()
else:
    stock_order = []
```

The same guard is unnecessary for MACE (sell/buy blocks are already
gated by `sell_shares.gt(0).any()` / `buy_shares.gt(0).any()`).

---

## 2. Test-Suite Gaps

The new `test_vec_env_regressions.py` is excellent for pinning the bugs
fixed in the original audit. I recommend adding the following before the
next refactor to lock down the less-obvious semantics:

### 2.1 Multi-step permanent-impact parity

Current parity tests run 2 steps. Add a 5–10 step test against the scalar
env using a non-trivial action sequence, and assert:

* `env.impact_model.get_perm_state_array()` matches element-wise.
* Cumulative `cost`, `turnover`, `total_buy_value`, `total_sell_value`
  match.
* `total_asset` trajectory matches within float32 tolerance.

This catches subtle drift in perm-state accounting that per-step parity
misses.

### 2.2 Margin adjustment cascade parity

The existing margin parity test uses `margin_adjust_period=30` (default)
and only runs 2 steps, so `_margin_adjust_long` / `_margin_adjust_short`
are never invoked. Add:

```python
params = MarginEnvParams(margin_adjust_period=1)  # trigger every step
```

And assert long_cash, loan, short_limit, short_credit, short_equity match
scalar after the cascade. Bonus: construct a state where the loan shortfall
forces liquidation, so the rank-based sell-loop in
`_margin_adjust_long` is exercised.

### 2.3 OW transient-state parity

Add:

```python
@pytest.mark.parametrize("impact_cls", [OWImpactModel])
def test_mace_scalar_vec_single_env_parity_ow(impact_cls):
    ...
```

Specifically assert `impact_model.get_perm_state_array()` matches after
3+ steps. This will catch the issue in §1.3 if it is ever "fixed" in
the wrong direction.

### 2.4 Multi-env action divergence

The two multi-env aggregate-trades tests use identical actions across
envs. Add one with **different** per-env actions and assert:

* Each env's `total_asset[i]` evolves independently.
* Aggregate `info["trades"]` reports the correct sum of per-env
  executed shares.

This pins that vectorisation actually *is* over the env dimension
(rather than accidentally broadcasting env 0 to all envs).

### 2.5 `num_envs > 1` normalizer isolation

Add a test that creates a single `MACEVecEnv(num_envs=4,
use_obs_normalizer=True)`, runs a few steps with identical actions in
every env, and asserts every env's observation is bit-identical (same
normaliser stats per batch).

---

## 3. Cleanups (low-risk, one-commit fixes)

| # | File | Change |
|---|---|---|
| 3.1 | `mace_vec_env.py` | Delete dead `_buy_stock` and `_sell_stock`; they are no longer called. |
| 3.2 | `mace_vec_env.py` | Return `obs.clone()` when `obs_normalizer is None` (§1.6). |
| 3.3 | `margin_vec_env.py` | Pass `turnover_percentiles` down from `step()` to `_execute_*` helpers (§1.2). |
| 3.4 | `margin_vec_env.py` | Short-circuit `stock_order` computation on all-HOLD steps (§1.7). |
| 3.5 | `runner_utils.py` | Upgrade the multi-worker normalizer warning to a raise when `use_obs_normalizer=True` (§1.4). |
| 3.6 | `README` | Document OW intra-step decay semantics (§1.3) and the single-worker restriction (§1.4). |

---

## 4. Recommended Priority Order

1. **§1.6** — return `obs.clone()` (one-line correctness fix; zero cost).
2. **§1.1** — delete dead `_buy_stock` / `_sell_stock` (reduces surface area;
   the regression test needs to drop its call into
   `env._buy_stock(...)` and be replaced with an `apply_trades_batched`
   edge-case test).
3. **§1.4** — raise on `num_workers > 1 and use_obs_normalizer` until
   shared-memory normaliser lands.
4. **§2.1 / 2.2** — multi-step parity + margin-cascade parity tests.
5. **§1.2** — pass `turnover_percentiles` through (perf).
6. **§1.3** + **§2.3** — decide OW semantics, lock with test.

---

## 5. Overall Assessment

The port is now functionally **complete**. All three critical-crash paths
are fixed, parameters propagate correctly, batched GPU execution is wired
up for MACE's hot path, and scalar↔vec parity is enforced by CI. The
remaining items are polish (§1.1, §1.7), perf (§1.2, §1.5), one subtle
correctness-hazard (§1.6), and test-coverage gaps (§2). Nothing on this
list blocks production use — all are safe to address in a follow-up
milestone.

One architectural observation: the Margin vec env still runs a Python
loop over stocks in `step()`. This is reasonable given the
cover-then-buy / sell-then-short position-flipping logic that doesn't
vectorise trivially. If Margin throughput becomes the bottleneck, the
right move is to refactor `_execute_*` into two batched kernels (one
for the long side, one for the short side) that operate on
`(num_envs, stock_dim)` matrices, mirroring the MACE pattern. That is a
~200-line refactor and deserves its own PR.
