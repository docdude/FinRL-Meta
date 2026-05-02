# Re-Audit Pass #3 - VEC_ENV_REAUDIT_V3

This pass verifies the follow-up fixes from the prior re-audit, confirms that
the remaining correctness hazards are closed, and separates the two leftover
performance items from blocking review concerns.

## 1. Verification of Previous Re-Audit Items

| Item | Status | Evidence |
| --- | --- | --- |
| 1.1 Delete dead `_buy_stock` / `_sell_stock` | Fixed | Helpers are gone from `meta/env_market_impact/vec/mace_vec_env.py` |
| 1.2 Thread `turnover_percentiles` through Margin helpers | Fixed | `_execute_sell_direction` and `_execute_buy_direction` now take `turnover_percentile`; `step()` computes it once |
| 1.3 OW intra-step decay semantics | Documented + tested | `test_mace_scalar_vec_single_env_parity_ow` pins permanent-state parity |
| 1.4 Multi-worker normalizer | Hardened | `train_with_epoch_evaluation()` now raises `ValueError("use num_workers=1 ...")`; pinned by `test_train_with_epoch_evaluation_rejects_mult_worker_normalizer` |
| 1.6 `obs.clone()` when normalizer disabled | Fixed | Final return is `obs.clone()` in `get_state()`; pinned by `test_mace_get_state_returns_clone_when_normalizer_disabled` |
| 1.7 `stock_order` short-circuit on HOLD | Fixed | `if trade_shares.any(): stock_order = ...` guard in Margin `step()` |
| 2.1 Multi-step perm-state parity | Added | `test_mace_multistep_permanent_impact_and_path_parity` checks permanent state, cumulative cost/turnover/buy/sell, and total asset across 5 steps |
| 2.2 Margin cascade parity | Added | `test_margin_adjustment_cascade_parity` uses `margin_adjust_period=1` plus a crafted `initial_margin_state` that forces liquidation |
| 2.3 OW transient parity | Added | `test_mace_scalar_vec_single_env_parity_ow` |
| 2.4 Multi-env action divergence | Added | `test_mace_multi_env_action_divergence_tracks_per_env_state` verifies per-env holdings plus aggregate trades |
| 2.5 Normalizer isolation | Added | `test_mace_multienv_normalizer_keeps_identical_env_rows_in_sync` |

All regressions from the original audit and the re-audit are now pinned by
tests.

Current test status:

- `unit_tests/test_vec_env_regressions.py` now contains 23 test functions.
- `pytest` currently collects 26 cases because the scalar/vec parity tests are
  parametrized across multiple impact models.
- Coverage spans scalar/vec parity for Baseline, AC, Sqrt, and OW, plus
  determinism, aggregate trades, normalizer transfer, margin cascade, and
  maintenance-warning robustness.

## 2. Remaining Performance Items

### 2.1 MACE buy-affordability per-stock loop

```python
for stock_idx in range(self.stock_dim):
    stock_shares = buy_shares[:, stock_idx]
    stock_total = requested_buy_total[:, stock_idx]
    can_buy = (stock_shares > 0) & (stock_total <= available_cash)
    accepted_buy_shares[can_buy, stock_idx] = stock_shares[can_buy]
    available_cash[can_buy] -= stock_total[can_buy]
```

Recommendation: keep as-is.

Reasoning:

- The scalar env iterates stocks in index order and short-circuits against the
  running cash balance.
- This loop reproduces that behavior exactly.
- `test_mace_scalar_vec_single_env_parity` and
  `test_mace_multistep_permanent_impact_and_path_parity` rely on that ordering.
- Fully vectorized alternatives such as scale-down or greedy-by-cost would
  change acceptance semantics and break parity.

Back-of-envelope cost estimate:

- At `stock_dim = 100`, `num_envs = 2048`, this adds roughly 100 small
  per-stock launches worth about 0.5-1 ms of launch overhead per step.
- Over a 252-step episode that is about 0.25 s per episode, which is well below
  1 percent of end-to-end PPO training time at the current training settings.

Low-risk future optimization if profiling ever makes this hot:

```python
@th.jit.script
def _sequential_affordability(buy_shares, requested_buy_total, cash):
    num_envs, stock_dim = buy_shares.shape
    accepted = th.zeros_like(buy_shares)
    available = cash.clone()
    for i in range(stock_dim):
        can_buy = (
            (buy_shares[:, i] > 0)
            & (requested_buy_total[:, i] <= available)
        )
        accepted[:, i] = th.where(
            can_buy,
            buy_shares[:, i],
            th.zeros_like(buy_shares[:, i]),
        )
        available = available - th.where(
            can_buy,
            requested_buy_total[:, i],
            th.zeros_like(available),
        )
    return accepted
```

That preserves the exact sequential semantics while reducing Python and launch
overhead. It is a reasonable future PR, but not a review blocker.

### 2.2 Margin per-stock loop in `step()`

```python
for stock_idx in stock_order:
    if (trade_shares[:, stock_idx] < 0).any():
        trade_value, trade_cost, trade_sides, aggregate_rows = self._execute_sell_direction(...)
```

Recommendation: keep as-is.

Reasoning:

- Margin position-flipping semantics are genuinely stateful per stock.
- The helpers mutate `self.stocks`, `self.long_cash`, `self.short_limit`,
  `self.short_credit`, and other state that later helpers observe.
- `test_margin_scalar_vec_single_env_parity` locks in that behavior.
- This is fundamentally more serial than MACE and should not be casually
  vectorized.

Cost estimate:

- The overhead is worse than MACE when `num_envs` is small because each stock
  can trigger several helper paths plus impact and maintenance checks.
- Once `num_envs` is large enough, the actual kernel work dominates and the
  relative overhead drops.

Reasonable optimization path if Margin throughput later becomes a bottleneck:

1. Split trades into four disjoint groups before the loop:
   `long_cover_group`, `long_buy_group`, `short_sell_long_group`, and
   `short_open_group`.
2. Run one batched `apply_trades_batched` per group in the same canonical
   sell-then-buy stock order.
3. Apply one batched affordability mask per group while preserving the same
   sequence as the scalar env.

That is a substantial refactor and belongs in a dedicated follow-up PR, not in
this audit closure.

## 3. Recommendation Summary

| Item | Recommendation |
| --- | --- |
| MACE buy-affordability loop | Keep for now. If profiling shows it is hot, wrap the sequential logic in `@torch.jit.script` as a zero-semantics-change optimization. |
| Margin per-stock step loop | Keep for now. If Margin becomes a bottleneck, refactor into grouped batched kernels while preserving scalar order semantics. |

Neither remaining item is blocking. Parity is more valuable than the marginal
speed-up at the current scale, and the current behavior is now locked down by
the regression suite.