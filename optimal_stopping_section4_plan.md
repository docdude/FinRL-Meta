# Section 4 Implementation Plan

This file maps Zhao, Tse and Zheng (2026), arXiv:2604.02035v1 Section 4 into concrete implementation steps for this repository.

## Scope Split

- `rl_optimal_stopping_ou_repro.py`: a paper-oriented OU reproduction.
- `rl_optimal_stopping_v5.py`: the Nasdaq adaptation, kept separate from the paper reproduction.

## Section 4.1 Model Setup

Paper item: use an OU signal process

Implementation:
- simulate offline signal paths from
  $dP_t = \theta(\bar p - P_t)dt + \sigma dW_t$
- use the paper baseline parameters by default:
  $\gamma=\iota=1$, $\Psi=0$, $R=1$, $\rho=0.05$, $\theta=0.1$, $\bar p=0$, $\sigma=0.2$, $\varpi=0.5$, $k=2$
- use a time grid with $\Delta t = 0.1$ and $L = 100$ steps for the reproduction script

Code decisions:
- generate OU paths directly instead of downloading market data
- use exact OU transitions instead of rolling z-scores
- keep the utility and payoff functions identical to the paper notation

## Section 4.2 HJB and Control Structure

Paper item: define
- $G(p,b) = U(p - b - \Psi - R)$
- $\Delta_1(p) = V_1(p,p) - V_0(p)$
- $\Delta_2(p,b) = G(p,b) - V_1(p,b)$
- Gibbs mean intensities on $[0, M]$

Implementation:
- keep `_G_t`, `_hjb_src`, `_mean_lam`, and entropy-cost helpers
- define entry intensity only from $\Delta_1$
- define exit intensity only from $\Delta_2$
- do not apply signal gates, soft priors, minimum hold rules, or heuristic exit boosts in the OU reproduction

Code decisions:
- the OU reproduction uses pure Gibbs mean intensities
- the Nasdaq adaptation should strip the extra live signal priors so it is closer to the paper control law even if the data layer stays non-paper

## Section 4.3.1 HJB Benchmark

Paper item: compare RL outputs against a finite-difference HJB benchmark

Implementation:
- solve the HJB benchmark directly in `rl_optimal_stopping_ou_repro.py`
- use the paper grid on $[p_{min}, p_{max}] \times [b_{min}, b_{max}] = [-4,4] \times [-4,4]$
  with step size $0.05$ by default
- solve $V_1(p,b)$ first row-by-row in `b` with finite differences and the paper boundary conditions:
  - $\partial_p V_1(p_{min}, b) = 0$
  - $V_1(p_{max}, b) = G(p_{max}, b)$
- solve $V_0(p)$ second using the diagonal values $V_1(p,p)$ with the paper boundary conditions:
  - $V_0(p_{min}) = V_1(p_{min}, p_{min})$
  - $\partial_p V_0(p_{max}) = 0$
- iterate each solve with first-order linearization of the HJB source term using the optimal mean intensity
- report benchmark free boundaries and compare RL against HJB on the interior region $[-3.2, 3.2]$

Code decisions:
- the benchmark is computed before RL training so the script stays theory-first
- the output includes value-function error metrics and boundary gaps between RL and HJB
- longer-horizon evaluation is reported separately and generated only after training so diagnostics do not perturb the benchmark RNG state

## Section 4.3.2 Offline Policy Iteration

Paper item: simulate `(J, B)` trajectories on offline signal paths and minimize one-step TD errors

Implementation:
- parameterize `V0(p)` and `V1(p,b)` with two-layer ReLU networks
- use 32 hidden units to match the paper configuration more closely
- for each offline signal path, simulate multiple `(J, B)` trajectories from Bernoulli entry and exit events using
  $q = 1 - \exp(-\bar\lambda \Delta t)$
- initialize half the simulated trajectories with `J=0, B=0`
- initialize the other half with `J=1` and sample `B` uniformly from the configured `b` range
- compute one-step TD errors:
  - regime 0 uses continuation to `V0` or `V1`
  - regime 1 uses continuation to `V1` or terminal payoff `G`
- jointly update both networks from the average TD-squared loss

Code decisions:
- the OU reproduction uses one-step TD only
- no minimum hold
- no target network smoothing
- no diagonal anchors, supervised exit anchors, or periodic re-anchoring
- no alternating freeze schedule

## Appendix B Algorithm Mapping

Algorithm 1 step 1:
- simulate `(J, B)` from offline `P` paths using the current value networks and closed-form Gibbs controls

Algorithm 1 step 2:
- compute `delta0` and `delta1` exactly as one-step TD errors on the simulated samples

Algorithm 1 step 3:
- minimize mean `(delta0^2 + delta1^2)` over all regime-0 and regime-1 samples

Algorithm 1 step 4:
- update `V0` and `V1` together with gradient descent

## Nasdaq Adaptation Cleanup

Completed changes in `rl_optimal_stopping_v5.py`:
- removed the soft entry prior layered on top of Gibbs entry intensity
- removed the soft exit boost layered on top of Gibbs exit intensity
- removed the mandatory minimum hold from the live decision path
- switched the active training loop to one-step offline TD on simulated `(J, B)` transitions
- disabled target-network smoothing, anchor penalties, and periodic re-anchoring in the default path

Current mismatches intentionally left in place:
- real-market Nasdaq rolling-z-score environment
- realized-return backtest reporting instead of HJB benchmarking

Current practical consequence to monitor:
- the more paper-like one-step TD core can materially change the learned entry region on Nasdaq data, so theory alignment now needs to be balanced against entry selectivity and realized trading behavior

## Validation Plan

- OU reproduction:
  - import check
  - quick training run with reduced path count and iteration count
  - verify HJB solver convergence, plots, boundary summary, and RL-vs-HJB error metrics
  - quick 40-iteration run currently yields entry-boundary gap about `0.03`, exit-boundary gap about `0.00` at `b=0`, and about `0.04` at `b=1`
  - full 1200-iteration benchmark now yields:
    - HJB entry boundary about `-0.495`
    - RL entry boundary about `-0.470`
    - `V0` mean absolute error about `0.016`
    - `V1` mean absolute error about `0.029`
    - exit-boundary gaps about `0.043`, `0.049`, `0.048` for `b=-1,0,1`
  - horizon diagnostics on the full run show that low finite-horizon completion is only partly censoring:
    - at `100` steps, `completion|entry` is about `5.8%` and `open_at_horizon` about `69.9%`
    - at `300` steps, `completion|entry` rises to about `41.7%`, but `open_at_horizon` is still about `52.0%`
    - interpretation: the learned OU policy is close to the HJB benchmark in value and boundary space, but under `M=50, eta=1e-5` it still implies a genuinely slow exit profile on Monte Carlo rollouts
- Nasdaq adaptation cleanup:
  - import check
  - run the existing script or a narrow function-level smoke test to ensure the simplified control law still executes