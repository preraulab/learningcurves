# learningcurves

MATLAB implementations of the Prerau Lab's state-space learning-curve estimators. Given a time series of trial-by-trial performance — binary (correct/incorrect), continuous (reaction time), or both recorded simultaneously — these estimators recover the underlying *cognitive state* (an unobservable latent variable representing the subject's understanding of the task) as it evolves across trials, with proper uncertainty quantification.

Methodologically, this is a **state-space** framework:

- **State equation.** The cognitive state `x_k` evolves as a Gaussian random walk (with optional AR(1) drift) across trials. You can't observe `x_k` directly.
- **Observation equation.** What you *can* observe are performance measures on each trial. Binary (correct/incorrect) observations follow a Bernoulli whose probability is a logistic function of `x_k`. Continuous observations (e.g., log reaction time) follow a Gaussian with mean linear in `x_k`.
- **Inference.** Expectation-maximization (or particle-filter Monte Carlo) produces a smoothed estimate `E[x_k | data]` with per-trial confidence bands.

The key conceptual contribution: fusing binary and continuous measures into one estimate gives a more credible and accurate cognitive-state trajectory than either measure analyzed separately.

## Methods in this toolbox

| Subdirectory | Observation model | Backend | When to use |
|---|---|---|---|
| [`binsmoother/`](#binsmoother) | Binary only | EM + forward/backward filter (Smith–Brown style) | You only have correct/incorrect per trial |
| [`mixedsmoother/`](#mixedsmoother) | Binary + continuous simultaneously | EM + forward/backward filter | You have paired binary + continuous (e.g., correct-or-not + reaction time) per trial |
| [`particle_filter/`](#particle-filter) | Binary, continuous, or mixed | Sequential Monte Carlo | Missing data in either channel; non-Gaussian state noise; or you want posterior particles rather than Gaussian summaries |

All three recover smoothed trajectories of the same underlying cognitive state — the difference is which observations they can consume and which approximations they make.

## Install

```matlab
addpath(genpath('/path/to/learningcurves'));
```

## Requirements

- **MATLAB R2020a or later**
- **Statistics and Machine Learning Toolbox** — for random sampling in the particle filter and some diagnostics
- No other toolboxes required

## binsmoother

Binary-only EM smoother. Fits a Gaussian-random-walk state model with a Bernoulli-logistic observation model. Canonical use: `k out of N` correct trials per session, recover a smooth `p(correct | trial)` curve with 5/50/95-percentile bands.

**Entry point:**

```matlab
[p05, p95, pmid, pmode, pmatrix, xnew, signewsq] = ...
    binsmoother(Responses, SigE, BackgroundProb, NumberSteps)
```

| Argument | Meaning |
|---|---|
| `Responses` | `1×K` number-correct-per-trial (supports "k out of N" when `max(Responses) > 1`) |
| `SigE` | initial guess for random-walk standard deviation of the latent state |
| `BackgroundProb` | chance-level correct probability |
| `NumberSteps` | max EM iterations (converges when state-variance change < 1e-8) |

Returns `p05`/`p95`/`pmid`/`pmode` curves on `p(correct)`, a per-trial `pmatrix` giving the posterior probability that performance exceeded chance, and the latent state `xnew` with variance `signewsq`.

Based on Smith, Brown et al., *J. Neurophysiol.* 2004. Our implementation was used as the binary-only special case in the mixed-filter work.

## mixedsmoother

Simultaneous binary + continuous EM smoother. This is the method introduced in Prerau et al., *Biological Cybernetics* 2008 and extended with the EM estimator in Prerau et al., *J. Neurophysiol.* 2009.

**Model:**

```
State:       x_k = ρ · x_{k-1} + v_k,       v_k ~ N(0, σ²_v)
Binary obs:  P(n_k correct) = 1 / (1 + exp(-(μ_1 + γ·x_k)))
Continuous:  z_k = α + β · x_k + e_k,        e_k ~ N(0, σ²_e)
```

All parameters `(α, β, γ, ρ, σ²_e, σ²_v)` are estimated by EM — no tuning required beyond initial guesses. `γ` is typically fixed to zero in the default implementation (binary probability is modulated by `μ_1` alone) but can be freed for joint-coupling models.

**Entry point:**

```matlab
[alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a] = ...
    mixedlearningcurve(N, Z, background_prob, rhog, alphag, betag, sig2eg, sig2vg)
```

| Argument | Meaning |
|---|---|
| `N` | `2×K` — row 1: number correct per trial; row 2: max possible correct per trial |
| `Z` | `1×K` continuous observations (typically `log(reaction_time)`) |
| `background_prob` | chance-level probability of correct response |
| `rhog, alphag, betag, sig2eg, sig2vg` | initial guesses for the EM |

Returns the EM-estimated model parameters, the smoothed latent state trajectory `xnew` with variance `signewsq`, the logit-transformed chance-bias `muone`, and the backward-smoother gain `a`.

Because the continuous channel tightens the otherwise-uncertain binary estimate, the mixed smoother typically produces **substantially narrower confidence bands and more accurate recovery of the true state** in simulation than the binary-only or continuous-only smoothers (Prerau et al. 2008, 2009).

## particle_filter

Sequential Monte Carlo estimator for cases where the EM smoothers' assumptions (Gaussian state noise, linear continuous observation model, complete data) don't hold. Produces a full posterior distribution as a set of weighted particles at each trial, rather than a Gaussian mean+variance.

**Entry point:**

```matlab
[param_ests, particles] = learningcurve_pfilter(times, data, ...
    num_particles, smoother, prog_bar, plot_on)
```

| Argument | Meaning |
|---|---|
| `times` | `1×T` observation times in seconds |
| `data` | `2×T` — row 1: binary observations (0/1), row 2: continuous; `NaN` marks missing data in either channel |
| `num_particles` | particle-population size (default: 5000) |
| `smoother` | `true`: forward + backward smoothing; `false`: forward-only filter (faster, higher posterior variance) |
| `prog_bar`, `plot_on` | UI options |

Returns `param_ests` (estimated parameters and posterior summaries) and `particles` (the raw particle matrix across time — useful for arbitrary posterior queries).

Model mirrors the mixedsmoother's (Gaussian-random-walk state, Bernoulli binary obs, Gaussian continuous obs) but the particle filter representation handles:

- **Missing data** — either channel can be `NaN` on any trial without special treatment
- **Non-Gaussian state noise** — just change the state-transition sampler
- **Non-linear continuous observation models** — just change the likelihood function

Specialized variants in this folder:

- `binary_learningcurve_pfilter_ptile.m` — binary-only particle filter with percentile extraction
- `learningcurve_pfilter_ptile.m` — mixed with percentile extraction on the posterior
- `learningcurve_pfilter_ptile_bino_only.m` — binary-only percentile variant
- `sleeponset_pfilterEEG_3states.m` — 3-state variant for EEG-based sleep-onset detection

## Citation

If you use this toolbox, please cite the appropriate methods paper(s):

### Primary: EM smoother for binary + continuous

> **Prerau MJ**, Smith AC, Eden UT, Kubota Y, Yanike M, Suzuki W, Graybiel AM, Brown EN.
> "Characterizing learning by simultaneous analysis of continuous and binary measures of performance."
> *Journal of Neurophysiology* 102(5):3060–3072, 2009.
> doi: [10.1152/jn.91251.2008](https://doi.org/10.1152/jn.91251.2008)

Applied to monkey within-session rapid association-learning data (Wirth/Suzuki) and mouse multi-day T-maze learning (Graybiel). Introduces formal definitions of the learning curve, reaction-time curve, ideal-observer curve, learning trial, and between-trial performance comparisons.

### Foundational: mixed-filter algorithm

> **Prerau MJ**, Smith AC, Eden UT, Yanike M, Suzuki WA, Brown EN.
> "A mixed filter algorithm for cognitive state estimation from simultaneously recorded continuous and binary measures of performance."
> *Biological Cybernetics* 99(1):1–14, 2008.
> doi: [10.1007/s00422-008-0227-z](https://doi.org/10.1007/s00422-008-0227-z)

Introduces the recursive mixed filter. Kalman filter (for continuous-only data) and the Smith–Brown binary recursive filter are shown as special cases.

A machine-readable citation is in [`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository" button uses it.

## Related concepts

The 2009 paper formalizes several trial-based concepts used downstream:

- **Learning curve** — `P(correct | trial)` over trials, with confidence bands.
- **Reaction-time curve** — posterior `E[reaction_time | trial]` over trials, also with bands.
- **Ideal-observer curve** — posterior probability that a trial is above chance given *all* data observed up to and including that trial (accounts for retrospective knowledge).
- **Learning trial** — the first trial at which the posterior probability of above-chance performance exceeds a user-specified criterion (e.g., 0.95).
- **Between-trial comparisons** — posterior probability that performance at trial `j` exceeds performance at trial `i < j`.

All are derived directly from the `xnew` / `signewsq` outputs of the smoothers in this toolbox.

## Documentation

Full API reference: **https://preraulab.github.io/learningcurves/**

Each function's full docstring is visible both in `help <function>` at the MATLAB prompt and on the hosted site.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
