# Results & Analysis — draft

> Draft for the ICLR-format report (Results + the analysis half of Method/Discussion).
> Markdown now for fast iteration; port to the LaTeX template once the narrative is
> agreed. Numbers are val/mse_ft (ft², mean over the 11 entities, the Kaggle-comparable
> metric) on the held-out fold0 split unless noted. Source: experiments.md.

## 1. Setup and metric

We forecast H=12 future steps for N=11 entities (10 players + ball) from C=8 context
frames. The evaluation metric is ADE expressed as the **mean squared error of a single
predicted trajectory** over the horizon, in feet². This single-prediction framing is
central to everything below: for a squared-error metric the optimal submission is the
**conditional mean** E[y | past], so multimodality can only be exploited insofar as it
sharpens that mean — a stochastic or best-of-K submission is not permitted.

We report local cross-validation throughout rather than relying on the Kaggle
leaderboard; our production model's val 3.07 corresponds to Kaggle 2.98 (the held-out
gap is small and favorable), confirming the local split is a faithful proxy.

## 2. Models and the production system

We compare a **graph/relational** family against a **geometric-equivariant** one:

- **EqMotion** — an exactly O(2)-equivariant trajectory network. Court-frame symmetry
  is hard-wired; reflection/rotation test-time augmentation is provably a no-op.
- **MART** — a relational transformer with K=20 multi-hypothesis decoder heads (not
  equivariant); entity-type embeddings distinguish TeamA/TeamB/ball.

Two design choices moved both models materially and generalize across the family:
**isotropic normalization** (a single shared positional scale, a prerequisite for valid
rotation augmentation — per-axis scaling turns rotation into shear) and adding the two
**hoops as static landmark nodes** (an explicit court frame). On MART we further apply an
**over-strong O(2) augmentation** (continuous rotation + reflection about court center);
because MART is only approximately invariant, this augmentation is a genuine regularizer
(on EqMotion it is a no-op). The 7.5M aug-MART reaches **val 3.11 / Kaggle 3.01**.

**Production** is a CSV-level blend **0.70·aug-MART + 0.30·EqMotion**, **val 3.07 /
Kaggle 2.98** — our best submission. (Headline report finding: a deliberately
over-strong O(2) prior plus explicit court-frame landmarks beats both plain equivariance
and learned-via-augmentation invariance.)

## 3. The central result: a task floor, not a modeling limit

The leaderboard top sits at Kaggle 2.6 (≈ val 2.68). We were unable to reach it, and the
interesting scientific content is **why**. Every lever we tried bottoms out at the same
place, and the evidence converges on a single explanation: **the residual error is a
property of the data, not the model.**

### 3.1 The error is almost entirely the ball

Decomposing the best model's val error by entity (ft²):

| entity group | val MSE |
|---|---|
| players (10) | 2.07 |
| ball (1) | 13.50 |
| **total (11)** | **3.11** |

Players are at their floor; the ball carries the error. The arithmetic of the gap to the
leader is stark — a total of 2.6 with players fixed implies a ball of ≈7.9, i.e. the
**entire 3.1→2.6 gap is the ball alone**.

### 3.2 The ball is genuinely multimodal, and its mode is not in the past

Three independent observations establish that the ball's error is irreducible from an
8-frame context rather than a capacity or optimization artifact:

1. **Oracle vs. mean gap.** Among MART's K=20 hypotheses, a per-entity *oracle* (pick the
   sample closest to ground truth) achieves **0.45 ft²** total — the model can almost
   always place *a* trajectory near the truth. But that good sample lies in the
   *low-density tail* of the sample cloud, not its center: every unsupervised aggregation
   (mean, median, trimmed-mean, closest-to-centroid, medoid, densest-cluster) lands within
   ±1% of the plain mean, and density-based selection moves the *wrong* way.

2. **A learned selector cannot recover it.** A supervised head-selector (per-agent MLP on
   the past + all-agent scene context + the K candidates, trained end-to-end) improves the
   ball by only ~5%, identically on a weak base model (16.8→15.8) and on our best model
   (13.5→12.9). The floor is invariant to base-model quality. Tellingly, *removing* the
   scene context improved the selector — so the configuration of the other players at t=C
   does **not** predict where the ball goes. The deciding signal is a *future decision*
   (pass left/right, drive, shoot) absent from the past.

3. **Possession structure is already exploited.** The ball is within 6 ft of its nearest
   player 72% of the horizon, yet anchoring the ball to that player's *ground-truth* future
   gives 26.9 ft² — twice the model's 13.5 — because the ball is near but never co-located
   with its handler (dribble offset, bounces, handoffs). Constant-velocity extrapolation is
   61.6 ft². The relational model already captures possession better than any hand-built
   prior.

### 3.3 The conditional mean is the floor, and we are at it

Because the metric rewards the conditional mean, the only way past 3.1 is a *sharper*
estimate of E[ball|past]. Two results show ours is already well-estimated:

- **Direct regression overfits and loses.** Training the model to regress the mean directly
  (`mean_mse` loss, identical recipe) reaches best val 3.20 then overfits to 3.58, *worse*
  than the 3.11 obtained by averaging the diverse best-of-K heads. The head-averaging is
  acting as a regularizer that yields a better-generalizing conditional-mean estimate than
  direct regression.
- **This also forecloses generative models.** Flow-matching / diffusion forecasters
  (MoFlow, LED, OmniTraj) win on the NBA benchmark on *minADE* (best-of-K). For a
  single-submission mean-MSE metric they too must emit the conditional mean — the same
  quantity `mean_mse` targets directly and cannot improve. Their headline advantage is
  structurally unavailable here.

### 3.4 Two architectures, one error

A cross-architecture diagnostic closes the case: EqMotion and MART — continuous O(2)
equivariance vs. relational attention, maximally different inductive biases — make
**the same errors on the same samples**, per-entity error correlation **ρ ≈ 0.93** across
all slices. No statistical combination of the two can bridge the gap (optimal blend weight
on MART is ~2%, *negative* on the ball).

### 3.5 Summary of the lever sweep

| lever | outcome |
|---|---|
| capacity (7.5M → 36M) | val 11 — bigger model spreads modes (helps minADE, wrecks the mean) |
| aggregation (6 reductions) | all within ±1% of the mean |
| learned selection (2 base models) | ~5% ball, floor invariant to base quality |
| possession anchor | 2× worse than the model (beats its own GT-handler oracle) |
| physics (constant velocity) | 61.6 ball |
| residual modeling | flat (min_ade) / overfit (mean_mse) |
| cross-architecture ensemble | ρ=0.93, optimal MART weight ~2% |
| direct mean regression | best 3.20, overfits to 3.58 |

Seven independent confirmations, two architectures: **~3.1 val / Kaggle ~2.98 is a task
floor for any single-prediction MSE model reading an 8-frame context.** The 2.6 leaderboard
result must exploit information or a submission mechanism outside this regime.

## 4. Takeaways

- **Graph structure helps where it should and not where it can't.** Court-frame landmarks
  and an over-strong symmetry prior are real, transferable gains for the *predictable*
  part of the motion (players, possession). They cannot manufacture information the past
  does not contain about the ball's discrete future decisions.
- **Metric-aware modeling matters more than architecture here.** The single largest
  conceptual error available on this task is conflating the three symmetry groups (the
  court's D2, the O(2) augmentation, the model's learned invariance) or training a
  best-of-K objective for a mean-of-K metric. Getting these right is what separates 3.5
  from 3.1; nothing separates 3.1 from 2.6 within the regime.
- **A rigorous negative result is the contribution.** We localize the irreducible error to
  ball multimodality and show, seven ways, that it is undetermined by the 8-frame past.

---
*Open items for the final draft: (i) port to ICLR LaTeX; (ii) pick 2 figures — suggest
the per-entity error bar chart and the oracle-vs-mean-vs-selector ball plot; (iii) decide
whether to fold the v3 aug-5k selector (val 3.05) into the reported production number or
keep it as an ablation.*
