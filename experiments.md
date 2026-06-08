# Experiments Log — NBA Trajectory Forecasting

Tracking what we tried, what changed in the pipeline, and the results. Metric is
`val/mse_ft` (mean squared error in feet², over the H=12 horizon), computed on the
**held-out validation fold** (`splits/fold0.json`, 497 sequences) unless noted as
Kaggle. Task: C=8 context → H=12 horizon, 11 entities (10 players + ball).

> **Headline (updated Jun 1):** best submission is a **0.70·aug-MART +
> 0.30·EqMotion blend → val 3.073, Kaggle 2.98 (confirmed) — PRODUCTION.** The
> backbone is **augmented + scaled MART** — isotropic norm + full O(2)
> augmentation (continuous rotation + reflections) + court-frame hoop nodes, 7.5M
> params, **5000 epochs** of cosine LR → honest val **3.11** (best ckpt 3.112),
> **Kaggle 3.01** on its own. aug-MART alone already beats the EqMotion 5-seed
> ensemble (val 3.25 / Kaggle 3.2) with a single model; blending in EqMotion at
> w=0.70 banks a further −0.04 (val 3.11→3.07, Kaggle 3.01→2.98) — the val gain
> held on the leaderboard. Note the val→Kaggle gap is *favorable* (~−0.10: Kaggle
> better than val), unlike EqMotion's roughly neutral gap. Overturns the earlier
> "task floor" verdict below. The previous best was EqMotion (iso + cosine + hoops): single
> val **3.33** / Kaggle **3.2**, 5-seed ensemble **3.25** (reflection TTA an
> exact no-op — empirical proof of O(2) equivariance). Starting context: ~3.7
> EqMotion, ~3.8 STGCNN, ~3.8 MART.
>
> **Why the "task floor" was wrong.** The earlier claim (below, struck through)
> rested on every EqMotion *and* old-MART intervention converging to the same
> error — but every one of those used a **300-epoch** schedule. The real binding
> constraint was the **training budget × augmentation**, not the data. EqMotion's
> hard O(2) equivariance makes geometric augmentation a *no-op* (TTA proves it),
> so it can't use augmentation to regularize a bigger model or absorb a longer
> schedule — it was stuck at *its* floor, not *the* floor. A **non-equivariant**
> MART can: full O(2) augmentation supplies the diversity that lets a 7.5M-param
> model train 5000 epochs with **zero overfitting** (train≈val≈0.026 at the end),
> and val fell 3.79 → 3.47 (1000 ep) → **3.11** (5000 ep).
>
> **~~Task-floor verdict (May 30–31) — SUPERSEDED, see above:~~** ~~the EqMotion
> ensemble at val 3.25 / Kaggle 3.2 sits at or near the task floor for both
> entity classes given the 8-frame past at H=12. Ball MSE bottoms out at ~13.7
> across every EqMotion intervention (capacity, weighting, ball-only loss) and
> ~15.8 across every MART intervention. Player MSE bottoms out at 2.19. Direct
> empirical closer: EqMotion ↔ MART error correlation ρ ≈ 0.93, so no
> cross-architecture averaging can bridge the gap.~~ This held only within the
> 300-epoch regime; the ρ≈0.93 figure was measured on the *old* under-trained
> MART, not the 3.11 aug-MART (correlation re-check on the new model is an open
> item). The 2.6 leaderboard top is reachable in principle (MART oracle min-of-K
> = 0.64); aug-MART at 3.11 closes ~40% of the 3.25→2.6 gap. Production
> submission decision pending the aug-MART Kaggle score
> (`submissions/solution_mart_aug_5k_best.csv`).

---

## 1. Metric & evaluation hygiene (important context)

Two measurement issues were found and fixed; numbers below are labelled by which
regime they were measured in.

- **Noisy single-window val.** The original val drew *one random window per
  sequence per epoch*, so `val/mse_ft` bounced ~3.2–3.9 between epochs and
  checkpoint-on-min caught lucky draws. **Fix:** `--full-val` (`NBAEvalSampler`,
  deterministic 8 evenly-spaced windows/seq) → stable selection signal.
- **Hoop nodes deflated the metric.** With `--add-hoops` the metric averaged over
  13 nodes incl. 2 *stationary* hoops (~0 error), deflating it: `2.82×13/11=3.33`.
  Kaggle scores only 11 entities. **Fix:** `n_landmarks` hparam strips landmark
  nodes from the val metric. Confirmed via `src/equivariance/diagnose_val.py`:
  2.82 (13 nodes) vs **3.33** (11 entities).
- **Result:** after both fixes, val tracks Kaggle well (3.33 val vs 3.2 Kaggle;
  3.46 vs 3.5 without hoops) → **val is a trustworthy selection metric**.
- The provided **test set has no labels** (8-frame context only), so it cannot be
  used as a local val set — the training-fold val is the only labelled proxy.

---

## 2. Results

### EqMotion (our best architecture)

| Run | Norm | Sched | Hoops | Val metric | val/mse_ft | Kaggle | Notes |
|---|---|---|---|---|---|---|---|
| baseline (pre-session) | aniso | none | no | single-window | ~3.7 | – | tuned hparams, lr=1e-4 defaults in `__main__` |
| aniso_sched | aniso | cosine | no | single-window | 3.50 | – | scheduler + tuned hparams |
| iso_sched | **iso** | cosine | no | single-window | **3.20** | 3.5 | lucky window draw (noisy metric) |
| iso_big | iso | cosine | no | single-window | 3.59 | – | hidden128/4-layer → overfit |
| aniso_sched_fv | aniso | cosine | no | full-val | 3.69 | – | |
| iso_sched_fv | **iso** | cosine | no | full-val | **3.46** | 3.5 | honest; val→Kaggle gap ~0.04 |
| iso_big_fv | iso | cosine | no | full-val | 3.88 | – | bigger still worse |
| iso_hoops_fv | iso | cosine | **yes** | full-val (13-node, buggy) | 2.82 | 3.2 | metric deflated by hoops |
| iso_hoops_fv | iso | cosine | **yes** | full-val (11-entity, honest) | **3.33** | **3.2** | best single model |
| **5-seed ensemble** | iso | cosine | yes | full-val (honest) | **3.25** | – | mean of 5 seeds; best overall |
| 5-seed ensemble + TTA | iso | cosine | yes | full-val (honest) | 3.25 | – | TTA = exact no-op (equivariance) |

**5 seeds:** 3.329 / 3.396 / 3.334 / 3.339 / 3.367 → mean 3.353, std 0.025 (low
variance confirms the metric is now stable). Ensemble averaging: 3.33 → **3.25**.

**Reflection TTA is an exact no-op** (single_plain == single_tta == 3.3294 to 4 dp;
ensemble_plain == ensemble_tta == 3.2497). This empirically proves EqMotion is
**exactly O(2)-equivariant** — the 4 court reflections leave predictions unchanged
(unlike STGCNN, which isn't equivariant and *did* benefit from reflection TTA).

Tuned hparams (Optuna study `eqmotion_v1`, trial #117): `hidden_nf=64,
hid_channel=64, n_layers=2, lr≈1.8e-3, weight_decay≈2e-6, batch_size=64,
grad_clip≈0.66`. The top-5 trials were val/loss 0.0134–0.0135 (essentially tied)
→ the model is **insensitive to these HPs** in range; structural changes (norm,
hoops) dominate.

### Other architectures (context, earlier work)

| Model | val/mse_ft | Notes |
|---|---|---|
| STGCNN (pos graph + autoreg decoder + TTA) | ~3.8–3.9 | best non-EqMotion; reflection TTA helped here (not equivariant) |
| STGCNN CVAE | ~4.3 | multimodal |
| SocialVAE | ~5.0 | multimodal |

**Why the stochastic/multimodal models lose:** the Kaggle metric is single-shot
mean MSE, which rewards predicting the *mean* mode. VAE-style models optimise
best-of-K (minADE), a different objective → they underperform on mean-MSE here.

### MART (fair comparison: 300 ep + cosine, schedule-matched to EqMotion)

Same `WindowEvalSampler` (8 deterministic windows/seq) and same `val/mse_ft`
definition (denormalized to feet², mean-of-K prediction, 11 real entities) as
EqMotion, so numbers are **directly comparable** in this row.

| Run | Loss | val/mse_ft | val/minADE (norm) | Notes |
|---|---|---|---|---|
| mart_minade_s1 | `min_ade` (paper-native) | **3.79** | 0.039 | fair MART baseline |
| mart_meanmse_s1 | `mean_mse` | 3.88 | 0.102 | "loss-aligned" — *worse* |

**MART loses to EqMotion by ~0.5 even after schedule alignment** (3.79 vs 3.33
single / 3.25 ensemble). Two findings worth flagging:

- **Undertraining was a real confound.** 100 ep (min_ade) → 4.22; 300 ep → 3.79.
  0.43 of the apparent MART deficit was schedule, not architecture.
- **The "obvious" loss switch backfired.** Swapping to `mean_mse` actually hurt
  (3.79 → 3.88), and val/minADE *tripled* (0.039 → 0.102). MART's K=20 decoder
  heads act as an **implicit ensemble**: `min_ade` keeps them *diverse* (each
  captures a distinct mode), and the mean-of-K prediction averages those modes
  into a sensible central estimate. `mean_mse` collapses all K to the same
  target → diversity dies, ensembling benefit vanishes, single mean prediction
  is worse. **Lesson:** for multimodal architectures with K decoder heads,
  loss-aligning to the leaderboard's single-shot metric can be counter-productive
  — the heads-as-ensemble structure was *load-bearing*.

### MART — augment + scale + train-long (NEW BEST, val 3.11) ⭐

The breakthrough run. Same `WindowEvalSampler` / `val/mse_ft` as above, so
directly comparable to every number in this doc. Recipe = give a **non-equivariant**
MART the same inductive structure that won for EqMotion (isotropic norm + court
symmetry + hoop nodes), but via **augmentation** instead of hard equivariance —
which (unlike EqMotion) lets capacity grow and the schedule lengthen.

| Run | Norm | Aug | Hoops | Params | Epochs | val/mse_ft |
|---|---|---|---|---|---|---|
| mart_minade_s1 (old baseline) | aniso | none | no | ~0.9M | 300 | 3.79 |
| mart_aug_iso_nohoops | **iso** | O(2)+mirror | no | 7.5M | 1000 | 3.55 |
| mart_aug_iso_hoops | **iso** | O(2)+mirror | **yes** | 7.5M | 1000 | 3.47 |
| **mart_aug_iso_hoops_5k** | **iso** | O(2)+mirror | **yes** | 7.5M | **5000** | **3.11** |
| — EqMotion single / ensemble (prev best) | iso | — | yes | 0.4M | 300 | 3.33 / 3.25 |

5k best-by-val ckpt = **3.112** (final epoch 3.123). Job 2954136, ~7h44m.

**Four load-bearing findings:**

1. **Augmentation enables the scale-up (the core mechanism).** At convergence
   train_loss ≈ val_loss ≈ 0.026 — *zero* overfitting on a 7.5M-param model over
   4.5k sequences, where every prior bigger-MART attempt overfit. Full continuous
   O(2) augmentation (rotation ∈ U(−180°,180°) + independent x/y reflections,
   applied as an exact isometry about the court center) supplies the diversity
   that regularizes both the larger model and the longer schedule. This is the
   lever EqMotion structurally *cannot* use — its hard O(2) equivariance makes
   the same augmentation an exact no-op (proven by the reflection-TTA test).
2. **Hoops transfer across architectures** (3.47 vs 3.55 at 1000 ep, +0.08). The
   court-frame lever is not EqMotion-specific — injecting the baskets as static
   nodes helps a relational transformer too. Good cross-arch ablation for the report.
3. **Epochs were the dominant lever, but only because augmentation unlocked
   them.** 300 ep (old) 3.79 → 1000 ep 3.47 → 5000 ep 3.11. Without augmentation a
   5000-epoch run on 4.5k sequences would overfit badly; with it, val keeps falling.
4. **Isotropic norm is a prerequisite, not just a nicety.** Rotation is only an
   isometry in the normalized frame under a *shared scalar* std; anisotropic
   per-axis std turns a rotation into a shear, so the augmented samples would be
   geometrically invalid. `--aug_rot_deg` auto-enables `--iso_norm`.

**Schedule-floor vs model-floor diagnostic (does the single-cosine LR cap us?).**
Because cosine LR and val both flatten near the end, a single run can't by itself
distinguish "model converged" from "LR ran out." Compared the two curves from the
5k log:

| epoch (% of run) | val/mse_ft | LR |
|---|---|---|
| 2499 (50%) | 3.373 | 2.55e-4 |
| 3499 (70%) | 3.208 | 1.11e-4 |
| 3999 (80%) | 3.134 | 5.69e-5 |
| 4499 (90%) | 3.138 | 2.20e-5 |
| 4749 (95%) | 3.123 | 1.30e-5 |
| 4808 (best) | **3.112** | — |
| 4894 | LR floor (1e-5) reached | 1.00e-5 |
| 4999 (final) | 3.123 | 1.00e-5 |

Val flattened at **~ep 4000–4500**, *before* the LR hit its 1e-5 floor at ep 4894
(Δval after LR-floor ≈ +0.004, i.e. flat). So 3.11 is a genuine **single-cosine
model floor, not a schedule strangle** — unlike the 1000-epoch run (where val was
still dropping when its schedule ended, which is exactly why extending to 5000 ep
paid). Implication: naïve "train even longer on one cosine" likely won't keep
paying. The unexplored variant is **warm restarts (SGDR)** — periodically
re-raising the LR to re-enter the productive mid-LR descent regime (the big gains
above happened at LR 1e-4 → 6e-5).

Submission generated: `submissions/solution_mart_aug_5k_best.csv` (verified: 1243
ids, no NaNs, coords in court range). Code: `--iso_norm --aug_rot_deg 180
--aug_court_mirror --aug_jitter` + `_augment_court()` in
[src/mart/main_nba_pt.py](src/mart/main_nba_pt.py) (isometry verified to float
precision) + best-by-val checkpointing + `configs/mart_nba_aug.yaml` (7.5M-param
config) + `jobs/train_mart_aug_5k.sh` (account=team-ai for >12h).

### MART — further experiments on aug-MART

Building on the best aug-MART (iso_norm + O(2) aug + hoops). All numbers are
`val/mse_ft` on the 497-sequence val fold, mean-of-K prediction.

**K-sweep — number of decoder heads**

| Model | K | Epochs | val/mse_ft |
|---|---|---|---|
| mart_exp_700ep (base) | 20 | 700 | 3.49 |
| *(manual)* | 5 | 700 | 3.23 |
| *(manual)* | 10 | 700 | 3.33 |
| *(manual)* | 40 | 700 | 6.31 |
| mart_k5_5k | 5 | 5000 | 3.16 |
| mart_k10_3k (curriculum) | 10 | 3000 | 3.14 |
| mart_k3_5k *(manual)* | 3 | 5000 | 3.12 |

At 700 epochs K=5 already beats K=20. K=40 completely failed — too many heads
to coordinate in a short schedule. With longer training K=3 edges out K=5 and
K=10; the diversity benefit of more heads is outweighed by the difficulty of
training them jointly.

**Loss variants — curriculum vs alternatives (K=20, 2k epochs)**

| Model | Loss | val/mse_ft |
|---|---|---|
| mart_curriculum_2k | curriculum (min-ADE → mean-MSE, sw 1k) | 3.09 |
| mart_soft_wta_2k | soft winner-takes-all (temp=0.5) | 3.32 |
| mart_laplace_nll_2k | Laplace NLL | 3.38 |

Curriculum is the clear winner. Soft-WTA converges to a similar basin as plain
min-ADE. Laplace NLL penalises uncertainty rather than directly optimising the
mean prediction, so it's misaligned with single-shot MSE scoring.

**CFI module — transplanting the cross-modal future interaction decoder**

| Model | K | CFI | Curriculum | Epochs | val/mse_ft |
|---|---|---|---|---|---|
| mart_cfi_700ep | 20 | yes | no | 700 | 3.33 |
| mart_k10_cfi_5k | 10 | yes | no | 5000 | 3.14 |
| mart_k10_curriculum_5k | 10 | no | yes | 5000 | 3.14 |
| mart_k10_cfi_curriculum_5k* | 10 | yes | yes | 5000 | 3.11 |
| mart_cfi_curriculum_5k | 20 | yes | yes | 5000 | 3.13 |

CFI at 700 epochs gives no benefit. At 5k, combining K=10 + CFI + curriculum
reaches 3.11 — the best CFI result — consistent with the K-sweep: K=10 is
better than K=20 at this compute scale. The CFI and curriculum gains appear
largely additive at K=10.

*(\*) checkpoint `mart_k10_cfi_curriculum_5k_best.ckpt`, no dedicated job script.*

**Curriculum tuning — developing the best single model**

| Model | Switch epoch | Phase-2 LR | Epochs | val/mse_ft |
|---|---|---|---|---|
| mart_curriculum_2k | 1000 | 0.0005 | 2000 | 3.09 |
| mart_curriculum_4k *(manual)* | 2000 | 0.0005 | 4000 | 3.11 |
| **mart_curriculum_4k_sw1k_lrdrop** | **1000** | **0.00005** | **4000** | **2.98** |

Switching at epoch 1000 (rather than 2000) gives phase 2 more time to directly
optimise MSE. The 4× LR drop with a fresh cosine schedule at the switch is the
decisive change — it prevents phase 2 from overshooting the mode structure built
in phase 1. This is the best standalone MART model.

---

## 3. What changed in the pipeline (code)

All in `src/equivariance/eqmotion_nba.py` unless noted.

1. **Cosine LR scheduler + warmup** in `configure_optimizers` (`--lr-scheduler`,
   `--warmup-epochs`). Scheduler + tuned hparams alone: 3.7 → 3.50.
2. **Isotropic normalization** (`--iso-norm`): single shared scalar std for x,y
   instead of per-axis (σ_x≈29.7 vs σ_y≈11.5 ft). Restores EqMotion's O(2)
   equivariance. 3.50 → 3.20 (single-window) / biggest pre-hoops lever.
3. **`__main__` is argparse-driven**; long training (300 ep) with `ModelCheckpoint`
   + `EarlyStopping` on `val/mse_ft`.
4. **Deterministic multi-window validation** (`NBAEvalSampler`, `--full-val`) for a
   stable, low-variance selection metric.
5. **Honest landmark-aware metric** (`n_landmarks`): static hoop nodes stripped
   from val ADE/FDE/MSE so it matches the 11-entity Kaggle metric.
6. **Court-frame hoop nodes** (`--add-hoops`): 2 static nodes at (±41.75, 0). Under
   iso norm: honest val 3.46 → 3.33, Kaggle 3.5 → 3.2. **Biggest single lever.**
7. **`--submit` loads the BEST checkpoint** (not the final-epoch model).
8. New scripts: `submit_eqmotion.py` (submit-from-checkpoint, no retrain),
   `diagnose_val.py` (metric-bug diagnostic), `ensemble_eqmotion.py` (multi-seed
   ensemble + reflection TTA eval/submit).
9. Job scripts now rsync **checkpoints and submissions** back to `$HOME` after each
   run (scratch is wiped by the next job's `rsync --delete`).
10. **Per-entity val diagnostics.** `validation_step` now also logs
    `val/mse_ball` and `val/mse_players` (ball = isplayer==0 among the first 11
    real entities; landmarks already stripped). Lets us see where residual MSE
    concentrates — and tells the ball-vs-players story above.
11. **Generalized court landmarks.** `LANDMARK_SETS` dict + `landmarks_from_spec()`
    helper + `--landmarks PRESET[,PRESET...]` CLI (e.g. `hoops,ft,3pt`). Each
    preset is a list of `(x, y, team_id)` tuples; distinct `team_id`s give each
    landmark type its own embedding. Back-compat `--add-hoops` kept as a
    shortcut for `--landmarks hoops`.
12. **Ball-weighted loss.** `MultiStepMSE(ball_weight=...)` + `--ball-weight`
    CLI. Weight=1 is plain MSE (matches prior behavior); weight>1 trains a
    ball-specialist while still seeing all agents as joint context.

---

## 4. Key insights

- **Equivariance group mismatch (and why it still helps).** EqMotion imposes full
  **O(2)** (continuous rotation + reflection) equivariance, but a basketball court
  only has **D2** symmetry (order 4). The over-strong prior is slightly misspecified
  yet wins because it's *exact and free* (zero params/samples, every orientation
  seen exactly), whereas teaching symmetry via augmentation is *approximate and
  paid* (costs capacity, only sampled angles). With ~4.5k sequences the
  variance-reduction dominates the small bias. At H=12 the misspecified part
  (basket-directed flow) barely bites — local kinematics dominate.
- **Three groups, don't conflate them (terminology — applies to both models).**
  It's tempting to say "we have D2 equivariance," but there are *three distinct*
  things:
  1. **The task's true symmetry is D2** (order 4: identity, reflect-length,
     reflect-width, 180° rot). A 37° rotation is NOT a court symmetry (it moves
     the baskets off-axis).
  2. **What we impose/augment with is O(2)** — the *full continuous circle* of
     rotations + reflections, *deliberately over-strong* vs the true D2.
     `--aug_rot_deg 180` draws θ ~ U(−180°,+180°) continuously (90° is just one
     of infinitely many sampled angles), NOT discrete 90°/180° flips.
  3. **What aug-MART actually ends up with is *approximate, learned* O(2)
     INVARIANCE — not equivariance.** It has zero architectural symmetry; the
     augmentation teaches soft invariance, paid in capacity/samples, exact only
     at sampled angles. Contrast EqMotion, which is *exactly* O(2)-EQUIVARIANT by
     construction (the reflection-TTA no-op proves predictions are identical
     under the group, for free). So "aug-MART is D2/O(2)-equivariant" is wrong on
     two counts: we impose the bigger O(2) (not D2), and it's approximate
     invariance (not exact equivariance). Why over-impose O(2) when truth is D2:
     local kinematics are genuinely rotation-isotropic (only the court *frame*
     breaks rotation symmetry), so the O(2) prior is right for the dynamics; the
     hoop nodes then re-inject the D2 court frame as *features*. Over-strong
     rotation buys variance reduction (why the 7.5M model trains 5000 ep at
     train≈val); the small D2↔O(2) bias is dominated by that win at H=12.
- **Anisotropic normalization silently breaks equivariance.** Per-axis std distorts
  rotations/reflections in the normalized frame the model operates in. Making it
  isotropic was a prerequisite for everything downstream.
- **Court-frame features beat "more equivariance."** The equivariant backbone is
  structurally blind to the absolute court frame (it's also translation-equivariant).
  Injecting the baskets as nodes gives it the D2 court structure *as features* —
  and crucially this only works once normalization is isotropic (the earlier
  "hoops don't help" verdict was confounded by aniso norm).
- **Bigger ≠ better here.** hidden128/4-layer consistently overfit 4.5k sequences.
- **Metric hygiene matters.** Two separate measurement artifacts (window noise,
  landmark deflation) each moved the apparent number by 0.2–0.5.
- **Ball is 6.4× harder than the average player, but the 10:1 entity count
  makes capacity reallocation a losing trade.** Total = (10·players + ball)/11.
  Each +0.1 ft² to players costs +0.09 in the total; each −1 to ball saves
  only 0.09. So to *net improve* by ball-weighting, ball must drop ~10× more
  than players rise. Empirically that ratio doesn't materialize — and the
  ball-only loss + capacity sweep showed *why*: under pure ball-only loss
  (no player gradient), ball MSE actually got **worse** (13.97 vs 13.70),
  and larger capacity didn't help either. The ball's predictability is
  *bounded by the joint architecture* — it depends on the player gradient
  to keep player embeddings sharp. So ball improvement isn't a
  capacity/optimization problem within EqMotion; it's an inductive-bias
  problem requiring a different architecture for the ball.

---

## 5. Open / next

- **⭐ DONE — augment + scale + train-long MART → val 3.11 (new best, beats the
  3.25 EqMotion ensemble).** See the "MART — augment + scale" section above for
  the full story. Overturns the prior task-floor verdict. **Kaggle 3.01
  (confirmed)** — now production (`submissions/solution_mart_aug_5k_best.csv`).
  The val→Kaggle gap was *favorable*: val 3.11 → Kaggle 3.01.
- **✅ DONE — aug-MART ↔ EqMotion ensemble re-check (job 2954940).** Cached
  aug-MART's K=20 preds on the same 3078 val windows; computed ρ + weighted blend
  + closed-form optimal weight. Analysis script validated against the OLD MART
  cache first (reproduced EqMotion 3.29 / MART 3.79 / ρ≈0.94 / w*=2.2% exactly).

  | | total | ball | players |
  |---|---|---|---|
  | EqMotion ensemble | 3.292 | 14.04 | 2.217 |
  | aug-MART 5k (mean-K) | 3.112 | 13.50 | 2.074 |
  | **best blend (0.70·augMART + 0.30·EqM)** | **3.073** | 13.30 | 2.051 |

  **Blend Kaggle-confirmed: val 3.073 → Kaggle 2.98** (val gain held on the
  leaderboard; favorable −0.09 val→LB gap, consistent with aug-MART's own). This
  is the production submission. ρ stayed high: all=0.941, ball=0.937, players=0.894. Closed-form optimal
  aug-MART weight: all **0.70**, ball 0.66, players 0.73. **Ensembling now helps
  (+0.039 over aug-MART alone), where it didn't with old MART** — and the reason
  corrects the earlier reasoning: averaging failed before NOT because ρ was high,
  but because old-MART (3.79) was far *worse* than EqMotion so any weight on it
  hurt (w*=2%). Optimal weight depends on ρ **and** relative error magnitude. Now
  aug-MART is *better* than EqMotion, so even at ρ=0.94 the optimal weight is 0.70
  and variance-reduction yields a real gain. High ρ still *caps* the gain at 0.04
  (correlated errors can only help each other so much) — but it's free (no training).
  **Caveat (resolved):** w=0.70 is val-tuned, so 3.073 was in-sample — but it
  held: Kaggle came back **2.98**. Script: `src/mart/corr_augmart_eqmotion.py`;
  cache: `cache/mart_aug_5k/`. Blend built CSV-level (the EqMotion CSV is
  byte-identical to the cached test preds, verified max diff 0.0000, so a by-id
  blend == blending the underlying feet predictions; no re-inference / ordering risk).
- **Highest-value next steps (ranked):**
  1. **✅ DONE — 0.70/0.30 blended submission → Kaggle 2.98** (new production,
     `submissions/solution_blend_augmart0.70_eqm0.30.csv`). The val +0.039 survived.
  2. **✅ DONE — SGDR warm restarts (job 2955274). Schedule shape is NOT the
     lever (clean negative result).** Budget-matched to the single-cosine 5k
     (same 5000 ep / 7.5M aug config / eta_min), 10 equal cosine cycles of 500
     ep (T_0=500, T_mult=1). **SGDR best val 3.116 vs single-cosine 3.112 — a
     tie** (Δ 0.004 = noise). Isolating LR trajectory at matched compute,
     periodic restarts don't beat one anneal → **3.11 is a genuine
     model/data/capacity floor, not a schedule strangle.** Confirms the 5k
     diagnostic (val flattened before the LR floor). The 10 cycle minima improved
     *monotonically* (3.556 → 3.119) rather than bouncing between diverse basins
     — a quality ladder, not diverse solutions.

     **Snapshot ensemble (job 2955532).** Averaging cycle minima: full 10-snap
     ensemble = 3.177 (WORSE — weak early cycles drag the mean up, as the
     monotone ladder predicted). But **best-3 (cycles 7+8+9, the late
     comparable-quality minima) = 3.101** — beats both the 3.119 best solo
     snapshot and the 3.112 baseline. The textbook snapshot-ensemble effect needs
     *diverse-but-comparable* members: the 3 late cycles qualify (different
     restart basins, similar quality), the early ones are just worse models and
     pollute the mean. Modest −0.011 vs baseline.

     **Kaggle-tested all 3 SGDR submissions (none beat the 2.98 production blend):**

     | Submission | val | Kaggle | val→LB gap |
     |---|---|---|---|
     | SGDR best single | 3.116 | 3.05 | −0.07 |
     | best-3 snapshot ens | 3.101 | 3.03 | −0.07 |
     | best-3 × EqMotion (w=0.73) | 3.069 | **3.00** | −0.07 |
     | — production (aug-MART × EqMotion) | 3.073 | **2.98** | −0.09 |

     **Stacking did NOT pay.** best-3×EqMotion tuned to val 3.069 (vs production
     3.073 — only −0.004, within noise), and on Kaggle it's 3.00 vs 2.98 — a
     touch *worse*. The snapshot ensemble's −0.011 val gain over single aug-MART
     mostly did not survive the EqMotion blend: EqMotion already absorbs the
     variance that snapshot-averaging reduces, so the two gains overlap rather
     than add. Also note the SGDR family carries a slightly smaller favorable
     val→Kaggle gap (−0.07 vs production's −0.09), which is why even the
     lower-val 3.069 blend lands above 2.98 on the board. **Production stays
     `solution_blend_augmart0.70_eqm0.30.csv` (Kaggle 2.98).** Code:
     `src/mart/{snapshot_ensemble_eval,blend_csvs,tune_best3_eqm_blend}.py`,
     `jobs/{train_mart_sgdr,snapshot_ensemble,submit_sgdr}.sh`. Snapshots:
     `checkpoints/mart_aug_iso_hoops_sgdr_snapshots/cycle_{00..09}.ckpt`. New
     CSVs: `solution_sgdr_best_val3.116.csv`,
     `solution_sgdr_snap_best3_val3.101.csv`,
     `solution_blend_best3_eqm_w0.73.csv`.
  3. **✅ D2 test-time augmentation (TTA) — WORKS, −0.029 free (job 2955757).**
     Average aug-MART's predictions over the 4 EXACT court symmetries (D2:
     identity, flip-x, flip-y, 180° rot), applied in normed space about the court
     center, predictions inverse-transformed back (each is an involution). The 4
     transforms give DIFFERENT per-transform val (identity 3.112, flip_x 3.111,
     flip_y 3.125, rot180 3.117) — direct proof aug-MART is only *approximately*
     O(2)-invariant (contrast EqMotion, where reflection TTA was an exact no-op).
     **D2 4-way avg = 3.083 (Δ −0.029 vs no-TTA 3.112)** — a bigger, free,
     no-training gain than the entire snapshot/blend effort, and ORTHOGONAL to
     both seed-ensembling and the EqMotion blend (TTA cuts aug-MART's own
     variance; the others add decorrelation on different axes) → should stack.
     The TTA gain itself *measures* the model's degree of non-equivariance — good
     report point. Code: `src/mart/tta_eval.py`, `jobs/tta_eval.sh`.
  4. **aug-MART multi-seed ensemble — STARTED then CANCELLED (jobs 2955751/52/53).**
     Seeds 2/3/4 launched, then cancelled ~ep 45: variance reduction caps at
     ~2.93 (see 4c) so it can't reach the val<2.6 goal, and it was tying up 3
     nodes the capacity test needs. Deprioritized in favor of the structural
     capacity test (4d). Still a valid −0.06–0.12 lever for a *Kaggle* push later.
  4c. **GOAL = val < 2.6.** Honest projection: variance reduction alone
     (TTA −0.029 + 4-seed ens ~−0.06–0.12 + EqMotion blend ~−0.04, all
     orthogonal) stacks to ~**2.93–2.98 val** — NOT 2.6. The 3.25→3.1→2.93 path
     is all *variance*; the residual to 2.6 is the *shared bias* both
     architectures hit (ρ≈0.94 ⇒ same errors on same samples ⇒ data/task floor,
     not model variance). Per the MART-oracle finding (min-of-K=0.64), 2.6 lives
     in future-mode info the 8-frame past underdetermines — unreachable by
     point-MSE averaging. So beating 2.6 needs a STRUCTURAL change (more
     capacity / longer context / richer features / different scoring), not more
     ensembling. **Next structural test = capacity (below).**
  4d. **Capacity-ceiling test — scaled aug-MART (job pending smoke 2955764).**
     ~3-4× bigger (256-512-8, vs 128-256-6) at the same 5000-ep aug recipe.
     Rationale: 7.5M showed ZERO overfit (train≈val), so capacity may not bind.
     Caveat: train≈val (equal, not train≪val) leans "data floor" over "capacity
     floor", so prior on a big gain is moderate. Diagnostic = watch train-vs-val:
     train collapsing below val ⇒ capacity now binds (helps); staying equal ⇒
     data floor confirmed. Config `mart_nba_aug_big.yaml`, `jobs/train_mart_aug_big.sh`.
     **LAUNCHED (job 2955786):** 36.1M params (4.8× the 7.5M backbone), 256-512-8,
     5000-ep cosine. Smoke confirmed builds/fits memory; per-epoch cost ≈ same as
     7.5M (data/val loop bound, not matmul) so ~8h, fits 24h wall. Diagnostic to
     watch: train-vs-val divergence (overfit ⇒ capacity binds ⇒ helps; train≈val
     persists ⇒ data floor, 3rd independent confirmation alongside schedule+ensembling).
  4e. **Goal/metric AUDIT (no-compute, decisive for target framing).** Findings:
     - **val/mse_ft is computed correctly** — mean of squared error over (T_f, N=11
       real, 2 coords) in feet², denormalized; matches the CLAUDE.md ADE spec and
       EqMotion's `compute_mse` exactly. No metric bug inflating our numbers.
     - **Submission is SINGLE-SHOT** (`data/sample_submission.csv`: 1 row/id, 264
       cols = 11×12×2, one trajectory/entity). So Kaggle = single-shot mean-MSE,
       confirming "predict the conditional mean" (averaging K heads) is correct
       and **multimodal/best-of-K submission is impossible** — kills the idea of
       submitting diverse modes. The oracle min-of-K=0.64 is therefore NOT
       directly exploitable; only better *mode selection from 8 frames* helps
       (and the learned selector already failed at that).
     - **Context hard-capped at 8 frames** (all 1243 test files are (8,11,4)) →
       the "longer context (C>8)" lever is DEAD; the model can't get more history.
     - **val→Kaggle gap is consistently ~+0.08** (5 confirmed pairs: 3.07→2.98,
       3.11→3.01, 3.12→3.05, 3.10→3.03, 3.07→3.00; val is *pessimistic*). So the
       Kaggle 2.6 LB top ≈ **val ~2.68**, and a literal "val < 2.6" ≈ Kaggle
       ~2.52 (below the top student team). Working target = **val ~2.68** =
       matching the leaderboard top. Context: this is a *class* competition, so
       2.6 is the top student group — likely a findable method lever, not SOTA.
     - **Surviving live levers toward ~2.68:** (a) capacity [4d, running];
       (b) basketball-specific / per-agent / handler-aware mode selection (the
       oracle gap proves headroom; generic MLP selector failed but handler-cued
       hasn't been tried); (c) a sharper-conditional-mean architecture from the
       cited papers (MoFlow flow-matching, LED). Variance levers (ensembling/TTA)
       are −0.03–0.12 each and cap ~2.93 — useful for Kaggle, insufficient for 2.68.
  4b. **10k single-cosine (job 2955275, RUNNING, ~ep 8800, best ~3.14).** The
     "more compute, same shape" ablation arm. Tracking ABOVE the 5k run's 3.11
     (its LR is nearly floored with ~1250 ep left) → looks like raw epochs aren't
     the lever either, consistent with SGDR. Confirms 3.11 as a real floor for
     the single 7.5M model; final number TBD.
  5. **Infra fix (DONE).** Job scripts now use per-job scratch
     (`/scratch/izar/$USER/job_$SLURM_JOB_ID`) — the shared-scratch `rsync
     --delete` collided and killed the first SGDR run (2955010) mid-training when
     the 10k job started. Patched in the mart_aug/sgdr/10k/snapshot scripts;
     other ~26 job scripts share the latent bug (only bites on concurrent runs).
  6. **Kaggle confirmed:** val→LB gap is favorable for aug-MART (−0.10), so val
     remains a trustworthy — slightly pessimistic — selector.
- **Done (EqMotion era):** 5-seed ensemble → val 3.25 (from 3.33); reflection TTA
  confirmed an exact no-op (EqMotion is exactly O(2)-equivariant). Submission:
  `submissions/solution_ens5_iso_hoops_val3.25.csv`.
- **More court landmarks — TRIED, all hurt (negative result).** Job 2952604,
  all under iso + cosine + full-val + honest 11-entity metric, single seed,
  vs the hoops-only control = **3.33**:

  | Landmark set | n_land | val/mse_ft |
  |---|---|---|
  | hoops (control) | 2 | **3.33** |
  | hoops + free-throw lines (±28, 0) | 4 | 3.358 |
  | hoops + 3-pt arc apex (±18, 0) | 4 | 3.372 |
  | hoops + ft + 3-pt | 6 | 3.417 |
  | hoops + 4 court corners (±47, ±25) | 6 | 3.422 |

  All worse than hoops-only; more nodes → monotonically worse. Plausible reason:
  ft and 3-pt apex sit on the basket axis already encoded by the two hoops, so
  distance/bearing to them is a linear function of distance/bearing to hoops —
  redundant. Corners are far from the action. The extra nodes dilute the
  message-passing without adding court information.

  Useful for the report: this rules out "hoops worked because of node count, not
  court information," and tightens the story to *"hoops are the right amount of
  D2 court structure."* Code: `LANDMARK_SETS` + `--landmarks PRESET[,PRESET...]`
  in [src/equivariance/eqmotion_nba.py](src/equivariance/eqmotion_nba.py).
- **MART baseline — done, loses to EqMotion under fair comparison.** Trained 300
  ep + cosine, same schedule as EqMotion, with both `min_ade` (paper-native) and
  `mean_mse` losses. Best MART = **3.79** (min_ade); mean_mse was *worse* (3.88).
  See the MART results table above for the implicit-K-ensemble lesson. Code
  alignment: `WindowEvalSampler` (deterministic 8 windows/seq) + denormalized
  feet² `val/mse_ft` on real entities → MART and EqMotion val numbers are now
  directly comparable. Submissions: `solution_mart_300ep_minade_val3.79.csv`,
  `solution_mart_300ep_meanmse_val3.88.csv`.
- **Per-entity error breakdown (ball is the dominant error source).** Added
  `val/mse_ball` / `val/mse_players` to `validation_step` and ran the diagnostic
  on the existing 5-seed iso+hoops ensemble (job 2952743):

  | Config | total | ball | players | ratio |
  |---|---|---|---|---|
  | single (iso_hoops_s0) | 3.33 | **14.30** | 2.23 | 6.41× |
  | 5-seed ensemble | 3.25 | **13.86** | 2.19 | 6.33× |

  Ball is ~6.4× harder than the average player and contributes ~39% of total
  MSE. *Theoretical* leverage: holding players fixed, cutting ball to 5 would
  drop total to ~2.45 (well past leaderboard top 2.6). So the ball is where the
  biggest concentrated headroom is — *if* it can be reduced.

- **Ball-weighted loss — TRIED, doesn't move the needle (single seed).**
  Added a `ball_weight` parameter to `MultiStepMSE` that weights the ball
  element's loss term ×N (gradient math verified: bw=5 → 3.8× stronger ball
  gradient, 0.76× weaker player gradient). Trained three new 300-epoch models
  with `--ball-weight 3/5/10` (job 2952755), identical to the 3.33 control
  otherwise. Per-entity table:

  | Run | total | ball | players | vs control |
  |---|---|---|---|---|
  | control (bw=1) | **3.33** | 14.30 | 2.23 | — |
  | iso_hoops_bw3 | 3.40 | 13.96 | 2.35 | ball ↓0.34, players ↑0.12 |
  | iso_hoops_bw5 | 3.36 | 13.70 | 2.33 | ball ↓0.60, players ↑0.10 |
  | iso_hoops_bw10 | 3.49 | 13.88 | 2.45 | ball ↓0.42, players ↑0.22 |

  Ball *does* drop slightly under weighting (≤ 4%) but the per-player rise is
  multiplied by 10 in the total, so the net is always worse. Even the best
  combined estimate — ball from bw5 specialist (13.70) + players from the 5-seed
  ensemble (2.19) — gives (10·2.19 + 13.70)/11 = **3.24**, only 0.01 better
  than the 3.25 ensemble. **The combine math doesn't pay**.

  Verified via wandb run history (300 epochs, 1019 val steps) that this isn't a
  checkpoint-selection artifact: minimum ball MSE *ever* seen during the bw5 run
  was 13.67, and the saved best-by-total checkpoint had ball=13.70 — same floor.

  Honest statement of the result: *with this architecture and single-seed
  weighting, ball MSE has a soft floor around 13.7; reallocating capacity via
  loss weight doesn't break it.* What's NOT proven: that this is an
  information-theoretic floor on the task — that would require larger model,
  ball-only loss, or different (non-equivariant) architecture.

  Lever ranking implied: weight reallocation alone is not the lever; capacity
  or inductive-bias change for the ball would be.

- **Ball-ONLY loss + capacity sweep — TRIED, confirms the floor is robust
  within EqMotion (and reveals *why*).** Job 2952799. Followed up the bw sweep
  by going to the limit: train with loss computed *only* on the ball (zero
  gradient on players), at three capacities. Checkpointed on `val/mse_ball`
  (since `val/mse_ft` is meaningless with players unconstrained).

  | Run | Arch (hidden / layers) | val/mse_ball |
  |---|---|---|
  | iso_hoops_s0 (joint, bw=1) | 64 / 2 | 14.30 |
  | iso_hoops_bw5 (joint, weighted ×5) | 64 / 2 | **13.70** |
  | iso_hoops_bo_small (ball-only) | 64 / 2 | 13.97 |
  | iso_hoops_bo_wide (ball-only) | 128 / 2 | 14.19 |
  | iso_hoops_bo_big (ball-only) | 128 / 4 | 14.02 |

  Three findings, each load-bearing:

  1. **Ball-only is *worse* than ball-weighted** (13.97 vs 13.70). Removing the
     player gradient hurts ball prediction.
  2. **Capacity doesn't help.** All three ball-only variants cluster at
     13.97–14.19; the smallest is tied for best.
  3. **The floor across every variant we've tried is 13.7–14.3** — ≤5% spread
     on every "ball specialist" intervention.

  **Mechanistic interpretation (why ball-only is worse).** The ball's future
  depends critically on player motion — passes follow handlers, the ball
  decelerates at the receiver, trajectories bend around player intent. Pure
  ball-only loss removes the gradient that keeps the *player* representations
  sharp inside the encoder. Player embeddings drift, and the ball prediction
  — which depends on them — suffers. **The bw=5 sweet spot is exactly the
  regime where there's *just enough* player gradient to keep player embeddings
  useful, with the extra ball weight focusing optimization on the ball.**
  Joint training is doing real work for the ball; you can't extract it by
  reweighting alone — the residual player loss is load-bearing.

  Combine math is unchanged: best ball (13.70 from bw5) + ensemble players
  (2.19) → (10·2.19 + 13.70)/11 = **3.24** vs current 3.25 ensemble. Real but
  marginal; ball MSE is *bounded* within the EqMotion family.

  **What this implies for next steps.** Capacity reallocation has been ruled
  out as a lever. Reducing ball error further requires a **non-equivariant
  architecture** specifically for the ball (the ball's dynamics — fast,
  abrupt, decision-driven — may genuinely not suit EqMotion's continuous
  O(2) prior, even with court-frame features). That's a meaningfully bigger
  build than what we've done so far. Alternative: accept the floor and
  spend effort on cross-architecture diversity (combine EqMotion ensemble
  with MART's predictions per-entity, see if MART's ball error is in a
  different regime).

- **MART per-entity diagnostic — HUGE finding, reframes the problem.**
  Job 2952871, `src/mart/diagnose_per_entity.py` on existing 300-ep MART
  checkpoints (canonical agent order: TeamA(5)+TeamB(5)+Ball, so ball=idx 10):

  | Run | Reduction | total11 | ball | players |
  |---|---|---|---|---|
  | mart_minade_s1 | mean-of-K | 3.67 | 15.90 | 2.44 |
  | mart_minade_s1 | **oracle min-of-K (=20)** | **0.64** | **3.10** | **0.39** |
  | mart_meanmse_s1 | mean-of-K | 3.90 | 16.78 | 2.61 |
  | mart_meanmse_s1 | oracle min-of-K (=20) | 3.47 | 14.67 | 2.35 |

  Three things matter here:

  1. **MART's mean-of-K ball (15.90) is WORSE than EqMotion's (13.86).** So
     the "MART's ball might be in a different regime" hypothesis was wrong
     when interpreted naively. *Within the mean-of-K reduction*, MART loses.
  2. **But MART's oracle ball MSE is 3.10** — 4.5× below EqMotion's 13.7
     floor. There exists, for every val sample, at least one head among
     K=20 that nails the ball almost perfectly. The K heads are doing real,
     diverse work; the mean reduction is destroying it. Oracle players is
     0.39 (6× below EqMotion's 2.19), and oracle total is 0.64 (5× below
     the 2.6 leaderboard top).
  3. **mean_mse training kills the diversity** (min-of-K ball jumps from
     3.10 → 14.67 vs minade). Re-confirms the earlier MART finding: aligning
     the loss to single-shot MSE collapses the heads-as-implicit-ensemble
     structure that produces the diverse modes.

  **This completely reframes the problem.** MART's bottleneck isn't
  capacity or arch — it's **head selection at inference**. The right modes
  exist in the K=20 hypotheses; mean-of-K is blurring them. Conditional on
  *any* reduction better than mean unlocking just a fraction of the oracle
  gap, MART becomes the architecture to push.

  Next experiments (cheap, no retraining required):
  - Median-of-K instead of mean-of-K (robust to outlier modes).
  - Trimmed-mean / mode (cluster K heads, take largest cluster's centroid).
  - "Closest-to-mean" head selection (pick the head nearest the consensus —
    the high-density mode rather than the centroid).
  - Physics-prior head selection (pick the head whose ball trajectory is
    most consistent with constant-velocity extrapolation).
  - Eventually: learn a head selector (small MLP on context features),
    or Gumbel-softmax mixture-of-experts at training.

  Decision: pursue K-reduction first on `mart_minade_s1`. If any simple
  reduction beats mean's 15.90 ball by even ~50% (i.e., reaches ~8), MART
  combined with EqMotion's player ensemble dramatically beats the 3.25
  current best. The combine math: ball at 8 + EqMotion players at 2.19 →
  (10·2.19 + 8)/11 = **2.72**, well past the leaderboard top.

- **K-reduction sweep — TRIED, simple reductions can't bridge the oracle
  gap.** Job 2952879, four point-estimate reductions of MART's K=20 heads
  evaluated on `mart_minade_s1`:

  | Reduction | total11 | ball | players |
  |---|---|---|---|
  | mean-of-K (baseline) | 3.67 | 15.90 | 2.44 |
  | median-of-K | 3.68 | 16.09 | 2.44 |
  | trimmed-mean (drop top/bottom 2 of 20) | 3.63 | 15.71 | 2.43 |
  | closest-to-mean (pick head closest to centroid) | 3.83 | 16.81 | 2.53 |
  | **oracle min-of-K (=20)** | **0.64** | **3.10** | **0.39** |

  All four simple reductions cluster within ±0.5 ball MSE — the best
  (trimmed-mean) is only 1% better than vanilla mean. "closest-to-mean" is
  *worse* because the K-centroid it picks against is itself unmoored from
  any real mode, so the closest-to-it head is the most-boring one.

  **Sharpened interpretation.** The K=20 heads aren't noisy perturbations
  of a central prediction (a 5× oracle gap would be impossible if they
  were) — they are **genuinely distinct modes** (pass-to-A vs pass-to-B vs
  drive vs shoot, etc.). The mean-of-K hurts the *ball* disproportionately
  because the ball has many discrete futures whose centroid is guaranteed
  to be far from every one of them; players have fewer modes per agent and
  averaging is closer to harmless. Bridging the oracle gap requires
  **predicting which mode obtains given the context** — that's a learning
  problem in itself, not something statistical aggregation can solve.

  **Real remaining options to bridge the gap:**
  - **Learned head selector** (the principled fix): small MLP/attention on
    the past trajectory + the K candidate predictions, outputs a soft
    distribution over the 20 heads. End-to-end MSE on head-weighted
    prediction. ~2–3 hour build; works only if past contains enough signal
    to predict the mode (often yes for basketball — handler identity
    strongly predicts ball destination).
  - **Per-head confidence calibration**: MART's min_ade already specializes
    heads to different samples. Add a confidence head, use confidence-
    weighted reduction at inference.
  - **Mode-of-K via clustering**: cheap, but if mean/median/trimmed-mean
    didn't help, clustering's centroid-of-largest-cluster probably won't
    either.

  Decision: park MART for now. The K=20 oracle ball of 3.10 is a tantalizing
  upper bound, but unlocking it needs a learned selector — a real build with
  uncertain return. Best current submission stays
  `solution_ens5_iso_hoops_val3.25.csv` (EqMotion ensemble). If time before
  June 10 permits, the learned head selector is the highest-upside remaining
  experiment.

- **Learned head selector — TRIED, hits a hard floor at ~5% ball reduction.**
  Built the principled fix: per-agent MLP that takes (past, agent_id,
  K=20 candidate trajectories) and outputs softmax weights over heads, end-
  to-end MSE on the head-weighted prediction. Cache stage runs MART forward
  on train+val (8 windows/seq each → 27K train / 3K val samples) and saves
  K predictions to disk so the selector iterates in minutes
  (`src/mart/cache_mart_preds.py`). Trainer iterates the cached predictions
  (`src/mart/train_head_selector.py`). Three variants tried:

  | Run | Arch / loss | best val/mse_ball |
  |---|---|---|
  | mean-of-K (baseline) | — | 16.78 |
  | v1: per-agent MLP, joint loss | hidden 128, no scene context | 15.79 |
  | v2: scene context, bigger MLP | hidden 256, all-agents context | 16.24 (overfit) |
  | v3: per-agent MLP, ball-only loss | hidden 128, gradient only on ball | 15.87 |
  | oracle min-of-K | — | 3.10 |

  Three different inductive bets (more context, more capacity, focused loss)
  converging on ~15.8–16.2 ball is strong evidence the signal isn't
  recoverable. v2 specifically *trained loss kept dropping* while val rose,
  classic overfit — meaning the model COULD memorize per-sample selections
  on training data but those don't generalize. **The K=20 heads encode
  *future decisions* (who the ballhandler will pass to and when) that
  aren't deterministically encoded in the 8-frame past.** The oracle gap is
  irreducible from past trajectories alone.

  Combine math (best selector + EqMotion players): ball 15.79 + players 2.19
  → 3.43 — still worse than the 3.25 EqMotion ensemble. **The selector
  approach is dead.** Production submission stays
  `solution_ens5_iso_hoops_val3.25.csv`.

  Scripts/checkpoints preserved for the report:
  `src/mart/checkpoints/head_selector_{v1,v2,v3}.pt`,
  `cache/mart_minade_s1/{train,val}.pt`,
  `src/mart/{cache_mart_preds,train_head_selector}.py`,
  `jobs/{train_head_selector,train_head_selector_only}.sh`.

  **Implication for the report narrative:** the EqMotion ball MSE floor at
  ~13.7 (capacity/reweighting ruled out) and MART's mean-of-K ball at ~16
  (head selection ruled out) both bottom out in roughly the same regime.
  Two independent architectural families with two independent intervention
  strategies converge on the same ball difficulty floor — strong evidence
  this is a **task floor**, not an architecture/optimization artifact. Ball
  trajectories at H=12 are *fundamentally multimodal-unpredictable* from an
  8-frame past, and no point-estimate model that minimizes single-shot MSE
  can beat the data's inherent uncertainty.

- **Residual MART for players — TRIED, both losses fail (final negative
  result that closes the loop on players too).** Built the boosting-style
  pipeline (`src/mart/{cache_eqmotion_residuals,train_mart_residual,
  submit_mart_residual}.py`, `jobs/train_mart_residual.sh`): cache EqMotion
  5-seed ensemble predictions on train+val+test windows, train fresh MART
  to predict the player residuals (`target − EqMotion_base`) with ball
  gradient zeroed, at inference combine `EqMotion_base + MART_residual` for
  players and keep `EqMotion_base` for ball. Two loss variants tested:

  | Run | Loss | best val/total11 | players |
  |---|---|---|---|
  | EqMotion-alone (baseline) | — | **3.292** | 2.217 |
  | mart_residual_v1 | min_ade (multimodal) | 3.295 (~flat) | 2.222 |
  | mart_residual_v2 | mean_mse (regression) | 3.357 (worse) | overfits to 2.78 |

  - **v1 (min_ade)** is essentially flat: K=20 heads find valid multimodal
    residuals but the mean-of-K reduction averages them back to ≈0. Same
    head-selection bottleneck as before.
  - **v2 (mean_mse)** overfits hard: training loss collapses 17× (0.066 →
    0.004) while val total11 climbs from 3.37 → 3.81. Model can fit per-
    sample training residuals but they're sample-specific noise, not a
    learnable systematic bias.

  **Together these two losses test orthogonal hypotheses and both fail.**
  No multimodal mode that's mean-recoverable; no systematic bias that's
  conditionally-learnable. **Player residuals are conditionally noise given
  the 8-frame past** — EqMotion's player MSE 2.19 is at the floor.

  Combined with the ball-side findings, the picture is now consistent: the
  EqMotion ensemble at val 3.25 / Kaggle 3.2 is essentially at the task
  floor for this dataset and horizon. The 2.6 leaderboard top is reachable
  in principle (we've seen MART's oracle min-of-K total = 0.64), but that
  requires bridging an oracle gap that 8 frames of past don't determine.

- **Cross-architecture ensemble correlation diagnostic — the closing
  evidence for the task-floor claim.** Used the existing val caches
  (EqMotion 5-seed ensemble predictions + MART mean-of-K predictions on
  the same 3078 deterministic windows) to compute per-entity error
  correlation between the two architectures. Results:

  | Slice | MSE EqM | MSE MART | ρ | MSE 50/50 avg | MSE optimal avg | optimal MART weight |
  |---|---|---|---|---|---|---|
  | All 11 | 3.29 | 3.79 | **0.929** | 3.41 | 3.29 | 2.2% |
  | Ball | 14.04 | 16.78 | **0.928** | 14.83 | 14.02 | **−9.0%** |
  | Players | 2.22 | 2.49 | **0.929** | 2.27 | 2.21 | 9.8% |

  Three structural conclusions:

  1. **ρ ≈ 0.93 is consistent across all three slices.** EqMotion and MART
     are making the same errors on the same samples regardless of which
     entities you slice on.
  2. **50/50 averaging is worse than EqMotion alone**; optimal weighting
     drives MART's weight to near zero (and *negative* on the ball).
  3. Two architectures with completely different inductive biases
     (continuous O(2) equivariance vs relational transformer attention)
     converging to the same errors on the same samples is direct empirical
     evidence that **the residual error is about the data, not the model**.

  This closes the case on cross-architecture averaging for this task: with
  ρ this high, no statistical combination of these two predictors can
  bridge the 3.25 → 2.6 gap. The 2.6 leaderboard top likely involves
  techniques outside the single-prediction MSE-trained regime —
  stochastic/multimodal submission strategies that exploit sample-specific
  structure not identifiable from the 8-frame past.

- **Other candidates:** larger model + ball weighting (isolates capacity from
  optimization); regularization HP search now that val is trustworthy;
  diverse cross-architecture ensemble (EqMotion + STGCNN + MART).
- **Report framing:** "an over-strong O(2) prior + explicit D2 court-frame features
  beats both plain equivariance and learned-via-augmentation," with the
  normalization and metric-hygiene ablations as supporting evidence. The ball
  remains the single concentrated source of irreducible-looking error; making
  it tractable likely needs a non-equivariant specialist, not capacity
  reallocation in the joint model.

### 5f. 2026-06-02 capacity + aggregation + possession sweep — three more floor confirmations

- **36M capacity test — FAILED, but diagnostic (job 2955786, killed at ep ~3736).**
  Scaled aug-MART 7.5M→36M (256-512-8), same iso+O(2)+hoops+5000ep recipe.
  Val/mse_ft pinned at **~11** while `val_minADE` (z-score, min-of-K) stayed
  **0.036 — identical to the small models**. Same sigma, so not a denorm bug:
  the bigger model finally had the capacity to do what `min_ade` actually asks
  (spread the K modes), so best-of-K stayed sharp (~0.8 ft) while the mean of the
  now-wide cloud blew up. **Capacity under `min_ade` is counterproductive for our
  mean-of-K metric.** The 7.5M models' good mean-of-K (3.11) is *accidental*
  mode-collapse (too small to spread), not objective alignment.

- **Mode-aggregation probe — reconfirms K-reduction is dead (`agg_probe.py`).**
  On 5k/10k aug checkpoints: sample diversity = **1.28 ft** (NOT collapsed), yet
  medoid (3.22) and densest-cluster-centroid (3.20) both *lose* to mean-of-K
  (3.11). The spread is unimodal jitter around a biased centre, not separable
  branches (no density valley for clustering to exploit). Per-entity **oracle =
  0.445 ft²** (RMS 0.94 ft) — the good sample sits in the low-density *tail*, so
  density selection moves the wrong way. Headroom is real but un-selectable;
  matches the earlier 4-reduction sweep and the 3-variant learned selector.

- **Possession-anchored ball — DEAD, model beats the handler-oracle
  (`ball_possession_probe.py` on aug 5k).** Current split: total **3.112**,
  players **2.074** (floor), ball **13.50** → the whole 3.1→2.6 gap is the ball
  (leader's ball ≈ 7.9). Tested anchoring the ball to its likely handler (player
  nearest the ball at t=C):

  | ball predictor | val ball MSE |
  |---|---|
  | model mean-of-K | **13.50** |
  | constant-velocity | 61.60 |
  | anchor to handler PREDICTED track | 27.48 |
  | anchor to handler GROUND-TRUTH track (oracle) | 26.92 |

  Possession persistence 0.72 (ball within 6 ft of t=C handler). **Even the
  GT-handler oracle (26.9) is 2× worse than the model (13.5)** — the ball is near
  but never co-located with its handler (dribble offset/bounces/handoffs), and
  the relational model already exploits possession better than any naive anchor.
  Constant-velocity (61.6) confirms the ball is strongly non-ballistic.

- **Verdict after this sweep:** every single-prediction lever is now exhausted —
  capacity (✗), aggregation (✗×5), selection (✗×3), possession (✗), physics (✗),
  residual (✗), cross-arch ensemble (✗, ρ=0.93). The ball at ~13.5 is irreducible
  for point-estimate MSE models from 8 frames.

- **Clean full-recipe `mean_mse` — LOST and OVERFIT (job 2958480, 7.5M, iso+O(2)
  +hoops+5000ep cosine, loss=mean_mse).** Definitive un-confounded conditional-
  mean-floor test. Best val ever = **3.20** (~ep 1200) vs `min_ade` **3.112**;
  then val *climbed* to ~3.59 while train collapsed to 0.0026 (≈1.3 ft² vs val
  3.59 — 2.7× train/val gap). `min_ade` never overfit (train≈val). **Conclusion:
  `min_ade`'s averaging of K diverse heads is a regularizer that gives a
  better-generalizing E[y|x] estimate than direct regression; 3.11 is the
  well-regularized conditional mean, NOT a `min_ade` artifact.**

- **This closes the generative lever too (by shared objective).** Every point-
  estimate generative model (MDN/GMM, flow-matching, diffusion) submits the
  conditional mean for our single-shot mean-MSE metric. `mean_mse` *is* that
  target optimized directly, and it can't beat 3.11 (overfits to 3.20). The
  published MoFlow [1] / LED / OmniTraj [2] NBA wins are all on **minADE**
  (best-of-K), which a single-submission mean-MSE metric structurally cannot
  exploit. **6th independent floor confirmation.** Path to 2.6 would require
  information beyond the 8-frame past or a submission mechanism we don't have;
  neither exists in this task.

- **Learned selector RE-RUN on the BEST model (aug-5k) — same ~5% floor (job
  2958657, `selector_aug5k.sh`).** The prior selector was on the inferior 300ep/
  no-aug `mart_minade_s1`; this re-ran the full cache→train pipeline on
  `mart_aug_iso_hoops_5k` (ball 13.50, the quantified-headroom case) with stronger
  regularization to combat the prior overfit. Baseline ball 13.50 →

  | variant | ball MSE | Δ |
  |---|---|---|
  | v1 scene-context | 13.10 | −3.0% |
  | v2 ball-only loss | 13.05 | −3.3% |
  | v3 no-scene + more reg | **12.88** | **−4.5%** |

  Best total 3.05 (vs 3.11) — real but tiny, nowhere near ball≈8 / total≈2.6.
  **The ~5% ball floor is invariant to base-model quality** — a cleaner,
  better-trained model did NOT make the modes more selectable. Tell-tale: v3
  *removed* scene context and won, so the t=C player configuration does NOT
  predict the ball's mode. **7th confirmation, and the strongest** — it kills the
  one lever with quantified headroom on the best model. The deciding info is a
  future decision (pass-left/right/drive/shoot) absent from the 8-frame past.

- **FINAL VERDICT.** val<2.6 is unreachable for a single-prediction model on this
  task; ~3.1 / Kaggle ~2.98 is the floor, 7-way confirmed across 2 architectures
  and every lever (capacity, aggregation×6, selection×2-models, possession,
  physics, residual, cross-arch, direct-regression). The entire 3.1→2.6 gap is
  irreducible ball multimodality not determined by 8 frames. The 2.6 leaderboard
  top must use out-of-regime information or a multi-submission mechanism this
  setup does not provide. Consolidate the floor story as the report's central
  critical-analysis contribution; production stays 0.70 aug-MART + 0.30 EqMotion
  (Kaggle 2.98). Optional micro-gain: fold the v3 aug-5k selector (total 3.05).

---

## 6. Best submission

- **`submissions/solution_blend_augmart0.70_eqm0.30.csv`** — 0.70·aug-MART +
  0.30·EqMotion blend, val **3.073**, **Kaggle 2.98 (confirmed) — PRODUCTION**.
  Best overall; upload this. Rebuild: blend the two CSVs below by id at w=0.70
  (CSV-level, no re-inference).
- `submissions/solution_mart_aug_5k_best.csv` — aug-MART 5000-ep, val 3.11 /
  Kaggle 3.01. Best single model; blend backbone. Regenerate: `sbatch jobs/submit_mart_aug.sh`.
- `submissions/solution_ens5_iso_hoops_val3.25.csv` — EqMotion 5-seed iso+hoops
  ensemble, val 3.25 / Kaggle 3.2 (blend's second component; former production).
- `submissions/solution_iso_hoops_fv_kaggle3.2.csv` — single EqMotion iso+hoops
  model, Kaggle 3.2 (confirmed).

Regenerate a single EqMotion CSV: `sbatch jobs/submit_eqmotion.sh --ckpt <path> --iso-norm`.
Rebuild the EqMotion ensemble CSV: `sbatch jobs/ensemble_eqmotion.sh`.
Rebuild the aug-MART CSV: `sbatch jobs/submit_mart_aug.sh`.
