# Experiments Log — NBA Trajectory Forecasting

Tracking what we tried, what changed in the pipeline, and the results. Metric is
`val/mse_ft` (mean squared error in feet², over the H=12 horizon), computed on the
**held-out validation fold** (`splits/fold0.json`, 497 sequences) unless noted as
Kaggle. Task: C=8 context → H=12 horizon, 11 entities (10 players + ball).

> **Headline (updated Jun 1):** the best model is now **augmented + scaled
> MART** — isotropic norm + full O(2) augmentation (continuous rotation +
> reflections) + court-frame hoop nodes, 7.5M params, **5000 epochs** of cosine
> LR → honest val **3.11** (best ckpt 3.112), **Kaggle 3.01 (confirmed)**. This
> **beats the EqMotion 5-seed ensemble (val 3.25) with a single model**, and
> beats EqMotion on the leaderboard too (Kaggle 3.01 vs 3.2). Note the val→Kaggle
> gap is *favorable* (−0.10: Kaggle better than val), unlike EqMotion's roughly
> neutral gap. **This is the new production submission.** Overturns the earlier
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
- **Highest-value next steps (ranked):**
  1. **Re-check aug-MART ↔ EqMotion error correlation (no GPU).** The ρ≈0.93 that
     killed cross-arch ensembling was measured on the *old, under-trained* MART.
     aug-MART (3.11 val / 3.01 Kaggle) is now the *strong* anchor; if its errors
     are even partly decorrelated from EqMotion's, an aug-MART-anchored ensemble
     could move toward the 2.6 top. Uses cached predictions.
  2. **SGDR warm restarts.** The diagnostic says 3.11 is a single-cosine model
     floor; re-raising the LR periodically tests whether the productive mid-LR
     descent (LR 1e-4→6e-5, where the big gains happened) repeats. Cheap GPU run.
  3. Ensemble multiple aug-MART seeds (EqMotion ensembling bought 3.33→3.25).
  4. **Kaggle confirmed (was item 1):** val→LB gap is favorable for this arch
     (−0.10), so val remains a trustworthy — slightly pessimistic — selector.
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

---

## 6. Best submission

- **`submissions/solution_mart_aug_5k_best.csv`** — aug-MART 5000-ep, honest val
  **3.11**, **Kaggle 3.01 (confirmed) — PRODUCTION**. Best overall; upload this.
  Regenerate: `sbatch jobs/submit_mart_aug.sh`.
- `submissions/solution_ens5_iso_hoops_val3.25.csv` — EqMotion 5-seed iso+hoops
  ensemble, honest val 3.25 / Kaggle 3.2 (former production, now superseded).
- `submissions/solution_iso_hoops_fv_kaggle3.2.csv` — single EqMotion iso+hoops
  model, Kaggle 3.2 (confirmed).

Regenerate a single EqMotion CSV: `sbatch jobs/submit_eqmotion.sh --ckpt <path> --iso-norm`.
Rebuild the EqMotion ensemble CSV: `sbatch jobs/ensemble_eqmotion.sh`.
Rebuild the aug-MART CSV: `sbatch jobs/submit_mart_aug.sh`.
