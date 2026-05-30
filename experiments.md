# Experiments Log — NBA Trajectory Forecasting

Tracking what we tried, what changed in the pipeline, and the results. Metric is
`val/mse_ft` (mean squared error in feet², over the H=12 horizon), computed on the
**held-out validation fold** (`splits/fold0.json`, 497 sequences) unless noted as
Kaggle. Task: C=8 context → H=12 horizon, 11 entities (10 players + ball).

> **Headline:** EqMotion with **isotropic normalization + cosine LR + court-frame
> hoop nodes** is our best single model — honest val **3.33**, Kaggle **3.2**.
> The **5-seed ensemble** improves to val **3.25** (reflection TTA confirmed
> as an exact no-op — empirical proof of O(2) equivariance). Starting point
> for context: ~3.7 val EqMotion, ~3.8 STGCNN, ~3.8 MART. Ball is the
> concentrated remaining error source (6.4× harder than players, ~39% of
> total) with a robust floor at **~13.7 ft²** across every intervention we've
> tried within EqMotion: ball-weighted loss, true ball-only loss, and larger
> capacity all converge to the same range. The mechanistic reason is the
> joint-training dependency — the ball's predictability depends on sharp
> *player* representations the joint gradient maintains. Breaking the floor
> would need a non-equivariant ball architecture, not capacity reallocation.

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

- **Done:** 5-seed ensemble → val 3.25 (from 3.33); reflection TTA confirmed an
  exact no-op (EqMotion is exactly O(2)-equivariant). Submission:
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

- **`submissions/solution_ens5_iso_hoops_val3.25.csv`** — 5-seed iso+hoops
  ensemble, honest val 3.25. Best overall; upload this.
- `submissions/solution_iso_hoops_fv_kaggle3.2.csv` — single iso+hoops model,
  Kaggle 3.2 (confirmed).

Regenerate a single checkpoint's CSV: `sbatch jobs/submit_eqmotion.sh --ckpt <path> --iso-norm`.
Rebuild the ensemble CSV: `sbatch jobs/ensemble_eqmotion.sh`.
