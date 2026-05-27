# Experiments Log — NBA Trajectory Forecasting

Tracking what we tried, what changed in the pipeline, and the results. Metric is
`val/mse_ft` (mean squared error in feet², over the H=12 horizon), computed on the
**held-out validation fold** (`splits/fold0.json`, 497 sequences) unless noted as
Kaggle. Task: C=8 context → H=12 horizon, 11 entities (10 players + ball).

> **Headline:** EqMotion with **isotropic normalization + cosine LR + court-frame
> hoop nodes** is our best single model — honest val **3.33**, Kaggle **3.2**
> (from a starting point of ~3.7 val / 3.9 for STGCNN). A multi-seed ensemble +
> TTA evaluation is in progress.

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
| iso_hoops_fv | iso | cosine | **yes** | full-val (11-entity, honest) | **3.33** | **3.2** | **best single model** |
| 5-seed ensemble (+TTA) | iso | cosine | yes | full-val (honest) | _in progress_ | – | jobs 2949519 → 2949521 |

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

---

## 5. Open / next

- **In progress:** 5-seed ensemble + reflection-TTA eval (jobs 2949519 → 2949521).
  Expectation: ensemble ~3.33 → ~3.1–3.2; TTA ≈ no-op for an equivariant model
  (a clean empirical equivariance check).
- **Candidates:** more court landmarks (free-throw lines, 3-pt arc, sidelines) —
  judge on Kaggle, not val (risk of court-frame overfit). Targeted regularization
  HP search now that val is trustworthy. Diverse cross-architecture ensemble.
- **Report framing:** "an over-strong O(2) prior + explicit D2 court-frame features
  beats both plain equivariance and learned-via-augmentation," with the
  normalization and metric-hygiene ablations as supporting evidence.

---

## 6. Best submission

`submissions/solution_iso_hoops_fv_kaggle3.2.csv` — EqMotion iso+hoops, from
`checkpoints/eqmotion/iso_hoops_fv/best.ckpt`. Kaggle 3.2.
Regenerate any checkpoint's CSV: `sbatch jobs/submit_eqmotion.sh --ckpt <path> --iso-norm`.
