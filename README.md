# NBA Trajectory Forecasting - EE-452 EPFL

Multi-agent trajectory prediction on NBA game data. Given 8 frames of past
positions for 10 players and the ball, predict the next 12 frames. Scored on
mean squared error in feet² (val/mse_ft).

**Best result:** 0.70 × curriculum-MART + 0.30 × EqMotion blend → **Kaggle 2.93**
Note: We got kaggle score of 2.92 using the entire dataset, which is just for the competetion!

---

## Setup

We use [uv](https://docs.astral.sh/uv/) for environment and dependency
management. Install the locked environment with:

```bash
uv sync
```

All scripts run inside that environment via `uv run` - e.g.
`uv run python src/mart/main_nba_pt.py ...`. The `python ...` commands below are
shorthand; prefix them with `uv run` (or activate `.venv` first).

### Wandb

Create a `.env` file at the project root:

```
WANDB_API_KEY=your_key_here
```

---

## Data

Download from Kaggle (requires Kaggle API credentials in `.env`):

```bash
uv run python src/utils/download_dataset.py
```

This places the data under `data/train/train/` and `data/test/test/` as `.pt`
sequence files. The train/val split is pre-defined in `splits/fold0.json`.

---

## Repository structure

```
network_ml_project/
├── data/               raw .pt sequence files (train + test)
├── splits/             fold0.json - train/val file lists
├── jobs/               SLURM job scripts for all experiments
├── submissions/        generated CSV files for Kaggle
├── src/
│   ├── utils/          shared metrics, dataset download, split utilities
│   ├── baselines/      provided non-graph temporal baseline (GRU / transformer)
│   ├── eda/            exploratory data analysis
│   ├── mart/           MART model - our primary architecture
│   ├── hht_cfi/        HHT-CFI model
│   ├── equivariance/   EqMotion model
│   ├── groupnet/       GroupNet model
│   ├── stgcnn/         ST-GCNN baseline
│   └── dynamic/        DNRI baseline
└── experiments.md      full experiment log with val/mse_ft numbers
```

---

## Running on SLURM

All experiments were run on the SCITAS (Izar) cluster. The `jobs/` folder holds
one SLURM batch script per experiment - training runs, ablations, probes, and
submission jobs (52 scripts in total). Submit any of them with `sbatch`:

```bash
sbatch jobs/train_mart_4k_curriculum_sw1k_lrdrop.sh   # the best model
sbatch jobs/train_eqmotion_5k.sh                       # EqMotion backbone
```

Each script requests its GPU/CPU/memory, rsyncs the repo to node-local scratch,
activates the environment, runs the underlying `src/` command, and syncs
checkpoints back. The exact flags for each run live inside the corresponding
script, so they double as a reproducible record of every experiment. To run
locally instead, use the `uv run python ...` commands shown per model below.

---

## Best model - curriculum-MART × EqMotion blend

Our best submission (**Kaggle 2.93**) is a weighted blend of two independently
trained models:

- **0.70 × curriculum-MART** - an augmented MART trained with a curriculum:
  min-ADE loss for the first 1000 epochs, then a 4× LR drop and a fresh cosine
  schedule for the remaining 3000.
- **0.30 × EqMotion** - the equivariant baseline (iso-norm + cosine + hoops),
  whose errors are partially uncorrelated with MART and so help on blend.

### Step 1 - train the curriculum-MART backbone

**On SLURM:**

```bash
sbatch jobs/train_mart_4k_curriculum_sw1k_lrdrop.sh
```

**Or run directly:**

```bash
cd src/mart
python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path ../../splits/fold0.json \
    --model_name mart_curriculum_4k_sw1k_lrdrop \
    --num_epochs 4000 \
    --lr 0.0002 \
    --iso_norm \
    --aug_rot_deg 180 \
    --aug_court_mirror \
    --use_hoops \
    --loss min_ade \
    --curriculum \
    --curriculum_switch 1000 \
    --curriculum_lr_reset \
    --curriculum_lr2 0.00005 \
    --scheduler_type CosineAnnealingLR \
    --gpu 0
```

Then write its test predictions:

```bash
python submit_nba_pt.py \
    --checkpoint checkpoints/mart_curriculum_4k_sw1k_lrdrop_best.ckpt \
    --test_dir ../../data/test/test \
    --out_csv ../../submissions/mart_curriculum_4k_sw1k_lrdrop.csv \
    --gpu 0
```

### Step 2 - train the EqMotion model

```bash
cd src/equivariance
python eqmotion_nba.py --iso-norm --add-hoops --run-name eqmotion_best
```

This writes an EqMotion submission to `submissions/` automatically. (To
regenerate it from a saved checkpoint, use `submit_eqmotion.py --ckpt <path>
--iso-norm`.)

### Step 3 - blend the two submissions

```bash
cd src/mart
python blend_csvs.py \
    --inputs ../../submissions/mart_curriculum_4k_sw1k_lrdrop.csv \
             ../../submissions/<eqmotion_submission>.csv \
    --weights 0.70 0.30 \
    --out ../../submissions/blend_curriculum_mart70_eqm30.csv
```

The resulting `blend_curriculum_mart70_eqm30.csv` is the **Kaggle 2.93**
submission.

---

## Models

### MART

MART (Multi-Agent Relational Transformer) uses a Relational Transformer (RT)
and Hierarchical RT (HRT) encoder stack with K parallel decoder heads.
Our adaptation adds isotropic normalization, full O(2) court augmentation,
static basket-hoop nodes, entity-type embeddings, and a CFI decoder variant.

**Files:**

| File | Description |
|---|---|
| `main_nba_pt.py` | Training script - all flags documented inline |
| `eval.py` | Evaluate a checkpoint on the val split (ADE / FDE / MSE in feet) |
| `submit_nba_pt.py` | Generate Kaggle submission CSV from a checkpoint |
| `models/mart.py` | Core MART architecture (RT + HRT + K decoders) |
| `models/mart_id.py` | MART variant with learned entity-type embeddings |
| `models/cfi.py` | Cross-modal Future Interaction decoder extension |
| `models/prt.py` | Pair Relational Transformer (upstream) |
| `models/hrt.py` | Hierarchical Relational Transformer (upstream) |
| `loaders/dataloader_nba_pt.py` | Dataset and sampler for NBA .pt files |
| `loaders/dataloader_nba_pt_hoops.py` | Same but appends 2 static hoop nodes (N=13) |
| `configs/mart_nba_aug.yaml` | Main config (6-layer, 128-dim, cosine LR) |
| `configs/mart_nba_pt.yaml` | Smaller base config (4-layer, 64-dim, 300 ep) |
| `utils.py` | Seed setup, config loader, threshold helper |

**Key training flags:**

```
--iso_norm              isotropic z-score normalization (required for rotation aug)
--aug_rot_deg 180       full O(2) rotation augmentation
--aug_court_mirror      random court reflection
--use_hoops             append static basket nodes as extra agents
--curriculum            min-ADE phase then mean-MSE phase
--curriculum_switch N   epoch to switch loss (default: num_epochs // 2)
--curriculum_lr_reset   rebuild cosine scheduler at switch with fresh T_max
--curriculum_lr2 LR     LR for phase 2 (default: same as phase 1)
--cfi                   enable CFI decoder (two branches)
--loss                  min_ade | mean_mse | min_mse | mean_ade
```

**Evaluate a checkpoint:**

```bash
cd src/mart
python eval.py \
    --checkpoint checkpoints/mart_curriculum_4k_sw1k_lrdrop_best.ckpt \
    --split_path ../../splits/fold0.json \
    --gpu 0
```

**Experiments:**

| Script | What it trains |
|---|---|
| `train_mart_2k_curriculum.sh` | 2k epochs, curriculum loss, hoops + full aug |
| `train_mart_3k_k10.sh` | 3k epochs, K=10 decoder heads |
| `train_mart_5k_k5.sh` | 5k epochs, K=5 decoder heads |
| `train_mart_5k_k10_curriculum.sh` | 5k epochs, K=10, curriculum |
| `train_mart_5k_k10_cfi.sh` | 5k epochs, K=10, CFI decoder |
| `train_mart_5k_cfi_curriculum.sh` | 5k epochs, CFI + curriculum |
| `train_mart_exp_cfi.sh` | 700 ep CFI ablation |
| `train_mart_2k_laplace_nll.sh` | Laplace NLL loss |
| `train_mart_2k_soft_wta.sh` | Soft winner-takes-all loss |
| `train_mart_4k_curriculum_sw1k_lrdrop.sh` | Best model |

For Optuna hyperparameter search:

```bash
cd src/mart
python optuna_search_mart.py
```

---

### HHT-CFI

HHT-CFI uses a Hypergraph Transformer encoder with a Cross-modal Future
Interaction decoder and Laplace NLL output distribution. We adapted it to the
NBA format with a per-agent normalization scheme and tested several inference
strategies at eval time.

**Files:**

| File | Description |
|---|---|
| `hht_cfi_nba.py` | Full pipeline: data, model wrapper, Lightning module, submission |
| `tune_hht_cfi.py` | Optuna hyperparameter search |
| `eval_and_experiment.ipynb` | Interactive evaluation and visualizations |
| `models.py` | MyTraj - upstream HHT-CFI model |
| `laplace_decoder_joint.py` | Laplace NLL decoder (upstream) |
| `train_best.py` | Train the best found configuration |

**Train:**

```bash
cd src/hht_cfi
python hht_cfi_nba.py
```

**Hyperparameter search:**

```bash
python tune_hht_cfi.py              # runs search, saves to optuna_hht_cfi.db
python tune_hht_cfi.py --report     # print best trial summary
```

**Submission** is generated inside `hht_cfi_nba.py`'s `__main__` block and
written to `submissions/`.

**Experiments:**

We ran 40 Optuna trials sweeping `hidden_size` (32/64/128), `x_encoder_layers`
(2-5), `x_encoder_head` (4/8), `lr`, `batch_size`, and `grad_clip`. At inference
time we tested min-scale mode selection, sigma-weighted mean, 4-way TTA, and
court-bound clamping - these are toggled via flags at the top of `hht_cfi_nba.py`.

---

### EqMotion

EqMotion is an equivariant motion prediction model. Its hard O(2) equivariance
means rotation/reflection augmentation is a no-op (confirmed via TTA), but it
serves as a strong baseline and blends well with MART.

**Files:**

| File | Description |
|---|---|
| `eqmotion_nba.py` | Full training + val + submission pipeline |
| `submit_eqmotion.py` | Generate Kaggle submission CSV |
| `ensemble_eqmotion.py` | Multi-seed ensemble averaging |
| `tune_eqmotion.py` | Optuna hyperparameter search |

**Train:**

```bash
cd src/equivariance
python eqmotion_nba.py --iso-norm --add-hoops --run-name eqmotion_best
```

(`--lr-scheduler` defaults to `cosine`; pass `--lr-scheduler plateau|none` to
change it. Training writes a Kaggle submission to `submissions/` unless
`--no-submit` is given.)

**Key flags:**

```
--iso-norm           isotropic z-score normalization
--add-hoops          inject 2 static basket nodes (court-frame structure)
--lr-scheduler       cosine | plateau | none (default: cosine)
--full-val           deterministic multi-window val for a stable val/mse_ft
--ball-weight W      up-weight the ball entity in the loss
--train-all          train on the full labelled set (no val) for the final model
--no-submit          skip writing the Kaggle submission
```

**Experiments:**

We explored isotropic normalization, cosine LR schedule, hoop nodes, capacity
scaling, ball-weighted loss, and multi-seed ensembles. Best single model uses
iso-norm + cosine + hoops. The 5-seed ensemble (`ensemble_eqmotion.py`) reached
Kaggle 3.2.

---

### GroupNet

GroupNet uses a multi-scale hypergraph neural network that models group
interactions between agents.

**Files:**

| File | Description |
|---|---|
| `train_hyper_nba_pt.py` | Training script |
| `submit_nba_pt.py` | Generate Kaggle submission CSV |
| `test_nba.py` | Evaluate on val split |
| `eval_k_sweep.py` | Sweep over K heads at eval time |
| `optuna_search.py` | Optuna hyperparameter search |
| `model/GroupNet_nba.py` | Base GroupNet architecture |
| `model/GroupNet_nba_id.py` | GroupNet with entity-type embeddings |

**Train:**

```bash
cd src/groupnet
uv run python train_hyper_nba_pt.py --split_path ../../splits/fold0.json --gpu 0
```

---

### ST-GCNN

Spatial-Temporal Graph Convolutional Network baseline.

**Files:**

| File | Description |
|---|---|
| `stgcnn_nba.py` | NBA-adapted ST-GCNN |
| `train.py` | Training script |
| `evaluate.py` | Evaluation script |
| `submit.py` | Generate Kaggle submission CSV |
| `tune_stgcnn.py` | Hyperparameter tuning |

**Train:**

```bash
cd src/stgcnn
python train.py --use_kinematics --use_edge_importance --epochs 100
```

Train-time court flips are on by default (`--no_augment` disables them). Other
flags: `--hidden_dim`, `--lr`, `--weight_decay`, `--batch_size`, `--step_size`,
`--gamma`, `--save_path`.

**Evaluate / submit:**

```bash
python evaluate.py --model_path best_model.pt --tta --clamp
python submit.py --model_path best_model.pt \
    --output_csv ../../submissions/submission_stgcnn.csv
```

---

### DNRI

Dynamic NRI learns the interaction graph structure jointly with trajectories.

**Files:**

| File | Description |
|---|---|
| `dnri_nba.py` | NBA adapter - training, validation, and submission |
| `tune_dnri.py` | Optuna hyperparameter search (writes `dnri_v1` study) |
| `retrain_topk.py` | Retrain the top-K Optuna trials |
| `dnri_ref/` | Upstream DNRI reference implementation |

**Train:**

```bash
cd src/dynamic
python dnri_nba.py --add-hoops --submit --run-name dnri_best
```

Flags: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--add-hoops`
(append 2 static hoop nodes), `--submit` (write a Kaggle CSV after training),
`--smoke` (run sanity checks and exit).

---

### Non-graph baseline (provided pipeline)

The `baselines/` folder holds the naive temporal baseline shipped with the
project (no graph prior) - the reference point our graph models are compared
against. It also includes the dataset/visualization utilities the rest of the
pipeline builds on.

**Files:**

| File | Description |
|---|---|
| `ref_script.py` | Original provided pipeline: dataset, sampler, naive trainer, submission |
| `ref_script_lightning.py` | Lightning rewrite with a `gru` / `transformer` backbone toggle |

**Train (parameters are set in the `__main__` block, no CLI flags):**

```bash
cd src/baselines
python ref_script.py            # naive temporal baseline + submission
python ref_script_lightning.py  # GRU / transformer backbone (edit BACKBONE)
```

Both write a Kaggle submission to `submissions/` after training.

---

## Blending submissions

To blend two submission CSVs (e.g. curriculum-MART + EqMotion at the production
weights):

```bash
cd src/mart
python blend_csvs.py \
    --inputs ../../submissions/mart_curriculum_4k_sw1k_lrdrop.csv \
             ../../submissions/eqmotion_best.csv \
    --weights 0.70 0.30 \
    --out ../../submissions/blend_curriculum_mart70_eqm30.csv
```

Weights are normalized internally; omit `--weights` for an equal-weight average.


