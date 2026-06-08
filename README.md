# NBA Trajectory Forecasting — EE-452 EPFL

Multi-agent trajectory prediction on NBA game data. Given 8 frames of past
positions for 10 players and the ball, predict the next 12 frames. Scored on
mean squared error in feet² (val/mse_ft).

**Best result:** 0.70 × aug-MART + 0.30 × EqMotion ensemble → **Kaggle 2.98**

---

## Setup

### Option A — conda

```bash
conda create -n nml python=3.12
conda activate nml
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning wandb optuna pandas python-box python-dotenv torch-geometric kagglehub tqdm
```

### Option B — uv

```bash
uv sync
```

### Wandb

Create a `.env` file at the project root:

```
WANDB_API_KEY=your_key_here
```

---

## Data

Download from Kaggle (requires Kaggle API credentials in `.env`):

```bash
python src/utils/download_dataset.py
```

This places the data under `data/train/train/` and `data/test/test/` as `.pt`
sequence files. The train/val split is pre-defined in `splits/fold0.json`.

---

## Repository structure

```
network_ml_project/
├── data/               raw .pt sequence files (train + test)
├── splits/             fold0.json — train/val file lists
├── jobs/               SLURM job scripts for all experiments
├── submissions/        generated CSV files for Kaggle
├── src/
│   ├── utils/          shared metrics, dataset download, split utilities
│   ├── mart/           MART model — our primary architecture
│   ├── hht_cfi/        HHT-CFI model
│   ├── equivariance/   EqMotion model
│   ├── groupnet/       GroupNet model
│   ├── stgcnn/         ST-GCNN baseline
│   └── dynamic/        DNRI baseline
└── experiments.md      full experiment log with val/mse_ft numbers
```

---

## Best model — MART curriculum

The best single model is an augmented MART with curriculum training: min-ADE
loss for the first 1000 epochs, then mean-MSE for the remaining 3000, with an
LR drop at the switch point.

**Train on SLURM:**

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

**Generate submission:**

```bash
cd src/mart
python submit_nba_pt.py \
    --checkpoint checkpoints/mart_curriculum_4k_sw1k_lrdrop_best.ckpt \
    --test_dir ../../data/test/test \
    --out_csv ../../submissions/mart_curriculum_4k_sw1k_lrdrop.csv \
    --gpu 0
```

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
| `main_nba_pt.py` | Training script — all flags documented inline |
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
| `models.py` | MyTraj — upstream HHT-CFI model |
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
(2–5), `x_encoder_head` (4/8), `lr`, `batch_size`, and `grad_clip`. At inference
time we tested min-scale mode selection, sigma-weighted mean, 4-way TTA, and
court-bound clamping — these are toggled via flags at the top of `hht_cfi_nba.py`.

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
python eqmotion_nba.py --iso-norm --add-hoops --cosine-sched --run-name eqmotion_best
```

**Experiments:**

We explored isotropic normalization, cosine LR schedule, hoop nodes, capacity
scaling, ball-weighted loss, and multi-seed ensembles. Best single model uses
iso-norm + cosine + hoops. The 5-seed ensemble reached Kaggle 3.2.

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
python train_hyper_nba_pt.py
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
python train.py --iso_norm --augment
```

---

### DNRI

Dynamic NRI learns the interaction graph structure jointly with trajectories.

**Files:**

| File | Description |
|---|---|
| `dnri_nba.py` | NBA adapter |
| `dnri_ref/` | Upstream DNRI reference implementation |

**Train:**

```bash
cd src/dynamic
python dnri_nba.py
```

---

## Blending submissions

To blend two submission CSVs (e.g. MART + EqMotion at the production weights):

```bash
cd src/mart
python blend_csvs.py \
    --csvs ../../submissions/mart_best.csv ../../submissions/eqmotion_ensemble.csv \
    --weights 0.70 0.30 \
    --out ../../submissions/blend_mart70_eqm30.csv
```

---

## Full experiment log

See `experiments.md` for the complete record of every run with val/mse_ft
numbers, what changed, and what worked or didn't.
