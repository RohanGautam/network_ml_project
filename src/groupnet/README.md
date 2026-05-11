# GroupNet Implementation on NBA Dataset


## Implementation

GroupNet algorithm has been implemented from the paper: Multiscale Hypergraph Neural Networks for Trajectory Prediction with Relational Reasoning. For results achieved on our dataset, check: [Presentation Slides](https://docs.google.com/presentation/d/12rBNnDbWdO8c8aaCXfiNe6oXmdsGIoPXkd70uYjaAxk/edit?usp=sharing). To run training, validation, etc., run the following scripts.

> **Note:** Update the paths for saved models, dataset locations, and `split.json` files according to your local setup.


#### Training

```bash
python train_hyper_nba_pt.py \
  --split_path ../network_ml_project/splits/fold0.json \
  --num_epochs 100 \
  --model_save_dir saved_models/nba_pt_run1 \
  --gpu 0
```


#### Hyperparameter Search

```bash
python optuna_search.py \
  --split_path ../network_ml_project/splits/fold0.json \
  --n_trials 20 \
  --trial_epochs 40 \
  --final_epochs 100 \
  --K_eval 50 \
  --final_save_dir saved_models/optuna_best \
  --gpu 0
```


#### Validation Sweep Over `k` (Stochastic Predictions)

```bash
python eval_k_sweep.py \
  --checkpoint saved_models/optuna_best/100.p \
  --split_path ../network_ml_project/splits/fold0.json \
  --ks 50 100 500 5000 10000 \
  --gpu 0
```


#### Generate Submission CSV

```bash
python submit_nba_pt.py \
  --checkpoint saved_models/optuna_best/100.p \
  --test_dir ../network_ml_project/data/test/test \
  --out_csv submissions/optuna_best_k50_cpu.csv \
  --sample_k 5000 \
  --gpu 0
```

