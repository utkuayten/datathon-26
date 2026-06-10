# Datathon

## Setup

```bash
conda env create -f environment.yml
conda activate datathon
```

## Data

Place `train.csv` and `test_x.csv` inside the `data/` folder.

## Run

```bash
# CatBoost (default)
python predict.py --model catboost

# LightGBM
python predict.py --model lightgbm
```

Output: `submission_catboost.csv` or `submission_lightgbm.csv`

## Hyperparameter Tuning

```bash
python tune.py
```
