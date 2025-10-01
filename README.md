
# bayes-glmm-fantasy-soccer
=======
# bayes-glmm-fantasy-soccer

Predictive Bayesian GLMM for fantasy soccer (World Cup demo).

## Project structure
- `scripts/` — data acquisition, feature engineering, model scripts
- `data/raw/` — raw StatsBomb JSON files (excluded from Git)
- `data/processed/` — processed CSVs (commit small sample)
- `data/model_output/` — saved model outputs (excluded)
- `notebooks/` — exploratory notebooks

## Quickstart
1. Create virtual environment and install:
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
