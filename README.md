# Formulation Dashboard

This Streamlit dashboard predicts **Entrapment Efficiency, Drug Content, and
Drug Release** using multiple models and statistical methods.

**Inputs:** Stearic acid, Tween 80
**Outputs (multi-target):** Entrapment efficiency, Drug content, Drug release, Particle size

## Features

- ANOVA analysis
- Response Surface Methodology (RSM)
- 3D response surface plots
- Contour plots
- Desirability optimization
- Experimental validation
- Outlier detection (independent + dependent variables)
- Machine learning models:
  - Linear Regression
  - Polynomial Regression (RSM)
  - Decision Tree
  - Random Forest
  - SVR
  - XGBoost

## Pages

- **Prediction** — enter Stearic acid / Tween 80 and get predictions (Entrapment efficiency, Drug content, Drug release, Particle size) from Random Forest, SVR, and Quadratic (RSM)
- **Reverse Prediction (Inverse Design)** — enter *target* outputs and the app searches the design space for the Stearic acid / Tween 80 combination that best achieves them
- **Model Comparison** — R², MAE, MSE, RMSE for all six models on the held-out test split
- **ANOVA Analysis** — OLS regression summary for any selected response
- **Response Surfaces** — 3D surface + contour plot (Random Forest) for any of the four responses (Entrapment efficiency, Drug content, Drug release, Particle size), over the full Stearic acid × Tween 80 design space
- **Optimization** — best experimental run by desirability = (EE × Drug content) / Particle size
- **Outlier Analysis** — boxplots and z-score outlier detection

## Project structure

```
formulation-app/
├── app.py
├── data/
│   └── data of the formulation.xlsx
├── requirements.txt
├── .gitignore
└── README.md
```

Trained models are saved to a local `models/` folder each time the app
starts (for inspection/reuse); this folder is git-ignored since it's
regenerated automatically — no need to commit it.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy to GitHub + Streamlit Community Cloud

1. Push this folder to a new GitHub repo:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: formulation prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)**, sign in with GitHub.
3. **New app** → select the repo/branch → main file path **`app.py`** → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically. Future
   `git push` to `main` auto-redeploys.

### Notes / known limitations

- The dataset has only **10 experimental runs**; with an 80/20 split that's
  8 train / 2 test rows, so the Model Comparison metrics are illustrative,
  not statistically robust — treat relative rankings loosely.
- Reverse Prediction is an inverse (many-to-few) search: 4 target responses,
  2 tunable inputs, so an exact match is usually impossible. The optimizer
  (differential evolution) finds the best achievable compromise within the
  bounded design space, and shows how close the result actually gets to
  each target.
- XGBoost and scikit-learn wheels are large; the first Streamlit Cloud build
  can take a few minutes.
- Training re-runs (cached via `st.cache_resource`) whenever `app.py` or the
  data file changes, so predictions always reflect the current dataset.
