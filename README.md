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

## Design

The app has a deliberate visual identity rather than default Streamlit
styling — a "lab instrument" look built for this specific dataset (a
lipid-nanoparticle design-of-experiments dashboard):

- **Palette:** cool near-white canvas (`#F5F7F8`), a single deep-teal accent
  (`#0E6E62`), amber (`#C97A2B`) used sparingly to mark experimental runs on
  charts.
- **Type:** Space Grotesk for headings, Inter for body text, IBM Plex Mono
  for every number/data table — figures read like instrument output rather
  than marketing copy.
- **Layout:** a dark teal/ink sidebar "console," a consistent eyebrow-label
  + title header on every page, and a monospace "spec strip" of key facts
  under the Dataset header.
- **Charts:** all plots (3D response surfaces, contour plots, boxplots,
  outlier scatter) are interactive Plotly charts in the matching palette,
  not static matplotlib images.

Theme colors are also set in `.streamlit/config.toml` so native widgets
(sliders, buttons, inputs) pick up the same teal accent automatically.

## Pages

- **Dataset** — the original 10-run experimental design matrix (all columns), summary statistics, and a CSV download
- **Prediction** — enter Stearic acid / Tween 80 and get predictions (Entrapment efficiency, Drug content, Drug release, Particle size) from all six tuned models, with the best-performing model (⭐) flagged
- **Reverse Prediction (Inverse Design)** — enter *target* outputs and the app searches the design space for the Stearic acid / Tween 80 combination that best achieves them, using the best-performing model by default
- **Model Comparison** — leave-one-out cross-validated R², MAE, MSE, RMSE for all six models, plus the tuned hyperparameters found for each
- **ANOVA Analysis** — OLS regression summary for any selected response
- **Response Surfaces** — 3D surface + contour plot for any of the four responses (Entrapment efficiency, Drug content, Drug release, Particle size), for any of the six models, over the full Stearic acid × Tween 80 design space
- **Optimization** — best experimental run by desirability = (EE × Drug content) / Particle size
- **Outlier Analysis** — boxplots and z-score outlier detection

## Model Performance

Model training and evaluation are built for a **10-run dataset**, where a
single random train/test split is too noisy to trust:

- **Leave-one-out cross-validation (LOOCV)** — every one of the 10
  experimental runs is held out and predicted exactly once by a model
  trained on the other 9, instead of an 80/20 split (which on 10 rows
  leaves only 1–2 test points).
- **Hyperparameter tuning** — `GridSearchCV` searches each model's
  hyperparameters using the same LOOCV splits (tree depth for Random
  Forest/Decision Tree, `C`/`gamma`/`epsilon` for SVR, learning
  rate/depth/estimators for XGBoost, regularization strength for the
  polynomial model), favoring settings that generalize instead of settings
  that just memorize the 9 training points.
- **Feature scaling** — every model runs inside a
  `Pipeline(StandardScaler → estimator)`, which matters for SVR and the
  regularized polynomial model given Stearic acid (120–360) and Tween 80
  (60–180) are on different scales.
- **Regularized polynomial RSM (Ridge)** — the quadratic response-surface
  model uses `Ridge` instead of plain `LinearRegression` on the degree-2
  features, since the ANOVA page's large condition number flags real
  multicollinearity in the quadratic design; Ridge stabilizes those
  coefficients.
- **Best-model tracking** — the app computes LOOCV R² for all six models
  and marks the winner (⭐) throughout the UI: it's the default in
  **Reverse Prediction** and **Response Surfaces**, and highlighted on
  **Model Comparison**.

In practice, the regularized polynomial (RSM) model tends to come out on
top — it matches the actual 2-factor design-of-experiments structure the
data was collected under, while more flexible models (Random Forest, SVR,
XGBoost, Decision Tree) have too few points to learn complex patterns
reliably. See the **Model Comparison** page for exact metrics and tuned
hyperparameters.

## Project structure

```
formulation-app/
├── app.py
├── data/
│   └── data of the formulation.xlsx
├── requirements.txt
├── .streamlit/
│   └── config.toml
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

- The dataset has only **10 experimental runs**. LOOCV makes the best use
  of that limited data for evaluation, but no amount of methodology
  substitutes for more experiments — treat all metrics as directional, not
  as production-grade accuracy guarantees.
- Reverse Prediction is an inverse (many-to-few) search: 4 target responses,
  2 tunable inputs, so an exact match is usually impossible. The optimizer
  (differential evolution) finds the best achievable compromise within the
  bounded design space, and shows how close the result actually gets to
  each target.
- XGBoost and scikit-learn wheels are large; the first Streamlit Cloud build
  can take a few minutes.
- Training + hyperparameter tuning (cached via `st.cache_resource`) runs
  once per session — expect a short delay (well under a minute) the first
  time the app loads or after `app.py`/the data file changes.
