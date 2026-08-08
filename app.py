import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import statsmodels.api as sm
from scipy.stats import zscore
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")  # small-sample sklearn convergence/feature-name warnings are just noise here

st.set_page_config(page_title="Formulation Lab | DoE Dashboard", page_icon="🧬", layout="wide")

# ----------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------
# A "lab instrument" identity for a DoE dashboard: cool near-white canvas
# (not the usual warm-cream AI-tool default), a single teal accent, and
# monospace numerals so data reads like a measurement readout rather than
# marketing copy. Tokens:
#   canvas #F5F7F8   panel #FFFFFF   ink #101820   slate #55606B
#   teal (primary)   #0E6E62        amber (highlight) #C97A2B
# Display face: Space Grotesk · Body: Inter · Data: IBM Plex Mono
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --canvas: #F5F7F8;
  --panel: #FFFFFF;
  --ink: #101820;
  --slate: #55606B;
  --teal: #0E6E62;
  --teal-dark: #0B5850;
  --amber: #C97A2B;
  --border: #E2E6EA;
}

.stApp { background-color: var(--canvas); }

h1, h2, h3, h4, .stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2, .stApp [data-testid="stMarkdownContainer"] h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}
html, body, [class*="css"], .stApp, p, label, .stMarkdown {
  font-family: 'Inter', sans-serif;
}
.stApp [data-testid="stMetricValue"], code, .stDataFrame, [data-testid="stDataFrame"] * {
  font-family: 'IBM Plex Mono', monospace !important;
}

/* Hero header */
.app-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 0.3rem;
  font-weight: 600;
}
.app-hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 2.1rem;
  color: var(--ink);
  margin: 0 0 0.3rem 0;
  line-height: 1.15;
}
.app-hero-sub { color: var(--slate); font-size: 0.96rem; margin-bottom: 0.7rem; max-width: 70ch; }
.app-hr { border: none; border-top: 1px solid var(--border); margin: 0 0 1.3rem 0; }

/* Spec strip: a mono "readout" line of key facts under a header */
.spec-strip {
  display: flex; gap: 1.6rem; flex-wrap: wrap;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--slate);
  border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
  padding: 0.55rem 0.1rem; margin: -0.6rem 0 1.5rem 0;
}
.spec-strip b { color: var(--ink); font-weight: 600; }

/* Sidebar as a dark console panel */
[data-testid="stSidebar"] { background-color: var(--ink); }
[data-testid="stSidebar"] * { color: #E8EDF0 !important; }
[data-testid="stSidebar"] .app-eyebrow { color: #57C7B8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.14); }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.14); }

/* Metric / KPI cards */
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  background-color: var(--teal); color: #fff !important; border: none !important;
  border-radius: 8px !important; font-weight: 600; padding: 0.5rem 1.1rem;
}
.stButton>button:hover, .stDownloadButton>button:hover { background-color: var(--teal-dark); }

/* Dataframes / tables */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* Tabs & selects: tidy borders */
[data-baseweb="select"] > div { border-radius: 8px !important; border-color: var(--border) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TEAL_SCALE = [[0.0, "#F5F7F8"], [0.35, "#9FD6CB"], [0.7, "#33978A"], [1.0, "#0E6E62"]]


def plot_config(filename):
    """Config for st.plotly_chart: always-visible toolbar with a PNG export
    button. This runs entirely client-side (Plotly.js), so it needs no
    server-side rendering package (e.g. kaleido+Chrome), which keeps the
    Streamlit Cloud deploy lightweight and reliable."""
    return {
        "displaylogo": False,
        "displayModeBar": True,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
    }


def render_header(eyebrow, title, subtitle=None, facts=None):
    """Consistent hero header used at the top of every page."""
    st.markdown(f'<div class="app-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-hero-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-hero-sub">{subtitle}</div>', unsafe_allow_html=True)
    if facts:
        strip = "".join(f"<span>{f}</span>" for f in facts)
        st.markdown(f'<div class="spec-strip">{strip}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<hr class="app-hr"/>', unsafe_allow_html=True)


# Ensure the folder for saved models exists (needed before joblib.dump)
os.makedirs("models", exist_ok=True)

DATA_PATH = os.path.join("data", "data of the formulation.xlsx")


# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


data = load_data()
X = data[["Stearic acid", "Tween 80"]]
y = data[["Entrapment efficiency", "Drug content", "Drug release", "Particle size"]]


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
# Every model is a Pipeline (StandardScaler + estimator) so all six models
# share one calling convention: models[name].predict([[x1, x2]]).
# Hyperparameters are tuned via leave-one-out cross-validation (LOOCV) —
# with only 10 experimental runs, a single train/test split is far too
# noisy to trust, so every fold gets to act as the test set exactly once.
def build_model_specs():
    return {
        "Linear Regression": (
            Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(LinearRegression()))]),
            {},
        ),
        "Polynomial Regression (RSM, Ridge)": (
            # Ridge (L2-regularized) instead of plain LinearRegression on the
            # degree-2 features: the raw quadratic RSM design is highly
            # collinear (confirmed by the large condition number on the
            # ANOVA page), and regularization stabilizes those coefficients.
            Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("scaler", StandardScaler()),
                ("reg", MultiOutputRegressor(Ridge())),
            ]),
            {"reg__estimator__alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(DecisionTreeRegressor(random_state=42)))]),
            {"reg__estimator__max_depth": [2, 3, None], "reg__estimator__min_samples_leaf": [1, 2]},
        ),
        "Random Forest": (
            Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))]),
            {"reg__estimator__max_depth": [2, 3], "reg__estimator__min_samples_leaf": [1, 2]},
        ),
        "SVR": (
            Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(SVR(kernel="rbf")))]),
            {"reg__estimator__C": [0.1, 1, 10], "reg__estimator__gamma": ["scale", 0.1],
             "reg__estimator__epsilon": [0.01, 0.1]},
        ),
        "XGBoost": (
            Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", random_state=42)))]),
            {"reg__estimator__n_estimators": [50, 100], "reg__estimator__max_depth": [2, 3],
             "reg__estimator__learning_rate": [0.1, 0.2]},
        ),
    }


# ----------------------------------------------------------------------
# Train + tune models (cached so this only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def train_models(X, y):
    loo = LeaveOneOut()
    model_specs = build_model_specs()

    models = {}
    best_params = {}
    metrics = {}

    for name, (pipe, grid) in model_specs.items():
        if grid:
            search = GridSearchCV(pipe, grid, cv=loo, scoring="r2", n_jobs=1)
            search.fit(X, y)
            best_est = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            best_est = pipe.fit(X, y)
            best_params[name] = {}

        # Out-of-fold (leave-one-out) predictions using the tuned hyperparameters
        # give an honest estimate of how the model performs on unseen runs,
        # using every one of the 10 experimental points as a held-out test case.
        oof_pred = cross_val_predict(best_est, X, y, cv=loo, n_jobs=1)
        metrics[name] = {
            "R² (LOOCV)": r2_score(y, oof_pred),
            "MAE (LOOCV)": mean_absolute_error(y, oof_pred),
            "MSE (LOOCV)": mean_squared_error(y, oof_pred),
            "RMSE (LOOCV)": np.sqrt(mean_squared_error(y, oof_pred)),
        }

        models[name] = best_est  # already refit on ALL data by GridSearchCV(refit=True)
        joblib.dump(best_est, f"models/{name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').lower()}.pkl")

    best_model_name = max(metrics, key=lambda n: metrics[n]["R² (LOOCV)"])
    return models, best_params, metrics, best_model_name


with st.spinner("Training and tuning models with leave-one-out cross-validation..."):
    models, best_params, metrics, best_model_name = train_models(X, y)


# ----------------------------------------------------------------------
# Multi-page dashboard
# ----------------------------------------------------------------------
st.sidebar.markdown('<div class="app-eyebrow">Lipid Nanoparticle DoE</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
    'font-size:1.35rem;margin-bottom:1rem;">🧬 Formulation Lab</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Go to",
    ["Dataset", "Prediction", "Reverse Prediction", "Model Comparison", "ANOVA Analysis",
     "Response Surfaces", "Optimization", "Outlier Analysis"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.14);'
    'border-radius:10px;padding:0.8rem 0.9rem;">'
    '<div class="app-eyebrow" style="margin-bottom:0.4rem;">Best model (LOOCV R²)</div>'
    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:600;font-size:1.05rem;'
    f'line-height:1.25;">{best_model_name}</div>'
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:#57C7B8;'
    f'margin-top:0.3rem;">↑ R² = {metrics[best_model_name]["R² (LOOCV)"]:.3f}</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------- Dataset ----------------
if page == "Dataset":
    render_header(
        "Raw Data",
        "Original Dataset",
        "The experimental design matrix used to train and validate every model in this app.",
        facts=[f"<b>{data.shape[0]}</b> runs", f"<b>{data.shape[1]}</b> columns",
               "<b>2</b> factors", "<b>4</b> responses"],
    )
    st.dataframe(data, width='stretch', hide_index=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(data.describe().T, width='stretch')

    st.download_button(
        "Download dataset as CSV",
        data.to_csv(index=False),
        file_name="data_of_the_formulation.csv",
        mime="text/csv",
    )

# ---------------- Prediction UI ----------------
elif page == "Prediction":
    render_header(
        "Forward Model",
        "Formulation Predictor",
        "Enter a candidate formulation and compare predictions across all six tuned models.",
    )
    st.sidebar.header("Input Parameters")
    stearic = st.sidebar.number_input("Stearic acid", min_value=60, max_value=400, step=10, value=240)
    tween = st.sidebar.number_input("Tween 80", min_value=60, max_value=200, step=10, value=120)

    if st.sidebar.button("Predict"):
        output_cols = ["Entrapment efficiency", "Drug content", "Drug release", "Particle size"]
        rows, index = [], []
        for name, model in models.items():
            rows.append(model.predict([[stearic, tween]])[0])
            index.append(f"⭐ {name}" if name == best_model_name else name)

        st.write("#### Predictions from all models")
        st.table(pd.DataFrame(rows, index=index, columns=output_cols))
        st.caption(
            f"⭐ = model with the best leave-one-out cross-validated R² "
            f"({best_model_name}, R² = {metrics[best_model_name]['R² (LOOCV)']:.3f}). "
            "See the Model Comparison page for full details."
        )

# ---------------- Reverse Prediction (Inverse Design) ----------------
elif page == "Reverse Prediction":
    render_header(
        "Inverse Design",
        "Reverse Prediction",
        "Specify the desired outcome and search the design space for the "
        "Stearic acid / Tween 80 combination that best achieves it.",
    )
    st.caption(
        "This is an inverse (many-to-few) problem: 4 targets, 2 tunable "
        "inputs, so an exact match usually isn't possible. The search finds "
        "the best achievable compromise (minimum weighted error) within the "
        "explored design space."
    )

    forward_model_name = st.selectbox(
        "Forward model to search against",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="The reverse search evaluates candidate (Stearic acid, Tween 80) "
             "pairs using this forward model's predictions. Defaults to the "
             "model with the best leave-one-out cross-validated R².",
    )

    st.markdown("### Target Responses")
    c1, c2, c3, c4 = st.columns(4)
    target_ee = c1.number_input(
        "Target Entrapment efficiency", value=float(data["Entrapment efficiency"].mean()),
        step=0.01, format="%.4f")
    target_dc = c2.number_input(
        "Target Drug content", value=float(data["Drug content"].mean()),
        step=0.01, format="%.4f")
    target_dr = c3.number_input(
        "Target Drug release", value=float(data["Drug release"].mean()),
        step=0.01, format="%.4f")
    target_ps = c4.number_input(
        "Target Particle size", value=float(data["Particle size"].mean()), step=1.0)

    st.markdown("### Response Weights (optional)")
    st.caption("Increase a weight to prioritize matching that response more closely.")
    w1, w2, w3, w4 = st.columns(4)
    weight_ee = w1.slider("Entrapment efficiency weight", 0.0, 5.0, 1.0, step=0.5)
    weight_dc = w2.slider("Drug content weight", 0.0, 5.0, 1.0, step=0.5)
    weight_dr = w3.slider("Drug release weight", 0.0, 5.0, 1.0, step=0.5)
    weight_ps = w4.slider("Particle size weight", 0.0, 5.0, 1.0, step=0.5)

    if st.button("Find Matching Formulation"):
        targets = np.array([target_ee, target_dc, target_dr, target_ps])
        weights = np.array([weight_ee, weight_dc, weight_dr, weight_ps])
        y_ranges = (y.max() - y.min()).to_numpy()
        y_ranges[y_ranges == 0] = 1.0  # avoid div-by-zero

        def forward_predict(x1, x2):
            return models[forward_model_name].predict([[x1, x2]])[0]

        def objective(params):
            x1, x2 = params
            pred = forward_predict(x1, x2)
            diff = (pred - targets) / y_ranges
            return np.sum(weights * diff ** 2)

        bounds = [
            (float(X["Stearic acid"].min()), float(X["Stearic acid"].max())),
            (float(X["Tween 80"].min()), float(X["Tween 80"].max())),
        ]
        with st.spinner("Searching the design space..."):
            result = differential_evolution(objective, bounds, seed=42, tol=1e-10, polish=True)
        x1_opt, x2_opt = result.x
        pred_opt = forward_predict(x1_opt, x2_opt)

        st.success("Best-matching formulation found", icon="✅")
        c1, c2 = st.columns(2)
        c1.metric("Stearic acid (mg)", f"{x1_opt:.2f}")
        c2.metric("Tween 80 (mg)", f"{x2_opt:.2f}")

        compare_df = pd.DataFrame({
            "Response": y.columns.tolist(),
            "Target": targets,
            "Predicted at solution": np.round(pred_opt, 4),
            "Abs. difference": np.round(np.abs(pred_opt - targets), 4),
        })
        st.markdown("### Target vs. Achieved")
        st.table(compare_df)

        st.download_button(
            "Download result as CSV",
            pd.DataFrame({
                "Stearic acid": [x1_opt], "Tween 80": [x2_opt],
                "Predicted Entrapment efficiency": [pred_opt[0]],
                "Predicted Drug content": [pred_opt[1]],
                "Predicted Drug release": [pred_opt[2]],
                "Predicted Particle size": [pred_opt[3]],
            }).to_csv(index=False),
            file_name="reverse_prediction_result.csv",
            mime="text/csv",
        )
        st.caption(
            "Note: the search is bounded to the experimentally explored "
            "range (Stearic acid "
            f"{X['Stearic acid'].min():.0f}–{X['Stearic acid'].max():.0f} mg, "
            f"Tween 80 {X['Tween 80'].min():.0f}–{X['Tween 80'].max():.0f} mg) "
            "so results stay within the validated design space."
        )

# ---------------- Model comparison ----------------
elif page == "Model Comparison":
    render_header(
        "Validation",
        "Model Comparison",
        "Leave-one-out cross-validated performance across all six tuned models.",
    )
    st.markdown(
        "Each of the 10 experimental runs is held out and predicted "
        "exactly once by a model trained on the other 9. This is far more "
        "reliable than a single random train/test split on a 10-row dataset, "
        "where the test set would only be 1–2 points."
    )

    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values("R² (LOOCV)", ascending=False)
    metrics_df = metrics_df.reset_index().rename(columns={"index": "Model"})

    def highlight_best(row):
        return ["background-color: #D6EFE9" if row["Model"] == best_model_name else "" for _ in row]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format(
            {c: "{:.4f}" for c in metrics_df.columns if c != "Model"}
        ),
        width='stretch',
        hide_index=True,
        column_config={"Model": st.column_config.TextColumn("Model", width="medium")},
    )
    st.success(
        f"Best model: **{best_model_name}** "
        f"(R² = {metrics[best_model_name]['R² (LOOCV)']:.3f}). "
        "This is the model used as the default in Reverse Prediction and "
        "the ⭐-marked row in Prediction.",
        icon="🏆",
    )

    st.markdown("#### Tuned Hyperparameters")
    st.caption(
        "Hyperparameters were selected via GridSearchCV using the same "
        "leave-one-out cross-validation splits, favoring settings that "
        "generalize rather than settings that just fit the training data best."
    )
    for name, params in best_params.items():
        if params:
            st.write(f"**{name}:** {params}")
        else:
            st.write(f"**{name}:** (no tunable hyperparameters)")

    st.caption(
        "Even with tuning, a 10-run dataset limits how accurate any model can "
        "be — the regularized Polynomial (RSM) model tends to generalize best "
        "here because it matches the underlying 2-factor design-of-experiments "
        "structure, while more flexible models (Random Forest, SVR, XGBoost, "
        "Decision Tree) have too little data to reliably learn complex patterns."
    )

# ---------------- ANOVA ----------------
elif page == "ANOVA Analysis":
    render_header("Statistics", "ANOVA Analysis", "OLS regression summary for any selected response.")
    response_name = st.selectbox("Response", y.columns.tolist())
    X_const = sm.add_constant(X)
    model = sm.OLS(y[response_name], X_const).fit()
    st.text(model.summary())

# ---------------- Response surfaces ----------------
elif page == "Response Surfaces":
    render_header(
        "Response Surface Methodology",
        "3D Surface & Contour Plots",
        "Model-predicted response across the full Stearic acid × Tween 80 design space.",
    )

    c1, c2 = st.columns(2)
    surface_model_name = c1.selectbox(
        "Model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="Defaults to the model with the best leave-one-out cross-validated R².",
    )
    target_options = {
        "Entrapment efficiency": 0,
        "Drug content": 1,
        "Drug release": 2,
        "Particle size": 3,
    }
    target_name = c2.selectbox("Response to plot", list(target_options.keys()))
    target_index = target_options[target_name]

    f1_range = np.linspace(X["Stearic acid"].min(), X["Stearic acid"].max(), 30)
    f2_range = np.linspace(X["Tween 80"].min(), X["Tween 80"].max(), 30)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)
    inputs = np.array([[a, b] for a, b in zip(np.ravel(f1_grid), np.ravel(f2_grid))])
    preds = models[surface_model_name].predict(inputs)[:, target_index].reshape(f1_grid.shape)

    tab_surface, tab_contour = st.tabs(["3D Surface", "2D Contour"])
    with tab_surface:
        fig3d = go.Figure(data=[go.Surface(
            x=f1_range, y=f2_range, z=preds, colorscale=TEAL_SCALE,
            colorbar=dict(title=target_name),
        )])
        fig3d.add_trace(go.Scatter3d(
            x=data["Stearic acid"], y=data["Tween 80"], z=data[target_name],
            mode="markers", marker=dict(size=5, color="#C97A2B"), name="Experimental runs",
        ))
        fig3d.update_layout(
            scene=dict(xaxis_title="Stearic acid (mg)", yaxis_title="Tween 80 (mg)", zaxis_title=target_name),
            height=560, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#101820"),
        )
        st.plotly_chart(fig3d, width='stretch', config=plot_config(f"3d_surface_{target_name.replace(' ', '_').lower()}"))
        st.caption("📷 Hover the chart and use the camera icon in the toolbar to download it as a PNG.")

    with tab_contour:
        figc = go.Figure(data=go.Contour(
            x=f1_range, y=f2_range, z=preds, colorscale=TEAL_SCALE,
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title=target_name),
        ))
        figc.add_trace(go.Scatter(
            x=data["Stearic acid"], y=data["Tween 80"], mode="markers",
            marker=dict(size=10, color="#C97A2B", symbol="x"), name="Experimental runs",
        ))
        figc.update_layout(
            xaxis_title="Stearic acid (mg)", yaxis_title="Tween 80 (mg)",
            height=520, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#101820"),
        )
        st.plotly_chart(figc, width='stretch', config=plot_config(f"contour_{target_name.replace(' ', '_').lower()}"))
        st.caption("📷 Hover the chart and use the camera icon in the toolbar to download it as a PNG.")

# ---------------- Optimization ----------------
elif page == "Optimization":
    render_header(
        "Desirability",
        "Formulation Optimization",
        "Best experimental run by desirability = (Entrapment efficiency × Drug content) / Particle size.",
    )
    desirability = (y["Entrapment efficiency"] * y["Drug content"]) / y["Particle size"]
    best_idx = desirability.idxmax()
    best_row = data.loc[best_idx]

    st.markdown("#### Best formulation (experimental validation)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Run", int(best_row["Runs"]))
    m2.metric("Stearic acid", f"{best_row['Stearic acid']:.0f} mg")
    m3.metric("Tween 80", f"{best_row['Tween 80']:.0f} mg")
    m4.metric("Entrapment eff.", f"{best_row['Entrapment efficiency']:.3f}")
    m5.metric("Drug content", f"{best_row['Drug content']:.3f}")
    m6.metric("Particle size", f"{best_row['Particle size']:.1f} nm")

    st.dataframe(data.loc[[best_idx]], width='stretch', hide_index=True)
    st.caption(
        "Desirability = (Entrapment efficiency × Drug content) / Particle size, "
        "evaluated on the actual experimental runs (not model-predicted)."
    )

# ---------------- Outlier Analysis ----------------
elif page == "Outlier Analysis":
    render_header("Quality Control", "Outlier Detection", "Boxplots and z-score screening across every variable.")

    st.markdown("#### Independent Variables")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Stearic acid", "Tween 80"])
    fig.add_trace(go.Box(y=data["Stearic acid"], name="Stearic acid", marker_color="#0E6E62", boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=data["Tween 80"], name="Tween 80", marker_color="#0E6E62", boxmean=True), row=1, col=2)
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=40, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Inter, sans-serif", color="#101820"))
    st.plotly_chart(fig, width='stretch', config=plot_config("independent_variables_boxplots"))
    st.caption("📷 Hover the chart and use the camera icon in the toolbar to download it as a PNG.")

    st.markdown("#### Dependent Variables")
    dep_cols = ["Entrapment efficiency", "Drug content", "Drug release", "Particle size"]
    fig2 = make_subplots(rows=1, cols=4, subplot_titles=dep_cols)
    for i, col in enumerate(dep_cols, start=1):
        fig2.add_trace(go.Box(y=data[col], name=col, marker_color="#C97A2B", boxmean=True), row=1, col=i)
    fig2.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#101820"))
    st.plotly_chart(fig2, width='stretch', config=plot_config("dependent_variables_boxplots"))
    st.caption("📷 Hover the chart and use the camera icon in the toolbar to download it as a PNG.")

    st.markdown("#### Z-Score Outlier Detection")
    numeric_data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    outliers = (z_scores > 2).any(axis=1)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(range(len(data))), y=data["Entrapment efficiency"], mode="markers",
        marker=dict(size=12, color=np.where(outliers, "#C97A2B", "#0E6E62")),
        text=[f"Run {r}" for r in data["Runs"]], hovertemplate="%{text}<br>%{y}<extra></extra>",
    ))
    fig3.update_layout(
        xaxis_title="Run Index", yaxis_title="Entrapment efficiency",
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#101820"),
    )
    st.plotly_chart(fig3, width='stretch', config=plot_config("zscore_outlier_detection"))
    st.caption("📷 Hover the chart and use the camera icon in the toolbar to download it as a PNG.")

    st.markdown("#### Outlier Runs (All Variables)")
    st.write(data[outliers])
