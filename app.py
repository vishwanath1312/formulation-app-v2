import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import statsmodels.api as sm
from scipy.stats import zscore
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Formulation Prediction App", page_icon="🧪", layout="wide")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------------------------------------------------------
# Train models (cached so the app doesn't retrain on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": MultiOutputRegressor(LinearRegression()),
        "Polynomial Regression (RSM)": MultiOutputRegressor(LinearRegression()),
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf")),
        "XGBoost": MultiOutputRegressor(XGBRegressor(objective="reg:squarederror")),
    }

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train)

    results = {}
    for name, model in models.items():
        if "Polynomial" in name:
            model.fit(X_poly, y_train)
            preds = model.predict(poly.transform(X_test))
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        results[name] = preds
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

    return models, poly, results


models, poly, results = train_models(X_train, X_test, y_train, y_test)


# ----------------------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ----------------------------------------------------------------------
# Multi-page dashboard
# ----------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Reverse Prediction", "Model Comparison", "ANOVA Analysis",
     "Response Surfaces", "Optimization", "Outlier Analysis"],
)

# ---------------- Prediction UI ----------------
if page == "Prediction":
    st.title("Formulation Prediction App")
    st.sidebar.header("Input Parameters")
    stearic = st.sidebar.number_input("Stearic acid", min_value=60, max_value=400, step=10, value=240)
    tween = st.sidebar.number_input("Tween 80", min_value=60, max_value=200, step=10, value=120)

    if st.sidebar.button("Predict"):
        rf_pred = models["Random Forest"].predict([[stearic, tween]])
        svr_pred = models["SVR"].predict([[stearic, tween]])
        quad_pred = models["Polynomial Regression (RSM)"].predict(poly.transform([[stearic, tween]]))

        output_cols = ["Entrapment efficiency", "Drug content", "Drug release", "Particle size"]
        st.write("### Predictions")
        st.table(pd.DataFrame(
            [rf_pred[0], svr_pred[0], quad_pred[0]],
            index=["Random Forest", "SVR", "Quadratic (RSM)"],
            columns=output_cols,
        ))

# ---------------- Reverse Prediction (Inverse Design) ----------------
elif page == "Reverse Prediction":
    st.title("Reverse Prediction (Inverse Design)")
    st.markdown(
        "Specify the **desired outcome** — target Entrapment efficiency, "
        "Drug content, Drug release, and Particle size — and this page "
        "searches the design space for the **Stearic acid / Tween 80** "
        "combination whose predicted outputs come closest to those targets."
    )
    st.caption(
        "This is an inverse (many-to-few) problem: 4 targets, 2 tunable "
        "inputs, so an exact match usually isn't possible. The search finds "
        "the best achievable compromise (minimum weighted error) within the "
        "explored design space."
    )

    forward_model_name = st.selectbox(
        "Forward model to search against",
        ["Random Forest", "SVR", "Polynomial Regression (RSM)", "XGBoost", "Linear Regression", "Decision Tree"],
        help="The reverse search evaluates candidate (Stearic acid, Tween 80) "
             "pairs using this forward model's predictions.",
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
            if forward_model_name == "Polynomial Regression (RSM)":
                return models[forward_model_name].predict(poly.transform([[x1, x2]]))[0]
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
    st.title("Model Comparison")
    metrics_table = {}
    for name, preds in results.items():
        metrics_table[name] = evaluate(y_test, preds)
    st.table(pd.DataFrame(metrics_table).T)
    st.caption(
        f"Metrics computed on a held-out test split ({len(X_test)} of {len(X)} runs). "
        "With a small 10-run dataset these numbers are illustrative rather than statistically robust."
    )

# ---------------- ANOVA ----------------
elif page == "ANOVA Analysis":
    st.title("ANOVA Analysis")
    response_name = st.selectbox("Response", y.columns.tolist())
    X_const = sm.add_constant(X)
    model = sm.OLS(y[response_name], X_const).fit()
    st.text(model.summary())

# ---------------- Response surfaces ----------------
elif page == "Response Surfaces":
    st.title("Response Surface & Contour Plots")

    def plot_surface(feature1, feature2, target_index, title):
        f1_range = np.linspace(X[feature1].min(), X[feature1].max(), 30)
        f2_range = np.linspace(X[feature2].min(), X[feature2].max(), 30)
        f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)
        inputs = np.array([[f1, f2] for f1, f2 in zip(np.ravel(f1_grid), np.ravel(f2_grid))])
        preds = models["Random Forest"].predict(inputs)[:, target_index].reshape(f1_grid.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(f1_grid, f2_grid, preds, cmap="viridis")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel("Response")
        ax.set_title(title)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        contour = ax2.contourf(f1_grid, f2_grid, preds, cmap="viridis")
        fig2.colorbar(contour)
        ax2.set_xlabel(feature1)
        ax2.set_ylabel(feature2)
        ax2.set_title(f"Contour Plot - {title}")
        st.pyplot(fig2)

    target_options = {
        "Entrapment efficiency": 0,
        "Drug content": 1,
        "Drug release": 2,
        "Particle size": 3,
    }
    target_name = st.selectbox("Response to plot", list(target_options.keys()))
    plot_surface("Stearic acid", "Tween 80", target_options[target_name], target_name)

# ---------------- Optimization ----------------
elif page == "Optimization":
    st.title("Desirability Optimization")
    desirability = (y["Entrapment efficiency"] * y["Drug content"]) / y["Particle size"]
    best_idx = desirability.idxmax()
    st.write("Best formulation (experimental validation):")
    st.write(data.loc[best_idx])
    st.caption(
        "Desirability = (Entrapment efficiency × Drug content) / Particle size, "
        "evaluated on the actual experimental runs (not model-predicted)."
    )

# ---------------- Outlier Analysis ----------------
elif page == "Outlier Analysis":
    st.title("Outlier Detection")

    st.subheader("Independent Variables - Boxplots")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].boxplot(data["Stearic acid"]); axes[0].set_title("Stearic acid")
    axes[1].boxplot(data["Tween 80"]); axes[1].set_title("Tween 80")
    st.pyplot(fig)

    st.subheader("Dependent Variables - Boxplots")
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
    axes2[0].boxplot(data["Entrapment efficiency"]); axes2[0].set_title("Entrapment efficiency")
    axes2[1].boxplot(data["Drug content"]); axes2[1].set_title("Drug content")
    axes2[2].boxplot(data["Drug release"]); axes2[2].set_title("Drug release")
    axes2[3].boxplot(data["Particle size"]); axes2[3].set_title("Particle size")
    st.pyplot(fig2)

    st.subheader("Z-Score Outlier Detection")
    numeric_data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    outliers = (z_scores > 2).any(axis=1)

    fig3, ax3 = plt.subplots()
    ax3.scatter(range(len(data)), data["Entrapment efficiency"], c=~outliers, cmap="coolwarm")
    ax3.set_xlabel("Run Index")
    ax3.set_ylabel("Entrapment efficiency")
    ax3.set_title("Outlier Detection (Entrapment efficiency)")
    st.pyplot(fig3)

    st.write("### Outlier Runs (All Variables)")
    st.write(data[outliers])
