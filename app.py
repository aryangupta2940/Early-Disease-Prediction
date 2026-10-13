import os
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1. PAGE SETUP + STYLING (dark healthcare theme, teal accents)
# ---------------------------------------------------------------
st.set_page_config(page_title="Early Disease Prediction", page_icon="🩺", layout="wide")

st.markdown("""
<style>
/* ---------- Base dark theme ---------- */
.stApp {background-color: #0b1519; color: #e2e8f0;}
[data-testid="stSidebar"] {background-color: #0e1d22;}
.block-container {padding-top: 1.5rem; max-width: 1100px;}
h1, h2, h3, h4 {color: #f1f5f9;}
[data-testid="stWidgetLabel"] p {color: #cbd5e1;}
[data-testid="stMetricValue"] {color: #5eead4;}
[data-testid="stMetricLabel"] p {color: #94a3b8;}
[data-testid="stCaptionContainer"] {color: #94a3b8;}
[data-testid="stSliderThumbValue"] {color: #5eead4;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: rgba(20,184,166,0.35) !important;}
button[data-baseweb="tab"] {font-size: 1.05rem;}

/* ---------- Hero ---------- */
.hero {background: linear-gradient(120deg, #0f3d3e, #0f766e); border: 1px solid #14b8a6;
       padding: 26px 32px; border-radius: 16px; margin-bottom: 14px;}
.hero h1 {margin: 0; color: #ffffff; font-size: 2rem;}
.hero h3 {margin: 4px 0 6px 0; color: #99f6e4; font-weight: 500; font-size: 1.1rem;}
.hero p {margin: 0; color: #d1fae5;}

/* ---------- Compact statistic cards ---------- */
.stat {background: #12252b; border: 1px solid #1f3f46; border-radius: 10px;
       padding: 8px 10px; text-align: center;}
.stat .num {font-size: 1.3rem; font-weight: 700; color: #5eead4;}
.stat .lbl {font-size: 0.78rem; color: #94a3b8;}

/* ---------- Generic text styles ---------- */
.card-title {font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0;}
.card-sub {color: #94a3b8; margin: 2px 0 8px 0;}
.form-label {font-weight: 600; color: #5eead4; margin: 12px 0 2px 0;}

/* ---------- Model cards ---------- */
.model-card {background: #12252b; border: 1px solid #1f3f46; border-radius: 12px;
             padding: 14px; min-height: 200px; font-size: 0.88rem; color: #cbd5e1;}
.model-card:hover {border-color: #14b8a6;}
.model-card h4 {margin: 0 0 6px 0; color: #e2e8f0; font-size: 1rem;}
.model-card p {margin: 0;}
.model-active {background: #0f3532; border: 2px solid #14b8a6;
               box-shadow: 0 0 14px rgba(20,184,166,0.45);}
.model-active h4 {color: #5eead4;}
.badge {display: inline-block; background: #14b8a6; color: #04201d; font-size: 0.72rem;
        font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-bottom: 6px;}

/* ---------- Result cards ---------- */
.result-high {background: rgba(239,68,68,0.12); border: 1px solid #ef4444; color: #fecaca;
              padding: 16px; border-radius: 14px; text-align: center;
              font-size: 1.4rem; font-weight: 700;}
.result-low {background: rgba(34,197,94,0.12); border: 1px solid #22c55e; color: #bbf7d0;
             padding: 16px; border-radius: 14px; text-align: center;
             font-size: 1.4rem; font-weight: 700;}

/* ---------- Predict button ---------- */
div.stButton > button {background: #14b8a6; color: #04201d; border: none;
    padding: 0.75rem 1rem; font-size: 1.15rem; font-weight: 700; border-radius: 12px;}
div.stButton > button:hover {background: #2dd4bf; color: #04201d;}

/* ---------- Workflow ---------- */
.flow-step {background: #12252b; border: 1px solid #1f3f46; border-radius: 10px;
            padding: 8px 12px; text-align: center; color: #e2e8f0; max-width: 340px; margin: 0 auto;}
.flow-hl {background: #0f3532; border: 2px solid #14b8a6;}
.flow-arrow {text-align: center; color: #14b8a6; font-size: 1.2rem; line-height: 1.3;}

/* ---------- Comparison table ---------- */
.cmp {width: 100%; border-collapse: collapse; font-size: 0.92rem;}
.cmp th {text-align: left; color: #5eead4; border-bottom: 1px solid #1f3f46; padding: 8px;}
.cmp td {color: #cbd5e1; border-bottom: 1px solid #16303a; padding: 8px;}
.cmp tr.row-active td {background: #0f3532; color: #5eead4; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# The 8 features in EXACTLY the same order used in the notebook
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# In the notebook, a 0 in these columns was treated as a missing value (NaN) and then
# filled by the median imputer. The app does the same for the patient's input.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DISCLAIMER = ("This tool is an educational machine learning project and is not a medical "
              "diagnosis. Please consult a qualified healthcare professional for medical advice.")

# Information about the five models evaluated in the notebook
# (only Random Forest is used by the app for prediction)
ACTIVE_MODEL = "Random Forest"
MODELS_INFO = [
    ("Logistic Regression", "A classification algorithm used to estimate the probability of a binary outcome."),
    ("Decision Tree", "A tree-based model that makes predictions using a sequence of decision rules."),
    ("Random Forest", "An ensemble of multiple decision trees used for classification."),
    ("KNN", "A distance-based algorithm that classifies a patient using nearby data points."),
    ("SVM", "A classification algorithm that finds a decision boundary to separate different classes."),
]

# Test-set results from the corrected notebook
# (name, accuracy, precision, recall, f1, used in app)
COMPARISON = [
    ("Logistic Regression", "70.78%", "60.00%", "50.00%", "54.55%", False),
    ("Decision Tree", "75.97%", "63.93%", "72.22%", "67.83%", False),
    ("Random Forest", "76.62%", "69.57%", "59.26%", "64.00%", True),
    ("KNN", "75.32%", "66.00%", "61.11%", "63.46%", False),
    ("SVM", "74.03%", "65.22%", "55.56%", "60.00%", False),
]


# ---------------------------------------------------------------
# 2. LOAD THE SAVED RANDOM FOREST MODEL + IMPUTER (once, cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_model_and_imputer():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "models", "random_forest_model.pkl")
    imputer_path = os.path.join(base, "models", "diabetes_imputer.pkl")
    rf_model = joblib.load(model_path)
    saved_imputer = joblib.load(imputer_path)
    return rf_model, saved_imputer

try:
    model, imputer = load_model_and_imputer()
except FileNotFoundError:
    st.error(
        "Model file not found. Run the model-saving cell in the notebook first "
        "(it creates models/random_forest_model.pkl and models/diabetes_imputer.pkl)."
    )
    st.stop()


# ---------------------------------------------------------------
# 3. SMALL HELPER FUNCTIONS FOR THE UI
# ---------------------------------------------------------------
def render_model_cards(active_text):
    """Show the five model cards in one row; only the Random Forest card is highlighted."""
    for col, (name, desc) in zip(st.columns(5), MODELS_INFO):
        if name == ACTIVE_MODEL:
            html = (f'<div class="model-card model-active"><span class="badge">{active_text}</span>'
                    f'<h4>{name}</h4><p>{desc}</p></div>')
        else:
            html = f'<div class="model-card"><h4>{name}</h4><p>{desc}</p></div>'
        col.markdown(html, unsafe_allow_html=True)


def render_probability_donut(prob_no, prob_yes):
    """Donut chart of the two probabilities (uses Streamlit's built-in Vega-Lite, no extra library)."""
    chart_data = pd.DataFrame({
        "Class": ["No Diabetes", "Diabetes"],
        "Probability": [prob_no * 100, prob_yes * 100],
        "Label": [f"{prob_no * 100:.1f}%", f"{prob_yes * 100:.1f}%"],
    })
    spec = {
        "background": "transparent",
        "height": 260,
        "encoding": {
            "theta": {"field": "Probability", "type": "quantitative", "stack": True},
            "color": {
                "field": "Class", "type": "nominal",
                "scale": {"domain": ["No Diabetes", "Diabetes"], "range": ["#14b8a6", "#f59e0b"]},
                "legend": {"orient": "bottom", "title": None},
            },
            "tooltip": [{"field": "Class", "type": "nominal"},
                        {"field": "Label", "type": "nominal", "title": "Probability"}],
        },
        "layer": [
            {"mark": {"type": "arc", "innerRadius": 60, "outerRadius": 105}},
            {"mark": {"type": "text", "radius": 83, "fontSize": 14, "fontWeight": "bold",
                      "color": "#04201d"},
             "encoding": {"text": {"field": "Label", "type": "nominal"}}},
        ],
        "config": {"view": {"stroke": None},
                   "legend": {"labelColor": "#cbd5e1", "labelFontSize": 13}},
    }
    st.vega_lite_chart(chart_data, spec, use_container_width=True, theme=None)


def comparison_table_html():
    """Build the model comparison table (values come from the corrected notebook results)."""
    rows = ""
    for name, acc, prec, rec, f1, used in COMPARISON:
        css = ' class="row-active"' if used else ""
        used_text = "Yes" if used else "No"
        rows += (f"<tr{css}><td>{name}</td><td>{acc}</td><td>{prec}</td>"
                 f"<td>{rec}</td><td>{f1}</td><td>{used_text}</td></tr>")
    head = ("<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th>"
            "<th>Recall</th><th>F1-score</th><th>Used in App</th></tr></thead>")
    return f'<table class="cmp">{head}<tbody>{rows}</tbody></table>'


# ---------------------------------------------------------------
# 4. SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("🩺 Early Disease Prediction")
st.sidebar.markdown("**Project Statistics**")
st.sidebar.markdown("- 768 Patient Records\n- 8 Health Features\n- 5 ML Models\n- 1 Prediction System")
st.sidebar.info(DISCLAIMER)


# ---------------------------------------------------------------
# 5. HERO + COMPACT STATISTICS
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🩺 Early Disease Prediction</h1>
<h3>AI-Powered Diabetes Risk Assessment</h3>
<p>Estimate diabetes risk using health parameters and machine learning.</p>
</div>
""", unsafe_allow_html=True)

stats = [("768", "Patient Records"), ("8", "Health Features"),
         ("5", "ML Models"), ("1", "Prediction System")]
for col, (num, label) in zip(st.columns(4), stats):
    col.markdown(f'<div class="stat"><div class="num">{num}</div>'
                 f'<div class="lbl">{label}</div></div>', unsafe_allow_html=True)

st.write("")

# Two main sections as tabs (Prediction is the first/default tab)
tab_pred, tab_info = st.tabs(["🩺 Prediction", "ℹ️ Information"])

# These stay None until a valid prediction is made
prediction = None
probability = None


# ===============================================================
# TAB 1: PREDICTION
# ===============================================================
with tab_pred:

    # ---------- A. Patient input form ----------
    with st.container(border=True):
        st.markdown('<p class="card-title">Patient Health Assessment</p>'
                    '<p class="card-sub">Enter the patient\'s health parameters to estimate diabetes risk.</p>',
                    unsafe_allow_html=True)

        st.markdown('<div class="form-label">👤 Patient Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        pregnancies = c1.slider("Pregnancies", 0, 20, 2, step=1,
                                help="Number of times pregnant")
        age = c2.slider("Age", 1, 120, 35, step=1, help="Age in years")

        st.markdown('<div class="form-label">🧪 Clinical Measurements</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        glucose = c1.slider("Glucose", 0, 300, 120, step=1,
                            help="Plasma glucose concentration")
        blood_pressure = c2.slider("Blood Pressure", 0, 200, 70, step=1,
                                   help="Diastolic blood pressure (mm Hg)")
        bmi = c3.slider("BMI", 0.0, 70.0, 28.5, step=0.1, help="Body Mass Index")
        c1, c2, c3 = st.columns(3)
        skin_thickness = c1.slider("Skin Thickness", 0, 100, 25, step=1,
                                   help="Triceps skin fold thickness (mm)")
        insulin = c2.slider("Insulin", 0, 900, 80, step=1,
                            help="2-hour serum insulin (mu U/ml)")

        st.markdown('<div class="form-label">🧬 Risk Indicator</div>', unsafe_allow_html=True)
        c1, _ = st.columns([1, 2])
        dpf = c1.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.35, step=0.01,
                        help="Diabetes hereditary risk indicator")

    st.write("")
    predict_clicked = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

    if predict_clicked:
        # --- basic validation ---
        errors = []
        if age <= 0:
            errors.append("Age must be greater than 0.")
        if glucose <= 0:
            errors.append("Glucose must be greater than 0.")
        if blood_pressure <= 0:
            errors.append("Blood Pressure must be greater than 0.")
        if bmi <= 0:
            errors.append("BMI must be greater than 0.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # Step 1: put the 8 values into a DataFrame, in the SAME order as training
            patient = pd.DataFrame(
                [[
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    dpf,
                    age
                ]],
                columns=FEATURES
            )

            # Step 2: same cleaning rule as the notebook - a 0 in these columns means "missing"
            # (this only matters for Skin Thickness and Insulin, because the others are validated > 0)
            patient[ZERO_MEANS_MISSING] = patient[ZERO_MEANS_MISSING].replace(0, float("nan"))

            # Step 3: apply the SAVED imputer (fitted on training data only, never refitted here)
            patient_imputed = imputer.transform(patient)

            # Step 4: the saved Random Forest receives the imputed values
            prediction = model.predict(patient_imputed)[0]
            probability = model.predict_proba(patient_imputed)[0]   # [P(no diabetes), P(diabetes)]

    # ---------- B, C, D. Result, chart, summary ----------
    if prediction is not None:
        st.write("")
        with st.container(border=True):
            st.markdown('<p class="card-title">Prediction Result</p>', unsafe_allow_html=True)
            st.write("")

            if prediction == 1:
                st.markdown('<div class="result-high">🔴 HIGHER PREDICTED RISK</div>',
                            unsafe_allow_html=True)
                st.write("")
                st.write("Based on the provided inputs, the machine learning model predicts "
                         "a higher likelihood of diabetes.")
            else:
                st.markdown('<div class="result-low">🟢 LOWER PREDICTED RISK</div>',
                            unsafe_allow_html=True)
                st.write("")
                st.write("Based on the provided inputs, the machine learning model predicts "
                         "a lower likelihood of diabetes.")

            st.markdown("#### Prediction Probability")
            chart_col, metric_col = st.columns([1, 1])
            with chart_col:
                render_probability_donut(probability[0], probability[1])
            with metric_col:
                st.write("")
                st.metric("Probability of No Diabetes", f"{probability[0] * 100:.1f}%")
                st.metric("Probability of Diabetes", f"{probability[1] * 100:.1f}%")
                st.progress(float(probability[1]), text="Estimated diabetes probability")

            st.caption("Model Prediction (Random Forest). Prediction model: Random Forest.")
            st.caption("Random Forest test accuracy: 76.62% — this is the model's performance on the "
                       "notebook's held-out test data. It is NOT the probability that an individual "
                       "patient has diabetes. This patient's probabilities are the two values shown "
                       "above, returned by model.predict_proba(patient_imputed).")
            st.caption(DISCLAIMER)

    # ---------- E. Model information ----------
    st.write("")
    st.markdown("### Machine Learning Models Used")
    render_model_cards("✓ ACTIVE MODEL")
    st.write("")
    st.caption("Five models were trained and evaluated in the notebook. "
               "Random Forest is the model currently used by the application for prediction.")


# ===============================================================
# TAB 2: INFORMATION
# ===============================================================
with tab_info:

    # ---------- A. About the project ----------
    with st.expander("📘 About the Project", expanded=True):
        st.markdown("**Early Disease Prediction Using Machine Learning on Health Data**")
        st.write("This project uses machine learning to estimate diabetes risk from "
                 "health-related parameters using the Pima Indians Diabetes Dataset.")

    # ---------- B. Dataset ----------
    with st.expander("📊 Dataset Information", expanded=True):
        d1, d2, d3 = st.columns(3)
        d1.metric("Patient Records", "768")
        d2.metric("Input Features", "8")
        d3.metric("Target", "Outcome")
        st.write("**Dataset:** Pima Indians Diabetes Dataset")
        st.write("**Outcome:** 0 = No Diabetes, 1 = Diabetes")
        st.write("**Input features:** " + ", ".join(FEATURES))
        st.write("**Data cleaning:** zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI "
                 "are treated as missing values and filled with the median (imputer fitted on the "
                 "training data only).")

    # ---------- C. Models ----------
    with st.expander("🤖 Machine Learning Models", expanded=True):
        render_model_cards("✓ Currently Used for Prediction")
        st.write("")
        st.write("**Random Forest Test Accuracy: 76.62%**")
        st.caption("Random Forest is the model currently used by the application.")

    # ---------- D. Comparison ----------
    with st.expander("📈 Model Comparison", expanded=True):
        st.markdown(comparison_table_html(), unsafe_allow_html=True)
        st.write("")
        st.caption("Results are from the corrected notebook's held-out test data and cover five "
                   "evaluated models. These are machine learning evaluation scores on this "
                   "dataset and are not the same as medical accuracy.")

    # ---------- E. How the system works ----------
    with st.expander("⚙️ How the System Works", expanded=True):
        flow = [("Patient Health Data", False), ("Median Imputation", False),
                ("Random Forest Model", True), ("Prediction", False),
                ("Probability", False)]
        html = ""
        for i, (step, highlight) in enumerate(flow):
            css = "flow-step flow-hl" if highlight else "flow-step"
            html += f'<div class="{css}">{step}</div>'
            if i < len(flow) - 1:
                html += '<div class="flow-arrow">↓</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.write("")
        st.write("**Random Forest = prediction model**")

    # ---------- F. Project workflow ----------
    with st.expander("🧭 Project Workflow", expanded=True):
        st.markdown("`Data Collection` → `Data Cleaning` → `EDA` → `Preprocessing` → "
                    "`Model Training` → `Model Evaluation` → `Prediction`")

    # ---------- G. Disclaimer ----------
    st.warning(DISCLAIMER)