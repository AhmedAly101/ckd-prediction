import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------
# Load the trained model and scaler (saved from the notebook)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("ckd_model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_artifacts()

# ---------------------------------------------------------------
# Must match the training notebook exactly
# ---------------------------------------------------------------
# Column order of X_train
FEATURE_ORDER = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet",
    "pe", "ane",
]

# Columns the scaler was fitted on (same order as numeric_cols in the notebook)
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
]

# LabelEncoder sorts classes alphabetically, so the first one is 0
ENCODING = {
    "rbc":   {"abnormal": 0, "normal": 1},
    "pc":    {"abnormal": 0, "normal": 1},
    "pcc":   {"notpresent": 0, "present": 1},
    "ba":    {"notpresent": 0, "present": 1},
    "htn":   {"no": 0, "yes": 1},
    "dm":    {"no": 0, "yes": 1},
    "cad":   {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe":    {"no": 0, "yes": 1},
    "ane":   {"no": 0, "yes": 1},
}

# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
st.title("Chronic Kidney Disease Prediction")
st.write(
    "Enter the patient's lab results and history, then press **Predict**. "
    "All fields are required."
)

with st.form("patient_form"):
    st.subheader("Patient")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 1, 100, 55)
    bp = c2.number_input("Blood pressure (mm/Hg)", 40, 200, 80)

    st.subheader("Urine tests")
    c1, c2, c3 = st.columns(3)
    sg = c1.selectbox("Specific gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=4)
    al = c2.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
    su = c3.selectbox("Sugar (su)", [0, 1, 2, 3, 4, 5])
    c1, c2 = st.columns(2)
    rbc = c1.selectbox("Red blood cells (rbc)", ["normal", "abnormal"])
    pc = c2.selectbox("Pus cells (pc)", ["normal", "abnormal"])
    c1, c2 = st.columns(2)
    pcc = c1.selectbox("Pus cell clumps (pcc)", ["notpresent", "present"])
    ba = c2.selectbox("Bacteria (ba)", ["notpresent", "present"])

    st.subheader("Blood tests")
    c1, c2, c3 = st.columns(3)
    bgr = c1.number_input("Blood glucose random (mg/dl)", 20, 600, 120)
    bu = c2.number_input("Blood urea (mg/dl)", 1.0, 400.0, 40.0)
    sc = c3.number_input("Serum creatinine (mg/dl)", 0.2, 80.0, 1.2, step=0.1)
    c1, c2, c3 = st.columns(3)
    sod = c1.number_input("Sodium (mEq/L)", 100.0, 170.0, 138.0)
    pot = c2.number_input("Potassium (mEq/L)", 1.5, 10.0, 4.4, step=0.1)
    hemo = c3.number_input("Hemoglobin (g/dl)", 3.0, 18.0, 15.0, step=0.1)
    c1, c2, c3 = st.columns(3)
    pcv = c1.number_input("Packed cell volume (pcv)", 9, 55, 45)
    wbcc = c2.number_input("White blood cell count (cells/cumm)", 2000, 27000, 8000)
    rbcc = c3.number_input("Red blood cell count (millions/cmm)", 2.0, 8.0, 5.2, step=0.1)

    st.subheader("History and symptoms")
    c1, c2, c3 = st.columns(3)
    htn = c1.selectbox("Hypertension (htn)", ["no", "yes"])
    dm = c2.selectbox("Diabetes mellitus (dm)", ["no", "yes"])
    cad = c3.selectbox("Coronary artery disease (cad)", ["no", "yes"])
    c1, c2, c3 = st.columns(3)
    appet = c1.selectbox("Appetite", ["good", "poor"])
    pe = c2.selectbox("Pedal edema (pe)", ["no", "yes"])
    ane = c3.selectbox("Anemia (ane)", ["no", "yes"])

    submitted = st.form_submit_button("Predict", type="primary")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    raw = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane,
    }

    # 1) one-row table in the training column order
    X = pd.DataFrame([raw])[FEATURE_ORDER]

    # 2) encode the categorical columns
    for col, mapping in ENCODING.items():
        X[col] = X[col].map(mapping)

    # 3) scale the numeric columns with the saved scaler
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS].astype(float))

    # 4) predict
    label = model.predict(X)[0]
    ckd_prob = model.predict_proba(X)[0][list(model.classes_).index("ckd")]

    st.divider()
    if label == "ckd":
        st.error("CKD detected")
    else:
        st.success("No CKD detected")
    st.metric("Probability of CKD", f"{ckd_prob:.1%}")
    st.progress(float(ckd_prob))

    st.caption(
        "This tool supports a clinician's judgement and is not a diagnosis. "
        "It was trained on 400 records from a single hospital."
    )
