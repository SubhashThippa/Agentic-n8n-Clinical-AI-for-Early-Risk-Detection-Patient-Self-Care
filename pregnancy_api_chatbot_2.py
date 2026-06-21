from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
# import os
# from datetime import datetime
# from typing import Optional, Dict, Any

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse

from jinja2 import Environment, FileSystemLoader
from jinja2 import Environment, select_autoescape

# from weasyprint import HTML
import pdfkit



# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Pregnancy Agentic AI (Preeclampsia + Macrosomia)")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORTS_DIR = os.path.join(BASE_DIR, "generated_reports")
WKHTMLTOPDF_PATH = r"C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"
PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)


os.makedirs(REPORTS_DIR, exist_ok=True)

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"])
)


# ============================================================
# STATIC UI FOLDER
# ============================================================
# Create a folder: ui/
# Put dashboard.html inside it.
if not os.path.exists("ui"):
    os.makedirs("ui", exist_ok=True)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# ============================================================
# ENV CONFIG
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/pregnancy_event")
MODELS_BASE = os.getenv("MODELS_BASE", "models")

# ============================================================
# IN-MEMORY STORES (DEMO PURPOSE)
# ============================================================
PATIENT_STORE: Dict[str, Dict[str, Any]] = {}
CHAT_MEMORY: Dict[str, list] = {}

# ============================================================
# GLOBAL MODEL HOLDERS
# ============================================================
PE_MODEL = None
MAC_MODEL = None
PE_FEATURES = None
MAC_FEATURES = None
MAC_MEAL_ENCODER = None

PE_MODEL_DIR = None
MAC_MODEL_DIR = None


# ============================================================
# UTILITIES
# ============================================================
def tier_from_score(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def tier_class(tier: str) -> str:
    t = str(tier).strip().lower()
    if t not in ["low", "medium", "high"]:
        return "medium"
    return t


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except:
        return default


def extract_meal_category(text: str) -> str:
    if text is None or str(text).strip() == "":
        return "unknown"

    t = str(text).lower()

    if any(k in t for k in ["bagel", "pasta", "white rice", "potatoes", "bread"]):
        return "high_carb"

    if any(k in t for k in ["juice", "soda", "milkshake"]):
        return "sugary"

    if any(k in t for k in ["egg", "eggs", "paneer", "fish", "yogurt", "chicken"]):
        return "protein"

    if any(k in t for k in ["cookies", "chips", "instant noodles"]):
        return "processed_snack"

    if any(k in t for k in ["chapati", "dal", "vegetables", "salad", "idli", "sambar", "brown rice"]):
        return "balanced"

    return "other"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def find_latest_model_dir(base_dir: str, prefix: str) -> str:
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Models base directory not found: {base_dir}")

    candidates = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith(prefix):
            m = re.search(r"(\d+)$", name)
            if m:
                v = int(m.group(1))
                candidates.append((v, full_path))

    if not candidates:
        raise FileNotFoundError(f"No model versions found for prefix '{prefix}' in {base_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_feature_df(payload: dict) -> pd.DataFrame:
    systolic_bp = safe_float(payload.get("systolic_bp"))
    diastolic_bp = safe_float(payload.get("diastolic_bp"))
    severe_headache = int(payload.get("severe_headache", 0))
    vision_changes = int(payload.get("vision_changes", 0))
    upper_abdominal_pain = int(payload.get("upper_abdominal_pain", 0))
    swelling_face_hands = int(payload.get("swelling_face_hands", 0))

    proteinuria_dipstick = safe_float(payload.get("proteinuria_dipstick"))

    fasting_glucose = safe_float(payload.get("fasting_glucose"))
    postprandial_1hr_glucose = safe_float(payload.get("postprandial_1hr_glucose"))

    post_meal_walk_minutes = safe_float(payload.get("post_meal_walk_minutes"))
    sleep_hours = safe_float(payload.get("sleep_hours"))

    weight_kg = safe_float(payload.get("weight_kg"))
    maternal_age = safe_float(payload.get("maternal_age"))
    pre_pregnancy_bmi = safe_float(payload.get("pre_pregnancy_bmi"))
    gdm_diagnosed = int(payload.get("gdm_diagnosed", 0))
    hba1c = safe_float(payload.get("hba1c"))

    gestational_age_weeks = safe_float(payload.get("gestational_age_weeks"))
    food_log_text = payload.get("food_log_text", "")

    # PE engineered
    pulse_pressure = np.nan
    map_val = np.nan
    bp_severe_flag = 0

    if not np.isnan(systolic_bp) and not np.isnan(diastolic_bp):
        pulse_pressure = systolic_bp - diastolic_bp
        map_val = (systolic_bp + 2 * diastolic_bp) / 3
        bp_severe_flag = int((systolic_bp >= 160) or (diastolic_bp >= 110))

    symptom_sum = severe_headache + vision_changes + upper_abdominal_pain + swelling_face_hands

    proteinuria_missing = int(np.isnan(proteinuria_dipstick))
    weight_missing = int(np.isnan(weight_kg))
    hba1c_missing = int(np.isnan(hba1c))

    # Macrosomia engineered
    glucose_spike = np.nan
    fasting_above_target_flag = 0
    postmeal_above_target_flag = 0
    high_spike_flag = 0

    if not np.isnan(fasting_glucose) and not np.isnan(postprandial_1hr_glucose):
        glucose_spike = postprandial_1hr_glucose - fasting_glucose
        fasting_above_target_flag = int(fasting_glucose >= 95)
        postmeal_above_target_flag = int(postprandial_1hr_glucose >= 140)
        high_spike_flag = int(postprandial_1hr_glucose >= 180)

    meal_category = extract_meal_category(food_log_text)

    row = {
        "gestational_age_weeks": gestational_age_weeks,
        "maternal_age": maternal_age,

        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "severe_headache": severe_headache,
        "vision_changes": vision_changes,
        "upper_abdominal_pain": upper_abdominal_pain,
        "swelling_face_hands": swelling_face_hands,
        "proteinuria_dipstick": proteinuria_dipstick,

        "fasting_glucose": fasting_glucose,
        "postprandial_1hr_glucose": postprandial_1hr_glucose,
        "post_meal_walk_minutes": post_meal_walk_minutes,
        "sleep_hours": sleep_hours,

        "weight_kg": weight_kg,
        "pre_pregnancy_bmi": pre_pregnancy_bmi,
        "gdm_diagnosed": gdm_diagnosed,
        "hba1c": hba1c,

        "food_log_text": food_log_text,
        "meal_category": meal_category,

        "pulse_pressure": pulse_pressure,
        "map": map_val,
        "symptom_sum": symptom_sum,
        "bp_severe_flag": bp_severe_flag,
        "proteinuria_missing": proteinuria_missing,
        "weight_missing": weight_missing,
        "hba1c_missing": hba1c_missing,

        "glucose_spike": glucose_spike,
        "fasting_above_target_flag": fasting_above_target_flag,
        "postmeal_above_target_flag": postmeal_above_target_flag,
        "high_spike_flag": high_spike_flag,
    }

    return pd.DataFrame([row])


def encode_meal(df_row: pd.DataFrame, encoder) -> pd.DataFrame:
    df_ = df_row.copy()
    meal_cat = str(df_.loc[0, "meal_category"]).strip().lower()

    if encoder is None:
        df_["meal_category_encoded"] = 0
        return df_

    # ✅ if meal category not in encoder classes, fallback to first known class
    if hasattr(encoder, "classes_"):
        classes = list(encoder.classes_)
        if meal_cat not in classes:
            meal_cat = classes[0]   # safest fallback

    df_["meal_category_encoded"] = int(encoder.transform([meal_cat])[0])
    return df_



# # ============================================================
# # LOAD MODELS
# # ============================================================
# def load_latest_models():
#     global PE_MODEL, MAC_MODEL, PE_FEATURES, MAC_FEATURES, MAC_MEAL_ENCODER
#     global PE_MODEL_DIR, MAC_MODEL_DIR

#     # PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
#     # MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

#     best_file = os.path.join(MODELS_BASE, "best_model.json")

#     if os.path.exists(best_file):
#         with open(best_file) as f:
#             best = json.load(f)

#         PE_MODEL_DIR = os.path.join(MODELS_BASE, best.get("preeclampsia"))
#         MAC_MODEL_DIR = os.path.join(MODELS_BASE, best.get("macrosomia"))

#     else:
#         # fallback if first run
#         PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
#         MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

#     pe_model_path = os.path.join(PE_MODEL_DIR, "preeclampsia_lgbm_model.pkl")
#     mac_model_path = os.path.join(MAC_MODEL_DIR, "macrosomia_lgbm_model.pkl")

#     pe_features_path = os.path.join(PE_MODEL_DIR, "feature_columns.json")
#     mac_features_path = os.path.join(MAC_MODEL_DIR, "feature_columns.json")

#     mac_encoder_path = os.path.join(MAC_MODEL_DIR, "meal_category_encoder.pkl")

#     # PE_MODEL = joblib.load(pe_model_path)
#     # MAC_MODEL = joblib.load(mac_model_path)

#     try:
#         PE_MODEL = joblib.load(pe_model_path)
#         MAC_MODEL = joblib.load(mac_model_path)

#     except Exception as e:
#         print("❌ Failed to load best model:", str(e))

#         # 🔁 ROLLBACK to previous working version
#         PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
#         MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

#         pe_model_path = os.path.join(PE_MODEL_DIR, "preeclampsia_lgbm_model.pkl")
#         mac_model_path = os.path.join(MAC_MODEL_DIR, "macrosomia_lgbm_model.pkl")

#         PE_MODEL = joblib.load(pe_model_path)
#         MAC_MODEL = joblib.load(mac_model_path)

#         print("⚠️ Rolled back to latest available model")

#     PE_FEATURES = load_json(pe_features_path)
#     MAC_FEATURES = load_json(mac_features_path)

#     MAC_MEAL_ENCODER = joblib.load(mac_encoder_path) if os.path.exists(mac_encoder_path) else None

#     print("✅ Latest models loaded")
#     print("PE:", PE_MODEL_DIR)
#     print("MAC:", MAC_MODEL_DIR)


# @app.on_event("startup")
# def startup_event():
#     load_latest_models()


# @app.post("/reload_models")
# def reload_models():
#     load_latest_models()
#     return {
#         "status": "reloaded",
#         "preeclampsia_model_dir": PE_MODEL_DIR,
#         "macrosomia_model_dir": MAC_MODEL_DIR
#     }


# ============================================================
# LOAD MODELS (WITH SAFE ROLLBACK)
# ============================================================
def load_latest_models():
    global PE_MODEL, MAC_MODEL, PE_FEATURES, MAC_FEATURES, MAC_MEAL_ENCODER
    global PE_MODEL_DIR, MAC_MODEL_DIR

    best_file = os.path.join(MODELS_BASE, "best_model.json")

    # ============================================================
    # STEP 1 — Decide which model to load
    # ============================================================
    if os.path.exists(best_file):
        try:
            with open(best_file) as f:
                best = json.load(f)

            pe_best = best.get("preeclampsia")
            mac_best = best.get("macrosomia")

            if pe_best:
                PE_MODEL_DIR = os.path.join(MODELS_BASE, pe_best)
            else:
                PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")

            if mac_best:
                MAC_MODEL_DIR = os.path.join(MODELS_BASE, mac_best)
            else:
                MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

        except Exception as e:
            print("⚠️ Error reading best_model.json:", str(e))
            PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
            MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

    else:
        # First run fallback
        PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
        MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

    # ============================================================
    # STEP 2 — Define paths
    # ============================================================
    pe_model_path = os.path.join(PE_MODEL_DIR, "preeclampsia_lgbm_model.pkl")
    mac_model_path = os.path.join(MAC_MODEL_DIR, "macrosomia_lgbm_model.pkl")

    pe_features_path = os.path.join(PE_MODEL_DIR, "feature_columns.json")
    mac_features_path = os.path.join(MAC_MODEL_DIR, "feature_columns.json")

    mac_encoder_path = os.path.join(MAC_MODEL_DIR, "meal_category_encoder.pkl")

    # ============================================================
    # STEP 3 — Try loading best model
    # ============================================================
    try:
        PE_MODEL = joblib.load(pe_model_path)
        MAC_MODEL = joblib.load(mac_model_path)

        PE_FEATURES = load_json(pe_features_path)
        MAC_FEATURES = load_json(mac_features_path)

        MAC_MEAL_ENCODER = joblib.load(mac_encoder_path) if os.path.exists(mac_encoder_path) else None

        print("✅ Loaded BEST model")
        print("PE:", PE_MODEL_DIR)
        print("MAC:", MAC_MODEL_DIR)

    # ============================================================
    # STEP 4 — ROLLBACK if anything fails
    # ============================================================
    except Exception as e:
        print("❌ Failed to load BEST model:", str(e))
        print("🔁 Rolling back to latest available version...")

        try:
            PE_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "preeclampsia_v")
            MAC_MODEL_DIR = find_latest_model_dir(MODELS_BASE, "macrosomia_v")

            pe_model_path = os.path.join(PE_MODEL_DIR, "preeclampsia_lgbm_model.pkl")
            mac_model_path = os.path.join(MAC_MODEL_DIR, "macrosomia_lgbm_model.pkl")

            pe_features_path = os.path.join(PE_MODEL_DIR, "feature_columns.json")
            mac_features_path = os.path.join(MAC_MODEL_DIR, "feature_columns.json")

            mac_encoder_path = os.path.join(MAC_MODEL_DIR, "meal_category_encoder.pkl")

            PE_MODEL = joblib.load(pe_model_path)
            MAC_MODEL = joblib.load(mac_model_path)

            PE_FEATURES = load_json(pe_features_path)
            MAC_FEATURES = load_json(mac_features_path)

            MAC_MEAL_ENCODER = joblib.load(mac_encoder_path) if os.path.exists(mac_encoder_path) else None

            print("⚠️ ROLLBACK SUCCESSFUL")
            print("PE:", PE_MODEL_DIR)
            print("MAC:", MAC_MODEL_DIR)

        except Exception as rollback_error:
            print("🚨 CRITICAL: Rollback also failed:", str(rollback_error))
            raise RuntimeError("No working model available. Check model files.")

# ============================================================
# STARTUP + RELOAD
# ============================================================
@app.on_event("startup")
def startup_event():
    load_latest_models()


@app.post("/reload_models")
def reload_models():
    load_latest_models()
    return {
        "status": "reloaded",
        "preeclampsia_model_dir": PE_MODEL_DIR,
        "macrosomia_model_dir": MAC_MODEL_DIR
    }


# ============================================================
# SCHEMAS
# ============================================================
class PatientData(BaseModel):
    patient_id: str = Field(...)

    timestamp: Optional[str] = None
    gestational_age_weeks: Optional[float] = None
    maternal_age: Optional[float] = None
    pre_pregnancy_bmi: Optional[float] = None
    gdm_diagnosed: Optional[int] = 0

    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    severe_headache: Optional[int] = 0
    vision_changes: Optional[int] = 0
    upper_abdominal_pain: Optional[int] = 0
    swelling_face_hands: Optional[int] = 0
    proteinuria_dipstick: Optional[float] = None

    fasting_glucose: Optional[float] = None
    postprandial_1hr_glucose: Optional[float] = None
    post_meal_walk_minutes: Optional[float] = None
    sleep_hours: Optional[float] = None
    weight_kg: Optional[float] = None
    hba1c: Optional[float] = None
    food_log_text: Optional[str] = ""


class ChatRequest(BaseModel):
    patient_id: str
    message: str


# ============================================================
# PREDICTION API
# ============================================================
@app.post("/predict_risk")
def predict_risk(data: PatientData):
    payload = data.dict()
    pid = payload["patient_id"]

    # ✅ Ensure timestamp exists
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().isoformat()

    df_row = build_feature_df(payload)

    # ✅ Meal encoding
    if MAC_MEAL_ENCODER is not None:
        df_row = encode_meal(df_row, MAC_MEAL_ENCODER)
    else:
        df_row["meal_category_encoded"] = 0

    # ✅ Prepare PE input
    pe_input = df_row.copy()
    for col in PE_FEATURES:
        if col not in pe_input.columns:
            pe_input[col] = np.nan
    pe_input = pe_input[PE_FEATURES]

    # ✅ Prepare Macrosomia input
    mac_input = df_row.copy()
    for col in MAC_FEATURES:
        if col not in mac_input.columns:
            mac_input[col] = np.nan
    mac_input = mac_input[MAC_FEATURES]

    # ✅ Predict scores
    # ✅ Use probability instead of class
    # ✅ For REGRESSOR model
    pe_score = float(np.clip(PE_MODEL.predict(pe_input)[0], 0, 1))
    mac_score = float(np.clip(MAC_MODEL.predict(mac_input)[0], 0, 1))

    # ✅ Build pred_json first
    pred_json = {
        "patient_id": pid,
        "timestamp": payload["timestamp"],
        "preeclampsia_risk_score": pe_score,
        "preeclampsia_risk_tier": tier_from_score(pe_score),
        "macrosomia_risk_score": mac_score,
        "macrosomia_risk_tier": tier_from_score(mac_score),
        "model_loaded": {
            "preeclampsia_dir": PE_MODEL_DIR,
            "macrosomia_dir": MAC_MODEL_DIR
        }
    }

    # ✅ Store in memory (AFTER pred_json exists)
    PATIENT_STORE[pid] = {
        "patient_data": payload,
        "last_prediction": pred_json
    }

    return JSONResponse(pred_json)

@app.post("/set_patient_data")
def set_patient_data(data: PatientData):
    payload = data.dict()
    pid = payload["patient_id"]

    # ✅ Ensure timestamp exists
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().isoformat()

    PATIENT_STORE[pid] = {
        "patient_data": payload,
        "last_prediction": None
    }

    return {"status": "stored", "patient_id": pid}


@app.get("/chat", response_class=HTMLResponse)
def chat_redirect():
    return FileResponse(os.path.join("ui", "dashboard.html"))


@app.get("/patients")
def patients():
    return {"patients": sorted(list(PATIENT_STORE.keys()))}


@app.get("/patient/{patient_id}")
def patient_details(patient_id: str):
    if patient_id not in PATIENT_STORE:
        return {"status": "fail", "message": "Patient not found in memory."}

    stored = PATIENT_STORE[patient_id]
    payload = stored.get("patient_data", {})
    pred_json = stored.get("last_prediction", None)

    # If prediction not available yet, compute once
    if pred_json is None and payload:
        pred = predict_risk(PatientData(**payload))
        pred_json = json.loads(pred.body.decode("utf-8"))

    return {
        "status": "ok",
        "patient_data": payload,
        "risk": pred_json
    }


@app.post("/send_to_n8n")
def send_to_n8n(event: dict):
    try:
        r = requests.post(N8N_WEBHOOK_URL, json=event, timeout=10)
        return {"status": "sent", "n8n_status": r.status_code, "response": r.text}
    except Exception as e:
        return {"status": "failed", "error": str(e), "webhook": N8N_WEBHOOK_URL}


@app.post("/generate_patient_pdf")
def generate_patient_pdf(payload: Dict[str, Any] = Body(...)):
    try:
        patient_id = payload.get("patient_id", "UNKNOWN")

        rs = payload.get("risk_summary_simple", {})
        pe_tier = rs.get("preeclampsia_tier", "unknown")
        mac_tier = rs.get("macrosomia_tier", "unknown")

        template = jinja_env.get_template("patient_report.html")

        html_str = template.render(
            patient_id=patient_id,
            report_title=payload.get("report_title", "Personalized Pregnancy Self-Care Plan"),

            pe_tier=pe_tier,
            pe_score=rs.get("preeclampsia_score", "NA"),
            mac_tier=mac_tier,
            mac_score=rs.get("macrosomia_score", "NA"),
            one_line_summary=rs.get("one_line_summary", ""),

            pe_tier_class=tier_class(pe_tier),
            mac_tier_class=tier_class(mac_tier),

            today_focus=payload.get("today_focus", []),

            diet_goal=payload.get("diet_plan", {}).get("goal", ""),
            breakfast_options=payload.get("diet_plan", {}).get("breakfast_options", []),
            lunch_options=payload.get("diet_plan", {}).get("lunch_options", []),
            dinner_options=payload.get("diet_plan", {}).get("dinner_options", []),
            snack_options=payload.get("diet_plan", {}).get("snack_options", []),
            avoid_or_limit=payload.get("diet_plan", {}).get("avoid_or_limit", []),
            hydration_tip=payload.get("diet_plan", {}).get("hydration_tip", ""),

            activity_minutes=payload.get("activity_plan", {}).get("recommended_walk_minutes", ""),
            post_meal_tip=payload.get("activity_plan", {}).get("post_meal_walk_tip", ""),
            safe_activity_note=payload.get("activity_plan", {}).get("safe_activity_note", ""),

            sleep_target=payload.get("sleep_plan", {}).get("sleep_target_hours", "7-9"),
            sleep_tip=payload.get("sleep_plan", {}).get("sleep_tip", ""),

            bp_tip=payload.get("blood_pressure_safety", {}).get("monitoring_tip", ""),
            bp_red_flags=payload.get("blood_pressure_safety", {}).get("red_flags", []),
            bp_action=payload.get("blood_pressure_safety", {}).get("what_to_do_if_red_flags", ""),

            glucose_tip=payload.get("glucose_safety", {}).get("monitoring_tip", ""),
            food_log_comment=payload.get("glucose_safety", {}).get("food_log_comment", ""),
            swap_suggestions=payload.get("glucose_safety", {}).get("simple_swap_suggestions", []),

            daily_checklist=payload.get("daily_checklist", []),
            next_steps=payload.get("next_steps", []),

            chatbot_link_label=payload.get("chatbot_link_label", "Need more help? Use your AI Assistant link:"),
            chatbot_link=payload.get("chatbot_link", "")
        )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_name = f"patient_report_{patient_id}_{ts}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_name)

        # ✅ Generate PDF (Windows friendly)
        # pdfkit.from_string(html_str, pdf_path)
        pdfkit.from_string(html_str, pdf_path, configuration=PDFKIT_CONFIG)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_name
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================
# RE-TRAIN API
# ============================================================

@app.post("/retrain_models")
def retrain_models(payload: dict):
    try:
        import pandas as pd
        import numpy as np
        import os
        import joblib
        import json
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        # ==========================
        # STEP 1 — LOAD NEW DATA
        # ==========================

        data = payload.get("data", [])

        if not data or len(data) < 5:
            return {"status": "failed", "reason": "Not enough new data"}

        new_df = pd.DataFrame(data)
        new_df = new_df.dropna(subset=["pe_label", "macro_label"])

        if new_df.empty:
            return {"status": "failed", "reason": "No valid labeled rows"}

        # ==========================
        # STEP 2 — LOAD OLD DATA
        # ==========================

        old_df = pd.read_csv("pregnancy_dataset_v2.csv")

        old_df = old_df.rename(columns={
            "preeclampsia_risk_tier": "pe_label",
            "macrosomia_risk_tier": "macro_label"
        })

        # ==========================
        # STEP 3 — COMBINE DATA
        # ==========================

        # Give more weight to new data
        df = pd.concat([old_df, new_df, new_df], ignore_index=True)

        # ==========================
        # STEP 4 — FEATURE ENGINEERING
        # ==========================

        def extract_meal_category(text):
            if pd.isna(text):
                return "unknown"
            t = str(text).lower()

            if any(k in t for k in ["rice", "bread", "pasta"]):
                return "high_carb"
            if any(k in t for k in ["juice", "soda"]):
                return "sugary"
            if any(k in t for k in ["egg", "paneer", "chicken"]):
                return "protein"
            return "balanced"

        df["meal_category"] = df.get("food_log_text", "").apply(extract_meal_category)

        # ==========================
        # STEP 5 — FEATURES
        # ==========================

        features = [
            "gestational_age_weeks",
            "maternal_age",
            "pre_pregnancy_bmi",
            "gdm_diagnosed",
            "systolic_bp",
            "diastolic_bp",
            "severe_headache",
            "vision_changes",
            "upper_abdominal_pain",
            "swelling_face_hands",
            "proteinuria_dipstick",
            "fasting_glucose",
            "postprandial_1hr_glucose",
            "post_meal_walk_minutes",
            "sleep_hours",
            "weight_kg",
            "hba1c"
        ]

        df = df.dropna(subset=features + ["pe_label", "macro_label"])

        X = df[features]
        y_pe = df["pe_label"]
        y_macro = df["macro_label"]

        # ==========================
        # STEP 6 — TRAIN TEST SPLIT
        # ==========================

        X_train, X_test, y_pe_train, y_pe_test = train_test_split(
            X, y_pe, test_size=0.2, random_state=42
        )

        _, _, y_macro_train, y_macro_test = train_test_split(
            X, y_macro, test_size=0.2, random_state=42
        )

        # ==========================
        # STEP 7 — TRAIN MODELS
        # ==========================

        pe_model = LGBMClassifier()
        mac_model = LGBMClassifier()

        pe_model.fit(X_train, y_pe_train)
        mac_model.fit(X_train, y_macro_train)

        # ==========================
        # STEP 8 — EVALUATE NEW MODEL
        # ==========================

        pe_pred = pe_model.predict(X_test)
        macro_pred = mac_model.predict(X_test)

        pe_acc = accuracy_score(y_pe_test, pe_pred)
        macro_acc = accuracy_score(y_macro_test, macro_pred)

        pe_report = classification_report(y_pe_test, pe_pred, output_dict=True)
        macro_report = classification_report(y_macro_test, macro_pred, output_dict=True)

        # ==========================
        # STEP 9 — LOAD OLD MODEL
        # ==========================

        def find_latest_model(base_dir, prefix):
            dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
            if not dirs:
                return None
            dirs.sort(key=lambda x: int(x.split("_v")[-1]))
            return os.path.join(base_dir, dirs[-1])

        old_pe_dir = find_latest_model("models", "preeclampsia_v")
        old_mac_dir = find_latest_model("models", "macrosomia_v")

        old_pe_acc = None
        old_mac_acc = None

        if old_pe_dir and old_mac_dir:
            try:
                old_pe_model = joblib.load(os.path.join(old_pe_dir, "preeclampsia_lgbm_model.pkl"))
                old_mac_model = joblib.load(os.path.join(old_mac_dir, "macrosomia_lgbm_model.pkl"))

                old_pe_pred = old_pe_model.predict(X_test)
                old_mac_pred = old_mac_model.predict(X_test)

                old_pe_acc = accuracy_score(y_pe_test, old_pe_pred)
                old_mac_acc = accuracy_score(y_macro_test, old_mac_pred)
            except:
                pass

        # ==========================
        # STEP 10 — ACCEPTANCE LOGIC
        # ==========================

        tolerance = 0.02

        accept_pe = True
        accept_mac = True

        if old_pe_acc is not None:
            accept_pe = pe_acc >= (old_pe_acc - tolerance)

        if old_mac_acc is not None:
            accept_mac = macro_acc >= (old_mac_acc - tolerance)

        # ==========================
        # STEP 11 — FORCE ACCEPT (ANTI-STAGNATION)
        # ==========================

        counter_file = "models/rejection_counter.json"

        if os.path.exists(counter_file):
            with open(counter_file) as f:
                counter = json.load(f)
        else:
            counter = {"count": 0}

        if not accept_pe or not accept_mac:
            counter["count"] += 1
        else:
            counter["count"] = 0

        if counter["count"] >= 3:
            accept_pe = True
            accept_mac = True
            counter["count"] = 0

        with open(counter_file, "w") as f:
            json.dump(counter, f)

        # ==========================
        # STEP 12 — VERSIONING
        # ==========================

        def get_next_version(base_dir, prefix):
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
            if not existing:
                return f"{prefix}1"
            nums = [int(d.split("_v")[-1]) for d in existing]
            return f"{prefix}{max(nums) + 1}"

        pe_version = get_next_version("models", "preeclampsia_v")
        mac_version = get_next_version("models", "macrosomia_v")

        pe_dir = os.path.join("models", pe_version)
        mac_dir = os.path.join("models", mac_version)

        os.makedirs(pe_dir, exist_ok=True)
        os.makedirs(mac_dir, exist_ok=True)

        saved_models = {}

        # ==========================
        # STEP 13 — SAVE MODELS (CONDITIONAL)
        # ==========================

        if accept_pe:
            joblib.dump(pe_model, os.path.join(pe_dir, "preeclampsia_lgbm_model.pkl"))
            with open(os.path.join(pe_dir, "feature_columns.json"), "w") as f:
                json.dump(features, f)
            saved_models["preeclampsia"] = pe_version
        else:
            saved_models["preeclampsia"] = "rejected"

        if accept_mac:
            joblib.dump(mac_model, os.path.join(mac_dir, "macrosomia_lgbm_model.pkl"))
            with open(os.path.join(mac_dir, "feature_columns.json"), "w") as f:
                json.dump(features, f)
            saved_models["macrosomia"] = mac_version
        else:
            saved_models["macrosomia"] = "rejected"


        # ==========================
        # STEP 13.5 — UPDATE BEST MODEL
        # ==========================

        best_file = "models/best_model.json"

        best_data = {}

        if os.path.exists(best_file):
            with open(best_file) as f:
                best_data = json.load(f)

        # Save only if accepted
        if accept_pe:
            best_data["preeclampsia"] = pe_version

        if accept_mac:
            best_data["macrosomia"] = mac_version

        with open(best_file, "w") as f:
            json.dump(best_data, f, indent=2)


        # ==========================
        # STEP 14 — SAVE METRICS
        # ==========================

        comparison = {
            "new_pe_accuracy": pe_acc,
            "old_pe_accuracy": old_pe_acc,
            "new_macro_accuracy": macro_acc,
            "old_macro_accuracy": old_mac_acc,
            "accepted_pe": accept_pe,
            "accepted_macro": accept_mac,
            "rows_used": len(df)
        }

        with open("models/model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # ==========================
        # DONE
        # ==========================

        return {
            "status": "completed",
            "saved_models": saved_models,
            "comparison": comparison
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================
# GEMINI CHAT
# ============================================================
def gemini_chat(system_prompt: str, user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "❌ Gemini API key not configured."

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"}]
            }
        ]
    }

    try:
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini API error: {str(e)}"


@app.post("/chat_api")
def chat_api(req: ChatRequest):
    pid = req.patient_id
    msg = req.message.strip()

    stored = PATIENT_STORE.get(pid, None)
    risk_context = ""

    # ✅ If patient data exists in memory, use it
    if stored and isinstance(stored, dict) and "patient_data" in stored:
        payload = stored.get("patient_data", {})

        # ✅ Use cached prediction if available, otherwise compute once
        pred_json = stored.get("last_prediction", None)

        if pred_json is None and payload:
            pred = predict_risk(PatientData(**payload))
            pred_json = json.loads(pred.body.decode("utf-8"))

            # ✅ store it back (so next chat doesn't recompute)
            PATIENT_STORE[pid]["last_prediction"] = pred_json

        if pred_json:

            vitals_context = json.dumps(payload, indent=2)

            risk_context = f"""
                    PATIENT CLINICAL DATA
                    ---------------------
                    {vitals_context}

                    RISK SCORES
                    -----------
                    Preeclampsia score={pred_json['preeclampsia_risk_score']:.3f}, tier={pred_json['preeclampsia_risk_tier']}
                    Macrosomia score={pred_json['macrosomia_risk_score']:.3f}, tier={pred_json['macrosomia_risk_tier']}
                    """

    system_prompt = f"""
You are an AI Pregnancy Assistant.
- If preeclampsia tier is HIGH: act urgent and recommend immediate medical evaluation.
- If macrosomia tier is HIGH: coach diet, glucose control, and adherence.
- Do not hallucinate diagnosis, only risk-based suggestions.

{risk_context}
"""

    reply = gemini_chat(system_prompt, msg)
    return {"patient_id": pid, "reply": reply}

@app.post("/generate_doctor_pdf")
def generate_doctor_pdf(payload: Dict[str, Any] = Body(...)):
    try:
        patient_id = payload.get("patient_id", "UNKNOWN")

        # ✅ Always use agent_output if present
        agent = payload.get("agent_output", payload)

        # ✅ risk values
        pe_score = agent.get("preeclampsia_risk_score", "NA")
        pe_tier  = agent.get("preeclampsia_risk_tier", "unknown")
        mac_score = agent.get("macrosomia_risk_score", "NA")
        mac_tier  = agent.get("macrosomia_risk_tier", "unknown")

        template = jinja_env.get_template("doctor_report.html")

        html_str = template.render(
            patient_id=patient_id,
            generated_time=datetime.utcnow().strftime("%d %b %Y, %I:%M %p UTC"),

            # ✅ context (use agent)
            gestational_age_weeks=agent.get("gestational_age_weeks", "NA"),
            maternal_age=agent.get("maternal_age", "NA"),
            pre_pregnancy_bmi=agent.get("pre_pregnancy_bmi", "NA"),
            gdm_diagnosed=agent.get("gdm_diagnosed", "NA"),

            # ✅ vitals (use agent)
            systolic_bp=agent.get("systolic_bp", "NA"),
            diastolic_bp=agent.get("diastolic_bp", "NA"),
            proteinuria_dipstick=agent.get("proteinuria_dipstick", "NA"),

            severe_headache=agent.get("severe_headache", "NA"),
            vision_changes=agent.get("vision_changes", "NA"),
            upper_abdominal_pain=agent.get("upper_abdominal_pain", "NA"),
            swelling_face_hands=agent.get("swelling_face_hands", "NA"),

            fasting_glucose=agent.get("fasting_glucose", "NA"),
            postprandial_1hr_glucose=agent.get("postprandial_1hr_glucose", "NA"),
            hba1c=agent.get("hba1c", "NA"),
            weight_kg=agent.get("weight_kg", "NA"),

            post_meal_walk_minutes=agent.get("post_meal_walk_minutes", "NA"),
            sleep_hours=agent.get("sleep_hours", "NA"),
            food_log_text=agent.get("food_log_text", ""),

            # ✅ risk display
            pe_tier=pe_tier,
            pe_score=pe_score,
            mac_tier=mac_tier,
            mac_score=mac_score,

            pe_tier_class=tier_class(pe_tier),
            mac_tier_class=tier_class(mac_tier),

            # ✅ summary/actions/messages
            summary=agent.get("summary", ""),
            recommended_action=agent.get("recommended_action", payload.get("action_type", "LOG_ONLY")),
            doctor_message=agent.get("doctor_message", payload.get("doctor_message", ""))
        )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_name = f"doctor_report_{patient_id}_{ts}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_name)

        # ✅ Generate PDF
        pdfkit.from_string(html_str, pdf_path, configuration=PDFKIT_CONFIG)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_name
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# ============================================================
# SERVE NEW UI FILE
# ============================================================
@app.get("/", response_class=HTMLResponse)
def root():
    # serve ui/dashboard.html
    dashboard_path = os.path.join("ui", "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h2>UI not found. Please create ui/dashboard.html</h2>")
