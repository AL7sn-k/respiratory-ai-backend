from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.database.db import engine
from app.database.models import Base
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import Depends
from app.database.db import get_db
from app.database.models import Patient, Diagnosis, Doctor, PatientAlert, Appointment
import json
import re
from app.services.symptom_scoring import score_symptoms
from app.services.fusion import fuse_predictions
import shutil
from pathlib import Path
from fastapi import UploadFile, File, Form
# from app.services.image_prediction import predict_image
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from passlib.context import CryptContext
from fastapi.responses import Response
from collections import Counter
from html import escape

app = FastAPI(
    title="Respiratory AI Diagnosis API",
    description="Backend API for doctor website, AI diagnosis, symptom analysis, fusion, reports, and virtual assistant.",
    version="1.0.0",
)

Base.metadata.create_all(bind=engine)


def ensure_mobile_columns():
    statements = [
        "ALTER TABLE patient_alerts ADD COLUMN IF NOT EXISTS patient_seen BOOLEAN DEFAULT FALSE",
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)",
    ]

    try:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
    except Exception:
        pass


ensure_mobile_columns()

BASE_DIR = Path(__file__).resolve().parents[1]

HEATMAP_DIR = BASE_DIR / "reports" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomScoreRequest(BaseModel):
    symptoms: List[str]


class FusionRequest(BaseModel):
    image_scores: dict
    symptom_scores: dict


class AssistantExplanationRequest(BaseModel):
    final_prediction: str
    final_confidence: float
    risk_level: str
    selected_symptoms: List[str]
    scan_type: str | None = None
    image_prediction: str | None = None
    image_confidence: float | None = None
    symptom_prediction: str | None = None
    symptom_confidence: float | None = None


class AssistantChatRequest(BaseModel):
    message: str
    doctor_id: int | None = None
    pending_request: str | None = None


class PatientCreateRequest(BaseModel):
    doctor_id: int | None = None
    full_name: str
    age: int | None = None
    gender: str | None = None
    phone: str | None = None
    national_id: str | None = None
    medical_notes: str | None = None
    email: str | None = None
    address: str | None = None
    date_of_birth: str | None = None


class PatientUpdateRequest(BaseModel):
    full_name: str
    age: int | None = None
    gender: str | None = None
    phone: str | None = None
    national_id: str | None = None
    medical_notes: str | None = None
    email: str | None = None
    address: str | None = None
    date_of_birth: str | None = None


class DiagnosisCreateRequest(BaseModel):
    patient_id: int | None = None
    scan_type: str | None = None
    image_path: str | None = None
    heatmap_path: str | None = None
    doctor_id: int | None = None
    doctor_name: str | None = None

    image_prediction: str | None = None
    image_confidence: float | None = None

    symptom_prediction: str | None = None
    symptom_confidence: float | None = None

    final_prediction: str
    final_confidence: float
    risk_level: str

    image_scores: dict
    symptom_scores: dict
    final_scores: dict
    selected_symptoms: list

    doctor_notes: str | None = None
    assistant_explanation: str | None = None


class DoctorRegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    specialization: str | None = None
    hospital_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None


class DoctorLoginRequest(BaseModel):
    email: str
    password: str


class DoctorUpdateRequest(BaseModel):
    full_name: str
    email: str
    specialization: str | None = None
    hospital_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None


class DoctorPasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class PatientAlertCreateRequest(BaseModel):
    patient_id: int
    doctor_id: int
    severity: str = "Medium"
    message: str | None = None
    symptoms: list[str] = []


class PatientAlertReplyRequest(BaseModel):
    doctor_reply: str


class PatientAlertStatusRequest(BaseModel):
    status: str


class AppointmentCreateRequest(BaseModel):
    patient_id: int
    doctor_id: int
    requested_date: str
    requested_time: str
    reason: str | None = None
    symptoms: list[str] = []


class AppointmentStatusUpdateRequest(BaseModel):
    status: str
    doctor_response: str | None = None


class PatientLoginRequest(BaseModel):
    email: str
    patient_id: int | None = None
    password: str | None = None
    login_value: str | None = None


class PatientChangePasswordRequest(BaseModel):
    patient_id: int
    new_password: str
    confirm_password: str


class PatientAppointmentRequest(BaseModel):
    preferred_date: str
    preferred_time: str | None = None
    appointment_type: str | None = None
    reason: str


class PatientAssistantMobileRequest(BaseModel):
    message: str
    fever: bool = False
    cough: bool = False
    chest_pain: bool = False
    shortness_of_breath: bool = False
    fatigue: bool = False
    night_sweats: bool = False
    weight_loss: bool = False
    blood_in_sputum: bool = False
    temperature: float | None = None
    oxygen_level: float | None = None


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


@app.get("/")
def root():
    return {
        "message": "Respiratory AI Diagnosis API is running",
        "modules": [
            "X-ray model",
            "CT model",
            "Symptom analysis",
            "Fusion prediction",
            "Virtual assistant",
            "Reports",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/symptom-score")
def symptom_score(request: SymptomScoreRequest):
    scores = score_symptoms(request.symptoms)
    top_prediction = max(scores, key=scores.get)
    top_confidence = scores[top_prediction]
    return {
        "selected_symptoms": request.symptoms,
        "symptom_scores": scores,
        "top_symptom_prediction": top_prediction,
        "top_symptom_confidence": top_confidence,
    }


@app.post("/fusion")
def fusion_prediction(request: FusionRequest):
    result = fuse_predictions(
        image_scores=request.image_scores, symptom_scores=request.symptom_scores
    )
    return result


RED_FLAG_SYMPTOMS = {
    "blood in sputum",
    "cough lasting 3 weeks or more",
    "cough that gets worse",
    "chronic cough",
    "shortness of breath",
    "difficulty breathing",
    "rapid breathing",
    "chest pain",
    "chest pain when breathing or coughing",
    "night sweats",
    "weight loss",
    "hoarseness",
    "recurrent chest infections",
}


DISEASE_CLINICAL_PROFILES = {
    "Pneumonia": {
        "symptoms": {
            "cough",
            "productive cough / sputum",
            "fever",
            "chills",
            "fatigue",
            "weakness",
            "shortness of breath",
            "difficulty breathing",
            "rapid breathing",
            "chest pain",
            "chest pain when breathing or coughing",
            "wheezing",
        },
        "rationale": (
            "Pneumonia is clinically supported when respiratory infection features are present, "
            "especially cough, fever or chills, fatigue, shortness of breath, rapid or difficult "
            "breathing, and chest pain during breathing or coughing."
        ),
        "verification": (
            "The doctor should review the chest image, assess breathing status and oxygen saturation, "
            "compare the result with physical examination findings, and request infection-related "
            "testing or additional imaging when clinically indicated."
        ),
    },
    "Tuberculosis": {
        "symptoms": {
            "persistent cough",
            "cough lasting 3 weeks or more",
            "productive cough / sputum",
            "blood in sputum",
            "chest pain",
            "fever",
            "chills",
            "night sweats",
            "weight loss",
            "loss of appetite",
            "fatigue",
            "weakness",
            "shortness of breath",
        },
        "rationale": (
            "Tuberculosis is clinically supported when chronic respiratory symptoms and systemic "
            "symptoms appear together, especially cough lasting three weeks or more, chest pain, "
            "coughing blood or sputum, fever, night sweats, weight loss, fatigue, weakness, or "
            "loss of appetite."
        ),
        "verification": (
            "The doctor should review the scan, evaluate exposure history and physical examination "
            "findings, and confirm using appropriate TB evaluation such as TB blood/skin testing, "
            "chest imaging review, sputum smear/culture, GeneXpert, or laboratory testing when "
            "clinically indicated."
        ),
    },
    "Lung Cancer": {
        "symptoms": {
            "chronic cough",
            "persistent cough",
            "cough that gets worse",
            "blood in sputum",
            "chest pain",
            "shortness of breath",
            "wheezing",
            "hoarseness",
            "weight loss",
            "loss of appetite",
            "fatigue",
            "weakness",
            "recurrent chest infections",
        },
        "rationale": (
            "Lung cancer is clinically concerning when warning symptoms are present, such as chronic "
            "or worsening cough, coughing blood, chest pain, shortness of breath, hoarseness, "
            "unexplained weight loss, fatigue, wheezing, or recurrent chest infections."
        ),
        "verification": (
            "The doctor should review the imaging carefully, compare the result with patient history "
            "and risk factors, and consider further evaluation such as CT imaging, specialist referral, "
            "sputum cytology, or biopsy when clinically indicated."
        ),
    },
    "Normal": {
        "symptoms": {
            "mild cough",
            "runny nose",
            "nasal congestion",
            "sneezing",
            "sore throat",
        },
        "rationale": (
            "A normal or mild respiratory result is more consistent with mild upper-respiratory "
            "symptoms such as mild cough, runny nose, nasal congestion, sneezing, or sore throat, "
            "especially when no strong red-flag symptoms are present."
        ),
        "verification": (
            "The doctor should still review the scan and clinical history. Follow-up is recommended "
            "if symptoms persist, worsen, or if red flags such as blood in sputum, severe shortness "
            "of breath, chest pain, weight loss, or night sweats appear."
        ),
    },
}


def _normalize_text(value) -> str:
    return str(value or "").strip().lower()


def _normalize_prediction_label(prediction) -> str:
    value = _normalize_text(prediction)

    if "pneumonia" in value:
        return "Pneumonia"

    if "tuberculosis" in value or value == "tb":
        return "Tuberculosis"

    if "lung cancer" in value or "lung_cancer" in value:
        return "Lung Cancer"

    if "normal" in value:
        return "Normal"

    return str(prediction or "Unknown")


def _format_list(items: list[str]) -> str:
    clean_items = [str(item).strip() for item in items if str(item).strip()]

    if not clean_items:
        return "none recorded"

    if len(clean_items) == 1:
        return clean_items[0]

    if len(clean_items) == 2:
        return f"{clean_items[0]} and {clean_items[1]}"

    return f"{', '.join(clean_items[:-1])}, and {clean_items[-1]}"


def _confidence_meaning(confidence) -> str:
    value = float(confidence or 0)

    if value >= 85:
        return "high confidence, but still requiring doctor confirmation"

    if value >= 70:
        return "moderate confidence, meaning the result is useful for support but should be interpreted carefully"

    return "low-to-moderate confidence, meaning the result should be treated as uncertain and reviewed carefully"


def build_basic_clinical_explanation(
    final_prediction,
    final_confidence,
    risk_level,
    selected_symptoms,
    scan_type: str | None = None,
    image_prediction: str | None = None,
    image_confidence: float | None = None,
    symptom_prediction: str | None = None,
    symptom_confidence: float | None = None,
) -> str:
    """
    Builds a doctor-facing explanation using only available system inputs:
    scan type, image model output, selected symptoms, fusion result, confidence, and risk level.
    It does not invent radiology findings such as lesions, cavities, opacity, or masses.
    """

    prediction = _normalize_prediction_label(final_prediction)
    profile = DISEASE_CLINICAL_PROFILES.get(
        prediction,
        {
            "symptoms": set(),
            "rationale": (
                "The result does not match one of the predefined disease explanation profiles. "
                "The doctor should interpret it carefully with the scan and clinical context."
            ),
            "verification": (
                "The doctor should review the scan, patient history, physical examination, and "
                "request appropriate confirmatory tests when clinically indicated."
            ),
        },
    )

    selected_symptoms = selected_symptoms or []
    selected_symptom_text = _format_list(selected_symptoms)

    profile_symptoms = profile["symptoms"]
    matched_symptoms = [
        symptom
        for symptom in selected_symptoms
        if _normalize_text(symptom) in profile_symptoms
    ]

    red_flags = [
        symptom
        for symptom in selected_symptoms
        if _normalize_text(symptom) in RED_FLAG_SYMPTOMS
    ]

    matched_text = (
        f"The selected symptoms most relevant to {prediction} in this case are: {_format_list(matched_symptoms)}."
        if matched_symptoms
        else (
            f"The entered symptoms do not strongly match the typical {prediction} symptom profile, "
            "so the scan model output and fusion result should be reviewed carefully."
        )
    )

    red_flag_text = (
        f"Red-flag symptoms were also selected: {_format_list(red_flags)}. These should increase the priority of doctor review."
        if red_flags
        else "No major red-flag symptoms were selected from the current symptom list."
    )

    scan_text = (
        f"The scan input was {scan_type or 'not specified'}."
        if not image_prediction
        else (
            f"The scan input was {scan_type or 'not specified'}, and the image model top prediction was "
            f"{image_prediction}"
            + (
                f" with {image_confidence}% confidence."
                if image_confidence is not None
                else "."
            )
        )
    )

    symptom_model_text = (
        ""
        if not symptom_prediction
        else (
            f"\nThe symptom model top prediction was {symptom_prediction}"
            + (
                f" with {symptom_confidence}% confidence."
                if symptom_confidence is not None
                else "."
            )
        )
    )

    return (
        "Clinical Interpretation:\n"
        f"The AI-assisted system suggests {prediction} with {final_confidence or 0}% confidence "
        f"and a {risk_level or 'not specified'} risk level. This represents {_confidence_meaning(final_confidence)}. "
        "The final result was generated from the image model output, selected symptoms, and fusion analysis.\n\n"
        "Input-Based Evidence:\n"
        f"{scan_text}{symptom_model_text}\n"
        f"The selected symptoms were: {selected_symptom_text}.\n"
        f"{matched_text}\n"
        f"{red_flag_text}\n\n"
        "Clinical Rationale:\n"
        f"{profile['rationale']}\n\n"
        "Recommended Doctor Verification:\n"
        f"{profile['verification']}\n\n"
        "Important Note:\n"
        "This output is clinical decision support only. It should not be treated as a final diagnosis "
        "without doctor review, patient history, physical examination, and appropriate confirmatory testing."
    )


@app.post("/assistant/explanation")
def assistant_explanation(request: AssistantExplanationRequest):
    explanation = build_basic_clinical_explanation(
        final_prediction=request.final_prediction,
        final_confidence=request.final_confidence,
        risk_level=request.risk_level,
        selected_symptoms=request.selected_symptoms,
        scan_type=request.scan_type,
        image_prediction=request.image_prediction,
        image_confidence=request.image_confidence,
        symptom_prediction=request.symptom_prediction,
        symptom_confidence=request.symptom_confidence,
    )

    return {"assistant_explanation": explanation}


def _extract_report_id_from_text(message: str) -> int | None:
    patterns = [
        r"\bD[-\s]?(\d+)\b",
        r"\breport\s*#?\s*(\d+)\b",
        r"\bdiagnosis\s*#?\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def _parse_json_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [item.strip() for item in value.split(",") if item.strip()]

    return []


def _diagnosis_context_text(
    diagnosis: Diagnosis, patient: Patient | None
) -> tuple[str, str, float, str, list[str]]:
    final_prediction = diagnosis.final_prediction or "N/A"
    final_confidence = diagnosis.final_confidence or 0
    risk_level = diagnosis.risk_level or "N/A"
    selected_symptoms = _parse_json_list(diagnosis.selected_symptoms_json)

    patient_name = patient.full_name if patient and patient.full_name else "N/A"
    patient_id = diagnosis.patient_id or "N/A"

    context = (
        f"Report D-{diagnosis.id} | Patient: {patient_name} (P-{patient_id}) | "
        f"Scan: {diagnosis.scan_type or 'N/A'} | Prediction: {final_prediction} | "
        f"Confidence: {final_confidence}% | Risk: {risk_level} | "
        f"Symptoms Count: {len(selected_symptoms)}."
    )

    return context, final_prediction, final_confidence, risk_level, selected_symptoms


DISEASE_SYMPTOM_PROFILES = {
    "Pneumonia": [
        "Mild cough",
        "Cough",
        "Productive cough / sputum",
        "Fever",
        "Chills",
        "Fatigue",
        "Weakness",
        "Shortness of breath",
        "Difficulty breathing",
        "Rapid breathing",
        "Chest pain",
        "Chest pain when breathing or coughing",
        "Wheezing",
    ],
    "Tuberculosis": [
        "Persistent cough",
        "Cough lasting 3 weeks or more",
        "Productive cough / sputum",
        "Blood in sputum",
        "Chest pain",
        "Fever",
        "Chills",
        "Night sweats",
        "Weight loss",
        "Loss of appetite",
        "Fatigue",
        "Weakness",
        "Shortness of breath",
    ],
    "Lung Cancer": [
        "Chronic cough",
        "Persistent cough",
        "Cough that gets worse",
        "Blood in sputum",
        "Chest pain",
        "Shortness of breath",
        "Wheezing",
        "Hoarseness",
        "Weight loss",
        "Loss of appetite",
        "Fatigue",
        "Weakness",
        "Recurrent chest infections",
    ],
    "Normal": [
        "Mild cough",
        "Runny nose",
        "Nasal congestion",
        "Sneezing",
        "Sore throat",
    ],
}


DISEASE_RATIONALE = {
    "Pneumonia": {
        "rationale": (
            "Pneumonia is clinically supported when respiratory infection features are present, "
            "especially cough, fever or chills, fatigue, shortness of breath, rapid or difficult "
            "breathing, and chest pain during breathing or coughing."
        ),
        "verification": (
            "The doctor should review the chest image, assess breathing status and oxygen saturation, "
            "compare with physical examination findings, and request infection-related tests or "
            "additional imaging when clinically indicated."
        ),
    },
    "Tuberculosis": {
        "rationale": (
            "Tuberculosis is clinically supported when chronic respiratory symptoms and systemic "
            "symptoms appear together, especially cough lasting three weeks or more, chest pain, "
            "coughing blood or sputum, fever, night sweats, weight loss, fatigue, weakness, or "
            "loss of appetite."
        ),
        "verification": (
            "The doctor should review the scan, evaluate exposure history and physical examination "
            "findings, and confirm using appropriate TB evaluation such as TB blood/skin testing, "
            "chest imaging review, sputum smear/culture, and laboratory testing when clinically indicated."
        ),
    },
    "Lung Cancer": {
        "rationale": (
            "Lung cancer is clinically concerning when warning symptoms are present, such as chronic "
            "or worsening cough, coughing blood, chest pain, shortness of breath, hoarseness, "
            "unexplained weight loss, fatigue, wheezing, or recurrent chest infections."
        ),
        "verification": (
            "The doctor should review the imaging carefully, compare with history and risk factors, "
            "and consider further evaluation such as CT imaging, specialist referral, sputum cytology, "
            "or biopsy when clinically indicated."
        ),
    },
    "Normal": {
        "rationale": (
            "A normal or mild respiratory result is more consistent with mild upper-respiratory "
            "symptoms such as mild cough, runny nose, nasal congestion, sneezing, or sore throat, "
            "especially when no strong red-flag symptoms are present."
        ),
        "verification": (
            "The doctor should still review the scan and clinical history. Follow-up is recommended "
            "if symptoms persist, worsen, or if red flags such as blood in sputum, severe shortness "
            "of breath, chest pain, weight loss, or night sweats appear."
        ),
    },
}


def _normalize_text(value):
    return str(value or "").strip().lower()


def _parse_json_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    return {}


def _format_percent(value):
    try:
        number = float(value)
    except Exception:
        number = 0.0

    if 0 <= number <= 1:
        number *= 100

    return f"{number:.2f}%"


def _top_score(scores):
    if not scores:
        return "N/A", 0.0

    return max(scores.items(), key=lambda item: float(item[1] or 0))


def _format_score_lines(scores):
    class_order = ["Pneumonia", "Tuberculosis", "Lung Cancer", "Normal"]

    if not scores:
        return "No symptom probability scores were saved for this report."

    lines = []
    for disease in class_order:
        score = scores.get(disease, 0)
        lines.append(f"- {disease}: {_format_percent(score)}")

    return "\n".join(lines)


def _classes_for_symptom(symptom):
    symptom_key = _normalize_text(symptom)
    matched_classes = []

    for disease, symptoms in DISEASE_SYMPTOM_PROFILES.items():
        normalized_symptoms = [_normalize_text(item) for item in symptoms]

        if symptom_key in normalized_symptoms:
            matched_classes.append(disease)

    return matched_classes


def _build_class_symptom_evidence(selected_symptoms):
    if not selected_symptoms:
        return "No selected symptoms were saved for this report."

    disease_order = ["Pneumonia", "Tuberculosis", "Lung Cancer", "Normal"]

    unique_by_class = {disease: [] for disease in disease_order}
    shared_symptoms = []
    unmapped_symptoms = []

    for symptom in selected_symptoms:
        matched_classes = _classes_for_symptom(symptom)

        if len(matched_classes) == 0:
            unmapped_symptoms.append(symptom)
        elif len(matched_classes) == 1:
            unique_by_class[matched_classes[0]].append(symptom)
        else:
            shared_symptoms.append((symptom, matched_classes))

    lines = []

    lines.append("Symptom Evidence by Disease Class:")
    for disease in disease_order:
        symptoms = unique_by_class[disease]

        if symptoms:
            lines.append(f"- {disease}: {_format_list(symptoms)}.")
        else:
            lines.append(f"- {disease}: no unique selected symptoms.")

    lines.append("")
    lines.append("Symptoms Shared Between Multiple Classes:")
    if shared_symptoms:
        for symptom, classes in shared_symptoms:
            lines.append(f"- {symptom}: {', '.join(classes)}.")
    else:
        lines.append("- No shared symptoms were selected.")

    if unmapped_symptoms:
        lines.append("")
        lines.append("Unmapped Symptoms:")
        for symptom in unmapped_symptoms:
            lines.append(f"- {symptom}: not mapped in the current symptom profile.")

    return "\n".join(lines)


def build_clinical_explanation(diagnosis, selected_symptoms):
    final_prediction = _normalize_prediction_label(diagnosis.final_prediction or "N/A")
    final_confidence = diagnosis.final_confidence or 0
    risk_level = diagnosis.risk_level or "N/A"
    scan_type = diagnosis.scan_type or "selected scan"

    image_scores = _parse_json_dict(diagnosis.image_scores_json)
    symptom_scores = _parse_json_dict(diagnosis.symptom_scores_json)

    image_top_prediction = diagnosis.image_prediction or _top_score(image_scores)[0]
    image_top_confidence = diagnosis.image_confidence or _top_score(image_scores)[1]

    if symptom_scores:
        symptom_top_prediction, symptom_top_confidence = _top_score(symptom_scores)
    else:
        symptom_top_prediction = diagnosis.symptom_prediction or "N/A"
        symptom_top_confidence = diagnosis.symptom_confidence or 0

    disease_info = DISEASE_RATIONALE.get(
        final_prediction,
        DISEASE_RATIONALE["Normal"],
    )

    return (
        "Clinical Interpretation:\n"
        f"The AI-assisted system suggests {final_prediction} with {_format_percent(final_confidence)} confidence "
        f"and a {risk_level} risk level. This represents {_confidence_meaning(final_confidence)}. "
        f"The final result was generated from the image model output, symptom model output, and fusion analysis.\n\n"
        "Input-Based Evidence:\n"
        f"The scan input was {scan_type}, and the image model top prediction was {image_top_prediction} "
        f"with {_format_percent(image_top_confidence)} confidence.\n"
        f"The symptom model top prediction was {symptom_top_prediction} with {_format_percent(symptom_top_confidence)} confidence.\n"
        f"Selected symptoms: {_format_list(selected_symptoms)}.\n\n"
        "Symptom Model Class Probabilities:\n"
        f"{_format_score_lines(symptom_scores)}\n\n"
        f"{_build_class_symptom_evidence(selected_symptoms)}\n\n"
        "Clinical Rationale:\n"
        f"{disease_info['rationale']}\n\n"
        "Important Note:\n"
        "This output is clinical decision support only. It should not be treated as a final diagnosis without "
        "doctor review, patient history, physical examination, and appropriate confirmatory testing."
    )


def _get_selected_red_flags(selected_symptoms):
    selected_red_flags = []

    for symptom in selected_symptoms:
        if _normalize_text(symptom) in RED_FLAG_SYMPTOMS:
            selected_red_flags.append(symptom)

    return selected_red_flags


def _build_symptom_review_references():
    return (
        "\nClinical References for Symptom-Based Clinical Review:\n"
        "- CDC TB Clinical and Laboratory Diagnosis: https://www.cdc.gov/tb/hcp/testing-diagnosis/clinical-and-laboratory-diagnosis.html\n"
        "- CDC TB Testing and Diagnosis: https://www.cdc.gov/tb/hcp/testing-diagnosis/index.html\n"
        "- IDSA CAP Clinical Pathway: https://www.idsociety.org/globalassets/idsa/practice-guidelines/community-acquired-pneumonia-in-adults/cap-clinical-pathway-final-online.pdf\n"
        "- NICE Suspected Lung Cancer Referral: https://cks.nice.org.uk/topics/lung-pleural-cancers-recognition-referral/management/referral-for-suspected-lung-or-pleural-cancer/"
    )


def _build_prediction_verification_references(final_prediction):
    final_prediction = _normalize_prediction_label(final_prediction)

    if final_prediction == "Pneumonia":
        return (
            "\nClinical References for Pneumonia Diagnostic Verification:\n"
            "- ATS/IDSA CAP Guideline: https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/\n"
            "- IDSA CAP Clinical Pathway: https://www.idsociety.org/globalassets/idsa/practice-guidelines/community-acquired-pneumonia-in-adults/cap-clinical-pathway-final-online.pdf"
        )

    if final_prediction == "Tuberculosis":
        return (
            "\nClinical References for Tuberculosis Diagnostic Verification:\n"
            "- CDC TB Clinical and Laboratory Diagnosis: https://www.cdc.gov/tb/hcp/testing-diagnosis/clinical-and-laboratory-diagnosis.html\n"
            "- CDC TB Testing and Diagnosis: https://www.cdc.gov/tb/hcp/testing-diagnosis/index.html\n"
            "- ATS/CDC/IDSA TB Diagnosis Guideline: https://academic.oup.com/cid/article/64/2/e1/2629583"
        )

    if final_prediction == "Lung Cancer":
        return (
            "\nClinical References for Lung Cancer Diagnostic Verification:\n"
            "- NICE Suspected Lung Cancer Referral: https://cks.nice.org.uk/topics/lung-pleural-cancers-recognition-referral/management/referral-for-suspected-lung-or-pleural-cancer/\n"
            "- ACCP Lung Cancer Diagnosis Guideline: https://pubmed.ncbi.nlm.nih.gov/23649436/\n"
            "- American Cancer Society Lung Cancer Diagnosis: https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/how-diagnosed.html"
        )

    return (
        "\nClinical References for Normal or Uncertain Diagnostic Verification:\n"
        "- IDSA CAP Clinical Pathway: https://www.idsociety.org/globalassets/idsa/practice-guidelines/community-acquired-pneumonia-in-adults/cap-clinical-pathway-final-online.pdf\n"
        "- CDC TB Clinical and Laboratory Diagnosis: https://www.cdc.gov/tb/hcp/testing-diagnosis/clinical-and-laboratory-diagnosis.html\n"
        "- NICE Suspected Lung Cancer Referral: https://cks.nice.org.uk/topics/lung-pleural-cancers-recognition-referral/management/referral-for-suspected-lung-or-pleural-cancer/"
    )


SYMPTOM_CLINICAL_REVIEW_ACTIONS = {
    "mild cough": "Review duration and progression, check for fever, dyspnea, sputum, and whether scan findings suggest lower respiratory disease.",
    "cough": "Review duration, severity, sputum production, fever, chest pain, breathing difficulty, and correlate with scan findings.",
    "persistent cough": "Review cough duration, sputum, fever, weight loss, smoking or exposure risk, and correlate with scan findings.",
    "cough lasting 3 weeks or more": "Review TB exposure/contact risk, sputum, fever, night sweats, weight loss, chest imaging, and consider TB testing when clinically indicated.",
    "chronic cough": "Review duration, smoking/risk history, recurrent infections, weight loss, hemoptysis, and whether further imaging or referral is needed.",
    "cough that gets worse": "Assess progression, fever pattern, sputum, chest pain, oxygen status, and whether repeat imaging or urgent review is needed.",
    "productive cough / sputum": "Assess sputum amount/character, fever, chest findings, and consider sputum testing or infection workup when clinically indicated.",
    "blood in sputum": "Assess amount, recurrence, vital signs, oxygen status, scan findings, and whether urgent respiratory evaluation, TB workup, or cancer referral is needed.",
    "shortness of breath": "Assess respiratory rate, oxygen saturation, work of breathing, clinical stability, and need for urgent evaluation.",
    "difficulty breathing": "Assess oxygen saturation, respiratory effort, respiratory rate, and whether urgent assessment is required.",
    "rapid breathing": "Check respiratory rate, oxygen saturation, fever, signs of respiratory distress, and clinical stability.",
    "chest pain": "Characterize severity and relation to breathing/coughing, assess vital signs and oxygen status, and consider urgent review if severe or persistent.",
    "chest pain when breathing or coughing": "Assess pleuritic pattern, fever, oxygen status, chest examination, and imaging correlation.",
    "fever": "Check temperature pattern/duration, infection signs, respiratory status, and whether blood or infection markers are needed.",
    "chills": "Assess fever pattern, infection severity, vital signs, respiratory status, and need for infection workup.",
    "night sweats": "Review TB risk, fever pattern, cough duration, weight loss, exposure history, and need for TB evaluation.",
    "weight loss": "Confirm unintentional weight loss and duration, review appetite, TB/cancer risk factors, and need for further evaluation.",
    "loss of appetite": "Review duration, weight change, systemic symptoms, TB/cancer/infection context, and clinical stability.",
    "fatigue": "Assess severity/duration, fever, weight loss, oxygen status, systemic illness, and clinical stability.",
    "weakness": "Assess severity, hydration, oxygen status, systemic illness, and whether urgent assessment is needed.",
    "wheezing": "Assess oxygen saturation, work of breathing, asthma/COPD history, infection signs, and scan findings.",
    "hoarseness": "Review duration, smoking/risk history, cough, hemoptysis, weight loss, and consider ENT or respiratory referral if persistent.",
    "recurrent chest infections": "Review frequency, previous imaging, risk factors, and whether CT imaging or specialist review is clinically indicated.",
    "runny nose": "Review duration and severity, check for lower respiratory symptoms or red flags, and correlate with scan findings.",
    "nasal congestion": "Review duration and severity, check for lower respiratory symptoms or red flags, and correlate with scan findings.",
    "sneezing": "Review whether symptoms are isolated upper-respiratory/allergic features or associated with fever, cough, or breathing difficulty.",
    "sore throat": "Review duration, fever, cough, breathing difficulty, systemic symptoms, and severity.",
}


def _build_symptom_based_clinical_review(selected_symptoms, selected_red_flags):
    lines = ["\n2. Symptom-Based Clinical Review:"]

    if not selected_symptoms:
        lines.extend(
            [
                "- No symptoms were recorded for this report.",
                "- Confirm the patient history directly and ask about respiratory and systemic symptoms before making a final decision.",
                "- If symptoms are added later, reassess the report using the updated symptom information.",
            ]
        )
        return "\n".join(lines)

    lines.append(f"- Selected symptoms: {_format_list(selected_symptoms)}.")

    if selected_red_flags:
        lines.append(
            f"- Red-flag symptoms selected: {_format_list(selected_red_flags)}. Prioritize clinical review and assess stability."
        )
    else:
        lines.append(
            "- Red-flag symptoms selected: none from the current symptom list."
        )

    lines.append("- Symptom-specific doctor checks:")

    for symptom in selected_symptoms:
        action = SYMPTOM_CLINICAL_REVIEW_ACTIONS.get(_normalize_text(symptom))

        if action:
            lines.append(f"  - {symptom}: {action}")
        else:
            lines.append(
                f"  - {symptom}: Review duration, severity, associated symptoms, and correlation with scan findings."
            )

    lines.append(
        "- Interpret symptoms together with scan review, patient history, examination findings, and final prediction because respiratory diseases can overlap clinically."
    )

    return "\n".join(lines)


def _build_prediction_based_verification(final_prediction):
    final_prediction = _normalize_prediction_label(final_prediction)

    if final_prediction == "Pneumonia":
        return (
            "\n3. Prediction-Based Diagnostic Verification:\n"
            "- Final prediction: Pneumonia.\n"
            "- Review the chest image/radiology findings and compare them with fever pattern, cough, sputum, oxygen saturation, and chest examination.\n"
            "- Consider pulse oximetry, blood/infection markers, sputum testing, microbiology, or additional imaging when clinically indicated."
        )

    if final_prediction == "Tuberculosis":
        return (
            "\n3. Prediction-Based Diagnostic Verification:\n"
            "- Final prediction: Tuberculosis.\n"
            "- Review cough duration, TB exposure/contact risk, systemic symptoms, infection-control precautions, and chest imaging.\n"
            "- Consider TB blood/skin testing, sputum smear microscopy, NAAT, culture, and drug-susceptibility testing when clinically indicated."
        )

    if final_prediction == "Lung Cancer":
        return (
            "\n3. Prediction-Based Diagnostic Verification:\n"
            "- Final prediction: Lung Cancer.\n"
            "- Review risk factors, hemoptysis, chronic or worsening cough, unexplained weight loss, hoarseness, recurrent infections, and imaging pattern.\n"
            "- Consider urgent respiratory referral, CT-based evaluation, bronchoscopy, CT-guided biopsy, sputum cytology, or tissue diagnosis when clinically indicated."
        )

    return (
        "\n3. Prediction-Based Diagnostic Verification:\n"
        f"- Final prediction: {final_prediction}.\n"
        "- Review scan quality, image-model confidence, patient history, and examination findings.\n"
        "- If clinical concern remains despite a normal or uncertain AI result, consider follow-up, repeat review, or further diagnostic evaluation when clinically indicated."
    )


def _build_next_steps(diagnosis, selected_symptoms):
    final_prediction = _normalize_prediction_label(diagnosis.final_prediction or "N/A")
    risk_level = diagnosis.risk_level or "N/A"
    scan_type = diagnosis.scan_type or "N/A"

    image_scores = _parse_json_dict(diagnosis.image_scores_json)
    symptom_scores = _parse_json_dict(diagnosis.symptom_scores_json)

    image_top_prediction = diagnosis.image_prediction or _top_score(image_scores)[0]
    image_top_confidence = diagnosis.image_confidence or _top_score(image_scores)[1]

    if symptom_scores:
        symptom_top_prediction, symptom_top_confidence = _top_score(symptom_scores)
    else:
        symptom_top_prediction = diagnosis.symptom_prediction or "N/A"
        symptom_top_confidence = diagnosis.symptom_confidence or 0

    selected_symptoms = selected_symptoms or []
    selected_red_flags = _get_selected_red_flags(selected_symptoms)

    steps = []

    steps.append(
        "1. Scan-Based Review:\n"
        f"- Scan type: {scan_type}\n"
        f"- Image model result: {image_top_prediction} ({_format_percent(image_top_confidence)} confidence)\n"
        f"- Final fused prediction: {final_prediction} ({_format_percent(diagnosis.final_confidence or 0)} confidence), risk level: {risk_level}\n"
        "- Manually review the uploaded scan and confirm that the image quality and scan type are clinically acceptable.\n"
        "- Compare the AI image result with the final fused prediction and the clinical picture before making a final decision."
    )

    steps.append(
        _build_symptom_based_clinical_review(selected_symptoms, selected_red_flags)
    )
    steps.append(_build_symptom_review_references())

    steps.append(_build_prediction_based_verification(final_prediction))
    steps.append(_build_prediction_verification_references(final_prediction))

    steps.append(
        "\n4. Clinical Safety Note:\n"
        "These recommendations are clinical decision-support suggestions only. The final decision must be made by the doctor after reviewing the scan, symptoms, patient history, examination findings, and any required tests."
    )

    return "\n".join(steps)




# ============================================================
# Patient mobile app compatibility endpoints
# Login uses Patient ID + registered email.
# These endpoints let the Flutter patient app connect to this backend
# without changing the doctor website login system.
# ============================================================

PATIENT_SYMPTOM_KEYWORDS = {
    "fever": ["fever", "temperature", "high temperature", "hot"],
    "cough": ["cough", "coughing"],
    "chest_pain": ["chest pain", "pain in chest", "chest pressure"],
    "shortness_of_breath": [
        "shortness of breath",
        "breathing difficulty",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "hard to breathe",
        "breathless",
        "low oxygen",
        "oxygen is low",
        "my oxygen is low",
        "oxygen level is low",
        "my oxygen level is low",
        "spo2 is low",
        "low spo2",
        "oxygen dropped",
        "oxygen drops",
    ],
    "fatigue": ["fatigue", "tired", "weak", "exhausted"],
    "night_sweats": ["night sweats", "sweating at night"],
    "weight_loss": ["weight loss", "lost weight", "losing weight"],
    "blood_in_sputum": [
        "blood in sputum",
        "blood in phlegm",
        "coughing blood",
        "blood when coughing",
        "bloody sputum",
        "bloody phlegm",
    ],
}


def _clean_email(value: str | None) -> str:
    return str(value or "").strip().lower()


def _patient_token(patient_id: int) -> str:
    return f"patient-{patient_id}"


def _patient_public(patient: Patient):
    return {
        "id": patient.id,
        "patient_id": patient.id,
        "doctor_id": patient.doctor_id,
        "full_name": patient.full_name,
        "age": patient.age,
        "gender": patient.gender,
        "phone": patient.phone,
        "national_id": patient.national_id,
        "email": patient.email,
        "address": patient.address,
        "date_of_birth": patient.date_of_birth,
        "medical_notes": patient.medical_notes,
        "created_at": patient.created_at,
    }


def _mobile_patient_from_auth(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> Patient:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Patient session missing. Please login again.")

    token = authorization.split(" ", 1)[1].strip()

    if not token.startswith("patient-"):
        raise HTTPException(status_code=401, detail="Invalid patient session. Please login again.")

    try:
        patient_id = int(token.split("-", 1)[1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid patient session. Please login again.")

    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.is_deleted == False)
        .first()
    )

    if not patient:
        raise HTTPException(status_code=401, detail="Patient account not found. Please login again.")

    return patient


def _map_diagnosis_for_mobile(diagnosis: Diagnosis):
    return {
        "id": diagnosis.id,
        "patient_id": diagnosis.patient_id,
        "doctor_id": diagnosis.doctor_id,
        "status": "Approved",
        "xray_path": diagnosis.image_path if diagnosis.scan_type and "x" in diagnosis.scan_type.lower() else None,
        "ct_path": diagnosis.image_path if diagnosis.scan_type and "ct" in diagnosis.scan_type.lower() else None,
        "xray_heatmap_path": diagnosis.heatmap_path if diagnosis.scan_type and "x" in diagnosis.scan_type.lower() else None,
        "ct_heatmap_path": diagnosis.heatmap_path if diagnosis.scan_type and "ct" in diagnosis.scan_type.lower() else None,
        "ai_diagnosis": diagnosis.final_prediction,
        "ai_confidence": (diagnosis.final_confidence or 0) / 100,
        "doctor_final_diagnosis": diagnosis.final_prediction,
        "doctor_notes": diagnosis.doctor_notes or diagnosis.assistant_explanation,
        "risk_level": diagnosis.risk_level,
        "report_path": f"diagnoses/{diagnosis.id}/report",
        "created_at": diagnosis.created_at,
        "reviewed_at": diagnosis.created_at,
    }


def _extract_mobile_oxygen(message: str) -> float | None:
    text_lower = (message or "").lower()
    patterns = [
        r"(?:oxygen|o2|spo2|saturation|sat)\s*(?:level|is|=|:)?\s*(\d{2,3})(?:\s*%)?",
        r"(\d{2,3})\s*%\s*(?:oxygen|o2|spo2|saturation|sat)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1))
            if 40 <= value <= 100:
                return value
    return None


def _extract_mobile_temperature(message: str) -> float | None:
    text_lower = (message or "").lower()
    patterns = [
        r"(?:temperature|temp|fever)\s*(?:is|=|:)?\s*(\d{2}(?:\.\d+)?)",
        r"(\d{2}(?:\.\d+)?)\s*(?:c|°c|degree|degrees)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1))
            if 30 <= value <= 45:
                return value
    return None


def _extract_mobile_symptoms(message: str) -> dict:
    text_lower = (message or "").lower()
    extracted = {
        "fever": False,
        "cough": False,
        "chest_pain": False,
        "shortness_of_breath": False,
        "fatigue": False,
        "night_sweats": False,
        "weight_loss": False,
        "blood_in_sputum": False,
    }

    for symptom, keywords in PATIENT_SYMPTOM_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            extracted[symptom] = True

    return extracted


def _mobile_symptom_names(symptoms: dict) -> list[str]:
    return [key.replace("_", " ").title() for key, value in symptoms.items() if value]


def _mobile_assistant_assessment(symptoms: dict, temperature: float | None, oxygen: float | None):
    severity_points = 0
    red_flags = []
    risk_reasons = []

    for key, value in symptoms.items():
        if value:
            severity_points += 1
            risk_reasons.append(f"{key.replace('_', ' ').title()} reported")

    if symptoms.get("shortness_of_breath"):
        severity_points += 4
        red_flags.append("shortness of breath / low oxygen concern")
        risk_reasons.append("Breathing difficulty or low oxygen concern is high-priority")

    if symptoms.get("chest_pain"):
        severity_points += 3
        red_flags.append("chest pain")
        risk_reasons.append("Chest pain needs medical attention if persistent or severe")

    if symptoms.get("blood_in_sputum"):
        severity_points += 8
        red_flags.append("blood in sputum")
        risk_reasons.append("Blood in sputum is a critical warning sign")

    if oxygen is not None:
        if oxygen < 90:
            severity_points += 10
            red_flags.append(f"very low oxygen level ({oxygen}%)")
            risk_reasons.append(f"Oxygen level {oxygen}% is emergency-level low")
        elif oxygen < 92:
            severity_points += 6
            red_flags.append(f"low oxygen level ({oxygen}%)")
            risk_reasons.append(f"Oxygen level {oxygen}% is below the safe monitoring threshold")
        elif oxygen < 95:
            severity_points += 2
            risk_reasons.append(f"Oxygen level {oxygen}% should be monitored")

    if temperature is not None:
        if temperature >= 39.5:
            severity_points += 4
            red_flags.append(f"very high fever ({temperature}°C)")
            risk_reasons.append(f"Temperature {temperature}°C is very high")
        elif temperature >= 38.5:
            severity_points += 2
            risk_reasons.append(f"Temperature {temperature}°C indicates fever")

    if symptoms.get("blood_in_sputum") or (oxygen is not None and oxygen < 90):
        severity = "Critical"
    elif symptoms.get("shortness_of_breath") and symptoms.get("chest_pain"):
        severity = "Critical"
    elif symptoms.get("shortness_of_breath") or symptoms.get("chest_pain") or (oxygen is not None and oxygen < 92):
        severity = "High"
    elif severity_points >= 5:
        severity = "High"
    elif severity_points >= 2:
        severity = "Medium"
    else:
        severity = "Low"

    trend = "Worsening" if severity in ["High", "Critical"] else "Stable" if severity == "Medium" else "Improving"

    if severity == "Critical":
        stage = "urgent_escalation"
        action_card = "Urgent alert sent to your doctor. Seek emergency care if symptoms are severe."
    elif severity == "High":
        stage = "doctor_alert"
        action_card = "Doctor alert sent. Monitor oxygen and symptoms closely while waiting for doctor review."
    elif severity == "Medium":
        stage = "monitoring"
        action_card = "Continue monitoring symptoms and repeat check-in if anything worsens."
    else:
        stage = "safe_guidance"
        action_card = "Continue normal monitoring, rest, fluids, and follow your doctor's instructions."

    return severity, trend, severity_points, red_flags, risk_reasons, stage, action_card


def _mobile_assistant_reply(patient_name: str, severity: str, red_flags: list[str]):
    if severity == "Critical":
        flags = ", ".join(red_flags) if red_flags else "dangerous symptoms"
        return (
            f"{patient_name}, your symptoms may be critical because of: {flags}. "
            "I have sent an urgent alert to your doctor. If you have severe breathing difficulty, chest pain, "
            "blue lips, confusion, fainting, or oxygen level below 90%, seek emergency medical care immediately."
        )

    if severity == "High":
        flags = ", ".join(red_flags) if red_flags else "high-risk symptoms"
        return (
            f"Your symptoms need medical attention because of: {flags}. "
            "I have sent an alert to your doctor so they can review your condition. "
            "If your oxygen is below 90%, or you have severe breathing difficulty, chest pain, blue lips, confusion, or fainting, seek emergency care immediately."
        )

    if severity == "Medium":
        return (
            "Your symptoms do not look critical right now, but they should be monitored. "
            "Drink fluids, rest, monitor your temperature and oxygen level, and follow your doctor's previous instructions."
        )

    return (
        "Your symptoms look mild based on the information you provided. Continue daily monitoring, rest, drink fluids, "
        "and follow your doctor's instructions. Tell me again if symptoms become worse."
    )


def _mobile_alert_title(severity: str) -> str:
    s = (severity or "Medium").lower()
    if "critical" in s:
        return "Critical symptom alert"
    if "high" in s:
        return "High symptom alert"
    return "Health monitoring alert"


def _mobile_alert_next_step(severity: str, doctor_reply: str | None):
    if doctor_reply:
        return "Follow the doctor's response shown inside this alert."
    s = (severity or "Medium").lower()
    if "critical" in s:
        return "Seek urgent care if symptoms are severe. Wait for your doctor review if stable."
    if "high" in s:
        return "Monitor your oxygen level and symptoms. Avoid physical effort until reviewed."
    return "Continue monitoring and submit another check-in if symptoms worsen."


@app.post("/auth/patient-login")
def patient_login(request: PatientLoginRequest, db: Session = Depends(get_db)):
    email = _clean_email(request.email)
    login_value = str(request.login_value or "").strip()
    password = str(request.password or "").strip()
    patient_id = request.patient_id

    if patient_id is None and login_value.isdigit():
        patient_id = int(login_value)

    patient = None

    if patient_id is not None:
        patient = (
            db.query(Patient)
            .filter(Patient.id == patient_id, Patient.is_deleted == False)
            .first()
        )

        if not patient or _clean_email(patient.email) != email:
            return {
                "success": False,
                "message": "Invalid patient ID or email. Please use the email registered in the doctor website.",
            }

    else:
        password_value = password or login_value

        if not password_value:
            return {
                "success": False,
                "message": "Enter your Patient ID or password.",
            }

        patient = (
            db.query(Patient)
            .filter(Patient.email == request.email, Patient.is_deleted == False)
            .first()
        )

        if not patient:
            patient = (
                db.query(Patient)
                .filter(Patient.is_deleted == False)
                .all()
            )
            patient = next((p for p in patient if _clean_email(p.email) == email), None)

        if not patient or not patient.password_hash or not pwd_context.verify(password_value, patient.password_hash):
            return {
                "success": False,
                "message": "Invalid patient email or password. If you did not set a password yet, login with Patient ID first.",
            }

    return {
        "success": True,
        "message": "Patient login successful",
        "access_token": _patient_token(patient.id),
        "token_type": "bearer",
        "patient_id": patient.id,
        "user": {
            "id": patient.id,
            "email": patient.email,
            "role": "patient",
            "full_name": patient.full_name,
            "patient_id": patient.id,
            "doctor_id": patient.doctor_id,
        },
        "patient": _patient_public(patient),
    }


@app.post("/patient/change-password")
def patient_change_password(
    request: PatientChangePasswordRequest,
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    if request.patient_id != patient.id:
        raise HTTPException(status_code=403, detail="Patient ID does not match your logged-in account.")

    if request.new_password != request.confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirm password do not match.")

    if len(request.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    patient.password_hash = pwd_context.hash(request.new_password)
    db.commit()

    return {
        "success": True,
        "message": "Password updated successfully. You can now login using email and password.",
    }

@app.get("/me")
def patient_me(patient: Patient = Depends(_mobile_patient_from_auth)):
    return {
        "id": patient.id,
        "email": patient.email,
        "role": "patient",
        "full_name": patient.full_name,
        "patient_id": patient.id,
        "doctor_id": patient.doctor_id,
    }


@app.get("/patient/dashboard")
def mobile_patient_dashboard(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    latest = (
        db.query(Diagnosis)
        .filter(Diagnosis.patient_id == patient.id)
        .order_by(Diagnosis.created_at.desc())
        .first()
    )

    unread_alerts = (
        db.query(PatientAlert)
        .filter(
            PatientAlert.patient_id == patient.id,
            PatientAlert.status == "New",
            PatientAlert.patient_seen == False,
        )
        .count()
    )

    pending_appointments = (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient.id, Appointment.status == "Pending")
        .count()
    )

    return {
        "patient": _patient_public(patient),
        "latest_case": _map_diagnosis_for_mobile(latest) if latest else None,
        "unread_alerts": unread_alerts,
        "pending_appointments": pending_appointments,
    }


@app.get("/patient/reports")
def mobile_patient_reports(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    diagnoses = (
        db.query(Diagnosis)
        .filter(Diagnosis.patient_id == patient.id)
        .order_by(Diagnosis.created_at.desc())
        .all()
    )

    return [_map_diagnosis_for_mobile(diagnosis) for diagnosis in diagnoses]


@app.get("/patient/alerts")
def mobile_patient_alerts(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    alerts = (
        db.query(PatientAlert)
        .filter(PatientAlert.patient_id == patient.id)
        .order_by(PatientAlert.created_at.desc())
        .limit(30)
        .all()
    )

    result = []

    for alert in alerts:
        result.append(
            {
                "id": alert.id,
                "severity": alert.severity or "Medium",
                "title": _mobile_alert_title(alert.severity),
                "message": alert.message or "Your symptoms were sent to your doctor for review.",
                "raw_message": alert.message,
                "status": "Doctor reviewed" if alert.status == "Reviewed" else "Sent to doctor",
                "next_step": _mobile_alert_next_step(alert.severity, alert.doctor_reply),
                "created_at": alert.created_at,
                "is_read": bool(alert.patient_seen),
                "case_id": None,
                "doctor_response": alert.doctor_reply,
                "doctor_response_at": alert.created_at if alert.doctor_reply else None,
                "doctor_name": None,
                "doctor_reviewed": alert.status == "Reviewed",
                "patient_seen": bool(alert.patient_seen),
                "symptoms": json.loads(alert.symptoms_json or "[]"),
            }
        )

    return result


@app.post("/patient/alerts/mark-read")
def mobile_mark_patient_alerts_read(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    alerts = (
        db.query(PatientAlert)
        .filter(PatientAlert.patient_id == patient.id, PatientAlert.patient_seen == False)
        .all()
    )

    for alert in alerts:
        alert.patient_seen = True

    db.commit()

    return {"status": "ok", "marked_read": len(alerts)}


@app.get("/patient/appointments")
def mobile_patient_appointments(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    appointments = (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient.id)
        .order_by(Appointment.created_at.desc())
        .all()
    )

    return [
        {
            "id": appointment.id,
            "patient_id": appointment.patient_id,
            "doctor_id": appointment.doctor_id,
            "preferred_date": appointment.requested_date,
            "preferred_time": appointment.requested_time,
            "appointment_type": "Follow-up",
            "reason": appointment.reason,
            "status": appointment.status,
            "doctor_response": appointment.doctor_response,
            "created_at": appointment.created_at,
        }
        for appointment in appointments
    ]


@app.post("/patient/appointments")
def mobile_create_patient_appointment(
    request: PatientAppointmentRequest,
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    if not patient.doctor_id:
        return {"success": False, "message": "This patient has no assigned doctor."}

    appointment = Appointment(
        patient_id=patient.id,
        doctor_id=patient.doctor_id,
        requested_date=request.preferred_date,
        requested_time=request.preferred_time or "Not specified",
        reason=(
            f"Appointment type: {request.appointment_type}\nReason: {request.reason}"
            if request.appointment_type
            else request.reason
        ),
        symptoms_json=json.dumps([]),
        status="Pending",
    )

    db.add(appointment)
    db.commit()
    db.refresh(appointment)

    return {
        "success": True,
        "message": "Appointment request sent successfully",
        "appointment_id": appointment.id,
    }


@app.get("/patient/symptoms/history")
def mobile_symptoms_history(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    alerts = (
        db.query(PatientAlert)
        .filter(PatientAlert.patient_id == patient.id)
        .order_by(PatientAlert.created_at.desc())
        .limit(20)
        .all()
    )

    history = []

    for alert in alerts:
        symptom_names = [str(s).lower().replace(" ", "_") for s in json.loads(alert.symptoms_json or "[]")]
        history.append(
            {
                "id": alert.id,
                "patient_id": patient.id,
                "fever": "fever" in symptom_names,
                "cough": "cough" in symptom_names,
                "chest_pain": "chest_pain" in symptom_names or "chest" in " ".join(symptom_names),
                "shortness_of_breath": "shortness_of_breath" in symptom_names or "breath" in " ".join(symptom_names),
                "fatigue": "fatigue" in symptom_names,
                "night_sweats": "night_sweats" in symptom_names,
                "weight_loss": "weight_loss" in symptom_names,
                "blood_in_sputum": "blood_in_sputum" in symptom_names or "blood" in " ".join(symptom_names),
                "temperature": None,
                "oxygen_level": None,
                "trend": "Worsening" if (alert.severity or "").lower() in ["high", "critical"] else "Stable",
                "created_at": alert.created_at,
            }
        )

    return history


@app.get("/patient/assistant-chat/history")
def mobile_assistant_history(patient: Patient = Depends(_mobile_patient_from_auth)):
    return []


@app.post("/patient/assistant-chat")
def mobile_assistant_chat(
    request: PatientAssistantMobileRequest,
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    extracted = _extract_mobile_symptoms(request.message)
    oxygen = request.oxygen_level if request.oxygen_level is not None else _extract_mobile_oxygen(request.message)
    temperature = request.temperature if request.temperature is not None else _extract_mobile_temperature(request.message)

    symptoms = {
        "fever": request.fever or extracted["fever"],
        "cough": request.cough or extracted["cough"],
        "chest_pain": request.chest_pain or extracted["chest_pain"],
        "shortness_of_breath": request.shortness_of_breath or extracted["shortness_of_breath"],
        "fatigue": request.fatigue or extracted["fatigue"],
        "night_sweats": request.night_sweats or extracted["night_sweats"],
        "weight_loss": request.weight_loss or extracted["weight_loss"],
        "blood_in_sputum": request.blood_in_sputum or extracted["blood_in_sputum"],
    }

    severity, trend, severity_points, red_flags, risk_reasons, stage, action_card = _mobile_assistant_assessment(symptoms, temperature, oxygen)
    symptom_names = _mobile_symptom_names(symptoms)

    alert = None
    alert_sent = False

    if severity in ["High", "Critical"] and patient.doctor_id:
        alert = PatientAlert(
            patient_id=patient.id,
            doctor_id=patient.doctor_id,
            severity=severity,
            message=(
                f"AI Assistant detected {severity.lower()} symptoms for {patient.full_name}. "
                f"Symptoms: {', '.join(symptom_names) if symptom_names else 'not specified'}. "
                f"Temperature: {temperature if temperature is not None else 'not provided'}. "
                f"Oxygen: {oxygen if oxygen is not None else 'not provided'}. "
                f"Patient message: {request.message}"
            ),
            symptoms_json=json.dumps(symptom_names),
            status="New",
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        alert_sent = True

    follow_up_questions = []
    if symptoms.get("shortness_of_breath") and oxygen is None:
        follow_up_questions.append("Do you know your oxygen level now?")
    if temperature is None:
        follow_up_questions.append("Do you know your temperature today?")
    if not symptoms.get("blood_in_sputum"):
        follow_up_questions.append("Is there any blood in your sputum or when coughing?")

    return {
        "assistant_reply": _mobile_assistant_reply(patient.full_name, severity, red_flags),
        "severity": severity,
        "trend": trend,
        "alert_sent": alert_sent,
        "alert_id": alert.id if alert else None,
        "alert_status": "Sent to doctor" if alert_sent else "No alert needed",
        "symptom_log_id": alert.id if alert else None,
        "severity_points": severity_points,
        "red_flags": red_flags,
        "risk_reasons": risk_reasons or ["No high-risk symptom was detected."],
        "follow_up_questions": follow_up_questions[:3],
        "conversation_stage": stage,
        "action_card": action_card,
        "extracted_symptoms": symptoms,
        "oxygen_level": oxygen,
        "temperature": temperature,
        "safe_note": "This assistant provides symptom tracking and safe guidance only. It does not replace a doctor or emergency care.",
    }


@app.get("/chat/messages")
def mobile_chat_messages(
    patient: Patient = Depends(_mobile_patient_from_auth),
    db: Session = Depends(get_db),
):
    alerts = (
        db.query(PatientAlert)
        .filter(PatientAlert.patient_id == patient.id, PatientAlert.doctor_reply.isnot(None))
        .order_by(PatientAlert.created_at.asc())
        .all()
    )

    return [
        {
            "id": alert.id,
            "patient_id": patient.id,
            "sender_id": alert.doctor_id,
            "receiver_id": patient.id,
            "message": alert.doctor_reply,
            "is_read": True,
            "created_at": alert.created_at,
        }
        for alert in alerts
    ]


@app.post("/assistant/chat")
def assistant_chat(request: AssistantChatRequest, db: Session = Depends(get_db)):
    user_message = request.message.strip()

    if not user_message:
        return {
            "success": False,
            "assistant_response": "Please write a message first.",
        }

    def extract_report_id(text: str, allow_plain_number: bool = False):
        text = text.strip()
        text_lower = text.lower()

        match = re.search(r"\bd\s*[-:]?\s*(\d+)\b", text_lower)
        if match:
            return int(match.group(1))

        match = re.search(
            r"\b(report|diagnosis|diagnose)\s*(id)?\s*[-:]?\s*(\d+)\b",
            text_lower,
        )
        if match:
            return int(match.group(3))

        if allow_plain_number and re.fullmatch(r"\d+", text):
            return int(text)

        return None

    def classify_request(text: str):
        text_lower = text.lower()

        if any(
            word in text_lower
            for word in [
                "report file",
                "send report",
                "open report",
                "download report",
                "pdf",
                "download pdf",
                "report pdf",
                "file",
            ]
        ):
            return "report_file"

        if any(
            word in text_lower
            for word in ["red flag", "red flags", "danger", "urgent", "warning"]
        ):
            return "red_flags"

        if any(word in text_lower for word in ["symptom", "symptoms", "summarize"]):
            return "symptoms"

        if any(
            word in text_lower
            for word in ["next", "step", "steps", "recommend", "recommendation"]
        ):
            return "next_steps"

        if any(
            word in text_lower
            for word in ["explain", "diagnosis", "diagnose", "result", "results"]
        ):
            return "explain"

        return None

    def parse_json_list(value):
        if not value:
            return []

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    pending_text = request.pending_request or ""
    action = classify_request(pending_text) or classify_request(user_message)

    diagnosis_id = extract_report_id(
        user_message,
        allow_plain_number=True if request.pending_request else False,
    )

    looks_like_patient_id = re.search(
        r"\bp\s*[-:]?\s*(\d+)\b",
        user_message.lower(),
    )

    if action and not diagnosis_id:
        action_label = {
            "explain": "explain",
            "symptoms": "summarize symptoms for",
            "red_flags": "identify red flags for",
            "next_steps": "suggest next steps for",
            "report_file": "prepare the report file for",
        }.get(action, "review")

        if request.pending_request:
            if looks_like_patient_id:
                wrong_patient_id = looks_like_patient_id.group(1)

                return {
                    "success": True,
                    "requires_report_id": True,
                    "pending_request": request.pending_request,
                    "assistant_response": (
                        f"It looks like you entered Patient ID P-{wrong_patient_id}, "
                        "but I need the diagnosis report ID.\n\n"
                        "Diagnosis report IDs start with D, for example: D-15.\n"
                        "Please check the Reports page and send the correct report ID."
                    ),
                }

            return {
                "success": True,
                "requires_report_id": True,
                "pending_request": request.pending_request,
                "assistant_response": (
                    "I could not recognize a valid diagnosis report ID.\n\n"
                    "Please send the report ID in this format: D-15.\n"
                    "You can find it in the Reports page under the ID column."
                ),
            }

        return {
            "success": True,
            "requires_report_id": True,
            "pending_request": user_message,
            "assistant_response": (
                f"Okay, but first tell me which diagnosis report you want me to {action_label}.\n\n"
                "Please send the diagnosis report ID, for example: D-15."
            ),
        }

    if diagnosis_id and not action:
        return {
            "success": True,
            "requires_report_id": False,
            "assistant_response": (
                f"I found report ID D-{diagnosis_id}, but I need to know what you want me to do with it.\n\n"
                "You can ask me to:\n"
                "- Explain diagnosis results\n"
                "- Summarize selected symptoms\n"
                "- Identify red flags\n"
                "- Suggest doctor-reviewed next steps"
            ),
        }

    if not action:
        return {
            "success": True,
            "requires_report_id": False,
            "assistant_response": (
                "I can help with diagnosis reports.\n\n"
                "Ask me something like:\n"
                "- Explain diagnosis results\n"
                "- Summarize selected symptoms\n"
                "- Identify red flags\n"
                "- Suggest doctor-reviewed next steps\n\n"
                "After that, I will ask you for the diagnosis report ID."
            ),
        }

    diagnosis_query = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id)

    if request.doctor_id is not None:
        diagnosis_query = diagnosis_query.filter(
            Diagnosis.doctor_id == request.doctor_id
        )

    diagnosis = diagnosis_query.first()

    if not diagnosis:
        return {
            "success": True,
            "requires_report_id": True,
            "pending_request": request.pending_request or user_message,
            "assistant_response": (
                f"I could not find diagnosis report D-{diagnosis_id}, or it does not belong to your account.\n\n"
                "Please check the Reports page and send the correct diagnosis report ID, for example: D-15."
            ),
        }

    patient = None
    if diagnosis.patient_id:
        patient = db.query(Patient).filter(Patient.id == diagnosis.patient_id).first()

    selected_symptoms = parse_json_list(diagnosis.selected_symptoms_json)

    patient_name = patient.full_name if patient else "N/A"
    symptoms_count = len(selected_symptoms)
    symptoms_text = (
        ", ".join(selected_symptoms) if selected_symptoms else "No symptoms saved"
    )

    context = (
        f"Report D-{diagnosis.id}\n"
        f"Patient: {patient_name} (P-{diagnosis.patient_id or 'N/A'})\n"
        f"Scan Type: {diagnosis.scan_type or 'N/A'}\n"
        f"Final Prediction: {diagnosis.final_prediction or 'N/A'}\n"
        f"Confidence: {diagnosis.final_confidence or 0}%\n"
        f"Risk Level: {diagnosis.risk_level or 'N/A'}\n"
        f"Selected Symptoms: {symptoms_count}"
    )

    disease_steps = {
        "Pneumonia": [
            "Review the scan manually before confirming the result.",
            "Check oxygen saturation and respiratory status.",
            "Consider infection markers or sputum testing when clinically needed.",
            "Assess severity and decide whether urgent care is required.",
        ],
        "Tuberculosis": [
            "Ask about cough duration, TB contact history, night sweats, and weight loss.",
            "Review scan findings carefully.",
            "Consider sputum AFB, culture, or GeneXpert when clinically appropriate.",
            "Use infection-control precautions if TB is clinically suspected.",
        ],
        "Lung Cancer": [
            "Review CT findings carefully.",
            "Check smoking history, weight loss, hemoptysis, and chronic cough.",
            "Consider referral to pulmonology or oncology.",
            "Consider further imaging or biopsy based on doctor judgment.",
        ],
        "Normal": [
            "Review scan quality and patient history.",
            "Check whether symptoms are severe or persistent.",
            "Consider follow-up evaluation if clinical suspicion remains.",
            "Document that the AI result is supportive only.",
        ],
    }

    red_flags = [
        "Severe shortness of breath",
        "Low oxygen saturation",
        "Coughing blood",
        "Severe or persistent chest pain",
        "Confusion or severe weakness",
        "High persistent fever",
        "Rapid clinical deterioration",
    ]

    final_prediction = diagnosis.final_prediction or "N/A"

    if action == "explain":
        explanation = build_clinical_explanation(
            diagnosis=diagnosis,
            selected_symptoms=selected_symptoms,
        )

        response = (
            f"{context}\n\n"
            f"{explanation}\n\n"
            "References are available in the Trusted Medical Knowledge Base section."
        )

    elif action == "symptoms":
        response = (
            f"{context}\n\n"
            "Selected Symptoms Summary:\n"
            f"{symptoms_text}\n\n"
            "These symptoms support the diagnostic reasoning but do not confirm the disease alone. "
            "Symptoms should be interpreted together with the scan, patient history, and doctor examination."
        )

    elif action == "red_flags":
        response = (
            f"{context}\n\n"
            "Red Flags to Check:\n"
            + "\n".join([f"- {item}" for item in red_flags])
            + "\n\nIf any red flag is present, the patient should be reviewed urgently by a qualified doctor."
        )

    elif action == "report_file":
        response = (
            f"{context}\n\n"
            "The diagnosis report file is ready.\n\n"
            "You can open the report preview or download it as a PDF."
        )

    elif action == "next_steps":
        response = (
            f"{context}\n\n"
            "Doctor-Reviewed Next Steps:\n"
            f"{_build_next_steps(diagnosis, selected_symptoms)}"
        )

    else:
        response = (
            f"{context}\n\n"
            "I can explain the result, summarize symptoms, identify red flags, or suggest doctor-reviewed next steps."
        )

    return {
        "success": True,
        "requires_report_id": False,
        "assistant_response": response,
        "context_diagnosis_id": diagnosis.id,
    }


@app.post("/predict-image")
def predict_uploaded_image(scan_type: str = Form(...), image: UploadFile = File(...)):
    image_path = UPLOAD_DIR / image.filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        from app.services.scan_validator import validate_scan_image
        scan_check = validate_scan_image(image_path)
    except Exception as e:
        return {
            "success": False,
            "error_type": "model_not_available_online",
            "message": (
                "Online image validation/prediction is not available on this deployment yet. "
                "Patients, appointments, alerts, replies, login, dashboard, and reports can still work. "
                f"Details: {str(e)}"
            ),
        }

    detected_type = scan_check["detected_type"]
    detected_confidence = scan_check["confidence"]

    if detected_type == "Invalid":
        return {
            "success": False,
            "error_type": "invalid_image",
            "detected_type": detected_type,
            "detected_confidence": detected_confidence,
            "scan_validation": scan_check,
            "message": "Invalid medical image. Please upload a valid chest X-ray or chest CT scan.",
        }

    if detected_type != scan_type:
        return {
            "success": False,
            "error_type": "scan_type_mismatch",
            "selected_type": scan_type,
            "detected_type": detected_type,
            "detected_confidence": detected_confidence,
            "scan_validation": scan_check,
            "message": f"Scan type mismatch. You selected {scan_type}, but the uploaded image appears to be {detected_type}.",
        }

    try:
        from app.services.image_prediction import predict_image
        result = predict_image(scan_type=scan_type, image_path=image_path)
    except Exception as e:
        return {
            "success": False,
            "error_type": "model_not_available_online",
            "scan_validation": scan_check,
            "message": (
                "Online AI image prediction is not available on this deployment yet. "
                "This usually means TensorFlow/model files are not deployed to Railway. "
                f"Details: {str(e)}"
            ),
        }

    result["success"] = True
    result["image_path"] = str(image_path)
    result["scan_validation"] = scan_check

    return result


@app.post("/patients")
def create_patient(request: PatientCreateRequest, db: Session = Depends(get_db)):
    patient = Patient(
        doctor_id=request.doctor_id,
        full_name=request.full_name,
        age=request.age,
        gender=request.gender,
        phone=request.phone,
        national_id=request.national_id,
        email=request.email,
        address=request.address,
        date_of_birth=request.date_of_birth,
        medical_notes=request.medical_notes,
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)

    return {
        "message": "Patient created successfully",
        "patient": {
            "id": patient.id,
            "doctor_id": patient.doctor_id,
            "full_name": patient.full_name,
            "age": patient.age,
            "gender": patient.gender,
            "phone": patient.phone,
            "national_id": patient.national_id,
            "email": patient.email,
            "address": patient.address,
            "date_of_birth": patient.date_of_birth,
            "medical_notes": patient.medical_notes,
            "created_at": patient.created_at,
        },
    }


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()

    if not patient:
        return {
            "success": False,
            "message": "Patient not found.",
        }

    patient.is_deleted = True

    db.commit()

    return {
        "success": True,
        "message": f"Patient P-{patient_id} removed from patient list.",
    }


@app.put("/patients/{patient_id}")
def update_patient(
    patient_id: int,
    request: PatientUpdateRequest,
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()

    if not patient:
        return {
            "success": False,
            "message": "Patient not found.",
        }

    patient.full_name = request.full_name
    patient.age = request.age
    patient.gender = request.gender
    patient.phone = request.phone
    patient.national_id = request.national_id
    patient.email = request.email
    patient.address = request.address
    patient.date_of_birth = request.date_of_birth
    patient.medical_notes = request.medical_notes

    db.commit()
    db.refresh(patient)

    return {
        "success": True,
        "message": f"Patient P-{patient_id} updated successfully.",
        "patient": {
            "doctor_id": patient.doctor_id,
            "id": patient.id,
            "full_name": patient.full_name,
            "age": patient.age,
            "gender": patient.gender,
            "phone": patient.phone,
            "national_id": patient.national_id,
            "medical_notes": patient.medical_notes,
            "created_at": patient.created_at,
            "email": patient.email,
            "address": patient.address,
            "date_of_birth": patient.date_of_birth,
            "is_deleted": patient.is_deleted,
        },
    }


@app.get("/patients")
def get_patients(doctor_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(Patient).filter(Patient.is_deleted == False)

    if doctor_id is not None:
        query = query.filter(Patient.doctor_id == doctor_id)

    patients = query.order_by(Patient.id.desc()).all()

    return {
        "patients": [
            {
                "doctor_id": patient.doctor_id,
                "id": patient.id,
                "full_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "phone": patient.phone,
                "national_id": patient.national_id,
                "medical_notes": patient.medical_notes,
                "created_at": patient.created_at,
                "email": patient.email,
                "address": patient.address,
                "date_of_birth": patient.date_of_birth,
                "is_deleted": patient.is_deleted,
            }
            for patient in patients
        ]
    }


@app.post("/diagnoses")
def create_diagnosis(request: DiagnosisCreateRequest, db: Session = Depends(get_db)):
    diagnosis = Diagnosis(
        patient_id=request.patient_id,
        doctor_id=request.doctor_id,
        doctor_name=request.doctor_name,
        scan_type=request.scan_type,
        image_path=request.image_path,
        heatmap_path=request.heatmap_path,
        image_prediction=request.image_prediction,
        image_confidence=request.image_confidence,
        symptom_prediction=request.symptom_prediction,
        symptom_confidence=request.symptom_confidence,
        final_prediction=request.final_prediction,
        final_confidence=request.final_confidence,
        risk_level=request.risk_level,
        image_scores_json=json.dumps(request.image_scores),
        symptom_scores_json=json.dumps(request.symptom_scores),
        final_scores_json=json.dumps(request.final_scores),
        selected_symptoms_json=json.dumps(request.selected_symptoms),
        doctor_notes=request.doctor_notes,
        assistant_explanation=request.assistant_explanation,
    )

    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)

    return {"message": "Diagnosis saved successfully", "diagnosis_id": diagnosis.id}


@app.get("/diagnoses")
def get_diagnoses(doctor_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(Diagnosis)

    if doctor_id is not None:
        query = query.filter(Diagnosis.doctor_id == doctor_id)

    diagnoses = query.order_by(Diagnosis.id.desc()).all()

    return {
        "diagnoses": [
            {
                "id": d.id,
                "patient_id": d.patient_id,
                "patient_name": d.patient.full_name if d.patient else None,
                "doctor_id": d.doctor_id,
                "doctor_name": d.doctor_name,
                "scan_type": d.scan_type,
                "final_prediction": d.final_prediction,
                "final_confidence": d.final_confidence,
                "risk_level": d.risk_level,
                "created_at": d.created_at,
            }
            for d in diagnoses
        ]
    }


@app.delete("/diagnoses/{diagnosis_id}")
def delete_diagnosis(diagnosis_id: int, db: Session = Depends(get_db)):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()

    if not diagnosis:
        return {
            "success": False,
            "message": "Diagnosis report not found.",
        }

    db.delete(diagnosis)
    db.commit()

    return {
        "success": True,
        "message": f"Report D-{diagnosis_id} deleted successfully.",
    }


@app.post("/patient-alerts")
def create_patient_alert(
    request: PatientAlertCreateRequest, db: Session = Depends(get_db)
):
    patient = (
        db.query(Patient)
        .filter(Patient.id == request.patient_id, Patient.is_deleted == False)
        .first()
    )

    if not patient:
        return {"success": False, "message": "Patient not found"}

    doctor = db.query(Doctor).filter(Doctor.id == request.doctor_id).first()

    if not doctor:
        return {"success": False, "message": "Doctor not found"}

    alert = PatientAlert(
        patient_id=request.patient_id,
        doctor_id=request.doctor_id,
        severity=request.severity,
        message=request.message,
        symptoms_json=json.dumps(request.symptoms),
        status="New",
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return {
        "success": True,
        "message": "Patient alert created successfully",
        "alert_id": alert.id,
    }


@app.get("/patient-alerts")
def get_patient_alerts(
    doctor_id: int | None = None,
    patient_id: int | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(PatientAlert)

    if doctor_id is not None:
        query = query.filter(PatientAlert.doctor_id == doctor_id)

    if patient_id is not None:
        query = query.filter(PatientAlert.patient_id == patient_id)

    alerts = query.order_by(PatientAlert.created_at.desc()).all()

    result = []

    for alert in alerts:
        patient = db.query(Patient).filter(Patient.id == alert.patient_id).first()

        result.append(
            {
                "id": alert.id,
                "patient_id": alert.patient_id,
                "doctor_id": alert.doctor_id,
                "patient_name": (
                    patient.full_name if patient else f"Patient P-{alert.patient_id}"
                ),
                "severity": alert.severity,
                "message": alert.message,
                "symptoms": json.loads(alert.symptoms_json or "[]"),
                "status": alert.status,
                "doctor_reply": alert.doctor_reply,
                "created_at": alert.created_at,
            }
        )

    return {"alerts": result}


@app.put("/patient-alerts/{alert_id}/reply")
def reply_to_patient_alert(
    alert_id: int,
    request: PatientAlertReplyRequest,
    db: Session = Depends(get_db),
):
    alert = db.query(PatientAlert).filter(PatientAlert.id == alert_id).first()

    if not alert:
        return {"success": False, "message": "Alert not found"}

    alert.doctor_reply = request.doctor_reply
    alert.status = "Reviewed"

    db.commit()
    db.refresh(alert)

    return {
        "success": True,
        "message": "Reply sent successfully",
        "alert": {
            "id": alert.id,
            "status": alert.status,
            "doctor_reply": alert.doctor_reply,
        },
    }


@app.put("/patient-alerts/{alert_id}/status")
def update_patient_alert_status(
    alert_id: int,
    request: PatientAlertStatusRequest,
    db: Session = Depends(get_db),
):
    alert = db.query(PatientAlert).filter(PatientAlert.id == alert_id).first()

    if not alert:
        return {"success": False, "message": "Alert not found"}

    if request.status not in ["New", "Reviewed"]:
        return {"success": False, "message": "Invalid alert status"}

    alert.status = request.status

    db.commit()
    db.refresh(alert)

    return {
        "success": True,
        "message": "Alert status updated successfully",
        "status": alert.status,
    }


@app.post("/appointments")
def create_appointment(
    request: AppointmentCreateRequest, db: Session = Depends(get_db)
):
    patient = (
        db.query(Patient)
        .filter(Patient.id == request.patient_id, Patient.is_deleted == False)
        .first()
    )

    if not patient:
        return {
            "success": False,
            "message": "Patient not found",
        }

    doctor = db.query(Doctor).filter(Doctor.id == request.doctor_id).first()

    if not doctor:
        return {
            "success": False,
            "message": "Doctor not found",
        }

    appointment = Appointment(
        patient_id=request.patient_id,
        doctor_id=request.doctor_id,
        requested_date=request.requested_date,
        requested_time=request.requested_time,
        reason=request.reason,
        symptoms_json=json.dumps(request.symptoms),
        status="Pending",
    )

    db.add(appointment)
    db.commit()
    db.refresh(appointment)

    return {
        "success": True,
        "message": "Appointment request sent successfully",
        "appointment_id": appointment.id,
    }


@app.get("/appointments")
def get_appointments(
    doctor_id: int | None = None,
    patient_id: int | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Appointment)

    if doctor_id is not None:
        query = query.filter(Appointment.doctor_id == doctor_id)

    if patient_id is not None:
        query = query.filter(Appointment.patient_id == patient_id)

    appointments = query.order_by(Appointment.created_at.desc()).all()

    return {
        "appointments": [
            {
                "id": appointment.id,
                "patient_id": appointment.patient_id,
                "doctor_id": appointment.doctor_id,
                "patient_name": (
                    appointment.patient.full_name
                    if appointment.patient
                    else "Unknown Patient"
                ),
                "requested_date": appointment.requested_date,
                "requested_time": appointment.requested_time,
                "reason": appointment.reason,
                "symptoms": json.loads(appointment.symptoms_json or "[]"),
                "status": appointment.status,
                "doctor_response": appointment.doctor_response,
                "created_at": appointment.created_at,
            }
            for appointment in appointments
        ]
    }


@app.put("/appointments/{appointment_id}/status")
def update_appointment_status(
    appointment_id: int,
    request: AppointmentStatusUpdateRequest,
    db: Session = Depends(get_db),
):
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()

    if not appointment:
        return {
            "success": False,
            "message": "Appointment not found",
        }

    allowed_statuses = ["Pending", "Approved", "Declined"]

    if request.status not in allowed_statuses:
        return {
            "success": False,
            "message": "Invalid appointment status",
        }

    appointment.status = request.status
    appointment.doctor_response = request.doctor_response

    db.commit()
    db.refresh(appointment)

    return {
        "success": True,
        "message": f"Appointment {request.status.lower()} successfully",
        "appointment": {
            "id": appointment.id,
            "status": appointment.status,
            "doctor_response": appointment.doctor_response,
        },
    }


@app.get("/dashboard/stats")
def get_dashboard_stats(doctor_id: int | None = None, db: Session = Depends(get_db)):
    total_patients = db.query(Patient).count()
    total_diagnoses = db.query(Diagnosis).count()
    high_risk_cases = db.query(Diagnosis).filter(Diagnosis.risk_level == "High").count()

    doctor_patients_query = db.query(Patient)
    doctor_diagnoses_query = db.query(Diagnosis)

    if doctor_id is not None:
        doctor_patients_query = doctor_patients_query.filter(
            Patient.doctor_id == doctor_id
        )
        doctor_diagnoses_query = doctor_diagnoses_query.filter(
            Diagnosis.doctor_id == doctor_id
        )

    doctor_patients = doctor_patients_query.count()
    doctor_diagnoses_count = doctor_diagnoses_query.count()

    doctor_high_risk = doctor_diagnoses_query.filter(
        Diagnosis.risk_level == "High"
    ).count()

    doctor_xray_cases = doctor_diagnoses_query.filter(
        Diagnosis.scan_type == "X-ray"
    ).count()

    doctor_ct_cases = doctor_diagnoses_query.filter(
        Diagnosis.scan_type == "CT Scan"
    ).count()

    doctor_diagnoses_list = doctor_diagnoses_query.all()

    if doctor_diagnoses_list:
        confidence_values = [
            d.final_confidence
            for d in doctor_diagnoses_list
            if d.final_confidence is not None
        ]

        doctor_average_confidence = (
            round(sum(confidence_values) / len(confidence_values), 2)
            if confidence_values
            else 0
        )

        diagnosis_counter = Counter(
            _normalize_prediction_label(d.final_prediction or "Unknown")
            for d in doctor_diagnoses_list
            if d.final_prediction
        )

        doctor_most_common_diagnosis = (
            diagnosis_counter.most_common(1)[0][0] if diagnosis_counter else "N/A"
        )

        doctor_diagnosis_distribution = [
            {"label": label, "count": count}
            for label, count in diagnosis_counter.most_common()
        ]
    else:
        doctor_average_confidence = 0
        doctor_most_common_diagnosis = "N/A"
        doctor_diagnosis_distribution = []

    recent_diagnoses = (
        doctor_diagnoses_query.order_by(Diagnosis.created_at.desc()).limit(10).all()
    )

    return {
        "total_patients": total_patients,
        "total_diagnoses": total_diagnoses,
        "high_risk_cases": high_risk_cases,
        "reports_generated": total_diagnoses,
        "doctor_patients": doctor_patients,
        "doctor_diagnoses": doctor_diagnoses_count,
        "doctor_high_risk": doctor_high_risk,
        "doctor_reports": doctor_diagnoses_count,
        "doctor_average_confidence": doctor_average_confidence,
        "doctor_most_common_diagnosis": doctor_most_common_diagnosis,
        "doctor_xray_cases": doctor_xray_cases,
        "doctor_ct_cases": doctor_ct_cases,
        "doctor_diagnosis_distribution": doctor_diagnosis_distribution,
        "recent_diagnoses": [
            {
                "id": d.id,
                "patient_id": d.patient_id,
                "patient_name": d.patient.full_name if d.patient else "Unknown Patient",
                "patient_name": d.patient.full_name if d.patient else None,
                "doctor_id": d.doctor_id,
                "doctor_name": d.doctor_name,
                "scan_type": d.scan_type,
                "final_prediction": d.final_prediction,
                "final_confidence": d.final_confidence,
                "risk_level": d.risk_level,
                "created_at": d.created_at,
            }
            for d in recent_diagnoses
        ],
    }


@app.get("/diagnoses/{diagnosis_id}/report")
def generate_pdf_report(
    diagnosis_id: int,
    download: bool = False,
    db: Session = Depends(get_db),
):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import inch
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>PDF engine is not available on this deployment.</h1><p>{escape(str(e))}</p>",
            status_code=503,
        )

    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()

    if not diagnosis:
        return {"error": "Diagnosis not found"}

    patient = None
    if diagnosis.patient_id:
        patient = db.query(Patient).filter(Patient.id == diagnosis.patient_id).first()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    pdf_path = reports_dir / f"diagnosis_report_{diagnosis.id}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=24,
        leftMargin=24,
        topMargin=22,
        bottomMargin=22,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.fontSize = 18
    title_style.leading = 22
    title_style.textColor = colors.HexColor("#0F172A")

    subtitle_style = styles["BodyText"]
    subtitle_style.fontSize = 8
    subtitle_style.leading = 10
    subtitle_style.textColor = colors.HexColor("#475569")

    heading_style = styles["Heading3"]
    heading_style.fontSize = 10
    heading_style.leading = 12
    heading_style.textColor = colors.HexColor("#0369A1")

    body_style = styles["BodyText"]
    body_style.fontSize = 8
    body_style.leading = 10

    small_style = styles["BodyText"]
    small_style.fontSize = 7
    small_style.leading = 9

    story = []

    def safe_text(value, fallback="N/A"):
        return str(value) if value not in [None, ""] else fallback

    def make_image(path_value, width=0.85 * inch, height=0.85 * inch):
        if not path_value:
            return Paragraph("Image not available", small_style)

        path = Path(path_value)

        if path.exists():
            return RLImage(str(path), width=width, height=height)

        return Paragraph("Image not available", small_style)

    try:
        selected_symptoms = json.loads(diagnosis.selected_symptoms_json or "[]")
    except Exception:
        selected_symptoms = []

    try:
        final_scores = json.loads(diagnosis.final_scores_json or "{}")
    except Exception:
        final_scores = {}

    symptoms_text = (
        ", ".join(selected_symptoms) if selected_symptoms else "No symptoms saved."
    )

    scores_text = "<br/>".join(
        [f"{label}: {round(value, 2)}%" for label, value in final_scores.items()]
    )

    if not scores_text:
        scores_text = "No fusion scores saved."

    original_img = make_image(diagnosis.image_path)

    if diagnosis.image_prediction == "Normal":
        second_image_title = "Normal Scan"
        second_img = make_image(diagnosis.image_path)
    else:
        second_image_title = "AI Heatmap"
        second_img = make_image(diagnosis.heatmap_path)

    header_table = Table(
        [
            [
                Paragraph(
                    "<b>AI-Assisted Respiratory Diagnosis Report</b>", title_style
                ),
                Paragraph(f"<b>Report D-{diagnosis.id}</b>", body_style),
            ],
            [
                Paragraph(
                    "Clinical Decision Support Report - Doctor Review Required",
                    subtitle_style,
                ),
                Paragraph(str(diagnosis.created_at).split(".")[0], subtitle_style),
            ],
        ],
        colWidths=[390, 130],
    )

    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#DBEAFE")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#93C5FD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(header_table)
    story.append(Spacer(1, 8))

    patient_info = Table(
        [
            [
                "Patient Name",
                safe_text(patient.full_name if patient else None),
                "Patient ID",
                f"P-{safe_text(diagnosis.patient_id)}",
            ],
            [
                "Age / Gender",
                f"{safe_text(patient.age if patient else None)} / {safe_text(patient.gender if patient else None)}",
                "Doctor",
                safe_text(diagnosis.doctor_name),
            ],
        ],
        colWidths=[75, 115, 70, 120],
    )

    patient_info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    image_table = Table(
        [
            ["Original", second_image_title],
            [original_img, second_img],
        ],
        colWidths=[65, 65],
    )

    image_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2FE")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("PADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    top_table = Table(
        [[patient_info, image_table]],
        colWidths=[380, 140],
    )

    top_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(Paragraph("Patient & Doctor Information", heading_style))
    story.append(top_table)
    story.append(Spacer(1, 8))

    summary_table = Table(
        [
            [
                "Final Prediction",
                safe_text(diagnosis.final_prediction),
                "Confidence",
                f"{safe_text(diagnosis.final_confidence)}%",
                "Risk",
                safe_text(diagnosis.risk_level),
            ],
            [
                "Scan Type",
                safe_text(diagnosis.scan_type),
                "Image Prediction",
                safe_text(diagnosis.image_prediction),
                "Image Confidence",
                f"{safe_text(diagnosis.image_confidence)}%",
            ],
        ],
        colWidths=[75, 110, 75, 95, 65, 85],
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTNAME", (4, 0), (4, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(Paragraph("Diagnosis Summary", heading_style))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    symptom_scores_table = Table(
        [
            [
                Paragraph("<b>Selected Symptoms</b>", body_style),
                Paragraph("<b>Final Fusion Scores</b>", body_style),
            ],
            [
                Paragraph(symptoms_text, body_style),
                Paragraph(scores_text, body_style),
            ],
        ],
        colWidths=[270, 270],
    )

    symptom_scores_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2FE")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(symptom_scores_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Virtual Assistant Explanation", heading_style))

    assistant_text = (
        diagnosis.assistant_explanation or "No assistant explanation saved."
    )

    story.append(
        Table(
            [[Paragraph(assistant_text, body_style)]],
            colWidths=[540],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        )
    )

    story.append(Spacer(1, 8))

    disclaimer = (
        "This AI report is for clinical decision support only. It does not replace radiology review, "
        "physical examination, laboratory tests, or doctor judgment. The final diagnosis must be confirmed "
        "by a qualified doctor."
    )

    story.append(
        Table(
            [[Paragraph(disclaimer, small_style)]],
            colWidths=[540],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#FDBA74")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        )
    )

    story.append(Spacer(1, 10))

    signature_table = Table(
        [
            [
                "Doctor Review",
                "________________________",
                "Signature",
                "________________________",
            ],
        ],
        colWidths=[90, 160, 70, 200],
    )

    signature_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(signature_table)

    doc.build(story)

    return FileResponse(
        path=str(pdf_path),
        filename=f"diagnosis_report_{diagnosis.id}.pdf",
        media_type="application/pdf",
        content_disposition_type="attachment" if download else "inline",
    )


def build_report_html(diagnosis: Diagnosis, patient: Patient | None):
    def safe(value, fallback="N/A"):
        if value is None or value == "":
            return escape(str(fallback))
        return escape(str(value))

    def percent(value):
        try:
            number = float(value or 0)
        except Exception:
            number = 0.0
        return f"{round(number, 2)}%"

    def parse_list(value):
        try:
            parsed = json.loads(value or "[]")
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def parse_dict(value):
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def image_block(path, route, alt):
        if not path:
            return '<div class="image-empty">No image available</div>'
        filename = Path(path).name
        return f'<img src="/{route}/{escape(filename)}" alt="{escape(alt)}">'

    def paragraph_text(value):
        clean = safe(value, "No assistant explanation saved.")
        clean = clean.replace("\n\n", "</p><p>")
        clean = clean.replace("\n", "<br>")
        return f"<p>{clean}</p>"

    selected_symptoms = parse_list(diagnosis.selected_symptoms_json)
    final_scores = parse_dict(diagnosis.final_scores_json)
    image_scores = parse_dict(diagnosis.image_scores_json)
    symptom_scores = parse_dict(diagnosis.symptom_scores_json)
    disease_order = ["Pneumonia", "Tuberculosis", "Lung Cancer", "Normal"]

    def score_rows(scores):
        if not scores:
            return '<div class="empty-card">No scores saved.</div>'
        ordered = [(label, scores[label]) for label in disease_order if label in scores]
        ordered.extend(
            (label, value)
            for label, value in scores.items()
            if label not in disease_order
        )
        rows = []
        for label, value in ordered:
            try:
                width = max(0, min(100, float(value or 0)))
            except Exception:
                width = 0
            rows.append(
                f'<div class="score-row"><div class="score-top"><span>{safe(label)}</span><strong>{percent(value)}</strong></div><div class="score-track"><div class="score-fill" style="width:{width}%"></div></div></div>'
            )
        return "".join(rows)

    symptoms_html = (
        "".join(
            f'<span class="symptom-chip">{safe(symptom)}</span>'
            for symptom in selected_symptoms
        )
        or '<span class="muted-text">No symptoms saved.</span>'
    )

    risk_class = re.sub(
        r"[^a-z0-9-]+", "-", str(diagnosis.risk_level or "medium").lower()
    ).strip("-")
    if "review" in risk_class:
        risk_class = "review-required"

    original_image_html = image_block(diagnosis.image_path, "uploads", "Original scan")
    heatmap_title = "AI Attention Heatmap"
    heatmap_note = "The heatmap highlights image regions that influenced the model. It is not a confirmed disease location."

    if str(diagnosis.image_prediction or "").lower() == "normal":
        heatmap_title = "Normal Scan Review"
        heatmap_note = "The image model predicted a normal scan, so no abnormal heatmap is displayed."
        heatmap_image_html = original_image_html
    else:
        heatmap_image_html = image_block(
            diagnosis.heatmap_path, "heatmaps", "AI heatmap"
        )

    report_date = str(diagnosis.created_at).split(".")[0]
    patient_name = patient.full_name if patient else "N/A"
    patient_age = patient.age if patient and patient.age is not None else "N/A"
    patient_gender = patient.gender if patient and patient.gender else "N/A"
    doctor_name = diagnosis.doctor_name or "N/A"
    final_prediction = diagnosis.final_prediction or "N/A"
    final_confidence = percent(diagnosis.final_confidence)
    image_confidence = percent(diagnosis.image_confidence)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Diagnosis Report D-{diagnosis.id}</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; padding: 28px; font-family: Arial, sans-serif; background: #e5e7eb; color: #0f172a; }}
            .report {{ width: 100%; max-width: 1020px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18); }}
            .hero {{ background: linear-gradient(135deg, #075985, #0f172a 62%); color: #ffffff; padding: 30px 36px; display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }}
            .hero h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.5px; }}
            .hero p {{ margin: 8px 0 0; color: #bae6fd; font-size: 15px; }}
            .report-id {{ min-width: 130px; text-align: center; padding: 12px 16px; border: 1px solid rgba(255, 255, 255, 0.28); background: rgba(255, 255, 255, 0.12); border-radius: 999px; font-weight: 900; }}
            .content {{ padding: 30px 36px 34px; }}
            .clinical-banner {{ display: grid; grid-template-columns: 1.3fr 0.7fr 0.7fr; gap: 14px; margin-bottom: 24px; }}
            .summary-card {{ border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 18px; padding: 18px; }}
            .summary-card span, .info-card span, .scan-card span {{ display: block; color: #64748b; font-size: 12px; font-weight: 800; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.03em; }}
            .summary-card strong {{ display: block; color: #0f172a; font-size: 28px; line-height: 1.1; }}
            .confidence-value {{ color: #0369a1 !important; }}
            .risk {{ display: inline-flex; align-items: center; justify-content: center; min-width: 94px; padding: 9px 14px; border-radius: 999px; font-weight: 900; font-size: 14px; }}
            .risk.high {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}
            .risk.medium {{ background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}
            .risk.low {{ background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }}
            .risk.review-required {{ background: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }}
            .section {{ margin-top: 24px; }}
            .section-title {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; padding-bottom: 10px; margin-bottom: 14px; border-bottom: 2px solid #e0f2fe; }}
            .section-title h2 {{ margin: 0; font-size: 18px; color: #0369a1; }}
            .info-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
            .info-card, .scan-card, .text-box, .score-box {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 16px; padding: 15px; }}
            .info-card strong, .scan-card strong {{ color: #0f172a; font-size: 15px; line-height: 1.35; }}
            .scan-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
            .scan-image-card {{ border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 18px; padding: 14px; }}
            .scan-image-card h3 {{ margin: 0 0 10px; color: #0369a1; font-size: 15px; }}
            .scan-image-card img {{ width: 100%; height: 260px; object-fit: contain; background: #020617; border-radius: 14px; border: 1px solid #cbd5e1; }}
            .image-empty {{ height: 260px; display: flex; align-items: center; justify-content: center; color: #64748b; background: #f1f5f9; border-radius: 14px; border: 1px dashed #cbd5e1; }}
            .image-note {{ margin: 10px 0 0; color: #475569; font-size: 13px; line-height: 1.5; }}
            .scan-meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px; }}
            .symptom-box {{ display: flex; flex-wrap: wrap; gap: 8px; }}
            .symptom-chip {{ display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; background: #e0f2fe; color: #0369a1; font-size: 13px; font-weight: 900; }}
            .scores-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
            .score-box h3 {{ margin: 0 0 14px; color: #0f172a; font-size: 15px; }}
            .score-row {{ margin-bottom: 12px; }}
            .score-top {{ display: flex; justify-content: space-between; gap: 12px; color: #334155; font-size: 13px; margin-bottom: 6px; }}
            .score-top strong {{ color: #0284c7; }}
            .score-track {{ height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }}
            .score-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0284c7, #22c55e); }}
            .text-box {{ line-height: 1.75; color: #334155; font-size: 14px; }}
            .text-box p {{ margin: 0 0 12px; }}
            .warning {{ margin-top: 24px; background: #fff7ed; border: 1px solid #fdba74; color: #9a3412; padding: 16px; border-radius: 16px; line-height: 1.6; font-weight: 700; }}
            .footer {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-top: 26px; padding-top: 18px; border-top: 1px solid #e2e8f0; }}
            .signature {{ color: #475569; font-size: 14px; }}
            .btn {{ display: inline-flex; align-items: center; justify-content: center; padding: 12px 18px; background: #0284c7; color: #ffffff; border-radius: 12px; text-decoration: none; font-weight: 900; border: none; }}
            .muted-text {{ color: #64748b; }}
            .empty-card {{ padding: 14px; border-radius: 12px; background: #f1f5f9; color: #64748b; text-align: center; }}
            @page {{ size: A4; margin: 7mm; }}
            @media print {{ body {{ background: #ffffff !important; padding: 0 !important; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }} .no-print {{ display: none !important; }} .report {{ box-shadow: none !important; border-radius: 0 !important; max-width: 100% !important; }} .content {{ padding: 22px 28px 26px !important; }} .hero {{ padding: 24px 28px !important; }} .scan-image-card img, .image-empty {{ height: 210px !important; }} .section {{ break-inside: avoid; margin-top: 18px !important; }} }}
        </style>
    </head>
    <body>
        <article class="report">
            <header class="hero">
                <div>
                    <h1>AI-Assisted Respiratory Diagnosis Report</h1>
                    <p>Clinical decision support report requiring doctor review and confirmation</p>
                </div>
                <div class="report-id">Report D-{diagnosis.id}</div>
            </header>
            <main class="content">
                <section class="clinical-banner">
                    <div class="summary-card"><span>Final Prediction</span><strong>{safe(final_prediction)}</strong></div>
                    <div class="summary-card"><span>Confidence</span><strong class="confidence-value">{final_confidence}</strong></div>
                    <div class="summary-card"><span>Risk Level</span><span class="risk {risk_class}">{safe(diagnosis.risk_level)}</span></div>
                </section>
                <section class="section">
                    <div class="section-title"><h2>Patient & Doctor Information</h2></div>
                    <div class="info-grid">
                        <div class="info-card"><span>Patient Name</span><strong>{safe(patient_name)}</strong></div>
                        <div class="info-card"><span>Patient ID</span><strong>P-{safe(diagnosis.patient_id)}</strong></div>
                        <div class="info-card"><span>Age / Gender</span><strong>{safe(patient_age)} / {safe(patient_gender)}</strong></div>
                        <div class="info-card"><span>Doctor</span><strong>{safe(doctor_name)}</strong></div>
                    </div>
                </section>
                <section class="section">
                    <div class="section-title"><h2>Scan Review</h2></div>
                    <div class="scan-grid">
                        <div class="scan-image-card"><h3>Original Scan</h3>{original_image_html}</div>
                        <div class="scan-image-card"><h3>{safe(heatmap_title)}</h3>{heatmap_image_html}<p class="image-note">{safe(heatmap_note)}</p></div>
                    </div>
                    <div class="scan-meta-grid">
                        <div class="scan-card"><span>Scan Type</span><strong>{safe(diagnosis.scan_type)}</strong></div>
                        <div class="scan-card"><span>Image Prediction</span><strong>{safe(diagnosis.image_prediction)}</strong></div>
                        <div class="scan-card"><span>Image Confidence</span><strong>{image_confidence}</strong></div>
                        <div class="scan-card"><span>Report Date</span><strong>{safe(report_date)}</strong></div>
                    </div>
                </section>
                <section class="section">
                    <div class="section-title"><h2>Selected Symptoms</h2></div>
                    <div class="text-box symptom-box">{symptoms_html}</div>
                </section>
                <section class="section">
                    <div class="section-title"><h2>Model Probability Review</h2></div>
                    <div class="scores-grid">
                        <div class="score-box"><h3>Final Fusion Scores</h3>{score_rows(final_scores)}</div>
                        <div class="score-box"><h3>Image Model Scores</h3>{score_rows(image_scores)}</div>
                        <div class="score-box"><h3>Symptom Model Scores</h3>{score_rows(symptom_scores)}</div>
                    </div>
                </section>
                <section class="section">
                    <div class="section-title"><h2>Virtual Assistant Explanation</h2></div>
                    <div class="text-box">{paragraph_text(diagnosis.assistant_explanation)}</div>
                </section>
                <div class="warning">This AI report is for clinical decision support only. It does not replace radiology review, physical examination, laboratory tests, or doctor judgment. The final diagnosis must be confirmed by a qualified doctor.</div>
                <footer class="footer">
                    <div class="signature">Doctor Review: ____________________________</div>
                    <a class="btn no-print" href="/diagnoses/{diagnosis.id}/html-pdf" target="_blank">Download PDF</a>
                </footer>
            </main>
        </article>
    </body>
    </html>
    """


@app.get("/diagnoses/{diagnosis_id}/preview", response_class=HTMLResponse)
def preview_report(diagnosis_id: int, db: Session = Depends(get_db)):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()

    if not diagnosis:
        return HTMLResponse("<h1>Diagnosis not found</h1>", status_code=404)

    patient = None
    if diagnosis.patient_id:
        patient = db.query(Patient).filter(Patient.id == diagnosis.patient_id).first()

    html = build_report_html(diagnosis, patient)

    return HTMLResponse(content=html)


@app.get("/diagnoses/{diagnosis_id}/html-pdf")
def download_html_pdf(diagnosis_id: int, db: Session = Depends(get_db)):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return HTMLResponse(
            content=(
                "<h1>HTML PDF export is not available on this online deployment.</h1>"
                f"<p>{escape(str(e))}</p>"
                "<p>Use the report preview or the ReportLab PDF endpoint instead.</p>"
            ),
            status_code=503,
        )

    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()

    if not diagnosis:
        return HTMLResponse("<h1>Diagnosis not found</h1>", status_code=404)

    patient = None
    if diagnosis.patient_id:
        patient = db.query(Patient).filter(Patient.id == diagnosis.patient_id).first()

    html = build_report_html(diagnosis, patient)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.set_content(html, wait_until="networkidle")

            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={
                    "top": "6mm",
                    "right": "6mm",
                    "bottom": "6mm",
                    "left": "6mm",
                },
            )

            browser.close()
    except Exception as e:
        return HTMLResponse(
            content=(
                "<h1>HTML PDF export failed on this online deployment.</h1>"
                f"<p>{escape(str(e))}</p>"
                "<p>This is usually because Playwright browsers are not installed on Railway.</p>"
            ),
            status_code=503,
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="diagnosis_report_{diagnosis.id}.pdf"'
        },
    )


@app.get("/diagnoses/{diagnosis_id}")
def get_diagnosis_details(diagnosis_id: int, db: Session = Depends(get_db)):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()

    if not diagnosis:
        return {"error": "Diagnosis not found"}

    patient = None
    if diagnosis.patient_id:
        patient = db.query(Patient).filter(Patient.id == diagnosis.patient_id).first()

    return {
        "id": diagnosis.id,
        "patient": {
            "id": patient.id if patient else None,
            "full_name": patient.full_name if patient else "N/A",
            "age": patient.age if patient else None,
            "gender": patient.gender if patient else "N/A",
            "phone": patient.phone if patient else "N/A",
            "national_id": patient.national_id if patient else "N/A",
        },
        "scan_type": diagnosis.scan_type,
        "image_path": diagnosis.image_path,
        "heatmap_path": diagnosis.heatmap_path,
        "image_prediction": diagnosis.image_prediction,
        "image_confidence": diagnosis.image_confidence,
        "symptom_prediction": diagnosis.symptom_prediction,
        "symptom_confidence": diagnosis.symptom_confidence,
        "final_prediction": diagnosis.final_prediction,
        "final_confidence": diagnosis.final_confidence,
        "risk_level": diagnosis.risk_level,
        "image_scores": diagnosis.image_scores_json,
        "symptom_scores": diagnosis.symptom_scores_json,
        "final_scores": diagnosis.final_scores_json,
        "selected_symptoms": diagnosis.selected_symptoms_json,
        "assistant_explanation": diagnosis.assistant_explanation,
        "doctor_notes": diagnosis.doctor_notes,
        "created_at": diagnosis.created_at,
    }


@app.post("/auth/register")
def register_doctor(request: DoctorRegisterRequest, db: Session = Depends(get_db)):
    if not request.full_name.strip():
        return {
            "success": False,
            "message": "Full name is required",
        }

    if not request.email.strip():
        return {
            "success": False,
            "message": "Email is required",
        }

    if not (request.specialization or "").strip():
        return {
            "success": False,
            "message": "Specialization is required",
        }

    if not (request.hospital_name or "").strip():
        return {
            "success": False,
            "message": "Hospital or clinic name is required",
        }

    if not request.date_of_birth:
        return {
            "success": False,
            "message": "Date of birth is required",
        }

    if not request.gender:
        return {
            "success": False,
            "message": "Gender is required",
        }

    if request.gender not in ["Male", "Female"]:
        return {
            "success": False,
            "message": "Invalid gender value",
        }

    if len(request.password) < 8:
        return {
            "success": False,
            "message": "Password must be at least 8 characters",
        }

    existing_doctor = (
        db.query(Doctor).filter(Doctor.email == request.email.strip()).first()
    )

    if existing_doctor:
        return {
            "success": False,
            "message": "Email is already registered",
        }

    password_hash = pwd_context.hash(request.password)

    doctor = Doctor(
        full_name=request.full_name.strip(),
        email=request.email.strip(),
        password_hash=password_hash,
        specialization=(
            request.specialization.strip() if request.specialization else None
        ),
        hospital_name=request.hospital_name.strip() if request.hospital_name else None,
        date_of_birth=request.date_of_birth,
        gender=request.gender,
    )

    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    return {
        "success": True,
        "message": "Doctor registered successfully",
        "doctor": {
            "id": doctor.id,
            "full_name": doctor.full_name,
            "email": doctor.email,
            "specialization": doctor.specialization,
            "hospital_name": doctor.hospital_name,
            "date_of_birth": doctor.date_of_birth,
            "gender": doctor.gender,
        },
    }


@app.put("/doctors/{doctor_id}/password")
def change_doctor_password(
    doctor_id: int,
    request: DoctorPasswordChangeRequest,
    db: Session = Depends(get_db),
):
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()

    if not doctor:
        return {
            "success": False,
            "message": "Doctor not found",
        }

    if not pwd_context.verify(request.current_password, doctor.password_hash):
        return {
            "success": False,
            "message": "Current password is incorrect",
        }

    if len(request.new_password) < 8:
        return {
            "success": False,
            "message": "New password must be at least 8 characters",
        }

    if request.new_password != request.confirm_password:
        return {
            "success": False,
            "message": "New password and confirm password do not match",
        }

    doctor.password_hash = pwd_context.hash(request.new_password)

    db.commit()

    return {
        "success": True,
        "message": "Password changed successfully",
    }


@app.put("/doctors/{doctor_id}")
def update_doctor_profile(
    doctor_id: int,
    request: DoctorUpdateRequest,
    db: Session = Depends(get_db),
):
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()

    if not doctor:
        return {
            "success": False,
            "message": "Doctor not found",
        }

    if not request.full_name.strip():
        return {
            "success": False,
            "message": "Doctor name is required",
        }

    if not request.email.strip():
        return {
            "success": False,
            "message": "Email is required",
        }

    if not (request.specialization or "").strip():
        return {
            "success": False,
            "message": "Specialization is required",
        }

    if not (request.hospital_name or "").strip():
        return {
            "success": False,
            "message": "Hospital or clinic name is required",
        }

    if request.gender and request.gender not in ["Male", "Female"]:
        return {
            "success": False,
            "message": "Invalid gender value",
        }

    existing_email = (
        db.query(Doctor)
        .filter(Doctor.email == request.email.strip(), Doctor.id != doctor_id)
        .first()
    )

    if existing_email:
        return {
            "success": False,
            "message": "Email is already used by another doctor",
        }

    doctor.full_name = request.full_name.strip()
    doctor.email = request.email.strip()
    doctor.specialization = (
        request.specialization.strip() if request.specialization else None
    )
    doctor.hospital_name = (
        request.hospital_name.strip() if request.hospital_name else None
    )
    doctor.date_of_birth = request.date_of_birth
    doctor.gender = request.gender

    db.commit()
    db.refresh(doctor)

    return {
        "success": True,
        "message": "Doctor profile updated successfully",
        "doctor": {
            "id": doctor.id,
            "full_name": doctor.full_name,
            "email": doctor.email,
            "specialization": doctor.specialization,
            "hospital_name": doctor.hospital_name,
            "date_of_birth": doctor.date_of_birth,
            "gender": doctor.gender,
        },
    }


@app.post("/auth/login")
def login_doctor(request: DoctorLoginRequest, db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == request.email.strip()).first()

    if not doctor:
        return {
            "success": False,
            "message": "Invalid email or password",
        }

    password_valid = pwd_context.verify(request.password, doctor.password_hash)

    if not password_valid:
        return {
            "success": False,
            "message": "Invalid email or password",
        }

    return {
        "success": True,
        "message": "Login successful",
        "doctor": {
            "id": doctor.id,
            "full_name": doctor.full_name,
            "email": doctor.email,
            "specialization": doctor.specialization,
            "hospital_name": doctor.hospital_name,
            "date_of_birth": doctor.date_of_birth,
            "gender": doctor.gender,
        },
    }
