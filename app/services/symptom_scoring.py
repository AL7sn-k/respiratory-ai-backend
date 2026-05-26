import math


DISEASES = ["Pneumonia", "Tuberculosis", "Lung Cancer", "Normal"]

PRIORS = {
    "Pneumonia": 0.25,
    "Tuberculosis": 0.25,
    "Lung Cancer": 0.25,
    "Normal": 0.25,
}

DEFAULT_LIKELIHOOD = {
    "Pneumonia": 0.04,
    "Tuberculosis": 0.04,
    "Lung Cancer": 0.04,
    "Normal": 0.04,
}

SYMPTOM_LIKELIHOODS = {
    "mild cough": {
        "Pneumonia": 0.35,
        "Tuberculosis": 0.18,
        "Lung Cancer": 0.15,
        "Normal": 0.70,
    },
    "cough": {
        "Pneumonia": 0.75,
        "Tuberculosis": 0.45,
        "Lung Cancer": 0.35,
        "Normal": 0.35,
    },
    "persistent cough": {
        "Pneumonia": 0.35,
        "Tuberculosis": 0.75,
        "Lung Cancer": 0.65,
        "Normal": 0.08,
    },
    "cough lasting 3 weeks or more": {
        "Pneumonia": 0.18,
        "Tuberculosis": 0.85,
        "Lung Cancer": 0.55,
        "Normal": 0.03,
    },
    "chronic cough": {
        "Pneumonia": 0.20,
        "Tuberculosis": 0.55,
        "Lung Cancer": 0.75,
        "Normal": 0.04,
    },
    "productive cough / sputum": {
        "Pneumonia": 0.65,
        "Tuberculosis": 0.45,
        "Lung Cancer": 0.25,
        "Normal": 0.08,
    },
    "blood in sputum": {
        "Pneumonia": 0.08,
        "Tuberculosis": 0.60,
        "Lung Cancer": 0.65,
        "Normal": 0.01,
    },
    "cough that gets worse": {
        "Pneumonia": 0.30,
        "Tuberculosis": 0.40,
        "Lung Cancer": 0.70,
        "Normal": 0.04,
    },
    "wheezing": {
        "Pneumonia": 0.35,
        "Tuberculosis": 0.18,
        "Lung Cancer": 0.45,
        "Normal": 0.12,
    },
    "hoarseness": {
        "Pneumonia": 0.12,
        "Tuberculosis": 0.12,
        "Lung Cancer": 0.55,
        "Normal": 0.10,
    },
    "shortness of breath": {
        "Pneumonia": 0.60,
        "Tuberculosis": 0.35,
        "Lung Cancer": 0.55,
        "Normal": 0.05,
    },
    "difficulty breathing": {
        "Pneumonia": 0.55,
        "Tuberculosis": 0.25,
        "Lung Cancer": 0.40,
        "Normal": 0.03,
    },
    "rapid breathing": {
        "Pneumonia": 0.55,
        "Tuberculosis": 0.18,
        "Lung Cancer": 0.25,
        "Normal": 0.02,
    },
    "chest pain": {
        "Pneumonia": 0.45,
        "Tuberculosis": 0.40,
        "Lung Cancer": 0.55,
        "Normal": 0.05,
    },
    "chest pain when breathing or coughing": {
        "Pneumonia": 0.55,
        "Tuberculosis": 0.35,
        "Lung Cancer": 0.35,
        "Normal": 0.03,
    },
    "fever": {
        "Pneumonia": 0.75,
        "Tuberculosis": 0.50,
        "Lung Cancer": 0.12,
        "Normal": 0.10,
    },
    "chills": {
        "Pneumonia": 0.60,
        "Tuberculosis": 0.30,
        "Lung Cancer": 0.05,
        "Normal": 0.08,
    },
    "night sweats": {
        "Pneumonia": 0.15,
        "Tuberculosis": 0.70,
        "Lung Cancer": 0.15,
        "Normal": 0.03,
    },
    "fatigue": {
        "Pneumonia": 0.55,
        "Tuberculosis": 0.55,
        "Lung Cancer": 0.50,
        "Normal": 0.18,
    },
    "weakness": {
        "Pneumonia": 0.45,
        "Tuberculosis": 0.45,
        "Lung Cancer": 0.40,
        "Normal": 0.15,
    },
    "runny nose": {
        "Pneumonia": 0.05,
        "Tuberculosis": 0.03,
        "Lung Cancer": 0.02,
        "Normal": 0.70,
    },
    "nasal congestion": {
        "Pneumonia": 0.06,
        "Tuberculosis": 0.03,
        "Lung Cancer": 0.02,
        "Normal": 0.70,
    },
    "sneezing": {
        "Pneumonia": 0.04,
        "Tuberculosis": 0.02,
        "Lung Cancer": 0.02,
        "Normal": 0.65,
    },
    "sore throat": {
        "Pneumonia": 0.10,
        "Tuberculosis": 0.05,
        "Lung Cancer": 0.05,
        "Normal": 0.55,
    },
    "weight loss": {
        "Pneumonia": 0.08,
        "Tuberculosis": 0.70,
        "Lung Cancer": 0.65,
        "Normal": 0.02,
    },
    "loss of appetite": {
        "Pneumonia": 0.15,
        "Tuberculosis": 0.55,
        "Lung Cancer": 0.45,
        "Normal": 0.05,
    },
    "recurrent chest infections": {
        "Pneumonia": 0.25,
        "Tuberculosis": 0.20,
        "Lung Cancer": 0.55,
        "Normal": 0.03,
    },
}

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

TEMPERATURE = 1.35


def normalize_symptom(symptom):
    return str(symptom or "").strip().lower()


def softmax_from_log_scores(log_scores):
    max_log_score = max(log_scores.values())

    exp_scores = {
        disease: math.exp((score - max_log_score) / TEMPERATURE)
        for disease, score in log_scores.items()
    }

    total = sum(exp_scores.values())

    if total <= 0:
        return {disease: round(100 / len(DISEASES), 2) for disease in DISEASES}

    scores = {
        disease: round((value / total) * 100, 2)
        for disease, value in exp_scores.items()
    }

    difference = round(100 - sum(scores.values()), 2)
    top_disease = max(scores, key=scores.get)
    scores[top_disease] = round(scores[top_disease] + difference, 2)

    return scores


def score_symptoms(selected_symptoms):
    selected = [
        normalize_symptom(symptom)
        for symptom in selected_symptoms
        if normalize_symptom(symptom)
    ]

    if not selected:
        return {disease: round(100 / len(DISEASES), 2) for disease in DISEASES}

    selected = list(dict.fromkeys(selected))

    log_scores = {}

    for disease in DISEASES:
        log_score = math.log(PRIORS[disease])

        for symptom in selected:
            likelihoods = SYMPTOM_LIKELIHOODS.get(symptom, {})
            likelihood = likelihoods.get(disease, DEFAULT_LIKELIHOOD[disease])
            likelihood = min(max(float(likelihood), 0.001), 0.999)
            log_score += math.log(likelihood)

        log_scores[disease] = log_score

    if any(symptom in RED_FLAG_SYMPTOMS for symptom in selected):
        log_scores["Normal"] += math.log(0.25)

    return softmax_from_log_scores(log_scores)


def calculate_symptom_scores(selected_symptoms):
    return score_symptoms(selected_symptoms)


def get_top_symptom_prediction(symptom_scores):
    if not symptom_scores:
        return "Unknown", 0.0

    top_label = max(symptom_scores, key=symptom_scores.get)
    top_confidence = symptom_scores[top_label]

    return top_label, top_confidence