def fuse_predictions(image_scores, symptom_scores, image_weight=0.70, symptom_weight=0.30):
    """
    Combines image model probabilities and symptom-based probabilities.

    image_scores example:
    {"Pneumonia": 80, "Tuberculosis": 15, "Normal": 5}

    symptom_scores example:
    {"Pneumonia": 60, "Tuberculosis": 35, "Lung Cancer": 20, "Normal": 0}
    """
    all_labels = set(image_scores.keys()) | set(symptom_scores.keys())
    final_scores = {}

    for label in all_labels:
        image_score = float(image_scores.get(label, 0))
        symptom_score = float(symptom_scores.get(label, 0))
        final_scores[label] = round((image_weight * image_score) + (symptom_weight * symptom_score), 2)

    final_prediction = max(final_scores, key=final_scores.get)
    final_confidence = final_scores[final_prediction]

    if final_confidence >= 75:
        risk_level = "High"
    elif final_confidence >= 50:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "final_scores": final_scores,
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "risk_level": risk_level
    }
