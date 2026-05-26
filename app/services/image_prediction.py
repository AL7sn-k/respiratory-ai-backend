from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from app.services.gradcam_plus_plus import (
    make_gradcam_plus_plus_heatmap,
    save_gradcam_overlay,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
HEATMAP_DIR = BASE_DIR / "reports" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)

XRAY_MODEL_PATH = MODELS_DIR / "xray_densenet_final.h5"
CT_MODEL_PATH = MODELS_DIR / "ct_multiclass_model.h5"

XRAY_CLASSES = ["normal", "pneumonia", "tuberculosis"]
CT_CLASSES = ["normal_ct", "lung_cancer", "pneumonia_ct"]

# ============================================================
# Image uncertainty thresholds
# Scores in this file are stored as percentages, not decimals.
# So 0.95 becomes 95.0, and 0.05 becomes 5.0.
# ============================================================

CT_MIN_CONFIDENCE = 95.0
CT_MIN_MARGIN = 5.0

XRAY_MIN_CONFIDENCE = 75.0
XRAY_MIN_MARGIN = 15.0

xray_model = None
ct_model = None


def load_model_by_scan_type(scan_type):
    global xray_model, ct_model

    scan_type = scan_type.lower()

    if scan_type == "x-ray":
        if xray_model is None:
            xray_model = tf.keras.models.load_model(XRAY_MODEL_PATH)
        return xray_model, XRAY_CLASSES

    if scan_type == "ct scan":
        if ct_model is None:
            ct_model = tf.keras.models.load_model(CT_MODEL_PATH)
        return ct_model, CT_CLASSES

    raise ValueError("Invalid scan type. Use 'X-ray' or 'CT Scan'.")


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def map_class_name(class_name):
    mapping = {
        "normal": "Normal",
        "pneumonia": "Pneumonia",
        "tuberculosis": "Tuberculosis",
        "normal_ct": "Normal",
        "lung_cancer": "Lung Cancer",
        "pneumonia_ct": "Pneumonia",
    }

    return mapping.get(class_name, class_name)


def get_uncertainty_thresholds(scan_type):
    scan_type = scan_type.lower()

    if scan_type == "ct scan":
        return CT_MIN_CONFIDENCE, CT_MIN_MARGIN

    if scan_type == "x-ray":
        return XRAY_MIN_CONFIDENCE, XRAY_MIN_MARGIN

    raise ValueError("Invalid scan type. Use 'X-ray' or 'CT Scan'.")


def apply_image_uncertainty_rule(
    image_scores,
    min_confidence,
    min_margin,
):
    """
    Selective classification reject option.

    The image model is considered uncertain if:
    1. Top class confidence is below min_confidence.
    2. The margin between top class and second class is below min_margin.

    If uncertain, the system should stop before symptom fusion and return:
    "Possible abnormality outside supported diseases"
    """

    sorted_scores = sorted(
        image_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_score = sorted_scores[0]

    if len(sorted_scores) > 1:
        second_label, second_score = sorted_scores[1]
    else:
        second_label, second_score = "N/A", 0

    margin = top_score - second_score

    if top_score < min_confidence:
        return {
            "is_image_uncertain": True,
            "top_image_prediction": "Possible abnormality outside supported diseases",
            "top_image_confidence": round(top_score, 2),
            "uncertainty_reason": (
                f"The image model confidence is low. "
                f"Top prediction was {top_label} with {round(top_score, 2)}% confidence. "
                f"Required minimum confidence is {min_confidence}%."
            ),
            "original_top_prediction": top_label,
            "original_top_confidence": round(top_score, 2),
            "second_prediction": second_label,
            "second_confidence": round(second_score, 2),
            "margin": round(margin, 2),
            "required_min_confidence": min_confidence,
            "required_min_margin": min_margin,
        }

    if margin < min_margin:
        return {
            "is_image_uncertain": True,
            "top_image_prediction": "Possible abnormality outside supported diseases",
            "top_image_confidence": round(top_score, 2),
            "uncertainty_reason": (
                f"The image model is confused between {top_label} and {second_label}. "
                f"Score margin is only {round(margin, 2)}%. "
                f"Required minimum margin is {min_margin}%."
            ),
            "original_top_prediction": top_label,
            "original_top_confidence": round(top_score, 2),
            "second_prediction": second_label,
            "second_confidence": round(second_score, 2),
            "margin": round(margin, 2),
            "required_min_confidence": min_confidence,
            "required_min_margin": min_margin,
        }

    return {
        "is_image_uncertain": False,
        "top_image_prediction": top_label,
        "top_image_confidence": round(top_score, 2),
        "uncertainty_reason": "",
        "original_top_prediction": top_label,
        "original_top_confidence": round(top_score, 2),
        "second_prediction": second_label,
        "second_confidence": round(second_score, 2),
        "margin": round(margin, 2),
        "required_min_confidence": min_confidence,
        "required_min_margin": min_margin,
    }


def predict_image(scan_type, image_path):
    model, classes = load_model_by_scan_type(scan_type)

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)[0]
    class_index = int(np.argmax(predictions))

    scores = {}

    for class_name, probability in zip(classes, predictions):
        mapped_name = map_class_name(class_name)
        scores[mapped_name] = round(float(probability) * 100, 2)

    min_confidence, min_margin = get_uncertainty_thresholds(scan_type)

    uncertainty_result = apply_image_uncertainty_rule(
        image_scores=scores,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )

    heatmap = make_gradcam_plus_plus_heatmap(
        model=model,
        image_array=processed_image,
        class_index=class_index,
    )

    heatmap_filename = f"heatmap_{Path(image_path).stem}.jpg"
    heatmap_path = HEATMAP_DIR / heatmap_filename

    save_gradcam_overlay(
        original_image_path=image_path,
        heatmap=heatmap,
        output_path=heatmap_path,
    )

    return {
        "scan_type": scan_type,
        "image_scores": scores,
        # This may become "Possible abnormality outside supported diseases"
        # if the image model is weak/confused.
        "top_image_prediction": uncertainty_result["top_image_prediction"],
        "top_image_confidence": uncertainty_result["top_image_confidence"],
        # Uncertainty fields
        "is_image_uncertain": uncertainty_result["is_image_uncertain"],
        "uncertainty_reason": uncertainty_result["uncertainty_reason"],
        "original_top_prediction": uncertainty_result["original_top_prediction"],
        "original_top_confidence": uncertainty_result["original_top_confidence"],
        "second_prediction": uncertainty_result["second_prediction"],
        "second_confidence": uncertainty_result["second_confidence"],
        "margin": uncertainty_result["margin"],
        "required_min_confidence": uncertainty_result["required_min_confidence"],
        "required_min_margin": uncertainty_result["required_min_margin"],
        "heatmap_path": str(heatmap_path),
        "heatmap_url": f"/heatmaps/{heatmap_filename}",
    }
