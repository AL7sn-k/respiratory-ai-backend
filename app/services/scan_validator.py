from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "scan_validator_model.h5"

IMG_SIZE = (224, 224)


CLASS_NAMES = ["xray", "ct", "invalid"]

DISPLAY_NAMES = {
    "xray": "X-ray",
    "ct": "CT Scan",
    "invalid": "Invalid",
}

_model = None


def load_scan_validator_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Scan validator model not found: {MODEL_PATH}")

        _model = tf.keras.models.load_model(str(MODEL_PATH))

    return _model


def validate_scan_image(image_path):
    model = load_scan_validator_model()

    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    top_index = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_index]
    confidence = float(preds[top_index] * 100)

    detected_type = DISPLAY_NAMES[top_class]

    return {
        "detected_type": detected_type,
        "confidence": round(confidence, 2),
        "raw_scores": {
            DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(preds[i] * 100), 2)
            for i in range(len(CLASS_NAMES))
        },
    }