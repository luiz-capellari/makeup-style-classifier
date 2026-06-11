"""
Inferência com o modelo ONNX exportado.
Roda em CPU — sem dependência de GPU no servidor.
"""

import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MakeupClassifier:
    def __init__(
        self,
        model_path: str = "models/makeup_classifier.onnx",
        classes_path: str = "models/classes.json",
    ):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        with open(classes_path) as f:
            self.classes = json.load(f)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 224, 224)

    def predict(self, image: Image.Image) -> dict:
        tensor  = self.preprocess(image)
        logits  = self.session.run(["logits"], {"image": tensor})[0][0]
        exp     = np.exp(logits - logits.max())
        probs   = exp / exp.sum()

        top_idx = int(probs.argmax())
        return {
            "style":      self.classes[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "all_scores": {
                cls: round(float(p), 4)
                for cls, p in zip(self.classes, probs)
            },
        }

