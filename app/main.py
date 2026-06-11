"""
API REST com FastAPI.
Endpoint principal: POST /classify — recebe imagem, retorna estilo.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

try:
    from classifier import MakeupClassifier
except ImportError:
    from app.classifier import MakeupClassifier


app = FastAPI(
    title="Makeup Style Classifier API",
    description="Classifica o estilo de maquiagem em fotos de rostos com EfficientNet-B0",
    version="1.0.0",
)

classifier = MakeupClassifier()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":   "EfficientNet-B0 (ONNX)",
        "classes": classifier.classes,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(..., description="Foto do rosto (jpg/png/webp)")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Formato não suportado. Use: {ALLOWED_TYPES}")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Não foi possível abrir a imagem.")

    result = classifier.predict(image)
    return result

