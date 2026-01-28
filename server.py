import json
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import models, transforms
from torch import nn

# -----------------------------
# CONFIG
# -----------------------------
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
CLASSES_PATH = ARTIFACTS_DIR / "classes.json"

# macOS: support du GPU Apple (Metal / MPS) si dispo, sinon CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# même preprocessing que l'entraînement (val_tf)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(title="Tank Classifier API")

model = None
classes = None


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_artifacts():
    global model, classes

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise RuntimeError(f"Missing classes file: {CLASSES_PATH}")

    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    num_classes = len(classes)

    # même archi que train.py
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()

    model = m


@app.on_event("startup")
def startup():
    load_artifacts()
    print(f"✅ Model loaded. classes={len(classes)} device={DEVICE}")


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "classes": len(classes) if classes else None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or classes is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use jpg/png/webp.")

    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = preprocess(img).unsqueeze(0).to(DEVICE)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        topk = torch.topk(probs, k=3)
        top_scores = topk.values.tolist()
        top_indices = topk.indices.tolist()

    top3 = [
        {"label": classes[idx], "score": float(score)}
        for idx, score in zip(top_indices, top_scores)
    ]

    return {
        "filename": file.filename,
        "top_1": top3[0],
        "top_3": top3,
    }