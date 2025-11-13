import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import numpy as np

from database import create_document, get_documents
from schemas import Analysis

app = FastAPI(title="Skin Disease Detector API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONDITIONS = [
    {
        "id": "acne",
        "name": "Acne",
        "about": "Acne is a common skin condition where hair follicles become clogged with oil and dead skin cells, causing whiteheads, blackheads or pimples.",
        "care": [
            "Wash gently twice daily with a mild cleanser",
            "Avoid picking or squeezing lesions",
            "Use non-comedogenic moisturizers and sunscreen",
            "Consider over-the-counter benzoyl peroxide or salicylic acid"
        ]
    },
    {
        "id": "eczema",
        "name": "Eczema (Atopic Dermatitis)",
        "about": "Eczema is an inflammatory skin condition causing dry, itchy patches that may crack and bleed.",
        "care": [
            "Moisturize frequently with fragrance-free creams",
            "Use gentle cleansers and lukewarm showers",
            "Avoid known triggers (wool, harsh soaps)",
            "Consider topical hydrocortisone for flares"
        ]
    },
    {
        "id": "psoriasis",
        "name": "Psoriasis",
        "about": "Psoriasis is an autoimmune condition that speeds up skin cell growth, leading to thick, scaly patches.",
        "care": [
            "Keep skin moisturized",
            "Use medicated shampoos or creams as directed",
            "Manage stress and avoid smoking",
            "Discuss phototherapy or biologics with a dermatologist"
        ]
    },
    {
        "id": "normal",
        "name": "Uncertain / Normal",
        "about": "The image does not clearly match common conditions in the model's classes. This is not a medical diagnosis.",
        "care": [
            "Monitor the area for changes",
            "Use sunscreen and gentle skincare",
            "Consult a dermatologist if symptoms persist or worsen"
        ]
    }
]

class PredictionResponse(BaseModel):
    condition: str
    confidence: float
    about: str
    care: List[str]
    alternatives: Optional[List[dict]] = None
    disclaimer: str


def dummy_model_predict(image: Image.Image) -> List[float]:
    """
    Placeholder for a real ML model. Returns pseudo-probabilities for 4 classes.
    In a real app, load a TensorFlow/PyTorch model and run inference here.
    """
    img = image.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    # Simple heuristic using mean color channels to create reproducible pseudo-scores
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    scores = np.array([
        max(0.0, (r - g + 0.5)),   # acne-ish
        max(0.0, (g - b + 0.5)),   # eczema-ish
        max(0.0, (b - r + 0.5)),   # psoriasis-ish
        0.5                        # uncertain baseline
    ], dtype=np.float32)
    probs = scores / (scores.sum() + 1e-8)
    return probs.tolist()


@app.get("/")
def read_root():
    return {"message": "Skin Disease Detector API running"}


@app.post("/analyze", response_model=PredictionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        bytes_data = await file.read()
        image = Image.open(io.BytesIO(bytes_data))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.")

    probs = dummy_model_predict(image)
    idx = int(np.argmax(probs))
    top = CONDITIONS[idx]
    confidence = float(round(probs[idx] * 100, 2))

    # build alternatives (top-3)
    order = np.argsort(probs)[::-1]
    alternatives = []
    for i in order[:3]:
        alternatives.append({
            "condition": CONDITIONS[int(i)]["name"],
            "confidence": float(round(probs[int(i)] * 100, 2))
        })

    response = PredictionResponse(
        condition=top["name"],
        confidence=confidence,
        about=top["about"],
        care=top["care"],
        alternatives=alternatives,
        disclaimer=(
            "This tool provides educational information only and is not a medical diagnosis. "
            "Consult a qualified dermatologist for evaluation and treatment."
        ),
    )

    # Persist to history if user_id provided and db available
    if user_id:
        try:
            doc = Analysis(
                user_id=user_id,
                condition=response.condition,
                confidence=response.confidence,
                about=response.about,
                care=response.care,
                alternatives=response.alternatives,
                image_name=file.filename or None,
            )
            create_document("analysis", doc)
        except Exception:
            # Do not block response if DB not available
            pass

    return response


@app.get("/history")
async def get_history(user_id: str = Query(..., description="User identifier"), limit: int = 20):
    try:
        docs = get_documents("analysis", {"user_id": user_id}, limit=limit)
    except Exception:
        # If db not available, return empty list
        return {"items": []}

    items = []
    for d in docs:
        items.append({
            "id": str(d.get("_id")),
            "condition": d.get("condition"),
            "confidence": d.get("confidence"),
            "created_at": d.get("created_at").isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at"),
            "image_name": d.get("image_name"),
        })
    # Newest first
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"items": items}


@app.get("/test")
def test_database():
    """Simple health check for environment."""
    import os
    return {
        "backend": "✅ Running",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
