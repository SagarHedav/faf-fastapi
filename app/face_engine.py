import insightface
import numpy as np
from PIL import Image
import io

print("Face engine module initialized")

# Global variable to hold the model (lazy loading)
_model = None
_loading = False

def get_model():
    """
    Lazy load the InsightFace model on first use.
    Thread-safe singleton pattern.
    """
    global _model, _loading
    
    if _model is not None:
        return _model
    
    if _loading:
        # Wait for other thread to finish loading
        import time
        while _loading and _model is None:
            time.sleep(0.1)
        return _model
    
    _loading = True
    print("Loading InsightFace model (buffalo_l)... This may take a minute on first start.")
    try:
        _model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )
        _model.prepare(ctx_id=0)
        print("✅ InsightFace model loaded successfully!")
    finally:
        _loading = False
    
    return _model

def extract_embedding(image_bytes):
    try:
        # Get model (lazy loaded on first call)
        model = get_model()
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (from RGBA or CMYK or PNG alpha)
        img = img.convert("RGB")

        # Convert to numpy
        img = np.array(img)

        # Run face detection
        faces = model.get(img)
        if len(faces) == 0:
            return None, "No face detected"

        # First detected face
        face = faces[0]
        embedding = face.embedding.tolist()

        return embedding, None

    except Exception as e:
        return None, str(e)