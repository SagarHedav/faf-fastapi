import insightface
import numpy as np
from PIL import Image
import io

print("Loading InsightFace model...")

# Use buffalo_l for accurate recognition
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

def extract_embedding(image_bytes):
    try:
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