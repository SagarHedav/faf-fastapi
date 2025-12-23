# app/faiss_manager.py
import os
import json
import numpy as np
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# REPLACE THE STRING BELOW WITH YOUR NEON CONNECTION STRING
# Example: "postgresql://user:pass@ep-xxxxx.region.aws.neon.tech/neondb?sslmode=require"
DB_URL = os.getenv("DATABASE_URL") or "postgresql://neondb_owner:npg_uVDzbl5sar0O@ep-cold-wildflower-a474dung-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")

class FaissManager:
    def __init__(self, d=512):
        self.dimension = d
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        self.id_map = []
        self.conn = None
        self._connect_db()

    def _connect_db(self):
        try:
            self.conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor, connect_timeout=10)
            self.conn.autocommit = True
        except Exception as e:
            print(f"DB Connection failed: {e}")
            self.conn = None

    def _ensure_connection(self):
        if self.conn is None or self.conn.closed:
            print("Reconnecting to DB...")
            self._connect_db()

    def _fetch_embeddings(self):
        self._ensure_connection()
        if self.conn is None:
            return None, []
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, face_embedding FROM students WHERE face_embedding IS NOT NULL")
                rows = cur.fetchall()
        except Exception as e:
            print(f"Error fetching embeddings: {e}")
            self.conn = None # Force reconnect next time
            return None, []

        embeddings = []
        ids = []
        for row in rows:
            emb = row["face_embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            
            emb_array = np.array(emb, dtype=np.float32)
            
            # Normalize for cosine similarity (L2 normalization)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                emb_array = emb_array / norm
            
            embeddings.append(emb_array)
            ids.append(row["id"])
        if embeddings:
            return np.vstack(embeddings), ids
        return None, []

    def build_index_from_db(self):
        try:
            vectors, ids = self._fetch_embeddings()
            if vectors is None:
                return False, "No embeddings found in DB"
            self.index.reset()
            self.index.add(vectors)
            self.id_map = ids
            self.save_to_disk()
            return True, f"Built index with {len(ids)} vectors"
        except Exception as e:
            return False, str(e)

    def save_to_disk(self):
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(INDEX_PATH + ".ids", "w") as f:
                json.dump(self.id_map, f)
        except Exception as e:
            print("FAISS save error:", e)

    def load_from_disk(self):
        if not os.path.exists(INDEX_PATH):
            return False, "Index file not found"
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(INDEX_PATH + ".ids", "r") as f:
                self.id_map = json.load(f)
            return True, f"Loaded index with {len(self.id_map)} vectors"
        except Exception as e:
            return False, str(e)

    def search(self, query_vec, k=1):
        if self.index.ntotal == 0:
            return None
        query = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        
        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        scores, indices = self.index.search(query, k)  # scores = inner products
        if indices[0][0] == -1:
            return None
        
        student_id = self.id_map[indices[0][0]]
        similarity = float(scores[0][0])  # Direct similarity score (cosine similarity)
        
        # Clip to [0, 1] range for safety
        similarity = max(0.0, min(1.0, similarity))
        
        return {"student_id": student_id, "similarity": similarity}

    def get_student_by_id(self, student_id):
        """Fetch full student record from DB given an ID"""
        self._ensure_connection()
        if self.conn is None:
            return None
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM students WHERE id = %s", (student_id,))
                row = cur.fetchone()
            return row
        except Exception as e:
            print(f"Error fetching student: {e}")
            self.conn = None
            return None

    def get_status(self):
        """Return basic status info about the FAISS index."""
        return {
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
            "index_path": INDEX_PATH,
            "loaded": os.path.exists(INDEX_PATH),
        }

_manager_instance = None

def get_faiss_manager():
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = FaissManager()
    return _manager_instance
