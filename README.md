# Face Recognition FastAPI Service

Face recognition service for the Fine Management System using InsightFace and FAISS.

## Features

- Face embedding extraction using InsightFace (buffalo_l model)
- FAISS-based similarity search for fast face recognition
- PostgreSQL integration for student data
- RESTful API with FastAPI

## Tech Stack

- **FastAPI**: Web framework
- **InsightFace**: Face recognition (buffalo_l model)
- **FAISS**: Vector similarity search
- **PostgreSQL**: Database (Neon)
- **Pillow**: Image processing

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL database (Neon recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/faf-fastapi.git
cd faf-fastapi
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

### Running Locally

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### `GET /`
Health check endpoint

### `POST /extract-embedding`
Extract face embedding from an image
- **Input**: Image file (multipart/form-data)
- **Output**: 512-dimensional face embedding

### `POST /search-face`
Search for matching face in database
- **Input**: Face embedding (512-dim array)
- **Output**: Best match with student info and similarity score

### `POST /rebuild-index`
Rebuild FAISS index from database

### `GET /faiss-status`
Get FAISS index status and statistics

## Environment Variables

```env
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
FACE_API_KEY=your_secret_api_key (optional)
```

## Deployment

### Railway (Recommended)

1. Create new project on Railway
2. Connect GitHub repository
3. Add `DATABASE_URL` environment variable
4. Railway will auto-deploy!

Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Memory Requirements

- **Model**: buffalo_l (~700MB)
- **Peak RAM**: ~1.2GB
- **Recommended**: At least 2GB RAM

## Model Information

Uses InsightFace's **buffalo_l** model:
- High accuracy face recognition
- 512-dimensional embeddings
- Suitable for production use

## License

MIT
