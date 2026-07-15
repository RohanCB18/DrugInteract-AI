# Drug Interaction Prediction — Comparative Study

> **GAT vs. ChemBERTa** — Comparing graph-based and text-based molecular representations  
> for drug-drug interaction prediction, with self-supervised pretraining variants.

## Architecture

```
React Frontend (Vite + Recharts)
        │
        ▼
FastAPI Backend (model serving + logging)
   ├── GAT model (Version A: baseline, Version B: contrastive pretrained)
   ├── ChemBERTa model (Version A: baseline, Version B: MLM pretrained)
   └── writes predictions → SQLite (local) / RDS Postgres (AWS)
        │
        ▼
Docker containers (frontend + backend)  →  EC2 + NGINX
```

## Quick Start (Local Development)

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note**: `torch_geometric` may need special install steps depending on your
> PyTorch + CUDA version. See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Prepare data

```bash
python -m backend.app.data.prepare_data
```

This downloads DrugBank DDI (via TDC) and ZINC-250K for pretraining. Takes ~5 min.

### 3. Train models

```bash
# GAT — Version A (baseline, from scratch)
python -m backend.app.training.train_gat --version A

# GAT — Contrastive pretraining on ZINC, then Version B
python -m backend.app.training.pretrain_gat
python -m backend.app.training.train_gat --version B

# ChemBERTa — Version A (fine-tune from HuggingFace weights)
python -m backend.app.training.train_chemberta --version A

# ChemBERTa — Additional MLM pretraining, then Version B
python -m backend.app.training.pretrain_chemberta
python -m backend.app.training.train_chemberta --version B
```

Use `--smoke-test` on any command for a quick sanity check (2-3 epochs, tiny subset).

### 4. Evaluate all models

```bash
python -m backend.app.training.evaluate
```

### 5. Start the API server

```bash
uvicorn backend.app.main:app --reload
```

API docs at: http://localhost:8000/docs

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict` | Predict interaction for a drug pair |
| GET | `/api/compare` | Get evaluation metrics for all models |
| GET | `/api/history` | Get recent prediction logs |
| GET | `/api/labels` | Get interaction class label map |
| GET | `/api/drugs` | Get known drug list for autocomplete |
| GET | `/api/health` | Health check |

## Model Variants

| Model | Type | Pretraining | Description |
|-------|------|-------------|-------------|
| GAT-A | Graph | None | GAT encoder trained from scratch on DDI labels |
| GAT-B | Graph | Contrastive (NT-Xent) on ZINC-250K | Pretrained encoder → fine-tuned on DDI |
| ChemBERTa-A | Text | HuggingFace base only | Fine-tuned directly on DDI |
| ChemBERTa-B | Text | Additional MLM on ZINC-250K | Domain-adapted → fine-tuned on DDI |

## Interaction Classes (Binned)

| ID | Label |
|----|-------|
| 0 | Increases effect |
| 1 | Decreases effect |
| 2 | Alters metabolism |
| 3 | Increases risk |
| 4 | Other interaction |

## Tech Stack

**Backend**: Python 3.11, PyTorch, PyTorch Geometric, HuggingFace Transformers, RDKit, FastAPI, SQLAlchemy  
**Frontend**: React 18, Vite, Recharts, React Router  
**Deployment**: Docker, Docker Compose, NGINX, AWS EC2 + RDS + S3

## Project Structure

```
gat/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── database.py          # SQLAlchemy ORM
│   │   ├── routes/              # API endpoints
│   │   ├── models/              # GAT + ChemBERTa architectures
│   │   ├── data/                # Featurizer, datasets, data prep
│   │   ├── training/            # Train/pretrain/evaluate scripts
│   │   └── explainability/      # Attention extraction
│   ├── checkpoints/             # Saved model weights
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/               # Page views
│   │   ├── api/                 # API client
│   │   └── utils/               # Molecule SVG renderer
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── README.md
```
