# 📊 Indic LLM Benchmark: Low-Resource Language Tracking Platform

This repository provides a modular, professional-grade benchmarking framework for evaluating Large Language Models (LLMs) on NLP tasks across 6 low-resource Indian languages: **Telugu, Tamil, Kannada, Malayalam, Marathi, and Hindi**.

Developed as part of a research internship, the platform combines a high-performance **FastAPI backend** with a stunning **Next.js research portal** to visualize model capabilities, sentence complexity, and semantic similarity metrics.

## 🚀 Key Features
*   **Professional Dashboard**: Live, interactive charts using Recharts and Primereact, proxied for production stability.
*   **Automated Research Pipeline**: End-to-end flow from data seeding to model evaluation and metric aggregation.
*   **Multi-Model Engine**: Support for Llama 3, Mistral, Gemma, and custom Indic architectures.
*   **Deep-Dive Analytics**: Correlates linguistic complexity (sentence length, token depth) with model performance.
*   **Hardened Architecture**: Reverse-proxy configurations and validated JSON error handling for maximum uptime.

---

## ⚙️ Installation & Setup

### 1. Backend Setup
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
pip install -e .
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

---

## 🛠 Usage: The Automated Pipeline

The platform is designed for fully automated data population. Follow these three steps to hydrate the dashboard with fresh research results:

### Step 1: Research Data Seeding
Generates raw simulated research corpora for the target languages.
```bash
python scripts/download_data.py
```

### Step 2: Dataset Building
Constructs schema-compliant JSONL shards with IndicNLP preprocessing.
```bash
python scripts/build_datasets.py
```

### Step 3: Core Benchmarking
Executes model inference and computes metrics (ROUGE, BERTScore, Complexity).
```bash
python src/evaluation/benchmark_runner.py --models gpt2_tiny --tasks summarization sentiment
```

---

## 🌐 Launching the Portal

### 1. Start the API (Port 8000)
```bash
.\venv\Scripts\python -m uvicorn src.api.main:app --reload
```

### 2. Start the Dashboard (Port 3000)
```bash
cd frontend
npm run dev
```
Navigate to `http://localhost:3000` to view the **Indic Model Leaderboard**.

---

## 📈 Data Ingestion Guide

The portal automatically monitors `results/benchmarks/` for the latest CSV artifacts.

*   **Leaderboard**: To add results, drop a `benchmark_summary_[timestamp].csv` file. Required columns: `Model`, `Task`, `Language`, `Samples`.
*   **Deep-Dive**: To populate scatter plots, use `sample_level_metrics_[timestamp].csv` with columns for `Prediction`, `Reference`, and `Complexity`.

---

## 📁 Project Architecture
```text
llm_indic_benchmark/
├── configs/                  # YAML configurations (models, tasks, pipeline)
├── data/
│   ├── raw/                  # Seeded research corpora  
│   └── processed/            # Structured JSONL Indic shards
├── frontend/                 # Next.js Research Portal (Port 3000)
├── results/
│   ├── predictions/          # Raw localized model outputs 
│   └── benchmarks/           # Aggregated metric CSVs (Dashboard Source)
├── scripts/                  # Data staging and building utilities
├── src/
│   ├── api/                  # FastAPI Backend (Port 8000)
│   ├── evaluation/           # Metric computation and complexity analysis
│   ├── models/               # Inference engine and model loaders
│   └── utils/                # Logging and Config Parsers
└── requirements.txt          # Python environment constraints
```

## 📝 License
This framework is intended for academic research use. Adhere to ethical usage guidelines for open-source model providers (Meta, Mistral, Google).
