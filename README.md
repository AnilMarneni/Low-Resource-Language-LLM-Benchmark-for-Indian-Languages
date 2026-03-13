# 📊 Benchmarking LLMs on Low-Resource Indian Languages

This repository provides a modular, reproducible, and robust benchmarking framework to evaluate Large Language Models (LLMs) on NLP tasks in low-resource Indian languages (e.g., Telugu, Kannada, Marathi, Tamil). 

It is designed following best practices used in academic NLP research, offering dynamic model instantiation, unified data preprocessing via IndicNLP, and an interactive Streamlit visualization dashboard.

## 🚀 Key Features
* **Modular Architecture:** Easy configuration via YAML (`configs/`).
* **Multi-Task Support:** Summarization, Question Answering, Sentiment Analysis, and Machine Translation.
* **Unified Pipeline:** End-to-end dataset scraping, pre-processing, and HF DataLoaders.
* **Multi-Model Inference:** Generative causal LMs safely batched using `device_map="auto"`.
* **Standardized Evaluation:** Automated ROUGE, SacreBLEU, F1, and Accuracy computation.
* **Interactive Visualizations:** High-quality Plotly graphics powered by Streamlit and Jupyter Notebooks.

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/AnilMarneni/Low-Resource-Language-LLM-Benchmark-for-Indian-Languages.git
cd Low-Resource-Language-LLM-Benchmark-for-Indian-Languages
```

2. **Create a virtual environment and load dependencies:**
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

pip install -e .
pip install -r requirements.txt
pip install accelerate # required for memory-mapping large models
```

3. **(Optional) Setup environment variables:**
To download gated HuggingFace models such as `meta-llama/Llama-3-8B-Instruct`, copy the environment template and insert your keys:
```bash
cp .env.example .env
```

## 🛠 Usage

### 1. Build the Dataset
Downloads the raw datasets and constructs a schema-compliant JSONL index via IndicNLP processors.
```bash
python scripts/download_data.py
python scripts/build_datasets.py
```

### 2. Run the Benchmarks
The central runner maps user requested models to task metrics across all low-resource languages. 
Parameters must match those designated in `configs/models.yaml` and `configs/tasks.yaml`.
```bash
python src/evaluation/benchmark_runner.py --models llama_3_8b_instruct mistral_7b_instruct --tasks summarization qa translation sentiment
```

### 3. Analyze Results Interactive Dashboard
Displays the benchmark aggregation `.csv` file geographically, mapping languages and model capabilities.
```bash
streamlit run src/visualization/dashboard.py
```

## 📁 Project Architecture
```text
llm_indic_benchmark/
├── configs/                  # YAML configurations (models, tasks, pipeline)
├── data/
│   ├── raw/                  # Scraped internet corpora  
│   └── processed/            # Serialized JSONL standardized schemas
├── notebooks/
│   └── analysis.ipynb        # Researcher sandbox for custom PCA / t-tests
├── results/
│   ├── predictions/          # Raw localized JSON model outputs 
│   └── benchmarks/           # Standardized metric CSV outputs
├── scripts/
│   └── build_datasets.py     # Aggregation algorithms
├── src/
│   ├── data/                 # IndicNLP data normalization and HTML cleaners
│   ├── evaluation/           # Evaluating wrappers (accuracy, ROUGE)
│   ├── models/               # Torch generators and dataloading
│   ├── tasks/                # Granular task wrappers mapped for inference
│   ├── visualization/        # Plotly logic and Streamlit engine
│   └── utils/                # Logging and Config Parsers
├── .env.example              # Secret key placeholder
├── setup.py                  # Module package initialization
└── requirements.txt          # Python environments constraints
```

## 📝 License
This benchmark repository is intended for academic research use. Please adhere to the ethical usage guidelines outlined by the respective open-source model providers (Llama 3, Mistral, Gemma, AI4Bharat).
