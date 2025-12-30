# 🚀 Learning Intelligence AI Tool  
**AI Kata – Data Science & Machine Learning Internship Assessment**

---
## 🔗 Live Demo & API Documentation

Explore and test the deployed FastAPI Swagger UI:

👉 Live API Docs:
https://learning-intelligence-ai-j2h8.onrender.com/docs

---

## 📌 Project Overview

The **Learning Intelligence AI Tool** is a production-style, executable AI system designed to analyze learner behavior data and provide **intelligent predictions and insights** for mentors and administrators on an internship or training platform.

Unlike notebook-based experimentation, this project delivers a **fully functional AI tool** that integrates machine learning into software pipelines and exposes predictions via **CLI and REST API interfaces**.

---

## 🎯 Objectives (As per Assignment)

This project demonstrates the ability to:

- Build a **usable AI tool**, not a notebook  
- Integrate machine learning into a software system  
- Design **production-grade AI pipelines**  
- Expose AI functionality via **CLI and API**  
- Move beyond experimentation toward **real-world AI deployment**  
- Use AI tools **responsibly and transparently**

> ⚠️ No notebooks are used for execution. All AI functionality is available via executable pipelines and interfaces.

---

## 🧠 AI Capabilities Implemented (Mandatory)

### ✅ 1. Course Completion Prediction
- **Problem Type:** Binary classification  
- **Output:** Probability of course completion  
- **Model:** Logistic Regression (trained offline, loaded during inference)

---

### ✅ 2. Early Risk Detection
- Learners with low completion probability are flagged early  
- **Risk threshold:** configurable (default = 0.4)  
- Output labels:
  - `High Risk`
  - `Low Risk`

---

### ✅ 3. Learning Intelligence via Feature Engineering
Chapter-level activity is converted into **student-level intelligence features**, including:
- Average and maximum time spent  
- Average and minimum scores  
- Chapters completed  
- Early performance (first 3 chapters)  

These features enable **early dropout detection**.

---

### ✅ 4. Human-Readable Insight Output
The tool generates:
- Completion probability  
- Risk labels  
- JSON responses (API)  
- Downloadable CSV reports (CLI & pipeline)

---

## 📥 Input Specification

The system accepts input as a **CSV file** with the following schema:

| Column Name | Description |
|------------|-------------|
| `student_id` | Unique learner identifier |
| `course_id` | Course identifier |
| `chapter_order` | Chapter sequence number |
| `time_spent` | Time spent on chapter |
| `score` | Assessment score |
| `completion_status` | Final completion flag (0/1) |

---

## 📤 Output Specification

Each learner receives:

| Output Field | Description |
|--------------|------------|
| `completion_probability` | Likelihood of completing the course |
| `risk_label` | High Risk / Low Risk |
| `student_id`, `course_id` | Identifiers |

---

## 🏗️ AI System Architecture

```
Data Ingestion
     ↓
Feature Engineering (Student-Level Aggregation)
     ↓
Preprocessing Pipeline
     ↓
Trained ML Model (Loaded)
     ↓
Risk Detection Logic
     ↓
CLI / API Output
```

---

## 🧪 Model Details

- **Algorithm:** Logistic Regression  
- **Why chosen:**
  - Interpretable  
  - Stable for small–medium datasets  
  - Suitable for binary classification  
- **Model persistence:** Saved and loaded using `joblib`  
- **Training:** Performed offline via training pipeline  
- **Inference:** Fully integrated into the AI tool  

---

## 🧑‍💻 Technology Stack

- **Language:** Python 3.10  
- **ML:** Scikit-learn, Pandas, NumPy  
- **API:** FastAPI  
- **CLI:** Typer  
- **Packaging:** setuptools, editable install  
- **Logging & Exceptions:** Custom logging and exception handling  
- **Deployment Ready:** Gunicorn / Uvicorn compatible  

---

## 📁 Project Structure

```
AI_Powered_Learning_Intelligence_System/
│
├── artifacts/
│   ├── model/
│   │   └── completion_model.pkl
│   └── reports/
│       └── risk_predictions.csv
│
├── data/
│   └── synthetic_learning_data.csv
│
├── logs/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model Training.ipynb
│   └── Model Training2.ipynb
│
├── src/
│   ├── __pycache__/
│   ├── app/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── main.py
│   │
│   ├── components/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipelines/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   │
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── venv/
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 Getting Started

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model (one-time)
```bash
python -m src.pipelines.training_pipeline
```

### 3️⃣ Run prediction pipeline
```bash
python -m src.pipelines.prediction_pipeline
```

### 4️⃣ Run CLI AI Tool
```bash
python -m src.app.cli
```

### 5️⃣ Run API (optional but recommended)
```bash
uvicorn src.app.main:app --reload
```

Open in browser:
```
http://127.0.0.1:8000/docs
```

---

## 🔌 Interfaces Provided

### ✅ Command Line Interface (CLI)
- Executable AI tool
- Accepts CSV input
- Generates downloadable reports

### ✅ REST API (FastAPI)
- File upload support
- JSON responses
- Swagger UI for testing

---

## 🧪 Testing & Validation

- Input validation via FastAPI
- Sanity checks on predictions
- Explicit error handling using custom exceptions
- Logging for traceability

---

## 🔍 Reproducibility

- Fixed random states
- Version-pinned dependencies
- Saved model artifacts
- Deterministic preprocessing pipelines

---

## ⚖️ Ethical AI & Responsible Usage

- Model predictions are probabilistic, not deterministic
- Risk labels are intended to support early intervention, not penalization
- No sensitive personal attributes are used
- Outputs are explainable and auditable

---

## 🤖 AI Usage Disclosure (Mandatory)

AI tools (including ChatGPT) were used responsibly for:

- Code structure guidance
- Documentation drafting
- Debugging assistance

All:

- Model logic
- Feature engineering
- Pipeline design
- Thresholding logic

were understood, verified, and implemented independently by the author.

No AI-generated code was used without validation.

---

## ✅ Compliance Statement

This submission:

- ❌ Does NOT include notebooks as deliverables
- ✅ Provides an executable AI tool
- ✅ Uses a trained ML model
- ✅ Exposes AI via CLI and API
- ✅ Meets all mandatory admin requirements

---

## 🏁 Final Note

This project focuses on AI engineering, not experimentation.
The system is designed to be usable, explainable, and deployable, aligning fully with the goals of the AI Kata assessment.

---

## 👤 Author

**Aditya Kumar Arya**

📢 **Follow & Connect:** If you liked this project, give it a ⭐ on GitHub and connect with me on [LinkedIn](https://www.linkedin.com/in/aditya-kumar-arya-25b154260/)!

🚀 **Happy Coding!** 🎯