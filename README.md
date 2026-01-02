# 🏥 Hospital Treatment Duration Prediction System

> **AI-powered clinical data extraction and treatment duration prediction system**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Recent Updates](#recent-updates)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system combines **Large Language Models (LLMs)** with **Machine Learning** to automatically extract clinical information from unstructured medical documents (PDFs, DOCX) and predict antibiotic treatment duration categories.

### The Problem It Solves

Healthcare professionals often need to:
- ✅ Extract structured data from unstructured clinical documents
- ✅ Handle documents containing single or multiple case studies
- ✅ Predict treatment duration based on patient demographics and medication details
- ✅ Validate and normalize extracted data for reliability

### The Solution

A production-ready pipeline that:
1. **Extracts** clinical features from documents using GPT-4o/Gemini
2. **Validates** and repairs extracted data with confidence scoring
3. **Normalizes** features to model-ready format
4. **Predicts** treatment duration using a trained Decision Tree classifier
5. **Returns** comprehensive results with full transparency

---

## ✨ Features

### 🤖 **LLM-Powered Extraction**
- Multi-provider support (OpenAI GPT-4o, Google Gemini)
- Automatic retry and fallback mechanisms
- Few-shot learning with 7+ clinical examples
- Confidence scoring for each extracted field

### 📄 **Multi-Format Document Support**
- PDF parsing with page-by-page processing
- DOCX/DOC document support
- TXT/MD file support
- File size validation (10MB limit)

### 🔍 **Multi-Case Study Detection** (NEW!)
- Automatically detects single or multiple case studies in one document
- Intelligent case boundary detection
- Processes each case independently
- Returns predictions for all detected cases

### 🛡️ **Robust Validation & Repair**
- JSON response validation and auto-repair
- Evidence-based extraction verification
- Range validation (age: 0-120, dosage: 0-10g)
- Unit conversion validation (mg → g)
- Confidence thresholds with human review flags

### 🎯 **ML Prediction**
- Trained Decision Tree Classifier (77.84% accuracy)
- Predicts: `<5 days`, `5-10 days`, `>10 days`
- Probability scores for each class
- Model metadata and timing information

### 🌐 **Full-Stack Application**
- **Backend**: FastAPI with async support
- **Frontend**: React + TypeScript with modern UI
- **File Upload**: Drag-and-drop interface
- **Results**: Comprehensive visualization with validation messages

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  • File Upload  • Results Display  • Validation UI         │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/JSON
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                            │
│  • /predict           - Single case prediction              │
│  • /predict-from-file - Multi-case document processing      │
│  • /health            - Health check                        │
└─────┬─────────────────────────────────────────────┬─────────┘
      │                                             │
      ▼                                             ▼
┌─────────────────┐                    ┌──────────────────────┐
│  LLM Extraction │                    │  ML Inference        │
│  • OpenAI GPT   │                    │  • Decision Tree     │
│  • Gemini       │                    │  • Probability       │
│  • Fallback     │                    │  • Metadata          │
└─────┬───────────┘                    └──────────────────────┘
      │
      ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Validation    │───▶│ Normalization │───▶│   Prediction    │
│   & Repair      │    │  (8 Features) │    │   (3 Classes)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- OpenAI API key OR Google AI API key

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/Hospital_Details_Analysis.git
cd Hospital_Details_Analysis

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...
```

### 2. Install Dependencies

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 3. Run the Application

**Terminal 1 - Backend:**
```bash
python -m uvicorn src.api.main:app --reload
# API available at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# UI available at http://localhost:5173
```

### 4. Test with Sample Documents

Upload `Case_Document.pdf` or `Case_Document_1.pdf` through the web interface!

---

## 📦 Installation

### Detailed Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; import openai; print('Setup successful!')"
```

### Frontend Setup

```bash
cd frontend
npm install

# Development build
npm run dev

# Production build
npm run build
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

#### Required API Keys

```bash
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-...

# Get from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=AIza...
```

#### Model Configuration

```bash
# Primary provider (openai or gemini)
PRIMARY_LLM_PROVIDER=openai

# Model selection
# Options: gpt-4o (recommended), o1, o1-preview, gpt-4-turbo
OPENAI_MODEL=gpt-4o

# Temperature (0.0 = deterministic, 2.0 = creative)
OPENAI_TEMPERATURE=0.0
```

#### Optional Settings

```bash
# Decision Tree model path
MODEL_PATH=best_dt_classifier_model.joblib

# API server
API_HOST=0.0.0.0
API_PORT=8000

# Caching
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_HOURS=24
```

For all options, see [`.env.example`](.env.example)

---

## 💻 Usage

### Web Interface (Recommended)

1. Start backend and frontend (see Quick Start)
2. Open http://localhost:5173
3. Drag & drop a PDF/DOCX file
4. View extracted features and predictions

### API Usage

#### Single Case Study

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "case_study_text": "A 52-year-old male patient was admitted with pneumonia. Started on IV ceftriaxone 1g once daily."
  }'
```

#### Multi-Case Document Upload

```bash
curl -X POST http://localhost:8000/predict-from-file \
  -F "file=@Case_Document.pdf"
```

#### Python SDK

```python
import requests

# Single case prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"case_study_text": "Patient is 45yo F with UTI. Given ciprofloxacin 500mg PO BD."}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Multi-case file upload
with open("Case_Document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-from-file",
        files={"file": f}
    )
cases = response.json()
print(f"Detected {len(cases)} case studies")
```

---

## 📚 API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /predict`

Single case study prediction from text.

**Request:**
```json
{
  "case_study_text": "Patient is a 62-year-old male..."
}
```

**Response:**
```json
{
  "prediction": "5-10 days",
  "confidence": 0.85,
  "probabilities": {
    "<5 days": 0.10,
    "5-10 days": 0.85,
    ">10 days": 0.05
  },
  "extracted_features": {
    "age": {"value": 62.0, "confidence": 1.0, "evidence": "62-year-old male"},
    "dosage": {"value": 1.0, "confidence": 1.0, "unit_conversion": "1g → 1.0g"},
    ...
  },
  "normalized_features": {
    "Age": 62.0,
    "Dosage (gram)": 1.0,
    "Gender_Male": 1,
    ...
  },
  "validation_messages": [...],
  "requires_human_review": false,
  "extraction_method": "llm",
  "inference_time_ms": 245.3,
  "model_version": "1.0"
}
```

#### `POST /predict-from-file`

Multi-case document processing.

**Request:**
- Form data with file upload
- Supports: PDF, DOCX, DOC, TXT, MD
- Max size: 10MB

**Response:**
```json
[
  {
    "prediction": "5-10 days",
    "confidence": 0.85,
    ...
    // Same structure as /predict response
  },
  {
    "prediction": "<5 days",
    "confidence": 0.92,
    ...
  }
]
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "llm_available": true
}
```

---

## 📁 Project Structure

```
Hospital_Details_Analysis/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   └── schema.py            # Pydantic data models
│   ├── services/
│   │   ├── prompts.py           # Single-case LLM prompts (UPDATED)
│   │   ├── multi_case_prompts.py # Multi-case prompts (FIXED)
│   │   ├── extractor.py         # LLM extraction service
│   │   ├── llm_validator.py     # Response validation
│   │   ├── normalizer.py        # Feature normalization
│   │   ├── inference.py         # ML model inference
│   │   └── fallback_extractor.py # Regex-based fallback
│   ├── utils/
│   │   └── file_parser.py       # Document parsing
│   └── config.py                # Configuration management
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # React application
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   ├── test_inference.py
│   ├── test_normalizer.py
│   └── ...
├── best_dt_classifier_model.joblib  # Trained ML model
├── .env.example                 # Environment template (UPDATED)
├── .gitignore
├── requirements.txt
├── README.md                    # This file
├── CHANGELOG.md                 # Version history (NEW)
└── CONTRIBUTING.md              # Contribution guide (NEW)
```

---

## 🆕 Recent Updates

### Version 2.0.0 (January 2026)

#### 🎉 **Multi-Case Detection Fixed**
- Fixed critical bug preventing detection of multiple case studies
- Updated prompt format to use `{"cases": [...]}` structure
- Now correctly handles both single and multiple case documents

#### 🚀 **Enhanced LLM Prompts**
- Improved confidence scoring with 6 detailed tiers
- Added 2 new examples (sparse info, complex abbreviations)
- Enhanced edge case handling (vague ages, missing units)
- Better medical abbreviation support

#### ⚙️ **Configuration Improvements**
- Model selection via environment variables
- Support for latest GPT models (gpt-4o, o1, o1-preview)
- Enhanced `.env.example` with detailed comments

#### 🐛 **Bug Fixes**
- Fixed route/frequency confusion in multi-case prompt
- Removed incorrect unit_conversion from non-dosage fields
- Standardized value formats (IV, OD, BD, TDS, QID)

See [CHANGELOG.md](CHANGELOG.md) for complete history.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt  # (if available)

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

---

## 📊 Model Details

### Decision Tree Classifier

- **Accuracy**: 77.84%
- **Features**: 8 (Age, Dosage, Gender, Route, Frequency)
- **Classes**: 3 (`<5 days`, `5-10 days`, `>10 days`)
- **Training**: GridSearchCV with 5-fold CV
- **Optimization**: Weighted F1-score

### Input Features

1. **Age** (float): Patient age in years
2. **Dosage (gram)** (float): Medication dosage in grams
3. **Gender_Male** (int): 1=Male, 0=Female
4. **Route_IV** (int): 1=Intravenous, 0=Other
5. **Route_Oral** (int): 1=Oral, 0=Other
6. **Frequency_OD** (int): 1=Once daily, 0=Other
7. **Frequency_QID** (int): 1=Four times daily, 0=Other
8. **Frequency_TDS** (int): 1=Three times daily, 0=Other

---

## 🔧 Troubleshooting

### Common Issues

**1. "LLM extraction service not available"**
- ✅ Check API keys in `.env` file
- ✅ Verify internet connection
- ✅ Ensure OpenAI/Google API quota is available

**2. "Model not found"**
- ✅ Verify `best_dt_classifier_model.joblib` exists in root directory
- ✅ Check `MODEL_PATH` in `.env`

**3. "Only detecting 1 case when there are multiple"**
- ✅ Update to latest version (includes multi-case fix)
- ✅ Restart the server after updating

**4. Frontend won't start**
- ✅ Run `npm install` in frontend directory
- ✅ Check Node.js version (16+ required)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

- **Development Team**: Hospital Details Analysis Contributors
- **LLM Integration**: OpenAI GPT-4o, Google Gemini
- **ML Framework**: Scikit-learn
- **Web Framework**: FastAPI, React

---

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/Hospital_Details_Analysis/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/Hospital_Details_Analysis/discussions)
- 📧 **Email**: your-email@example.com

---

## 🔮 Roadmap

- [ ] Support for more document formats (RTF, HTML)
- [ ] Real-time collaborative editing
- [ ] Custom model training interface
- [ ] Export to EHR systems (HL7, FHIR)
- [ ] Multi-language support
- [ ] Mobile application

---

**⭐ If you find this project useful, please star it on GitHub!**

---

*Last Updated: January 2026*
*Version: 2.0.0*
