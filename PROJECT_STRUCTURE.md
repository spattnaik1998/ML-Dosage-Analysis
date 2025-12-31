# Hospital Treatment Duration Prediction - Project Structure

## Overview

This project implements a production-grade ML pipeline for predicting antibiotic treatment duration from unstructured clinical case studies using LLM-based feature extraction.

## Directory Structure

```
Hospital_Details_Analysis/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── schema.py                # Pydantic data models
│   └── services/
│       ├── __init__.py
│       ├── prompts.py               # LLM extraction prompts
│       ├── extractor.py             # LLM extraction service
│       ├── transformer.py           # Feature transformation
│       └── validator.py             # Extraction validation
├── tests/                           # Unit tests (to be added)
├── demo.py                          # Interactive demo script
├── best_dt_classifier_model.joblib  # Trained ML model
├── decision_tree_classifier.py      # Original training script
├── Hopsital Dataset.csv             # Training dataset
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example environment variables
├── README.md                        # Model documentation
└── PROJECT_STRUCTURE.md             # This file

```

## Core Components

### 1. Data Models (`src/models/schema.py`)

**ExtractedField**
- Structure for individual LLM-extracted fields
- Contains: value, raw_value, evidence, confidence

**RawFeatures**
- LLM extraction output format
- Fields: age, dosage, gender, route, frequency
- Each field is an ExtractedField with confidence scores

**ModelInput**
- Canonical ML model input schema
- 8 features: Age, Dosage (gram), Gender_Male, Route_IV, Route_Oral, Frequency_OD, Frequency_QID, Frequency_TDS
- Includes validation rules and encoding logic

**PredictionOutput**
- Model prediction result format
- Contains: duration_category, probabilities, model_version, inference_time_ms

### 2. LLM Services

**Prompts (`src/services/prompts.py`)**
- `SYSTEM_PROMPT`: Comprehensive instructions for LLM
- `USER_PROMPT_TEMPLATE`: Template for case study input
- `FEW_SHOT_EXAMPLES`: 5 curated examples for in-context learning
- `PromptBuilder`: Formats prompts for OpenAI/Gemini APIs

**Extractor (`src/services/extractor.py`)**
- `LLMExtractor`: Main extraction service
  - Supports OpenAI GPT-4o and Google Gemini
  - Automatic retry with exponential backoff
  - Fallback between providers
- `CachedLLMExtractor`: Extends with caching support
  - Reduces API calls and costs
  - Configurable TTL

**Transformer (`src/services/transformer.py`)**
- `FeatureTransformer`: Converts RawFeatures → ModelInput
  - Parses age from various formats
  - Converts dosage units (mg → g, mcg → g)
  - Encodes categorical variables (gender, route, frequency)
  - Handles missing values with median imputation

**Validator (`src/services/validator.py`)**
- `ExtractionValidator`: Validates LLM output
  - Checks evidence exists in original text
  - Validates confidence scores
  - Verifies unit conversions
  - Detects hallucinations
- `ValidationResult`: Structured validation report

### 3. Configuration (`src/config.py`)

**Settings**
- Loads configuration from environment variables
- Validates API keys and model paths
- Supports .env file for local development

**Key Settings:**
- LLM provider selection (primary/fallback)
- Model paths and versions
- Cache configuration
- API server settings
- Feature flags

## Data Flow

```
┌─────────────────────┐
│  Case Study Text    │
│  (Unstructured)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLM Extractor      │  ← Uses prompts.py
│  (OpenAI/Gemini)    │  ← Retry + fallback
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RawFeatures        │  ← With evidence & confidence
│  (age, dosage, etc) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Validator          │  ← Check hallucinations
│  (Evidence-based)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Transformer        │  ← Normalize & encode
│  (Feature Eng)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ModelInput         │  ← 8 validated features
│  (Canonical Schema) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Decision Tree      │  ← Trained model
│  Classifier         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Prediction Output  │
│  (<5, 5-10, >10 d)  │
└─────────────────────┘
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
```

### 3. Run Demo

```bash
python demo.py
```

The demo will:
1. Load the trained model
2. Extract features from sample case studies using LLM
3. Validate extractions
4. Transform to model input
5. Generate predictions

## Usage Examples

### Basic Usage

```python
from src.services.extractor import LLMExtractor
from src.services.transformer import FeatureTransformer
from src.services.validator import ExtractionValidator
import joblib

# Load model
model = joblib.load("best_dt_classifier_model.joblib")

# Initialize extractor
extractor = LLMExtractor(primary_provider="openai")

# Extract features
case_study = "Patient is 60yo male, prescribed 500mg oral antibiotic once daily"
result = extractor.extract(case_study)
raw_features = result["raw_features"]

# Validate
validation = ExtractionValidator.validate_all(raw_features, case_study)
if not validation.is_valid:
    print("Validation errors:", validation.errors)

# Transform
model_input = FeatureTransformer.transform(raw_features)

# Predict
prediction = model.predict(model_input.to_dataframe())[0]
print(f"Predicted duration: {prediction}")
```

### With Caching

```python
from src.services.extractor import CachedLLMExtractor
import redis

# Setup Redis cache
cache = redis.Redis(host='localhost', port=6379, db=0)

# Use cached extractor
extractor = CachedLLMExtractor(
    cache_backend=cache,
    primary_provider="openai"
)

# First call hits API
result1 = extractor.extract(case_study)
print(f"From cache: {result1['from_cache']}")  # False

# Second call uses cache
result2 = extractor.extract(case_study)
print(f"From cache: {result2['from_cache']}")  # True
```

## Key Features

### 1. Production-Grade Prompts
- Strict JSON output format
- Anti-hallucination safeguards
- Confidence scoring (0.0 - 1.0)
- Evidence extraction from source text
- 5 few-shot examples for in-context learning

### 2. Robust Extraction
- Automatic retry with exponential backoff
- Fallback between OpenAI and Gemini
- Comprehensive error handling
- Request/response logging

### 3. Validation Pipeline
- Evidence verification against source text
- Confidence score validation
- Unit conversion verification
- Hallucination detection

### 4. Feature Engineering
- Automatic unit conversion (mg/mcg → grams)
- Categorical encoding (gender, route, frequency)
- Missing value imputation (median)
- Consistent transformation logic

### 5. Modular Architecture
- Clear separation of concerns
- Easy to extend or replace components
- Comprehensive type hints
- Pydantic validation throughout

## Next Steps

### Immediate (Done ✓)
- [x] Core schema definitions
- [x] LLM extraction service
- [x] Feature transformation
- [x] Validation pipeline
- [x] Demo script

### Short-term (To Do)
- [ ] Unit tests for all components
- [ ] FastAPI REST API endpoint
- [ ] Batch processing support
- [ ] Redis caching integration
- [ ] Database persistence layer

### Long-term
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Monitoring and alerting
- [ ] Web UI dashboard
- [ ] Docker containerization
- [ ] Kubernetes deployment

## Development Guidelines

### Adding New Features

1. Update schema in `src/models/schema.py`
2. Modify extraction prompt in `src/services/prompts.py`
3. Update transformer logic in `src/services/transformer.py`
4. Add validation rules in `src/services/validator.py`
5. Write tests

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_extractor.py
```

### Code Style

- Follow PEP 8
- Use type hints throughout
- Document with docstrings (Google style)
- Keep functions small and focused
- Use Pydantic for validation

## Troubleshooting

### Common Issues

**LLM API Errors**
- Check API keys in `.env`
- Verify API quota/limits
- Check network connectivity
- Review rate limiting

**Validation Failures**
- Check extraction evidence
- Verify confidence scores
- Review unit conversions
- Inspect original case study text

**Model Prediction Errors**
- Ensure 8 features are provided
- Check feature value ranges
- Verify feature order matches training
- Check for NaN/null values

## Contributing

1. Follow the modular architecture
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
5. Update documentation

## License

See LICENSE file for details.

---

**Version:** 1.0.0
**Last Updated:** 2024
**Maintainer:** ML Engineering Team
