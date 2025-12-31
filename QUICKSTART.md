# Quick Start Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file from the example:

```bash
copy .env.example .env
```

Edit `.env` and add your API keys:

```env
# At minimum, add one of these:
OPENAI_API_KEY=sk-your-key-here
# OR
GOOGLE_API_KEY=your-google-key-here
```

## Running the Demo

```bash
python demo.py
```

The demo will:
- ✅ Load the trained Decision Tree model
- ✅ Extract features from 3 sample case studies using LLM
- ✅ Validate all extractions
- ✅ Transform features to model input format
- ✅ Generate predictions with probabilities

## Testing Individual Components

### Test LLM Extraction Only

```python
from dotenv import load_dotenv
load_dotenv()

from src.services.extractor import LLMExtractor

extractor = LLMExtractor(primary_provider="openai")

case_study = """
Patient is a 62-year-old male with pneumonia.
Prescribed ceftriaxone 1g IV twice daily.
"""

result = extractor.extract(case_study)
print(result["raw_features"])
```

### Test Feature Transformation

```python
from src.services.transformer import FeatureTransformer
from src.models.schema import RawFeatures, ExtractedField

# Create sample raw features
raw = RawFeatures(
    age=ExtractedField(value=60.0, raw_value="60", evidence="60 years old", confidence=1.0),
    dosage=ExtractedField(value=0.5, raw_value="500mg", evidence="500mg dose", confidence=1.0, unit_conversion="500mg → 0.5g"),
    gender=ExtractedField(value="male", raw_value="male", evidence="male patient", confidence=1.0),
    route=ExtractedField(value="oral", raw_value="PO", evidence="PO route", confidence=1.0),
    frequency=ExtractedField(value="BD", raw_value="twice daily", evidence="twice daily", confidence=1.0)
)

model_input = FeatureTransformer.transform(raw)
print(model_input)
```

### Test Model Prediction

```python
import joblib
from src.models.schema import ModelInput

# Load model
model = joblib.load("best_dt_classifier_model.joblib")

# Create input
model_input = ModelInput(
    Age=60.0,
    **{"Dosage (gram)": 0.5},
    Gender_Male=1,
    Route_IV=0,
    Route_Oral=1,
    Frequency_OD=0,
    Frequency_QID=0,
    Frequency_TDS=0
)

# Predict
prediction = model.predict(model_input.to_dataframe())[0]
probabilities = model.predict_proba(model_input.to_dataframe())[0]

print(f"Prediction: {prediction}")
print(f"Probabilities: {dict(zip(model.classes_, probabilities))}")
```

## Project Structure

```
src/
├── models/
│   └── schema.py          # Data models (RawFeatures, ModelInput, etc.)
├── services/
│   ├── prompts.py         # LLM prompts and few-shot examples
│   ├── extractor.py       # LLM extraction service
│   ├── transformer.py     # Feature transformation
│   └── validator.py       # Extraction validation
└── config.py              # Configuration management

demo.py                    # Interactive demo script
```

## Common Use Cases

### 1. Single Prediction

```python
from src.services.extractor import LLMExtractor
from src.services.transformer import FeatureTransformer
import joblib

# Initialize
extractor = LLMExtractor(primary_provider="openai")
model = joblib.load("best_dt_classifier_model.joblib")

# Process
case_study = "Your case study text here..."
extraction = extractor.extract(case_study)
model_input = FeatureTransformer.transform(extraction["raw_features"])
prediction = model.predict(model_input.to_dataframe())[0]

print(f"Duration: {prediction}")
```

### 2. Batch Processing

```python
case_studies = [
    "Case 1 text...",
    "Case 2 text...",
    "Case 3 text..."
]

results = []
for case_study in case_studies:
    extraction = extractor.extract(case_study)
    model_input = FeatureTransformer.transform(extraction["raw_features"])
    prediction = model.predict(model_input.to_dataframe())[0]
    results.append(prediction)

print(results)
```

### 3. With Validation

```python
from src.services.validator import ExtractionValidator

extraction = extractor.extract(case_study)
raw_features = extraction["raw_features"]

# Validate extraction
validation = ExtractionValidator.validate_all(raw_features, case_study)

if validation.is_valid:
    print("✓ Extraction valid")
    model_input = FeatureTransformer.transform(raw_features)
    # ... continue with prediction
else:
    print("✗ Validation errors:")
    for error in validation.errors:
        print(f"  - {error}")
```

## Troubleshooting

### "No API keys found"
- Make sure `.env` file exists in the project root
- Check that you've added `OPENAI_API_KEY` or `GOOGLE_API_KEY`
- Restart your Python session after creating `.env`

### "Model file not found"
- Ensure `best_dt_classifier_model.joblib` is in the project root
- Check the `MODEL_PATH` in your `.env` file

### "Validation failed"
- Review the validation errors/warnings
- Check if the LLM extracted evidence from the actual text
- Verify confidence scores are reasonable
- Inspect the original case study for the required information

### LLM extraction timeout
- Check your internet connection
- Verify your API key is valid
- Check API rate limits
- Try using the fallback provider

## Next Steps

1. ✅ Run `demo.py` to see the full pipeline
2. 📖 Read `PROJECT_STRUCTURE.md` for detailed architecture
3. 🔧 Customize prompts in `src/services/prompts.py` for your use case
4. 🚀 Build a REST API using the services
5. 📊 Monitor extraction quality and model performance

## Getting Help

- Check `PROJECT_STRUCTURE.md` for architecture details
- Review `README.md` for model documentation
- Inspect the code - everything is well-documented!

---

Happy predicting! 🏥📊
