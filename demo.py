"""
Demo script showcasing the complete ML pipeline.

This script demonstrates:
1. LLM-based feature extraction from clinical text
2. Validation of extracted features
3. Transformation to model input format
4. Prediction using the trained model
"""

import os
import json
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.services.extractor import LLMExtractor
from src.services.transformer import FeatureTransformer
from src.services.validator import ExtractionValidator
from src.models.schema import PredictionOutput


def load_model(model_path: str = "best_dt_classifier_model.joblib"):
    """Load the trained Decision Tree model"""
    print(f"\n{'='*70}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*70}")
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Features: {model.n_features_in_}")
    print(f"  - Classes: {model.classes_}")
    return model


def demonstrate_pipeline(case_study_text: str, model):
    """
    Demonstrate the complete pipeline for a single case study.

    Pipeline steps:
    1. Extract features using LLM
    2. Validate extraction
    3. Transform to model input
    4. Predict duration category
    """
    print(f"\n{'='*70}")
    print("PIPELINE DEMONSTRATION")
    print(f"{'='*70}")

    print("\n📄 Input Case Study:")
    print(f"   {case_study_text}")

    # Step 1: Extract features using LLM
    print(f"\n{'─'*70}")
    print("STEP 1: LLM Feature Extraction")
    print(f"{'─'*70}")

    extractor = LLMExtractor(
        primary_provider="openai",
        fallback_provider="gemini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    try:
        extraction_result = extractor.extract(case_study_text)
        raw_features = extraction_result["raw_features"]

        print(f"✓ Extraction completed")
        print(f"  - Provider: {extraction_result['provider_used']}")
        print(f"  - Time: {extraction_result['extraction_time_ms']:.2f}ms")

        if extraction_result.get('error'):
            print(f"  ⚠ Warning: {extraction_result['error']}")

        print(f"\n  Extracted Features:")
        for field_name in ["age", "dosage", "gender", "route", "frequency"]:
            field = getattr(raw_features, field_name)
            print(f"    • {field_name.capitalize()}:")
            print(f"        Value: {field.value}")
            print(f"        Raw: {field.raw_value}")
            print(f"        Evidence: {field.evidence}")
            print(f"        Confidence: {field.confidence}")
            if hasattr(field, 'unit_conversion') and field.unit_conversion:
                print(f"        Conversion: {field.unit_conversion}")

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return

    # Step 2: Validate extraction
    print(f"\n{'─'*70}")
    print("STEP 2: Validation")
    print(f"{'─'*70}")

    validation_result = ExtractionValidator.validate_all(raw_features, case_study_text)

    if validation_result.is_valid:
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")

    if validation_result.errors:
        print(f"\n  Errors ({len(validation_result.errors)}):")
        for error in validation_result.errors:
            print(f"    • {error}")

    if validation_result.warnings:
        print(f"\n  Warnings ({len(validation_result.warnings)}):")
        for warning in validation_result.warnings:
            print(f"    • {warning}")

    if not validation_result.errors:
        print("\n  ✓ No validation errors")

    # Step 3: Transform to model input
    print(f"\n{'─'*70}")
    print("STEP 3: Feature Transformation")
    print(f"{'─'*70}")

    try:
        transform_result = FeatureTransformer.transform_with_metadata(raw_features)
        model_input = transform_result["model_input"]
        metadata = transform_result["transformation_metadata"]

        print("✓ Transformation completed")
        print(f"\n  Model Input Features:")
        print(f"    • Age: {model_input.Age}")
        print(f"    • Dosage (gram): {model_input.Dosage_gram}")
        print(f"    • Gender_Male: {model_input.Gender_Male}")
        print(f"    • Route_IV: {model_input.Route_IV}")
        print(f"    • Route_Oral: {model_input.Route_Oral}")
        print(f"    • Frequency_OD: {model_input.Frequency_OD}")
        print(f"    • Frequency_QID: {model_input.Frequency_QID}")
        print(f"    • Frequency_TDS: {model_input.Frequency_TDS}")

        print(f"\n  Transformation Summary:")
        for field_name, field_meta in metadata.items():
            if field_meta.get('imputed'):
                print(f"    ⚠ {field_name}: Value was imputed (not found in text)")

    except Exception as e:
        print(f"✗ Transformation failed: {e}")
        return

    # Step 4: Predict using model
    print(f"\n{'─'*70}")
    print("STEP 4: Model Prediction")
    print(f"{'─'*70}")

    try:
        import time
        import numpy as np

        start_time = time.time()

        # Convert to DataFrame for prediction
        input_df = model_input.to_dataframe()

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        inference_time = (time.time() - start_time) * 1000

        # Map probabilities to class names
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(model.classes_, probabilities)
        }

        print("✓ Prediction completed")
        print(f"  - Inference time: {inference_time:.2f}ms")
        print(f"\n  📊 PREDICTION: {prediction}")
        print(f"\n  Probability Distribution:")
        for class_name in sorted(prob_dict.keys()):
            prob = prob_dict[class_name]
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"    {class_name:>10}: [{bar}] {prob:.1%}")

        # Create prediction output
        prediction_output = PredictionOutput(
            duration_category=prediction,
            probabilities=prob_dict,
            model_version="1.0",
            inference_time_ms=inference_time
        )

        print(f"\n  JSON Output:")
        print(f"  {json.dumps(prediction_output.model_dump(), indent=2)}")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")


def main():
    """Main demo function"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║   Hospital Treatment Duration Prediction - Pipeline Demo         ║
║   ML-powered prediction from unstructured clinical text           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Check API keys
    print("\n🔑 Checking API Keys...")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not openai_key and not google_key:
        print("❌ ERROR: No API keys found!")
        print("   Please set OPENAI_API_KEY or GOOGLE_API_KEY in .env file")
        return

    if openai_key:
        print(f"   ✓ OpenAI API key found (***{openai_key[-4:]})")
    if google_key:
        print(f"   ✓ Google API key found (***{google_key[-4:]})")

    # Load model
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return

    # Test case studies
    case_studies = [
        """Patient is a 68-year-old female presenting with community-acquired pneumonia.
Started on IV ceftriaxone 2g once daily. Patient has history of diabetes mellitus.""",

        """45yo M with acute UTI. Commenced ciprofloxacin 500mg PO BD for 7 days.
Vitals stable, responding well to treatment.""",

        """Elderly woman with sepsis. Prescribed gentamicin 80mg intravenously
three times daily. Close monitoring required."""
    ]

    # Run pipeline for each case study
    for i, case_study in enumerate(case_studies, 1):
        print(f"\n\n{'#'*70}")
        print(f"# CASE STUDY {i} of {len(case_studies)}")
        print(f"{'#'*70}")

        demonstrate_pipeline(case_study, model)

        if i < len(case_studies):
            input("\n⏸  Press Enter to continue to next case study...")

    print(f"\n\n{'='*70}")
    print("DEMO COMPLETED")
    print(f"{'='*70}")
    print("\nThank you for trying the Hospital Treatment Duration Prediction pipeline!")
    print("For more information, see README.md")


if __name__ == "__main__":
    main()
