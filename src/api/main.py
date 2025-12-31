"""
FastAPI Backend for Clinical ML Pipeline.

Exposes a single /predict endpoint that runs the complete pipeline:
1. LLM extraction of clinical features
2. Validation and repair
3. Normalization to model input
4. ML inference

Returns comprehensive results including intermediate steps for transparency.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import traceback
from pathlib import Path

from src.services import (
    HybridExtractor,
    LLMResponseValidator,
    FeatureNormalizer,
    ModelInferenceEngine,
    InferenceService,
    NormalizationIssue,
    ValidationSeverity
)
from src.models.schema import RawFeatures, ModelInput
from src.config import get_settings


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    case_study_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Clinical case study text"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "case_study_text": "Patient is a 65-year-old male admitted with severe pneumonia. Started on IV ceftriaxone 2g once daily. Patient has comorbidities including diabetes and hypertension."
            }
        }


class ExtractedFeatureDetail(BaseModel):
    """Detail about a single extracted feature"""
    value: Optional[Any]
    raw_value: Optional[str]
    evidence: Optional[str]
    confidence: float
    unit_conversion: Optional[str] = None


class ValidationMessage(BaseModel):
    """Validation warning or error"""
    severity: str
    field: str
    message: str
    repaired: bool = False


class PredictionResponse(BaseModel):
    """Complete pipeline response"""
    # Final prediction
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

    # Intermediate steps for transparency
    extracted_features: Dict[str, ExtractedFeatureDetail]
    normalized_features: Dict[str, Any]
    validation_messages: List[ValidationMessage]

    # Flags
    requires_human_review: bool
    extraction_method: str  # "llm", "fallback", or "hybrid"

    # Metadata
    inference_time_ms: float
    model_version: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    stage: str  # Which pipeline stage failed


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Clinical ML Inference API",
    description="Production-grade API for antibiotic treatment duration prediction",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

settings = get_settings()

# Initialize extractors
try:
    hybrid_extractor = HybridExtractor(
        primary_provider=settings.primary_llm_provider,
        fallback_provider="gemini" if settings.primary_llm_provider == "openai" else "openai"
    )
except Exception as e:
    print(f"Warning: Could not initialize LLM extractors: {e}")
    hybrid_extractor = None

# Initialize validator
llm_validator = LLMResponseValidator()

# Initialize normalizer
normalizer = FeatureNormalizer()

# Initialize inference engine
try:
    model_path = Path(settings.model_path)
    if model_path.exists():
        InferenceService.initialize(str(model_path))
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": InferenceService._engine is not None if hasattr(InferenceService, '_engine') else False,
        "llm_available": hybrid_extractor is not None
    }


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_duration(request: PredictionRequest):
    """
    Run complete ML pipeline on clinical case study text.

    Pipeline stages:
    1. Extract features using LLM (with fallback to regex)
    2. Validate and repair extracted features
    3. Normalize to model input format
    4. Run ML inference

    Returns comprehensive results including all intermediate steps.
    """

    case_study_text = request.case_study_text.strip()

    # ========================================================================
    # STAGE 1: EXTRACTION
    # ========================================================================

    try:
        if hybrid_extractor is None:
            raise HTTPException(
                status_code=503,
                detail="LLM extraction service not available. Check API keys."
            )

        extraction_result = hybrid_extractor.extract(case_study_text)
        extraction_method = extraction_result.get("extraction_method", "unknown")

        # Convert to RawFeatures
        raw_features = RawFeatures(**extraction_result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Feature extraction failed",
                detail=str(e),
                stage="extraction"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 2: NORMALIZATION
    # ========================================================================

    try:
        normalization_result = normalizer.normalize(raw_features)

        if not normalization_result.success:
            # Check for critical errors
            critical_errors = [
                issue for issue in normalization_result.issues
                if issue.severity == ValidationSeverity.ERROR
            ]

            if critical_errors:
                raise HTTPException(
                    status_code=422,
                    detail=ErrorResponse(
                        error="Feature normalization failed",
                        detail=f"Critical validation errors: {[e.message for e in critical_errors]}",
                        stage="normalization"
                    ).model_dump()
                )

        model_input = normalization_result.model_input

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Feature normalization failed",
                detail=str(e),
                stage="normalization"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 3: INFERENCE
    # ========================================================================

    try:
        prediction_result = InferenceService.predict(
            model_input,
            return_probabilities=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Model inference failed",
                detail=str(e),
                stage="inference"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 4: BUILD RESPONSE
    # ========================================================================

    # Extract feature details
    extracted_features = {}
    for field_name in ['age', 'dosage', 'gender', 'route', 'frequency']:
        field_data = getattr(raw_features, field_name)
        if field_data:
            extracted_features[field_name] = ExtractedFeatureDetail(
                value=field_data.value,
                raw_value=field_data.raw_value,
                evidence=field_data.evidence,
                confidence=field_data.confidence,
                unit_conversion=field_data.unit_conversion
            )

    # Normalized features (the 8 model inputs)
    normalized_features = {
        "Age": model_input.Age,
        "Dosage (gram)": model_input.Dosage_gram,
        "Gender_Male": model_input.Gender_Male,
        "Route_IV": model_input.Route_IV,
        "Route_Oral": model_input.Route_Oral,
        "Frequency_OD": model_input.Frequency_OD,
        "Frequency_QID": model_input.Frequency_QID,
        "Frequency_TDS": model_input.Frequency_TDS
    }

    # Validation messages
    validation_messages = []
    requires_human_review = False

    for issue in normalization_result.issues:
        validation_messages.append(ValidationMessage(
            severity=issue.severity.value,
            field=issue.field,
            message=issue.message,
            repaired=issue.auto_repaired
        ))

        # Flag for human review if critical issues
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
            requires_human_review = True

    # Check for low confidence extractions
    for field_name, field_detail in extracted_features.items():
        if field_detail.confidence < 0.7:
            validation_messages.append(ValidationMessage(
                severity="WARNING",
                field=field_name,
                message=f"Low extraction confidence ({field_detail.confidence:.2f})",
                repaired=False
            ))
            requires_human_review = True

    # Build response
    response = PredictionResponse(
        prediction=prediction_result.prediction,
        confidence=prediction_result.confidence,
        probabilities=prediction_result.probabilities,
        extracted_features=extracted_features,
        normalized_features=normalized_features,
        validation_messages=validation_messages,
        requires_human_review=requires_human_review,
        extraction_method=extraction_method,
        inference_time_ms=prediction_result.inference_time_ms,
        model_version=prediction_result.model_metadata.model_version
    )

    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to prevent stack trace leakage"""
    return {
        "error": "Internal server error",
        "detail": "An unexpected error occurred. Please contact support.",
        "stage": "unknown"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
