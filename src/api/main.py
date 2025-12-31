"""
FastAPI Backend for Clinical ML Pipeline.

Exposes a single /predict endpoint that runs the complete pipeline:
1. LLM extraction of clinical features
2. Validation and repair
3. Normalization to model input
4. ML inference

Returns comprehensive results including intermediate steps for transparency.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import traceback
from pathlib import Path
import json

from src.services import (
    HybridExtractor,
    LLMResponseValidator,
    FeatureNormalizer,
    ModelInferenceEngine,
    InferenceService,
    NormalizationIssue,
    ValidationSeverity
)
from src.services.multi_case_prompts import MultiCasePromptBuilder
from src.models.schema import RawFeatures, ModelInput
from src.config import get_settings
from src.utils import DocumentParser, FileParsingError


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
hybrid_extractor = None
extractor_error = None
llm_extractor = None

try:
    # First create the LLM extractor
    from src.services.extractor import LLMExtractor
    llm_extractor = LLMExtractor(
        primary_provider=settings.primary_llm_provider,
        fallback_provider="gemini" if settings.primary_llm_provider == "openai" else "openai"
    )

    # Then create the hybrid extractor
    hybrid_extractor = HybridExtractor(
        llm_extractor=llm_extractor,
        confidence_threshold=0.6,
        use_fallback_for_missing=True
    )
    print(f"[OK] LLM extractors initialized (primary: {settings.primary_llm_provider})")
except Exception as e:
    extractor_error = str(e)
    print(f"[WARNING] Could not initialize LLM extractors: {e}")
    print(f"  Make sure API keys are set in .env file:")
    print(f"  - OPENAI_API_KEY (for OpenAI GPT-4)")
    print(f"  - GOOGLE_API_KEY (for Google Gemini)")
    hybrid_extractor = None
    llm_extractor = None

# Initialize validator
llm_validator = LLMResponseValidator()

# Initialize normalizer
normalizer = FeatureNormalizer()

# Initialize inference engine
try:
    model_path = Path(settings.model_path)
    if model_path.exists():
        InferenceService.initialize(str(model_path))
        print(f"[OK] Model loaded from {model_path}")
    else:
        print(f"[WARNING] Model not found at {model_path}")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "model_loaded": InferenceService._engine is not None if hasattr(InferenceService, '_engine') else False,
        "llm_available": hybrid_extractor is not None
    }

    if hybrid_extractor is None and extractor_error:
        health_status["llm_error"] = extractor_error
        health_status["status"] = "degraded"

    return health_status


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

        if not normalization_result.is_valid:
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
            repaired=bool(issue.rule_applied)  # If rule was applied, it was repaired
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
# FILE UPLOAD ENDPOINT (MULTI-CASE STUDY EXTRACTION)
# ============================================================================

@app.post("/predict-from-file", response_model=List[PredictionResponse])
async def predict_from_file(file: UploadFile = File(...)):
    """
    Extract multiple case studies from uploaded document and predict for each.

    Supports: PDF, DOCX, TXT files
    Handles: Documents containing 1 or more case studies

    Returns array of predictions, one per case study found.
    """

    # ========================================================================
    # STAGE 1: FILE VALIDATION
    # ========================================================================

    # Validate file extension
    filename = file.filename
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
    file_ext = Path(filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Unsupported file type",
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}",
                stage="file_validation"
            ).model_dump()
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="File read failed",
                detail=str(e),
                stage="file_read"
            ).model_dump()
        )

    # Validate file size (10 MB limit)
    if not DocumentParser.validate_file_size(len(file_content), max_size_mb=10):
        raise HTTPException(
            status_code=413,
            detail=ErrorResponse(
                error="File too large",
                detail="Maximum file size is 10 MB",
                stage="file_validation"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 2: FILE PARSING
    # ========================================================================

    try:
        document_text = DocumentParser.parse_file(filename, file_content)
    except FileParsingError as e:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error="File parsing failed",
                detail=str(e),
                stage="file_parsing"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 3: MULTI-CASE EXTRACTION
    # ========================================================================

    try:
        if hybrid_extractor is None or llm_extractor is None:
            error_detail = "LLM extraction service not available."
            if extractor_error:
                error_detail += f" Initialization error: {extractor_error}"
            error_detail += " Please configure OPENAI_API_KEY or GOOGLE_API_KEY in your .env file."

            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error="LLM Service Unavailable",
                    detail=error_detail,
                    stage="initialization"
                ).model_dump()
            )

        # Use multi-case prompt
        if settings.primary_llm_provider == "openai":
            messages = MultiCasePromptBuilder.build_openai_messages(document_text)
            response = llm_extractor._openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}  # Note: Will return object, we'll parse array
            )
            result_text = response.choices[0].message.content
        else:  # gemini
            prompt = MultiCasePromptBuilder.build_gemini_prompt(document_text)
            response = llm_extractor._gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0}
            )
            result_text = response.text

        # Parse JSON array
        try:
            # Handle both array and object responses
            parsed = json.loads(result_text)
            if isinstance(parsed, dict):
                # If single object, wrap in array
                case_studies = [parsed]
            elif isinstance(parsed, list):
                case_studies = parsed
            else:
                raise ValueError("Unexpected response format")
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="LLM response parsing failed",
                    detail=f"Could not parse JSON: {str(e)}",
                    stage="extraction"
                ).model_dump()
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Multi-case extraction failed",
                detail=str(e),
                stage="extraction"
            ).model_dump()
        )

    # ========================================================================
    # STAGE 4: PROCESS EACH CASE STUDY
    # ========================================================================

    results = []

    for idx, case_data in enumerate(case_studies, 1):
        try:
            # Extract fields from case data
            raw_features_dict = {
                "age": case_data.get("age"),
                "dosage": case_data.get("dosage"),
                "gender": case_data.get("gender"),
                "route": case_data.get("route"),
                "frequency": case_data.get("frequency")
            }

            raw_features = RawFeatures(**raw_features_dict)

            # Normalize
            normalization_result = normalizer.normalize(raw_features)

            if not normalization_result.is_valid:
                critical_errors = [
                    issue for issue in normalization_result.issues
                    if issue.severity == ValidationSeverity.ERROR
                ]
                if critical_errors:
                    # Skip this case, add error result
                    results.append(PredictionResponse(
                        prediction="ERROR",
                        confidence=0.0,
                        probabilities={},
                        extracted_features={},
                        normalized_features={},
                        validation_messages=[ValidationMessage(
                            severity="ERROR",
                            field="normalization",
                            message=f"Case {idx} failed normalization: {[e.message for e in critical_errors]}",
                            repaired=False
                        )],
                        requires_human_review=True,
                        extraction_method="llm_multi_case",
                        inference_time_ms=0.0,
                        model_version="1.0"
                    ))
                    continue

            model_input = normalization_result.model_input

            # Run inference
            prediction_result = InferenceService.predict(model_input, return_probabilities=True)

            # Build response (similar to single case)
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

            validation_messages = []
            requires_human_review = False

            for issue in normalization_result.issues:
                validation_messages.append(ValidationMessage(
                    severity=issue.severity.value,
                    field=issue.field,
                    message=issue.message,
                    repaired=bool(issue.rule_applied)  # If rule was applied, it was repaired
                ))
                if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
                    requires_human_review = True

            # Check confidence
            for field_name, field_detail in extracted_features.items():
                if field_detail.confidence < 0.7:
                    validation_messages.append(ValidationMessage(
                        severity="WARNING",
                        field=field_name,
                        message=f"Low extraction confidence ({field_detail.confidence:.2f})",
                        repaired=False
                    ))
                    requires_human_review = True

            # Add case ID to validation
            case_id = case_data.get("case_id", f"case_{idx}")
            validation_messages.insert(0, ValidationMessage(
                severity="INFO",
                field="case_id",
                message=f"Case Study ID: {case_id}",
                repaired=False
            ))

            results.append(PredictionResponse(
                prediction=prediction_result.prediction,
                confidence=prediction_result.confidence,
                probabilities=prediction_result.probabilities,
                extracted_features=extracted_features,
                normalized_features=normalized_features,
                validation_messages=validation_messages,
                requires_human_review=requires_human_review,
                extraction_method="llm_multi_case",
                inference_time_ms=prediction_result.inference_time_ms,
                model_version=prediction_result.model_metadata.model_version
            ))

        except Exception as e:
            # Add error result for this case
            results.append(PredictionResponse(
                prediction="ERROR",
                confidence=0.0,
                probabilities={},
                extracted_features={},
                normalized_features={},
                validation_messages=[ValidationMessage(
                    severity="ERROR",
                    field="processing",
                    message=f"Case {idx} processing failed: {str(e)}",
                    repaired=False
                )],
                requires_human_review=True,
                extraction_method="llm_multi_case",
                inference_time_ms=0.0,
                model_version="1.0"
            ))

    return results


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
