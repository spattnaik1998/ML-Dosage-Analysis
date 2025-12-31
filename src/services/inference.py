"""
ML Inference Service for Treatment Duration Prediction.

This module loads the pre-trained Decision Tree classifier and provides
production-grade inference with input validation, error handling, and
comprehensive response metadata.
"""

import joblib
import time
from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from src.models.schema import ModelInput, PredictionOutput


# ============================================================================
# MODEL METADATA
# ============================================================================

@dataclass
class ModelMetadata:
    """Metadata about the loaded model"""
    model_path: str
    model_type: str
    n_features: int
    feature_names: List[str]
    classes: List[str]
    model_version: str = "1.0"
    scikit_learn_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "classes": self.classes,
            "model_version": self.model_version,
            "scikit_learn_version": self.scikit_learn_version
        }


# ============================================================================
# PREDICTION RESULT
# ============================================================================

@dataclass
class PredictionResult:
    """Complete prediction result with metadata"""
    prediction: str
    probabilities: dict[str, float]
    confidence: float  # Max probability
    inference_time_ms: float
    model_metadata: ModelMetadata
    input_features: dict
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "inference_time_ms": self.inference_time_ms,
            "model_metadata": self.model_metadata.to_dict(),
            "input_features": self.input_features,
            "warnings": self.warnings
        }


# ============================================================================
# INFERENCE ERRORS
# ============================================================================

class InferenceError(Exception):
    """Base exception for inference errors"""
    pass


class ModelLoadError(InferenceError):
    """Error loading model file"""
    pass


class InputValidationError(InferenceError):
    """Error validating input features"""
    pass


class PredictionError(InferenceError):
    """Error during prediction"""
    pass


# ============================================================================
# MODEL INFERENCE ENGINE
# ============================================================================

class ModelInferenceEngine:
    """
    Production-grade inference engine for Decision Tree classifier.

    Features:
    - Lazy model loading (load on first prediction)
    - Input schema validation
    - Feature order enforcement
    - Type checking
    - Comprehensive error handling
    - Performance metrics
    """

    # Expected feature names in exact order
    EXPECTED_FEATURES = [
        'Age',
        'Dosage (gram)',
        'Gender_Male',
        'Route_IV',
        'Route_Oral',
        'Frequency_OD',
        'Frequency_QID',
        'Frequency_TDS'
    ]

    # Expected data types
    EXPECTED_DTYPES = {
        'Age': (int, float, np.integer, np.floating),
        'Dosage (gram)': (int, float, np.integer, np.floating),
        'Gender_Male': (int, np.integer),
        'Route_IV': (int, np.integer),
        'Route_Oral': (int, np.integer),
        'Frequency_OD': (int, np.integer),
        'Frequency_QID': (int, np.integer),
        'Frequency_TDS': (int, np.integer)
    }

    # Expected classes
    EXPECTED_CLASSES = ['<5 days', '5-10 days', '>10 days']

    def __init__(self, model_path: str, model_version: str = "1.0"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to joblib model file
            model_version: Model version identifier
        """
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self._is_loaded = False

    def load_model(self) -> ModelMetadata:
        """
        Load model from disk and extract metadata.

        Returns:
            ModelMetadata object

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._is_loaded:
            return self.metadata

        try:
            # Check file exists
            if not self.model_path.exists():
                raise ModelLoadError(f"Model file not found: {self.model_path}")

            # Load model
            self.model = joblib.load(self.model_path)

            # Extract metadata
            self.metadata = self._extract_metadata()

            # Validate model
            self._validate_model()

            self._is_loaded = True
            return self.metadata

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from loaded model"""
        # Get model type
        model_type = type(self.model).__name__

        # Get number of features
        n_features = self.model.n_features_in_

        # Get feature names if available
        feature_names = (
            list(self.model.feature_names_in_)
            if hasattr(self.model, 'feature_names_in_')
            else self.EXPECTED_FEATURES
        )

        # Get classes
        classes = list(self.model.classes_) if hasattr(self.model, 'classes_') else self.EXPECTED_CLASSES

        # Get scikit-learn version
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except:
            sklearn_version = None

        return ModelMetadata(
            model_path=str(self.model_path),
            model_type=model_type,
            n_features=n_features,
            feature_names=feature_names,
            classes=classes,
            model_version=self.model_version,
            scikit_learn_version=sklearn_version
        )

    def _validate_model(self):
        """Validate loaded model has required attributes"""
        required_methods = ['predict', 'predict_proba']

        for method in required_methods:
            if not hasattr(self.model, method):
                raise ModelLoadError(f"Model missing required method: {method}")

        # Validate number of features
        if self.metadata.n_features != len(self.EXPECTED_FEATURES):
            raise ModelLoadError(
                f"Model expects {self.metadata.n_features} features, "
                f"but {len(self.EXPECTED_FEATURES)} are configured"
            )

        # Validate classes (with some flexibility in ordering)
        model_classes_set = set(self.metadata.classes)
        expected_classes_set = set(self.EXPECTED_CLASSES)

        if model_classes_set != expected_classes_set:
            raise ModelLoadError(
                f"Model classes {model_classes_set} don't match expected {expected_classes_set}"
            )

    def predict(
        self,
        model_input: ModelInput,
        return_probabilities: bool = True
    ) -> PredictionResult:
        """
        Run inference on input features.

        Args:
            model_input: ModelInput with validated features
            return_probabilities: Whether to return class probabilities

        Returns:
            PredictionResult with prediction and metadata

        Raises:
            InputValidationError: If input validation fails
            PredictionError: If prediction fails
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Convert to DataFrame
            input_df = self._prepare_input(model_input)

            # Validate input
            self._validate_input(input_df)

            # Run prediction
            prediction = self.model.predict(input_df)[0]

            # Get probabilities
            if return_probabilities:
                probabilities_array = self.model.predict_proba(input_df)[0]
                probabilities = {
                    class_name: float(prob)
                    for class_name, prob in zip(self.metadata.classes, probabilities_array)
                }
                confidence = float(max(probabilities_array))
            else:
                probabilities = {}
                confidence = 1.0

            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000

            # Prepare input features dict
            input_features = model_input.model_dump()

            return PredictionResult(
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
                model_metadata=self.metadata,
                input_features=input_features,
                warnings=[]
            )

        except (InputValidationError, PredictionError):
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}") from e

    def predict_batch(
        self,
        model_inputs: List[ModelInput],
        return_probabilities: bool = True
    ) -> List[PredictionResult]:
        """
        Run batch inference on multiple inputs.

        Args:
            model_inputs: List of ModelInput objects
            return_probabilities: Whether to return class probabilities

        Returns:
            List of PredictionResult objects
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        results = []

        for model_input in model_inputs:
            try:
                result = self.predict(model_input, return_probabilities)
                results.append(result)
            except Exception as e:
                # Create error result
                results.append(PredictionResult(
                    prediction=None,
                    probabilities={},
                    confidence=0.0,
                    inference_time_ms=0.0,
                    model_metadata=self.metadata,
                    input_features=model_input.model_dump(),
                    warnings=[f"Prediction failed: {str(e)}"]
                ))

        return results

    def _prepare_input(self, model_input: ModelInput) -> pd.DataFrame:
        """
        Convert ModelInput to DataFrame with correct column order.

        Args:
            model_input: Validated ModelInput object

        Returns:
            DataFrame with features in correct order
        """
        return model_input.to_dataframe()

    def _validate_input(self, input_df: pd.DataFrame):
        """
        Validate input DataFrame.

        Args:
            input_df: Input DataFrame

        Raises:
            InputValidationError: If validation fails
        """
        # Check number of features
        if len(input_df.columns) != len(self.EXPECTED_FEATURES):
            raise InputValidationError(
                f"Expected {len(self.EXPECTED_FEATURES)} features, "
                f"got {len(input_df.columns)}"
            )

        # Check feature names and order
        actual_features = list(input_df.columns)
        if actual_features != self.EXPECTED_FEATURES:
            raise InputValidationError(
                f"Feature mismatch. Expected {self.EXPECTED_FEATURES}, "
                f"got {actual_features}"
            )

        # Check data types
        for col in input_df.columns:
            expected_types = self.EXPECTED_DTYPES.get(col)
            if expected_types:
                actual_value = input_df[col].iloc[0]
                if not isinstance(actual_value, expected_types):
                    raise InputValidationError(
                        f"Feature '{col}' has invalid type. "
                        f"Expected {expected_types}, got {type(actual_value)}"
                    )

        # Check for NaN/None values
        if input_df.isnull().any().any():
            null_cols = input_df.columns[input_df.isnull().any()].tolist()
            raise InputValidationError(
                f"Features contain null values: {null_cols}"
            )

        # Check binary feature values
        binary_features = [
            'Gender_Male', 'Route_IV', 'Route_Oral',
            'Frequency_OD', 'Frequency_QID', 'Frequency_TDS'
        ]

        for col in binary_features:
            value = input_df[col].iloc[0]
            if value not in [0, 1]:
                raise InputValidationError(
                    f"Binary feature '{col}' must be 0 or 1, got {value}"
                )

        # Check numeric ranges
        age = input_df['Age'].iloc[0]
        if not (0 <= age <= 120):
            raise InputValidationError(
                f"Age {age} is out of valid range [0, 120]"
            )

        dosage = input_df['Dosage (gram)'].iloc[0]
        if not (0 < dosage <= 10):
            raise InputValidationError(
                f"Dosage {dosage}g is out of valid range (0, 10]"
            )


# ============================================================================
# INFERENCE SERVICE (Singleton)
# ============================================================================

class InferenceService:
    """
    Singleton inference service for global model access.
    """

    _instance: Optional['InferenceService'] = None
    _engine: Optional[ModelInferenceEngine] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_path: str, model_version: str = "1.0"):
        """
        Initialize the inference service with a model.

        Args:
            model_path: Path to model file
            model_version: Model version
        """
        instance = cls()
        instance._engine = ModelInferenceEngine(model_path, model_version)
        instance._engine.load_model()
        return instance

    @classmethod
    def get_engine(cls) -> ModelInferenceEngine:
        """Get the inference engine"""
        instance = cls()
        if instance._engine is None:
            raise RuntimeError("InferenceService not initialized. Call initialize() first.")
        return instance._engine

    @classmethod
    def predict(cls, model_input: ModelInput, **kwargs) -> PredictionResult:
        """Run prediction using singleton engine"""
        engine = cls.get_engine()
        return engine.predict(model_input, **kwargs)

    @classmethod
    def predict_batch(cls, model_inputs: List[ModelInput], **kwargs) -> List[PredictionResult]:
        """Run batch prediction using singleton engine"""
        engine = cls.get_engine()
        return engine.predict_batch(model_inputs, **kwargs)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_model(model_path: str, model_version: str = "1.0") -> ModelInferenceEngine:
    """
    Convenience function to load model.

    Args:
        model_path: Path to model file
        model_version: Model version

    Returns:
        Initialized ModelInferenceEngine
    """
    engine = ModelInferenceEngine(model_path, model_version)
    engine.load_model()
    return engine


def predict_duration(
    model_input: ModelInput,
    engine: Optional[ModelInferenceEngine] = None,
    model_path: str = "best_dt_classifier_model.joblib"
) -> PredictionResult:
    """
    Convenience function for single prediction.

    Args:
        model_input: ModelInput object
        engine: Optional pre-loaded engine (creates new if None)
        model_path: Path to model file (used if engine is None)

    Returns:
        PredictionResult
    """
    if engine is None:
        engine = load_model(model_path)

    return engine.predict(model_input)
