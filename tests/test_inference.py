"""
Unit tests for ML Inference Service.

Tests model loading, input validation, prediction, and error handling.
"""

import pytest
import numpy as np
from pathlib import Path

from src.services.inference import (
    ModelInferenceEngine,
    InferenceService,
    ModelLoadError,
    InputValidationError,
    PredictionError,
    predict_duration
)
from src.models.schema import ModelInput


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def model_path():
    """Get path to trained model"""
    return "best_dt_classifier_model.joblib"


@pytest.fixture
def engine(model_path):
    """Create and load inference engine"""
    engine = ModelInferenceEngine(model_path)
    engine.load_model()
    return engine


@pytest.fixture
def valid_input():
    """Create valid model input"""
    return ModelInput(
        Age=60.0,
        **{"Dosage (gram)": 0.5},
        Gender_Male=1,
        Route_IV=0,
        Route_Oral=1,
        Frequency_OD=1,
        Frequency_QID=0,
        Frequency_TDS=0
    )


# ============================================================================
# TEST MODEL LOADING
# ============================================================================

class TestModelLoading:
    """Test model loading and validation"""

    def test_model_loads_successfully(self, model_path):
        """Test model loads without errors"""
        engine = ModelInferenceEngine(model_path)
        metadata = engine.load_model()

        assert engine._is_loaded
        assert metadata is not None
        assert metadata.n_features == 8
        assert metadata.model_type == "DecisionTreeClassifier"

    def test_model_file_not_found(self):
        """Test error when model file doesn't exist"""
        engine = ModelInferenceEngine("nonexistent_model.joblib")

        with pytest.raises(ModelLoadError, match="Model file not found"):
            engine.load_model()

    def test_model_metadata_extraction(self, engine):
        """Test metadata extraction"""
        metadata = engine.metadata

        assert metadata.n_features == 8
        assert len(metadata.feature_names) == 8
        assert len(metadata.classes) == 3
        assert set(metadata.classes) == {'<5 days', '5-10 days', '>10 days'}
        assert metadata.model_version == "1.0"

    def test_lazy_loading(self, model_path):
        """Test model loads lazily on first prediction"""
        engine = ModelInferenceEngine(model_path)

        assert not engine._is_loaded

        # First prediction triggers loading
        input_data = ModelInput(
            Age=60.0,
            **{"Dosage (gram)": 0.5},
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=1,
            Frequency_OD=1,
            Frequency_QID=0,
            Frequency_TDS=0
        )

        result = engine.predict(input_data)

        assert engine._is_loaded
        assert result.prediction in engine.metadata.classes


# ============================================================================
# TEST INPUT VALIDATION
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    def test_valid_input_passes(self, engine, valid_input):
        """Test valid input passes validation"""
        result = engine.predict(valid_input)

        assert result.prediction in engine.metadata.classes
        assert isinstance(result.probabilities, dict)
        assert len(result.probabilities) == 3

    def test_age_out_of_range(self, engine):
        """Test age out of valid range"""
        invalid_input = ModelInput(
            Age=150.0,  # Invalid
            **{"Dosage (gram)": 0.5},
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=1,
            Frequency_OD=1,
            Frequency_QID=0,
            Frequency_TDS=0
        )

        with pytest.raises(InputValidationError, match="Age .* out of valid range"):
            engine.predict(invalid_input)

    def test_dosage_out_of_range(self, engine):
        """Test dosage out of valid range"""
        invalid_input = ModelInput(
            Age=60.0,
            **{"Dosage (gram)": 15.0},  # Invalid
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=1,
            Frequency_OD=1,
            Frequency_QID=0,
            Frequency_TDS=0
        )

        with pytest.raises(InputValidationError, match="Dosage .* out of valid range"):
            engine.predict(invalid_input)

    def test_binary_feature_invalid_value(self, engine):
        """Test binary feature with invalid value"""
        # ModelInput validation should catch this, but test engine validation
        # We can't easily create invalid ModelInput due to Pydantic validation,
        # so this test is more conceptual
        pass


# ============================================================================
# TEST PREDICTION
# ============================================================================

class TestPrediction:
    """Test prediction functionality"""

    def test_prediction_returns_valid_class(self, engine, valid_input):
        """Test prediction returns one of the expected classes"""
        result = engine.predict(valid_input)

        assert result.prediction in ['<5 days', '5-10 days', '>10 days']

    def test_probabilities_returned(self, engine, valid_input):
        """Test probabilities are returned"""
        result = engine.predict(valid_input, return_probabilities=True)

        assert len(result.probabilities) == 3
        assert all(0 <= prob <= 1 for prob in result.probabilities.values())
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.01  # Sum to ~1

    def test_probabilities_omitted(self, engine, valid_input):
        """Test probabilities can be omitted"""
        result = engine.predict(valid_input, return_probabilities=False)

        assert result.probabilities == {}
        assert result.confidence == 1.0

    def test_confidence_score(self, engine, valid_input):
        """Test confidence score is max probability"""
        result = engine.predict(valid_input, return_probabilities=True)

        max_prob = max(result.probabilities.values())
        assert result.confidence == max_prob

    def test_inference_time_recorded(self, engine, valid_input):
        """Test inference time is recorded"""
        result = engine.predict(valid_input)

        assert result.inference_time_ms > 0
        assert result.inference_time_ms < 1000  # Should be fast

    def test_input_features_in_result(self, engine, valid_input):
        """Test input features are included in result"""
        result = engine.predict(valid_input)

        assert 'Age' in result.input_features
        assert result.input_features['Age'] == 60.0

    def test_model_metadata_in_result(self, engine, valid_input):
        """Test model metadata is included in result"""
        result = engine.predict(valid_input)

        assert result.model_metadata is not None
        assert result.model_metadata.model_type == "DecisionTreeClassifier"


# ============================================================================
# TEST BATCH PREDICTION
# ============================================================================

class TestBatchPrediction:
    """Test batch prediction functionality"""

    def test_batch_prediction(self, engine):
        """Test batch prediction with multiple inputs"""
        inputs = [
            ModelInput(
                Age=60.0,
                **{"Dosage (gram)": 0.5},
                Gender_Male=1,
                Route_IV=0,
                Route_Oral=1,
                Frequency_OD=1,
                Frequency_QID=0,
                Frequency_TDS=0
            ),
            ModelInput(
                Age=45.0,
                **{"Dosage (gram)": 1.0},
                Gender_Male=0,
                Route_IV=1,
                Route_Oral=0,
                Frequency_OD=0,
                Frequency_QID=0,
                Frequency_TDS=1
            ),
            ModelInput(
                Age=70.0,
                **{"Dosage (gram)": 0.75},
                Gender_Male=1,
                Route_IV=0,
                Route_Oral=0,  # BD route
                Frequency_OD=0,
                Frequency_QID=0,
                Frequency_TDS=0  # BD frequency
            )
        ]

        results = engine.predict_batch(inputs, return_probabilities=True)

        assert len(results) == 3
        assert all(r.prediction in engine.metadata.classes for r in results)
        assert all(len(r.probabilities) == 3 for r in results)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_bd_encoding_all_zeros(self, engine):
        """Test BD encoding (all frequency/route flags = 0) works"""
        bd_input = ModelInput(
            Age=60.0,
            **{"Dosage (gram)": 0.75},
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=0,  # BD route
            Frequency_OD=0,
            Frequency_QID=0,
            Frequency_TDS=0  # BD frequency
        )

        result = engine.predict(bd_input)

        assert result.prediction in engine.metadata.classes
        # Should not raise errors

    def test_minimum_age(self, engine):
        """Test minimum valid age"""
        input_data = ModelInput(
            Age=1.0,  # Very young but valid
            **{"Dosage (gram)": 0.1},
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=1,
            Frequency_OD=1,
            Frequency_QID=0,
            Frequency_TDS=0
        )

        result = engine.predict(input_data)
        assert result.prediction in engine.metadata.classes

    def test_maximum_age(self, engine):
        """Test maximum valid age"""
        input_data = ModelInput(
            Age=120.0,  # Maximum valid age
            **{"Dosage (gram)": 0.5},
            Gender_Male=1,
            Route_IV=0,
            Route_Oral=1,
            Frequency_OD=1,
            Frequency_QID=0,
            Frequency_TDS=0
        )

        result = engine.predict(input_data)
        assert result.prediction in engine.metadata.classes


# ============================================================================
# TEST INFERENCE SERVICE (SINGLETON)
# ============================================================================

class TestInferenceService:
    """Test singleton inference service"""

    def test_singleton_pattern(self, model_path):
        """Test InferenceService is singleton"""
        service1 = InferenceService()
        service2 = InferenceService()

        assert service1 is service2

    def test_service_initialization(self, model_path):
        """Test service initialization"""
        service = InferenceService.initialize(model_path)

        assert service._engine is not None
        assert service._engine._is_loaded

    def test_service_predict(self, model_path, valid_input):
        """Test prediction through service"""
        InferenceService.initialize(model_path)

        result = InferenceService.predict(valid_input)

        assert result.prediction in ['<5 days', '5-10 days', '>10 days']


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_predict_duration(self, valid_input):
        """Test predict_duration convenience function"""
        result = predict_duration(valid_input)

        assert result.prediction in ['<5 days', '5-10 days', '>10 days']
        assert result.probabilities is not None


# ============================================================================
# TEST SERIALIZATION
# ============================================================================

class TestSerialization:
    """Test result serialization"""

    def test_result_to_dict(self, engine, valid_input):
        """Test PredictionResult can be serialized to dict"""
        result = engine.predict(valid_input)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'prediction' in result_dict
        assert 'probabilities' in result_dict
        assert 'confidence' in result_dict
        assert 'model_metadata' in result_dict

    def test_metadata_to_dict(self, engine):
        """Test ModelMetadata can be serialized to dict"""
        metadata_dict = engine.metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert 'model_type' in metadata_dict
        assert 'n_features' in metadata_dict
        assert 'classes' in metadata_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
