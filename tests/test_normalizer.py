"""
Unit tests for Feature Normalizer.

Tests normalization rules, edge cases, conflict resolution, and validation.
"""

import pytest
from src.services.normalizer import FeatureNormalizer, ValidationSeverity
from src.models.schema import RawFeatures, ExtractedField


class TestAgeNormalization:
    """Test age normalization"""

    def test_valid_age(self):
        """Test with valid age value"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=62.0, raw_value="62", evidence="62 years", confidence=1.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Age == 62.0
        assert len(result.errors) == 0

    def test_age_out_of_range(self):
        """Test age outside valid range"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=150.0, raw_value="150", evidence="150", confidence=1.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert not result.is_valid  # Error for out of range
        assert len(result.errors) > 0
        assert any("out of valid range" in e.message for e in result.errors)

    def test_age_missing_with_default(self):
        """Test missing age uses default"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Age == FeatureNormalizer.DEFAULT_AGE
        assert len(result.warnings) > 0
        assert any("default" in w.message.lower() for w in result.warnings)

    def test_age_parsed_from_raw_value(self):
        """Test age parsed from raw_value when value is None"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=None, raw_value="45yo", evidence="45yo", confidence=0.8),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Age == 45.0
        assert any("Parsed age from raw_value" in w.message for w in result.warnings)


class TestDosageNormalization:
    """Test dosage normalization and unit conversion"""

    def test_valid_dosage_in_grams(self):
        """Test valid dosage already in grams"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Dosage_gram == 1.0

    def test_dosage_unit_error_correction(self):
        """Test correction of suspected unit error (mg not converted)"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=500.0, raw_value="500mg", evidence="500mg", confidence=1.0),  # Wrong!
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        # Should auto-correct 500 to 0.5
        assert result.model_input.Dosage_gram == 0.5
        assert any("seems to be in mg" in w.message for w in result.warnings)

    def test_dosage_parsed_from_raw_value(self):
        """Test dosage parsed from raw_value with units"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=None, raw_value="750mg", evidence="750mg", confidence=0.8),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Dosage_gram == 0.75
        assert any("Parsed dosage from raw_value" in w.message for w in result.warnings)

    def test_dosage_mcg_conversion(self):
        """Test microgram to gram conversion"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=None, raw_value="250mcg", evidence="250mcg", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="OD", raw_value="OD", evidence="OD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Dosage_gram == 0.00025


class TestGenderNormalization:
    """Test gender encoding"""

    def test_gender_male_variations(self):
        """Test various male gender representations"""
        normalizer = FeatureNormalizer()

        test_cases = ["male", "M", "man", "gentleman", "he"]

        for gender_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value=gender_val, raw_value=gender_val, evidence=gender_val, confidence=1.0),
                route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
                frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Gender_Male == 1, f"Failed for: {gender_val}"

    def test_gender_female_variations(self):
        """Test various female gender representations"""
        normalizer = FeatureNormalizer()

        test_cases = ["female", "F", "woman", "lady", "she"]

        for gender_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value=gender_val, raw_value=gender_val, evidence=gender_val, confidence=1.0),
                route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
                frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Gender_Male == 0, f"Failed for: {gender_val}"

    def test_gender_default(self):
        """Test default gender when missing"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Gender_Male == 1  # Default to male
        assert any("defaulting to male" in w.message.lower() for w in result.warnings)


class TestRouteNormalization:
    """Test route encoding"""

    def test_route_iv(self):
        """Test IV route encoding"""
        normalizer = FeatureNormalizer()

        test_cases = ["IV", "intravenous", "intravenously"]

        for route_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
                route=ExtractedField(value=route_val, raw_value=route_val, evidence=route_val, confidence=1.0),
                frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Route_IV == 1, f"Failed for: {route_val}"
            assert result.model_input.Route_Oral == 0, f"Failed for: {route_val}"

    def test_route_oral(self):
        """Test oral route encoding"""
        normalizer = FeatureNormalizer()

        test_cases = ["oral", "PO", "orally", "by mouth"]

        for route_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
                route=ExtractedField(value=route_val, raw_value=route_val, evidence=route_val, confidence=1.0),
                frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Route_IV == 0, f"Failed for: {route_val}"
            assert result.model_input.Route_Oral == 1, f"Failed for: {route_val}"

    def test_route_bd_encoding(self):
        """Test BD/other route (both flags 0)"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
            dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
            gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
            route=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=1.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Route_IV == 0
        assert result.model_input.Route_Oral == 0
        assert any("BD" in w.message for w in result.warnings)


class TestFrequencyNormalization:
    """Test frequency encoding"""

    def test_frequency_od(self):
        """Test OD (once daily) encoding"""
        normalizer = FeatureNormalizer()

        test_cases = ["OD", "qd", "once daily", "once", "q24h"]

        for freq_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
                route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
                frequency=ExtractedField(value=freq_val, raw_value=freq_val, evidence=freq_val, confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Frequency_OD == 1, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_QID == 0, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_TDS == 0, f"Failed for: {freq_val}"

    def test_frequency_bd_all_zeros(self):
        """Test BD (twice daily) - ALL ZEROS"""
        normalizer = FeatureNormalizer()

        test_cases = ["BD", "bid", "twice daily", "twice", "q12h"]

        for freq_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
                route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
                frequency=ExtractedField(value=freq_val, raw_value=freq_val, evidence=freq_val, confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Frequency_OD == 0, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_QID == 0, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_TDS == 0, f"Failed for: {freq_val}"
            # BD is valid! Should have a warning noting this is BD encoding
            assert any("BD" in w.message for w in result.warnings)

    def test_frequency_tds(self):
        """Test TDS (three times daily) encoding"""
        normalizer = FeatureNormalizer()

        test_cases = ["TDS", "tid", "three times daily", "q8h"]

        for freq_val in test_cases:
            raw = RawFeatures(
                age=ExtractedField(value=60.0, raw_value="60", evidence="60", confidence=1.0),
                dosage=ExtractedField(value=1.0, raw_value="1g", evidence="1g", confidence=1.0),
                gender=ExtractedField(value="male", raw_value="male", evidence="male", confidence=1.0),
                route=ExtractedField(value="IV", raw_value="IV", evidence="IV", confidence=1.0),
                frequency=ExtractedField(value=freq_val, raw_value=freq_val, evidence=freq_val, confidence=1.0)
            )

            result = normalizer.normalize(raw)

            assert result.is_valid
            assert result.model_input.Frequency_OD == 0, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_QID == 0, f"Failed for: {freq_val}"
            assert result.model_input.Frequency_TDS == 1, f"Failed for: {freq_val}"


class TestConflictResolution:
    """Test conflict resolution and validation"""

    def test_route_conflict_both_one(self):
        """Test error when both route flags would be 1"""
        # This shouldn't happen in normal flow, but test validation
        # We'll need to create an invalid state manually for testing
        pass  # Note: This would require mocking internal state

    def test_frequency_conflict_multiple_active(self):
        """Test error when multiple frequency flags are 1"""
        # Similar to above - would require mocking
        pass


class TestEdgeCases:
    """Test edge cases and unusual inputs"""

    def test_all_fields_missing(self):
        """Test with all fields missing"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            dosage=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            gender=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            route=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            frequency=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid  # Should use defaults
        assert result.model_input.Age == FeatureNormalizer.DEFAULT_AGE
        assert result.model_input.Dosage_gram == FeatureNormalizer.DEFAULT_DOSAGE
        assert result.model_input.Gender_Male == 1
        assert result.model_input.Route_IV == 0
        assert result.model_input.Route_Oral == 0
        assert len(result.warnings) >= 5  # One warning per field

    def test_mixed_confidence_levels(self):
        """Test with mixed confidence levels"""
        normalizer = FeatureNormalizer()

        raw = RawFeatures(
            age=ExtractedField(value=62.0, raw_value="62", evidence="62", confidence=1.0),
            dosage=ExtractedField(value=None, raw_value="500mg", evidence="500mg", confidence=0.3),
            gender=ExtractedField(value="male", raw_value="M", evidence="M", confidence=0.8),
            route=ExtractedField(value=None, raw_value=None, evidence=None, confidence=0.0),
            frequency=ExtractedField(value="BD", raw_value="BD", evidence="BD", confidence=0.9)
        )

        result = normalizer.normalize(raw)

        assert result.is_valid
        assert result.model_input.Age == 62.0
        assert result.model_input.Dosage_gram == 0.5  # Parsed from raw_value
        assert result.model_input.Gender_Male == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
