"""
Extraction Validator for validating LLM extraction outputs.

This module validates extracted data against original text to catch
hallucinations, missing evidence, and incorrect conversions.
"""

import re
from typing import Optional
from src.models.schema import RawFeatures, ExtractedField


class ValidationResult:
    """Result of validation with warnings and errors"""

    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Returns True if no errors exist"""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """Returns True if warnings exist"""
        return len(self.warnings) > 0

    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)

    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "warnings": self.warnings,
            "errors": self.errors
        }

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, {len(self.warnings)} warnings, {len(self.errors)} errors)"


class ExtractionValidator:
    """
    Post-extraction validation to catch hallucinations and errors.
    """

    @staticmethod
    def validate_extraction(raw_features: RawFeatures, original_text: str) -> ValidationResult:
        """
        Validate extracted data against original text.

        Args:
            raw_features: RawFeatures object from LLM extraction
            original_text: Original case study text

        Returns:
            ValidationResult with warnings/errors
        """
        result = ValidationResult()

        # Validate each field
        ExtractionValidator._validate_field(
            "age", raw_features.age, original_text, result
        )
        ExtractionValidator._validate_field(
            "dosage", raw_features.dosage, original_text, result
        )
        ExtractionValidator._validate_field(
            "gender", raw_features.gender, original_text, result
        )
        ExtractionValidator._validate_field(
            "route", raw_features.route, original_text, result
        )
        ExtractionValidator._validate_field(
            "frequency", raw_features.frequency, original_text, result
        )

        # Validate dosage conversion specifically
        if raw_features.dosage:
            ExtractionValidator._validate_dosage_conversion(
                raw_features.dosage, result
            )

        return result

    @staticmethod
    def _validate_field(
        field_name: str,
        field_data: Optional[ExtractedField],
        original_text: str,
        result: ValidationResult
    ):
        """
        Validate a single extracted field.

        Args:
            field_name: Name of the field being validated
            field_data: ExtractedField object
            original_text: Original case study text
            result: ValidationResult to accumulate warnings/errors
        """
        if field_data is None:
            result.add_error(f"{field_name}: Field is missing from extraction")
            return

        # Check 1: If value exists, evidence should exist
        if field_data.value is not None:
            if not field_data.evidence:
                result.add_error(
                    f"{field_name}: Has value '{field_data.value}' but no evidence"
                )

        # Check 2: Evidence must be substring of original text (case-insensitive)
        if field_data.evidence:
            # Normalize both for comparison
            evidence_lower = field_data.evidence.lower().strip()
            text_lower = original_text.lower()

            if evidence_lower not in text_lower:
                result.add_error(
                    f"{field_name}: Evidence '{field_data.evidence}' not found in original text"
                )

        # Check 3: Confidence must match value presence
        if field_data.value is None:
            if field_data.confidence > 0.5:
                result.add_warning(
                    f"{field_name}: Null value but confidence {field_data.confidence} > 0.5"
                )
        else:
            # Value exists, confidence should be reasonable
            if field_data.confidence < 0.3:
                result.add_warning(
                    f"{field_name}: Has value but low confidence {field_data.confidence}"
                )

        # Check 4: Raw value should be related to evidence
        if field_data.raw_value and field_data.evidence:
            raw_lower = str(field_data.raw_value).lower().strip()
            evidence_lower = field_data.evidence.lower()

            # Check if raw value is substring of evidence
            if raw_lower not in evidence_lower:
                # Could be abbreviation or different form, just warn
                result.add_warning(
                    f"{field_name}: Raw value '{field_data.raw_value}' not directly in evidence '{field_data.evidence}'"
                )

    @staticmethod
    def _validate_dosage_conversion(dosage_field: ExtractedField, result: ValidationResult):
        """
        Validate dosage unit conversion is mathematically correct.

        Args:
            dosage_field: ExtractedField for dosage
            result: ValidationResult to accumulate warnings/errors
        """
        if dosage_field.value is None or dosage_field.raw_value is None:
            return  # Nothing to validate

        value_grams = dosage_field.value
        raw_value = str(dosage_field.raw_value)

        # Extract number from raw value
        match = re.search(r'(\d+\.?\d*)', raw_value)
        if not match:
            result.add_warning("dosage: Could not extract numeric value from raw_value for validation")
            return

        raw_number = float(match.group(1))

        # Check conversion based on unit
        raw_lower = raw_value.lower()

        expected_value = None
        if 'mg' in raw_lower and 'mcg' not in raw_lower:
            expected_value = raw_number / 1000
        elif 'mcg' in raw_lower or 'µg' in raw_lower:
            expected_value = raw_number / 1000000
        elif 'g' in raw_lower and 'mg' not in raw_lower:
            expected_value = raw_number
        else:
            # No unit found, assume mg for common antibiotic dosages
            if raw_number > 10:  # Likely in mg if number is large
                expected_value = raw_number / 1000
                result.add_warning(
                    f"dosage: No unit specified, assuming mg. Raw: '{raw_value}' → {value_grams}g"
                )
            else:
                expected_value = raw_number

        # Validate conversion
        if expected_value is not None:
            tolerance = max(0.001, expected_value * 0.01)  # 1% tolerance
            if abs(value_grams - expected_value) > tolerance:
                result.add_error(
                    f"dosage: Conversion appears incorrect - "
                    f"'{raw_value}' → {value_grams}g (expected ~{expected_value}g)"
                )

    @staticmethod
    def validate_confidence_scores(raw_features: RawFeatures) -> ValidationResult:
        """
        Validate that confidence scores are reasonable.

        Args:
            raw_features: RawFeatures object from LLM extraction

        Returns:
            ValidationResult with warnings/errors about confidence scores
        """
        result = ValidationResult()

        fields = {
            "age": raw_features.age,
            "dosage": raw_features.dosage,
            "gender": raw_features.gender,
            "route": raw_features.route,
            "frequency": raw_features.frequency
        }

        for field_name, field_data in fields.items():
            if field_data is None:
                continue

            # Check confidence is in valid range
            if not (0.0 <= field_data.confidence <= 1.0):
                result.add_error(
                    f"{field_name}: Confidence {field_data.confidence} not in range [0.0, 1.0]"
                )

            # Check consistency: null value should have 0 confidence
            if field_data.value is None and field_data.confidence != 0.0:
                result.add_warning(
                    f"{field_name}: Null value should have confidence 0.0, got {field_data.confidence}"
                )

            # Check: very high confidence should have evidence
            if field_data.confidence >= 0.9 and not field_data.evidence:
                result.add_warning(
                    f"{field_name}: High confidence {field_data.confidence} but no evidence provided"
                )

        return result

    @staticmethod
    def validate_all(raw_features: RawFeatures, original_text: str) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            raw_features: RawFeatures object from LLM extraction
            original_text: Original case study text

        Returns:
            Combined ValidationResult
        """
        # Validate extraction against original text
        result = ExtractionValidator.validate_extraction(raw_features, original_text)

        # Validate confidence scores
        confidence_result = ExtractionValidator.validate_confidence_scores(raw_features)

        # Merge results
        result.warnings.extend(confidence_result.warnings)
        result.errors.extend(confidence_result.errors)

        return result
