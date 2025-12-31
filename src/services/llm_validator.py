"""
LLM Response Validator and Repair Module.

This module validates JSON responses from LLM APIs, detects errors,
repairs recoverable issues, and flags unrecoverable cases for human review.
"""

import json
import re
from enum import Enum
from typing import Any, Optional, Union
from dataclasses import dataclass, field
from src.models.schema import RawFeatures, ExtractedField


# ============================================================================
# ERROR TAXONOMY
# ============================================================================

class ErrorSeverity(str, Enum):
    """Severity levels for validation errors"""
    INFO = "info"              # Informational, no action needed
    WARNING = "warning"        # Fixable issue, auto-repaired
    ERROR = "error"           # Recoverable with repair
    CRITICAL = "critical"     # Unrecoverable, needs human review


class ErrorCategory(str, Enum):
    """Categories of validation errors"""
    # JSON Structure Errors
    MALFORMED_JSON = "malformed_json"
    MISSING_FIELD = "missing_field"
    INVALID_SCHEMA = "invalid_schema"

    # Data Quality Errors
    LOW_CONFIDENCE = "low_confidence"
    MISSING_EVIDENCE = "missing_evidence"
    EVIDENCE_MISMATCH = "evidence_mismatch"

    # Data Type Errors
    INVALID_TYPE = "invalid_type"
    OUT_OF_RANGE = "out_of_range"
    INVALID_VALUE = "invalid_value"

    # Conversion Errors
    UNIT_CONVERSION_ERROR = "unit_conversion_error"
    ENCODING_ERROR = "encoding_error"

    # Completeness Errors
    PARTIAL_EXTRACTION = "partial_extraction"
    NULL_CRITICAL_FIELD = "null_critical_field"


class RepairStrategy(str, Enum):
    """Strategies for repairing errors"""
    AUTO_FIX = "auto_fix"                    # Automatically repair
    IMPUTE = "impute"                        # Use default/median values
    NORMALIZE = "normalize"                  # Normalize format/casing
    CONVERT = "convert"                      # Convert units/types
    INFER = "infer"                         # Infer from context
    HUMAN_REVIEW = "human_review"           # Flag for manual review
    SKIP = "skip"                           # Skip this field


# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

@dataclass
class ConfidenceThresholds:
    """Confidence score thresholds for different actions"""

    # Minimum confidence to accept a value without warning
    ACCEPTABLE: float = 0.7

    # Minimum confidence for critical fields (age, dosage)
    CRITICAL_FIELD: float = 0.6

    # Below this, trigger human review
    HUMAN_REVIEW: float = 0.4

    # Below this, reject the extraction entirely
    REJECT: float = 0.2

    # Confidence for imputed values
    IMPUTED_DEFAULT: float = 0.5


# ============================================================================
# VALIDATION ERROR
# ============================================================================

@dataclass
class ValidationError:
    """Represents a single validation error"""
    field_name: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_value: Any = None
    repair_strategy: Optional[RepairStrategy] = None
    repaired_value: Any = None
    confidence_impact: float = 0.0  # How much to reduce confidence by

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "field": self.field_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_value": self.original_value,
            "repair_strategy": self.repair_strategy.value if self.repair_strategy else None,
            "repaired_value": self.repaired_value
        }


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class LLMValidationResult:
    """Result of LLM response validation and repair"""

    is_valid: bool
    requires_human_review: bool
    errors: list[ValidationError] = field(default_factory=list)
    repaired_data: Optional[dict] = None
    original_data: Optional[dict] = None
    overall_confidence: float = 0.0
    repair_count: int = 0

    @property
    def critical_errors(self) -> list[ValidationError]:
        """Get only critical errors"""
        return [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]

    @property
    def warnings(self) -> list[ValidationError]:
        """Get only warnings"""
        return [e for e in self.errors if e.severity == ErrorSeverity.WARNING]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "requires_human_review": self.requires_human_review,
            "overall_confidence": self.overall_confidence,
            "repair_count": self.repair_count,
            "errors": [e.to_dict() for e in self.errors],
            "critical_errors": len(self.critical_errors),
            "warnings": len(self.warnings),
            "repaired_data": self.repaired_data
        }


# ============================================================================
# LLM RESPONSE VALIDATOR
# ============================================================================

class LLMResponseValidator:
    """
    Validates and repairs LLM JSON responses.
    Handles malformed JSON, missing fields, and low-confidence extractions.
    """

    # Critical fields that must be present
    CRITICAL_FIELDS = ["age", "dosage"]

    # All expected fields
    REQUIRED_FIELDS = ["age", "dosage", "gender", "route", "frequency"]

    # Valid values for categorical fields
    VALID_GENDER = ["male", "female", None]
    VALID_ROUTE = ["IV", "oral", "other", None]
    VALID_FREQUENCY = ["OD", "BD", "TDS", "QID", "other", None]

    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """
        Initialize validator.

        Args:
            thresholds: Custom confidence thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or ConfidenceThresholds()

    def validate_and_repair(
        self,
        llm_response: str,
        original_text: Optional[str] = None
    ) -> LLMValidationResult:
        """
        Validate and repair LLM response.

        Args:
            llm_response: Raw LLM response string (should be JSON)
            original_text: Original case study text for evidence checking

        Returns:
            LLMValidationResult with validation status and repairs
        """
        result = LLMValidationResult(
            is_valid=True,
            requires_human_review=False,
            original_data=None
        )

        # Step 1: Parse JSON
        parsed_data = self._parse_json(llm_response, result)
        if parsed_data is None:
            result.is_valid = False
            result.requires_human_review = True
            return result

        result.original_data = parsed_data.copy()

        # Step 2: Validate schema structure
        self._validate_schema(parsed_data, result)

        # Step 3: Validate and repair each field
        repaired_data = self._validate_fields(parsed_data, result)

        # Step 4: Check evidence against original text
        if original_text:
            self._validate_evidence(repaired_data, original_text, result)

        # Step 5: Calculate overall confidence
        result.overall_confidence = self._calculate_overall_confidence(repaired_data)

        # Step 6: Determine if human review is needed
        result.requires_human_review = self._needs_human_review(result)

        # Step 7: Store repaired data
        result.repaired_data = repaired_data

        return result

    def _parse_json(self, response: str, result: LLMValidationResult) -> Optional[dict]:
        """
        Parse JSON with error recovery.

        Args:
            response: Raw response string
            result: Validation result to populate

        Returns:
            Parsed JSON dict or None if unrecoverable
        """
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            result.errors.append(ValidationError(
                field_name="__json__",
                category=ErrorCategory.MALFORMED_JSON,
                severity=ErrorSeverity.ERROR,
                message=f"JSON decode error: {str(e)}",
                original_value=response[:100] + "..."
            ))

        # Repair Strategy 1: Extract JSON from markdown code blocks
        if "```json" in response or "```" in response:
            try:
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    result.errors[-1].repair_strategy = RepairStrategy.AUTO_FIX
                    result.errors[-1].repaired_value = "Extracted from markdown"
                    result.errors[-1].severity = ErrorSeverity.WARNING
                    result.repair_count += 1
                    return parsed
            except:
                pass

        # Repair Strategy 2: Find JSON object in text
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                parsed = json.loads(json_str)
                result.errors[-1].repair_strategy = RepairStrategy.AUTO_FIX
                result.errors[-1].repaired_value = "Extracted JSON object"
                result.errors[-1].severity = ErrorSeverity.WARNING
                result.repair_count += 1
                return parsed
        except:
            pass

        # Repair Strategy 3: Fix common JSON errors
        try:
            # Replace single quotes with double quotes
            fixed = response.replace("'", '"')
            # Remove trailing commas
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)

            parsed = json.loads(fixed)
            result.errors[-1].repair_strategy = RepairStrategy.AUTO_FIX
            result.errors[-1].repaired_value = "Fixed JSON syntax"
            result.errors[-1].severity = ErrorSeverity.WARNING
            result.repair_count += 1
            return parsed
        except:
            pass

        # Unrecoverable - escalate to critical
        result.errors[-1].severity = ErrorSeverity.CRITICAL
        result.errors[-1].repair_strategy = RepairStrategy.HUMAN_REVIEW
        return None

    def _validate_schema(self, data: dict, result: LLMValidationResult):
        """
        Validate that response has correct schema structure.

        Args:
            data: Parsed JSON data
            result: Validation result to populate
        """
        # Check for missing fields
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                severity = (ErrorSeverity.CRITICAL if field_name in self.CRITICAL_FIELDS
                           else ErrorSeverity.ERROR)

                result.errors.append(ValidationError(
                    field_name=field_name,
                    category=ErrorCategory.MISSING_FIELD,
                    severity=severity,
                    message=f"Required field '{field_name}' is missing",
                    repair_strategy=RepairStrategy.IMPUTE
                ))

                # Add empty field structure
                data[field_name] = {
                    "value": None,
                    "raw_value": None,
                    "evidence": None,
                    "confidence": 0.0
                }
                result.repair_count += 1

    def _validate_fields(self, data: dict, result: LLMValidationResult) -> dict:
        """
        Validate and repair individual fields.

        Args:
            data: Parsed JSON data
            result: Validation result to populate

        Returns:
            Repaired data dictionary
        """
        repaired = data.copy()

        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                continue

            field_data = data[field_name]

            # Ensure field has correct structure
            if not isinstance(field_data, dict):
                result.errors.append(ValidationError(
                    field_name=field_name,
                    category=ErrorCategory.INVALID_SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message=f"Field '{field_name}' should be object, got {type(field_data)}",
                    original_value=field_data,
                    repair_strategy=RepairStrategy.AUTO_FIX
                ))

                # Repair: Wrap in correct structure
                repaired[field_name] = {
                    "value": field_data if field_data is not None else None,
                    "raw_value": str(field_data) if field_data is not None else None,
                    "evidence": None,
                    "confidence": self.thresholds.IMPUTED_DEFAULT
                }
                result.repair_count += 1
                field_data = repaired[field_name]

            # Validate required subfields
            for subfield in ["value", "raw_value", "evidence", "confidence"]:
                if subfield not in field_data:
                    field_data[subfield] = None if subfield != "confidence" else 0.0

            # Validate confidence score
            confidence = field_data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                result.errors.append(ValidationError(
                    field_name=field_name,
                    category=ErrorCategory.INVALID_TYPE,
                    severity=ErrorSeverity.WARNING,
                    message=f"Invalid confidence score: {confidence}",
                    original_value=confidence,
                    repair_strategy=RepairStrategy.AUTO_FIX,
                    repaired_value=0.0
                ))
                field_data["confidence"] = 0.0
                result.repair_count += 1

            # Validate specific fields
            if field_name == "age":
                self._validate_age(field_data, result)
            elif field_name == "dosage":
                self._validate_dosage(field_data, result)
            elif field_name == "gender":
                self._validate_gender(field_data, result)
            elif field_name == "route":
                self._validate_route(field_data, result)
            elif field_name == "frequency":
                self._validate_frequency(field_data, result)

            # Check confidence thresholds
            self._check_confidence_threshold(field_name, field_data, result)

        return repaired

    def _validate_age(self, field_data: dict, result: LLMValidationResult):
        """Validate age field"""
        value = field_data.get("value")

        if value is None:
            # Critical field is null
            if field_data.get("confidence", 0) < self.thresholds.CRITICAL_FIELD:
                result.errors.append(ValidationError(
                    field_name="age",
                    category=ErrorCategory.NULL_CRITICAL_FIELD,
                    severity=ErrorSeverity.ERROR,
                    message="Critical field 'age' is null with low confidence",
                    repair_strategy=RepairStrategy.IMPUTE
                ))
            return

        # Type validation
        if not isinstance(value, (int, float)):
            try:
                repaired = float(value)
                field_data["value"] = repaired
                result.repair_count += 1
            except:
                result.errors.append(ValidationError(
                    field_name="age",
                    category=ErrorCategory.INVALID_TYPE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Age must be numeric, got {type(value)}",
                    original_value=value,
                    repair_strategy=RepairStrategy.HUMAN_REVIEW
                ))
                return

        # Range validation
        if not (0 <= value <= 120):
            result.errors.append(ValidationError(
                field_name="age",
                category=ErrorCategory.OUT_OF_RANGE,
                severity=ErrorSeverity.CRITICAL,
                message=f"Age {value} is out of valid range [0, 120]",
                original_value=value,
                repair_strategy=RepairStrategy.HUMAN_REVIEW
            ))

    def _validate_dosage(self, field_data: dict, result: LLMValidationResult):
        """Validate dosage field with unit conversion check"""
        value = field_data.get("value")
        raw_value = field_data.get("raw_value")

        if value is None:
            if field_data.get("confidence", 0) < self.thresholds.CRITICAL_FIELD:
                result.errors.append(ValidationError(
                    field_name="dosage",
                    category=ErrorCategory.NULL_CRITICAL_FIELD,
                    severity=ErrorSeverity.ERROR,
                    message="Critical field 'dosage' is null with low confidence",
                    repair_strategy=RepairStrategy.IMPUTE
                ))
            return

        # Type validation
        if not isinstance(value, (int, float)):
            try:
                repaired = float(value)
                field_data["value"] = repaired
                result.repair_count += 1
            except:
                result.errors.append(ValidationError(
                    field_name="dosage",
                    category=ErrorCategory.INVALID_TYPE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Dosage must be numeric, got {type(value)}",
                    original_value=value,
                    repair_strategy=RepairStrategy.HUMAN_REVIEW
                ))
                return

        # Range validation
        if not (0 < value <= 10):
            result.errors.append(ValidationError(
                field_name="dosage",
                category=ErrorCategory.OUT_OF_RANGE,
                severity=ErrorSeverity.ERROR,
                message=f"Dosage {value}g seems unusual (expected 0-10g)",
                original_value=value,
                repair_strategy=RepairStrategy.HUMAN_REVIEW
            ))

        # Unit conversion validation
        if raw_value and value:
            expected = self._calculate_expected_dosage(raw_value)
            if expected and abs(value - expected) > 0.01:
                result.errors.append(ValidationError(
                    field_name="dosage",
                    category=ErrorCategory.UNIT_CONVERSION_ERROR,
                    severity=ErrorSeverity.WARNING,
                    message=f"Dosage conversion may be incorrect: {raw_value} → {value}g (expected ~{expected}g)",
                    original_value=value,
                    repair_strategy=RepairStrategy.AUTO_FIX,
                    repaired_value=expected
                ))
                field_data["value"] = expected
                result.repair_count += 1

    def _validate_gender(self, field_data: dict, result: LLMValidationResult):
        """Validate and normalize gender field"""
        value = field_data.get("value")

        if value is None:
            return

        # Normalize casing
        if isinstance(value, str):
            normalized = value.lower().strip()

            # Repair common variations
            if normalized in ["m", "male", "man"]:
                if value != "male":
                    field_data["value"] = "male"
                    result.repair_count += 1
            elif normalized in ["f", "female", "woman"]:
                if value != "female":
                    field_data["value"] = "female"
                    result.repair_count += 1
            elif normalized not in self.VALID_GENDER:
                result.errors.append(ValidationError(
                    field_name="gender",
                    category=ErrorCategory.INVALID_VALUE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Invalid gender value: '{value}'",
                    original_value=value,
                    repair_strategy=RepairStrategy.HUMAN_REVIEW
                ))

    def _validate_route(self, field_data: dict, result: LLMValidationResult):
        """Validate and normalize route field"""
        value = field_data.get("value")

        if value is None:
            return

        if isinstance(value, str):
            normalized = value.lower().strip()

            # Normalize to standard values
            if normalized in ["iv", "intravenous"]:
                field_data["value"] = "IV"
                result.repair_count += 1
            elif normalized in ["po", "oral", "mouth"]:
                field_data["value"] = "oral"
                result.repair_count += 1
            elif value not in self.VALID_ROUTE:
                field_data["value"] = "other"
                result.repair_count += 1
                result.errors.append(ValidationError(
                    field_name="route",
                    category=ErrorCategory.ENCODING_ERROR,
                    severity=ErrorSeverity.WARNING,
                    message=f"Non-standard route '{value}' normalized to 'other'",
                    original_value=value,
                    repair_strategy=RepairStrategy.NORMALIZE,
                    repaired_value="other"
                ))

    def _validate_frequency(self, field_data: dict, result: LLMValidationResult):
        """Validate and normalize frequency field"""
        value = field_data.get("value")

        if value is None:
            return

        if isinstance(value, str):
            normalized = value.lower().strip()

            # Normalize variations
            mapping = {
                "od": "OD", "once daily": "OD", "once": "OD", "qd": "OD",
                "bd": "BD", "twice daily": "BD", "q12h": "BD", "bid": "BD",
                "tds": "TDS", "three times daily": "TDS", "tid": "TDS",
                "qid": "QID", "four times daily": "QID", "q6h": "QID"
            }

            if normalized in mapping:
                field_data["value"] = mapping[normalized]
                result.repair_count += 1
            elif value not in self.VALID_FREQUENCY:
                field_data["value"] = "other"
                result.repair_count += 1
                result.errors.append(ValidationError(
                    field_name="frequency",
                    category=ErrorCategory.ENCODING_ERROR,
                    severity=ErrorSeverity.WARNING,
                    message=f"Non-standard frequency '{value}' normalized to 'other'",
                    original_value=value,
                    repair_strategy=RepairStrategy.NORMALIZE,
                    repaired_value="other"
                ))

    def _check_confidence_threshold(
        self,
        field_name: str,
        field_data: dict,
        result: LLMValidationResult
    ):
        """Check if confidence meets thresholds"""
        confidence = field_data.get("confidence", 0.0)
        value = field_data.get("value")

        # Critical fields need higher confidence
        threshold = (self.thresholds.CRITICAL_FIELD if field_name in self.CRITICAL_FIELDS
                    else self.thresholds.ACCEPTABLE)

        if value is not None and confidence < threshold:
            severity = ErrorSeverity.WARNING if confidence >= self.thresholds.HUMAN_REVIEW else ErrorSeverity.ERROR

            result.errors.append(ValidationError(
                field_name=field_name,
                category=ErrorCategory.LOW_CONFIDENCE,
                severity=severity,
                message=f"Low confidence {confidence:.2f} (threshold: {threshold:.2f})",
                confidence_impact=-0.1
            ))

    def _validate_evidence(
        self,
        data: dict,
        original_text: str,
        result: LLMValidationResult
    ):
        """Validate that evidence exists in original text"""
        original_lower = original_text.lower()

        for field_name, field_data in data.items():
            if not isinstance(field_data, dict):
                continue

            evidence = field_data.get("evidence")
            value = field_data.get("value")

            if value is not None and evidence:
                evidence_lower = evidence.lower().strip()

                if evidence_lower not in original_lower:
                    result.errors.append(ValidationError(
                        field_name=field_name,
                        category=ErrorCategory.EVIDENCE_MISMATCH,
                        severity=ErrorSeverity.CRITICAL,
                        message=f"Evidence '{evidence}' not found in original text",
                        original_value=evidence,
                        repair_strategy=RepairStrategy.HUMAN_REVIEW,
                        confidence_impact=-0.3
                    ))

    def _calculate_expected_dosage(self, raw_value: str) -> Optional[float]:
        """Calculate expected dosage in grams from raw value"""
        if not raw_value:
            return None

        match = re.search(r'(\d+\.?\d*)', raw_value)
        if not match:
            return None

        number = float(match.group(1))
        raw_lower = raw_value.lower()

        if 'mg' in raw_lower and 'mcg' not in raw_lower:
            return number / 1000
        elif 'mcg' in raw_lower or 'µg' in raw_lower:
            return number / 1000000
        elif 'g' in raw_lower:
            return number

        return None

    def _calculate_overall_confidence(self, data: dict) -> float:
        """Calculate overall confidence score"""
        confidences = []

        for field_name in self.REQUIRED_FIELDS:
            if field_name in data and isinstance(data[field_name], dict):
                conf = data[field_name].get("confidence", 0.0)
                # Weight critical fields more heavily
                weight = 1.5 if field_name in self.CRITICAL_FIELDS else 1.0
                confidences.append(conf * weight)

        if not confidences:
            return 0.0

        # Weighted average
        total_weight = len([f for f in self.REQUIRED_FIELDS if f in self.CRITICAL_FIELDS]) * 1.5
        total_weight += len([f for f in self.REQUIRED_FIELDS if f not in self.CRITICAL_FIELDS])

        return sum(confidences) / total_weight

    def _needs_human_review(self, result: LLMValidationResult) -> bool:
        """Determine if extraction needs human review"""
        # Any critical errors trigger review
        if result.critical_errors:
            return True

        # Overall confidence too low
        if result.overall_confidence < self.thresholds.HUMAN_REVIEW:
            return True

        # Too many repairs needed
        if result.repair_count > 3:
            return True

        # Missing critical fields with low confidence
        for field in self.CRITICAL_FIELDS:
            if result.repaired_data and field in result.repaired_data:
                field_data = result.repaired_data[field]
                if isinstance(field_data, dict):
                    if field_data.get("value") is None:
                        if field_data.get("confidence", 0) < self.thresholds.CRITICAL_FIELD:
                            return True

        return False
