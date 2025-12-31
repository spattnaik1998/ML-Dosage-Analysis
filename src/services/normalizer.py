"""
Feature Normalization Pipeline for ML Model Input.

This module converts raw extracted clinical features into the exact
8-feature model input format with explicit encoding rules, edge case
handling, and conflict resolution.
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

from src.models.schema import RawFeatures, ModelInput, ExtractedField


# ============================================================================
# VALIDATION SEVERITY
# ============================================================================

class ValidationSeverity(str, Enum):
    """Severity levels for normalization issues"""
    WARNING = "warning"  # Non-critical, can proceed
    ERROR = "error"      # Critical, may affect prediction quality


# ============================================================================
# NORMALIZATION ISSUE
# ============================================================================

@dataclass
class NormalizationIssue:
    """Represents a validation issue during normalization"""
    field: str
    severity: ValidationSeverity
    message: str
    original_value: any = None
    normalized_value: any = None
    rule_applied: str = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "severity": self.severity.value,
            "message": self.message,
            "original_value": str(self.original_value),
            "normalized_value": str(self.normalized_value),
            "rule_applied": self.rule_applied
        }


# ============================================================================
# NORMALIZATION RESULT
# ============================================================================

@dataclass
class NormalizationResult:
    """Result of feature normalization"""
    model_input: Optional[ModelInput]
    issues: List[NormalizationIssue] = field(default_factory=list)
    is_valid: bool = True

    @property
    def errors(self) -> List[NormalizationIssue]:
        """Get only error-level issues"""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[NormalizationIssue]:
        """Get only warning-level issues"""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_warning(self, field: str, message: str, **kwargs):
        """Add a warning"""
        self.issues.append(NormalizationIssue(
            field=field,
            severity=ValidationSeverity.WARNING,
            message=message,
            **kwargs
        ))

    def add_error(self, field: str, message: str, **kwargs):
        """Add an error"""
        self.issues.append(NormalizationIssue(
            field=field,
            severity=ValidationSeverity.ERROR,
            message=message,
            **kwargs
        ))
        self.is_valid = False


# ============================================================================
# FEATURE NORMALIZER
# ============================================================================

class FeatureNormalizer:
    """
    Normalizes raw extracted features into model input format.

    Converts:
    - Age (numeric) → Age (float)
    - Dosage (with units) → Dosage (gram) (float)
    - Gender (string) → Gender_Male (0/1)
    - Route (string) → Route_IV (0/1), Route_Oral (0/1)
    - Frequency (string) → Frequency_OD (0/1), Frequency_QID (0/1), Frequency_TDS (0/1)
    """

    # Default values for imputation
    DEFAULT_AGE = 55.0  # Median from training data
    DEFAULT_DOSAGE = 0.75  # Median from training data

    # Valid ranges
    AGE_MIN = 0.0
    AGE_MAX = 120.0
    DOSAGE_MIN = 0.0
    DOSAGE_MAX = 10.0

    def normalize(self, raw_features: RawFeatures) -> NormalizationResult:
        """
        Normalize raw features to model input.

        Args:
            raw_features: RawFeatures from extraction

        Returns:
            NormalizationResult with model input and validation issues
        """
        result = NormalizationResult(model_input=None)

        # Normalize each field
        age = self._normalize_age(raw_features.age, result)
        dosage = self._normalize_dosage(raw_features.dosage, result)
        gender_male = self._normalize_gender(raw_features.gender, result)
        route_iv, route_oral = self._normalize_route(raw_features.route, result)
        freq_od, freq_qid, freq_tds = self._normalize_frequency(raw_features.frequency, result)

        # Validate encoding conflicts
        self._validate_route_encoding(route_iv, route_oral, result)
        self._validate_frequency_encoding(freq_od, freq_qid, freq_tds, result)

        # Create model input if no critical errors
        if result.is_valid:
            try:
                result.model_input = ModelInput(
                    Age=age,
                    **{"Dosage (gram)": dosage},
                    Gender_Male=gender_male,
                    Route_IV=route_iv,
                    Route_Oral=route_oral,
                    Frequency_OD=freq_od,
                    Frequency_QID=freq_qid,
                    Frequency_TDS=freq_tds
                )
            except Exception as e:
                result.add_error(
                    "model_input",
                    f"Failed to create ModelInput: {str(e)}"
                )

        return result

    # ========================================================================
    # AGE NORMALIZATION
    # ========================================================================

    def _normalize_age(self, age_field: ExtractedField, result: NormalizationResult) -> float:
        """
        Normalize age to float.

        Rules:
        1. If value is numeric and valid, use it
        2. If value is None but raw_value exists, try to parse
        3. If unparseable, use default with warning
        4. Validate range [0, 120]
        """
        # Case 1: Value already provided
        if age_field.value is not None:
            try:
                age = float(age_field.value)
            except (ValueError, TypeError):
                result.add_error(
                    "age",
                    f"Invalid age value: {age_field.value}",
                    original_value=age_field.value
                )
                return self.DEFAULT_AGE

            # Validate range
            if not (self.AGE_MIN <= age <= self.AGE_MAX):
                result.add_error(
                    "age",
                    f"Age {age} out of valid range [{self.AGE_MIN}, {self.AGE_MAX}]",
                    original_value=age
                )
                # Clamp to valid range as fallback
                age = max(self.AGE_MIN, min(age, self.AGE_MAX))
                result.add_warning(
                    "age",
                    f"Age clamped to {age}",
                    normalized_value=age,
                    rule_applied="range_clamping"
                )

            return age

        # Case 2: Try to parse from raw_value
        if age_field.raw_value:
            parsed_age = self._parse_age_from_string(age_field.raw_value)
            if parsed_age is not None:
                result.add_warning(
                    "age",
                    "Parsed age from raw_value",
                    original_value=age_field.raw_value,
                    normalized_value=parsed_age,
                    rule_applied="string_parsing"
                )
                return parsed_age

        # Case 3: Use default
        result.add_warning(
            "age",
            f"Age not found, using default {self.DEFAULT_AGE}",
            normalized_value=self.DEFAULT_AGE,
            rule_applied="imputation"
        )
        return self.DEFAULT_AGE

    def _parse_age_from_string(self, age_str: str) -> Optional[float]:
        """Parse age from string like '60yo', '60 years old'"""
        if not age_str:
            return None

        # Extract first number
        match = re.search(r'(\d+\.?\d*)', str(age_str))
        if match:
            try:
                age = float(match.group(1))
                if self.AGE_MIN <= age <= self.AGE_MAX:
                    return age
            except ValueError:
                pass

        return None

    # ========================================================================
    # DOSAGE NORMALIZATION
    # ========================================================================

    def _normalize_dosage(self, dosage_field: ExtractedField, result: NormalizationResult) -> float:
        """
        Normalize dosage to grams.

        Rules:
        1. If value is numeric, use it (assume already in grams)
        2. If raw_value has units, parse and convert
        3. Validate conversion if unit_conversion is provided
        4. Validate range (0, 10]
        """
        # Case 1: Value already provided (should be in grams)
        if dosage_field.value is not None:
            try:
                dosage = float(dosage_field.value)
            except (ValueError, TypeError):
                result.add_error(
                    "dosage",
                    f"Invalid dosage value: {dosage_field.value}",
                    original_value=dosage_field.value
                )
                return self.DEFAULT_DOSAGE

            # Validate range
            if not (self.DOSAGE_MIN < dosage <= self.DOSAGE_MAX):
                result.add_error(
                    "dosage",
                    f"Dosage {dosage}g out of valid range ({self.DOSAGE_MIN}, {self.DOSAGE_MAX}]",
                    original_value=dosage
                )
                # If extremely wrong, suspect unit error
                if dosage > 100:
                    # Likely mg not converted to g
                    corrected = dosage / 1000
                    result.add_warning(
                        "dosage",
                        f"Dosage {dosage}g seems to be in mg, converted to {corrected}g",
                        original_value=dosage,
                        normalized_value=corrected,
                        rule_applied="suspected_unit_error"
                    )
                    return corrected

                return dosage

            return dosage

        # Case 2: Parse from raw_value with units
        if dosage_field.raw_value:
            parsed_dosage = self._parse_dosage_from_string(dosage_field.raw_value)
            if parsed_dosage is not None:
                result.add_warning(
                    "dosage",
                    "Parsed dosage from raw_value",
                    original_value=dosage_field.raw_value,
                    normalized_value=parsed_dosage,
                    rule_applied="string_parsing_with_units"
                )
                return parsed_dosage

        # Case 3: Use default
        result.add_warning(
            "dosage",
            f"Dosage not found, using default {self.DEFAULT_DOSAGE}g",
            normalized_value=self.DEFAULT_DOSAGE,
            rule_applied="imputation"
        )
        return self.DEFAULT_DOSAGE

    def _parse_dosage_from_string(self, dosage_str: str) -> Optional[float]:
        """Parse dosage with unit conversion to grams"""
        if not dosage_str:
            return None

        dosage_lower = str(dosage_str).lower()

        # Extract number
        match = re.search(r'(\d+\.?\d*)', dosage_lower)
        if not match:
            return None

        try:
            number = float(match.group(1))
        except ValueError:
            return None

        # Determine unit and convert
        if 'mg' in dosage_lower and 'mcg' not in dosage_lower:
            # Milligrams to grams
            return number / 1000
        elif 'mcg' in dosage_lower or 'µg' in dosage_lower:
            # Micrograms to grams
            return number / 1000000
        elif 'g' in dosage_lower and 'mg' not in dosage_lower:
            # Already grams
            return number
        else:
            # No clear unit - assume mg if > 10, else grams
            if number > 10:
                return number / 1000
            else:
                return number

    # ========================================================================
    # GENDER NORMALIZATION
    # ========================================================================

    def _normalize_gender(self, gender_field: ExtractedField, result: NormalizationResult) -> int:
        """
        Normalize gender to binary encoding.

        Gender_Male:
        - 1 = Male
        - 0 = Female

        Rules:
        1. Map common variations to male/female
        2. Default to 1 (male) if ambiguous
        """
        if gender_field.value is None:
            if gender_field.raw_value:
                # Try to parse from raw_value
                gender = self._parse_gender(gender_field.raw_value)
                result.add_warning(
                    "gender",
                    f"Parsed gender from raw_value: {gender_field.raw_value} → {gender}",
                    original_value=gender_field.raw_value,
                    normalized_value="male" if gender == 1 else "female",
                    rule_applied="string_parsing"
                )
                return gender

            # Default to male
            result.add_warning(
                "gender",
                "Gender not found, defaulting to male (Gender_Male=1)",
                normalized_value=1,
                rule_applied="default_to_male"
            )
            return 1

        # Parse from value
        gender = self._parse_gender(str(gender_field.value))
        return gender

    def _parse_gender(self, gender_str: str) -> int:
        """
        Parse gender string to binary.

        Returns:
            1 for male, 0 for female
        """
        gender_lower = str(gender_str).lower().strip()

        # Male variations
        if gender_lower in ['male', 'm', 'man', 'gentleman', 'boy', 'he', 'his', 'mr', 'mr.']:
            return 1

        # Female variations
        if gender_lower in ['female', 'f', 'woman', 'lady', 'girl', 'she', 'her', 'mrs', 'mrs.', 'ms', 'ms.']:
            return 0

        # Default to male if unclear
        return 1

    # ========================================================================
    # ROUTE NORMALIZATION
    # ========================================================================

    def _normalize_route(self, route_field: ExtractedField, result: NormalizationResult) -> Tuple[int, int]:
        """
        Normalize route to binary encoding.

        Route_IV, Route_Oral:
        - IV: (1, 0)
        - Oral: (0, 1)
        - BD/Other: (0, 0)

        Rules:
        1. Map common variations to IV/Oral
        2. If both flags would be 1, resolve conflict (error)
        3. If neither, assume BD (0, 0) - valid encoding
        """
        if route_field.value is None:
            if route_field.raw_value:
                route_iv, route_oral = self._parse_route(route_field.raw_value)
                result.add_warning(
                    "route",
                    f"Parsed route from raw_value: {route_field.raw_value}",
                    original_value=route_field.raw_value,
                    normalized_value=f"IV={route_iv}, Oral={route_oral}",
                    rule_applied="string_parsing"
                )
                return route_iv, route_oral

            # Default to BD (both 0)
            result.add_warning(
                "route",
                "Route not found, defaulting to BD/Other (0, 0)",
                normalized_value="BD",
                rule_applied="default_to_bd"
            )
            return 0, 0

        # Parse from value
        route_iv, route_oral = self._parse_route(str(route_field.value))
        return route_iv, route_oral

    def _parse_route(self, route_str: str) -> Tuple[int, int]:
        """
        Parse route string to binary encoding.

        Returns:
            (Route_IV, Route_Oral)
        """
        route_lower = str(route_str).lower().strip()

        # IV variations
        if route_lower in ['iv', 'intravenous', 'intravenously']:
            return 1, 0

        # Oral variations
        if route_lower in ['oral', 'po', 'orally', 'by mouth', 'mouth']:
            return 0, 1

        # Other routes (IM, SC, etc.) or BD
        # All map to (0, 0)
        return 0, 0

    # ========================================================================
    # FREQUENCY NORMALIZATION
    # ========================================================================

    def _normalize_frequency(self, freq_field: ExtractedField, result: NormalizationResult) -> Tuple[int, int, int]:
        """
        Normalize frequency to binary encoding.

        Frequency_OD, Frequency_QID, Frequency_TDS:
        - OD (once daily): (1, 0, 0)
        - QID (4x daily): (0, 1, 0)
        - TDS (3x daily): (0, 0, 1)
        - BD (twice daily): (0, 0, 0) - IMPORTANT: All zeros is valid!

        Rules:
        1. Map common variations to OD/BD/TDS/QID
        2. If multiple flags would be 1, resolve conflict (error)
        3. BD is represented by all zeros (not an error!)
        """
        if freq_field.value is None:
            if freq_field.raw_value:
                freq_od, freq_qid, freq_tds = self._parse_frequency(freq_field.raw_value)
                result.add_warning(
                    "frequency",
                    f"Parsed frequency from raw_value: {freq_field.raw_value}",
                    original_value=freq_field.raw_value,
                    normalized_value=f"OD={freq_od}, QID={freq_qid}, TDS={freq_tds}",
                    rule_applied="string_parsing"
                )
                return freq_od, freq_qid, freq_tds

            # Default to BD (all zeros)
            result.add_warning(
                "frequency",
                "Frequency not found, defaulting to BD (0, 0, 0)",
                normalized_value="BD",
                rule_applied="default_to_bd"
            )
            return 0, 0, 0

        # Parse from value
        freq_od, freq_qid, freq_tds = self._parse_frequency(str(freq_field.value))
        return freq_od, freq_qid, freq_tds

    def _parse_frequency(self, freq_str: str) -> Tuple[int, int, int]:
        """
        Parse frequency string to binary encoding.

        Returns:
            (Frequency_OD, Frequency_QID, Frequency_TDS)
        """
        freq_lower = str(freq_str).lower().strip()

        # OD (once daily)
        if freq_lower in ['od', 'qd', 'once daily', 'once', 'daily', 'q24h']:
            return 1, 0, 0

        # QID (four times daily)
        if freq_lower in ['qid', 'four times daily', 'four times', 'q6h']:
            return 0, 1, 0

        # TDS (three times daily)
        if freq_lower in ['tds', 'tid', 'three times daily', 'three times', 'thrice daily', 'q8h']:
            return 0, 0, 1

        # BD (twice daily) - ALL ZEROS
        if freq_lower in ['bd', 'bid', 'twice daily', 'twice', 'q12h']:
            return 0, 0, 0

        # Unknown - default to BD (all zeros)
        return 0, 0, 0

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_route_encoding(self, route_iv: int, route_oral: int, result: NormalizationResult):
        """
        Validate route encoding.

        Rules:
        - Both 0: Valid (BD/Other route)
        - One 1: Valid
        - Both 1: CONFLICT (Error)
        """
        if route_iv == 1 and route_oral == 1:
            result.add_error(
                "route",
                "Conflict: Both Route_IV and Route_Oral are 1 (must be mutually exclusive)",
                original_value=f"IV={route_iv}, Oral={route_oral}",
                rule_applied="mutual_exclusion_check"
            )

    def _validate_frequency_encoding(self, freq_od: int, freq_qid: int, freq_tds: int, result: NormalizationResult):
        """
        Validate frequency encoding.

        Rules:
        - All 0: Valid (BD - twice daily)
        - One 1: Valid
        - Multiple 1s: CONFLICT (Error)
        """
        num_active = freq_od + freq_qid + freq_tds

        if num_active > 1:
            result.add_error(
                "frequency",
                f"Conflict: Multiple frequency flags are 1 (OD={freq_od}, QID={freq_qid}, TDS={freq_tds}). Only one can be active.",
                original_value=f"OD={freq_od}, QID={freq_qid}, TDS={freq_tds}",
                rule_applied="mutual_exclusion_check"
            )
        elif num_active == 0:
            # All zeros is VALID - represents BD
            result.add_warning(
                "frequency",
                "All frequency flags are 0, representing BD (twice daily) - this is valid",
                normalized_value="BD",
                rule_applied="bd_encoding"
            )
