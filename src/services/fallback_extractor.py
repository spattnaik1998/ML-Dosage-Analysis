"""
Rule-Based Fallback Extractor for Clinical Case Studies.

This module provides regex-based extraction when LLM output is invalid,
low-confidence, or unavailable. Uses pattern matching and heuristics
to extract clinical features with confidence scoring.
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from src.models.schema import RawFeatures, ExtractedField


# ============================================================================
# REGEX PATTERNS
# ============================================================================

class ClinicalPatterns:
    """
    Comprehensive regex patterns for clinical data extraction.
    Patterns are ordered by specificity (most specific first).
    """

    # ========================================================================
    # AGE PATTERNS
    # ========================================================================
    AGE_PATTERNS = [
        # "62-year-old", "62 year old", "62 years old"
        (r'(\d{1,3})[- ]year[s]?[- ]old', 0.95),

        # "62yo", "62 yo", "62y/o", "62 y/o"
        (r'(\d{1,3})\s*y/?o\b', 0.90),

        # "age 62", "age: 62", "aged 62"
        (r'age[d]?\s*:?\s*(\d{1,3})\b', 0.90),

        # "Patient is 62", "Pt is 62"
        (r'(?:patient|pt)[^.]*?is\s+(\d{1,3})\b', 0.85),

        # "62M", "62F" (age with gender)
        (r'\b(\d{1,3})\s*[MFmf]\b', 0.80),

        # Standalone age at start of sentence
        (r'^(\d{1,3})\s+(?:year|yr)', 0.75),

        # Age ranges: "60-65 years"
        (r'(\d{1,3})-\d{1,3}\s*year', 0.70),

        # Contextual: any 2-3 digit number near age keywords
        (r'(?:age|years?|yo)\D*(\d{1,3})', 0.60),
    ]

    # ========================================================================
    # DOSAGE PATTERNS
    # ========================================================================
    DOSAGE_PATTERNS = [
        # "500mg", "500 mg", "0.5g", "0.5 g"
        (r'(\d+\.?\d*)\s*(mg|g|gm|gram|milligram|mcg|µg|microgram)s?\b', 0.95),

        # "dose: 500mg", "dosage: 500mg"
        (r'dos(?:e|age)\s*:?\s*(\d+\.?\d*)\s*(mg|g|gm)', 0.95),

        # Drug name followed by dose: "ceftriaxone 1g"
        (r'[a-z]+(?:cillin|mycin|floxacin|azole)\s+(\d+\.?\d*)\s*(mg|g|gm)', 0.90),

        # "1g IV", "500mg PO"
        (r'(\d+\.?\d*)\s*(mg|g|gm)\s+(?:IV|PO|IM|SC)', 0.90),

        # "given 500mg", "prescribed 1g"
        (r'(?:given|prescribed|started)\s+(\d+\.?\d*)\s*(mg|g|gm)', 0.85),

        # Standalone dose with units
        (r'\b(\d+\.?\d*)\s*(mg|g|gm|mcg)\b', 0.70),
    ]

    # ========================================================================
    # GENDER PATTERNS
    # ========================================================================
    GENDER_PATTERNS = [
        # "male patient", "female patient"
        (r'\b(male|female)\s+patient', 0.95, None),

        # "M", "F" with word boundaries
        (r'\b([MF])\b', 0.90, None),

        # "man", "woman", "gentleman", "lady"
        (r'\b(man|woman|gentleman|lady|male|female)\b', 0.85,
         {'man': 'male', 'gentleman': 'male', 'woman': 'female', 'lady': 'female'}),

        # "he", "she", "his", "her"
        (r'\b(he|she|his|her)\b', 0.70,
         {'he': 'male', 'his': 'male', 'she': 'female', 'her': 'female'}),

        # "Mr.", "Mrs.", "Ms."
        (r'\b(Mr\.|Mrs\.|Ms\.)', 0.80,
         {'Mr.': 'male', 'Mrs.': 'female', 'Ms.': 'female'}),
    ]

    # ========================================================================
    # ROUTE PATTERNS
    # ========================================================================
    ROUTE_PATTERNS = [
        # "intravenous", "IV"
        (r'\b(intravenous|IV)\b', 0.95, 'IV'),

        # "oral", "PO", "by mouth"
        (r'\b(oral|PO|by mouth)\b', 0.95, 'oral'),

        # "intramuscular", "IM"
        (r'\b(intramuscular|IM)\b', 0.90, 'other'),

        # "subcutaneous", "SC", "SubQ"
        (r'\b(subcutaneous|SC|SubQ)\b', 0.90, 'other'),

        # Route after dose: "500mg IV"
        (r'\d+\.?\d*\s*(?:mg|g)\s+(IV|PO|oral)', 0.95, None),

        # "given IV", "administered orally"
        (r'(?:given|administered)\s+(IV|orally|intravenously)', 0.90,
         {'orally': 'oral', 'intravenously': 'IV'}),
    ]

    # ========================================================================
    # FREQUENCY PATTERNS
    # ========================================================================
    FREQUENCY_PATTERNS = [
        # Standard abbreviations: "OD", "BD", "TDS", "QID"
        (r'\b(OD|BD|TDS|QID)\b', 0.95, None),

        # "once daily", "twice daily", etc.
        (r'\b(once|twice|three times|four times)\s+(?:a\s+)?daily', 0.95,
         {'once': 'OD', 'twice': 'BD', 'three times': 'TDS', 'four times': 'QID'}),

        # "q12h", "q8h", "q6h" (every X hours)
        (r'\b(q\d{1,2}h)\b', 0.90,
         {'q24h': 'OD', 'q12h': 'BD', 'q8h': 'TDS', 'q6h': 'QID'}),

        # "every 12 hours", "every 8 hours"
        (r'every\s+(\d{1,2})\s+hours?', 0.85,
         {'24': 'OD', '12': 'BD', '8': 'TDS', '6': 'QID'}),

        # Latin abbreviations: "QD", "BID", "TID"
        (r'\b(QD|BID|TID|QID)\b', 0.90,
         {'QD': 'OD', 'BID': 'BD', 'TID': 'TDS'}),

        # "x times daily", "x times a day"
        (r'(\d)\s+times\s+(?:a\s+)?da(?:ily|y)', 0.85,
         {'1': 'OD', '2': 'BD', '3': 'TDS', '4': 'QID'}),

        # After dose: "500mg BD"
        (r'\d+\.?\d*\s*(?:mg|g)\s+(?:IV|PO)?\s*(OD|BD|TDS|QID)', 0.95, None),
    ]


# ============================================================================
# EXTRACTION RESULT
# ============================================================================

@dataclass
class PatternMatch:
    """Result of a single pattern match"""
    value: str
    raw_value: str
    pattern: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str  # Surrounding text for evidence


# ============================================================================
# FALLBACK EXTRACTOR
# ============================================================================

class FallbackExtractor:
    """
    Rule-based extractor using regex patterns and heuristics.
    Used as fallback when LLM output is invalid or low-confidence.
    """

    def __init__(self):
        """Initialize fallback extractor"""
        self.patterns = ClinicalPatterns()

    def extract(self, case_study_text: str) -> RawFeatures:
        """
        Extract clinical features using regex patterns.

        Args:
            case_study_text: Clinical case study text

        Returns:
            RawFeatures object with extracted data
        """
        text = case_study_text.strip()

        # Extract each field
        age_field = self._extract_age(text)
        dosage_field = self._extract_dosage(text)
        gender_field = self._extract_gender(text)
        route_field = self._extract_route(text)
        frequency_field = self._extract_frequency(text)

        return RawFeatures(
            age=age_field,
            dosage=dosage_field,
            gender=gender_field,
            route=route_field,
            frequency=frequency_field
        )

    def _extract_age(self, text: str) -> ExtractedField:
        """Extract age using patterns"""
        matches = self._find_all_matches(text, self.patterns.AGE_PATTERNS)

        if not matches:
            return ExtractedField(
                value=None,
                raw_value=None,
                evidence=None,
                confidence=0.0
            )

        # Select best match (highest confidence)
        best_match = max(matches, key=lambda m: m.confidence)

        # Parse age value
        try:
            age_value = float(best_match.value)

            # Sanity check
            if not (0 <= age_value <= 120):
                # Reduce confidence for unusual ages
                best_match.confidence *= 0.5

        except ValueError:
            age_value = None
            best_match.confidence *= 0.3

        return ExtractedField(
            value=age_value,
            raw_value=best_match.raw_value,
            evidence=best_match.context,
            confidence=best_match.confidence
        )

    def _extract_dosage(self, text: str) -> ExtractedField:
        """Extract dosage with unit normalization to grams"""
        matches = self._find_all_matches_with_units(
            text,
            self.patterns.DOSAGE_PATTERNS
        )

        if not matches:
            return ExtractedField(
                value=None,
                raw_value=None,
                evidence=None,
                confidence=0.0,
                unit_conversion=None
            )

        # Select best match
        best_match = max(matches, key=lambda m: m.confidence)

        # Parse and convert to grams
        dosage_value, unit_conversion = self._normalize_dosage(
            best_match.value,
            best_match.raw_value
        )

        # Sanity check
        if dosage_value and not (0 < dosage_value <= 10):
            best_match.confidence *= 0.6

        return ExtractedField(
            value=dosage_value,
            raw_value=best_match.raw_value,
            evidence=best_match.context,
            confidence=best_match.confidence,
            unit_conversion=unit_conversion
        )

    def _extract_gender(self, text: str) -> ExtractedField:
        """Extract gender"""
        matches = self._find_all_matches_with_mapping(
            text,
            self.patterns.GENDER_PATTERNS
        )

        if not matches:
            return ExtractedField(
                value=None,
                raw_value=None,
                evidence=None,
                confidence=0.0
            )

        # Select best match
        best_match = max(matches, key=lambda m: m.confidence)

        # Normalize to "male" or "female"
        gender_value = self._normalize_gender(best_match.value)

        return ExtractedField(
            value=gender_value,
            raw_value=best_match.raw_value,
            evidence=best_match.context,
            confidence=best_match.confidence
        )

    def _extract_route(self, text: str) -> ExtractedField:
        """Extract administration route"""
        matches = self._find_all_matches_with_mapping(
            text,
            self.patterns.ROUTE_PATTERNS
        )

        if not matches:
            return ExtractedField(
                value=None,
                raw_value=None,
                evidence=None,
                confidence=0.0
            )

        best_match = max(matches, key=lambda m: m.confidence)

        # Already normalized to "IV", "oral", or "other"
        route_value = best_match.value

        return ExtractedField(
            value=route_value,
            raw_value=best_match.raw_value,
            evidence=best_match.context,
            confidence=best_match.confidence
        )

    def _extract_frequency(self, text: str) -> ExtractedField:
        """Extract dosing frequency"""
        matches = self._find_all_matches_with_mapping(
            text,
            self.patterns.FREQUENCY_PATTERNS
        )

        if not matches:
            return ExtractedField(
                value=None,
                raw_value=None,
                evidence=None,
                confidence=0.0
            )

        best_match = max(matches, key=lambda m: m.confidence)

        # Normalize to standard abbreviations
        freq_value = self._normalize_frequency(best_match.value)

        return ExtractedField(
            value=freq_value,
            raw_value=best_match.raw_value,
            evidence=best_match.context,
            confidence=best_match.confidence
        )

    def _find_all_matches(
        self,
        text: str,
        patterns: List[Tuple[str, float]]
    ) -> List[PatternMatch]:
        """Find all matches for simple patterns"""
        matches = []

        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1)
                raw_value = match.group(0)

                # Get context (30 chars before and after)
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()

                matches.append(PatternMatch(
                    value=value,
                    raw_value=raw_value,
                    pattern=pattern,
                    confidence=base_confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context
                ))

        return matches

    def _find_all_matches_with_units(
        self,
        text: str,
        patterns: List[Tuple[str, float]]
    ) -> List[PatternMatch]:
        """Find matches for patterns with units (dosage)"""
        matches = []

        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Group 1: number, Group 2: unit
                value = match.group(1)
                unit = match.group(2) if match.lastindex >= 2 else None
                raw_value = match.group(0)

                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()

                # Store both number and unit in raw_value
                if unit:
                    raw_value = f"{value}{unit}"

                matches.append(PatternMatch(
                    value=value,
                    raw_value=raw_value,
                    pattern=pattern,
                    confidence=base_confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context
                ))

        return matches

    def _find_all_matches_with_mapping(
        self,
        text: str,
        patterns: List[Tuple[str, float, Optional[dict]]]
    ) -> List[PatternMatch]:
        """Find matches for patterns with value mapping"""
        matches = []

        for pattern, base_confidence, mapping in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1)
                raw_value = match.group(0)

                # Apply mapping if exists
                if mapping:
                    value = mapping.get(value.lower(), mapping.get(value, value))

                # Get context
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()

                matches.append(PatternMatch(
                    value=value,
                    raw_value=raw_value,
                    pattern=pattern,
                    confidence=base_confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context
                ))

        return matches

    def _normalize_dosage(self, value: str, raw_value: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Normalize dosage to grams.

        Args:
            value: Numeric value as string
            raw_value: Full raw value with units

        Returns:
            (normalized_value_in_grams, conversion_string)
        """
        try:
            number = float(value)
        except ValueError:
            return None, None

        raw_lower = raw_value.lower()

        # Determine unit and convert
        if 'mg' in raw_lower and 'mcg' not in raw_lower:
            # Milligrams to grams
            converted = number / 1000
            conversion = f"{raw_value} → {converted}g"
        elif 'mcg' in raw_lower or 'µg' in raw_lower or 'microgram' in raw_lower:
            # Micrograms to grams
            converted = number / 1000000
            conversion = f"{raw_value} → {converted}g"
        elif 'g' in raw_lower or 'gram' in raw_lower:
            # Already in grams
            converted = number
            conversion = f"{raw_value} → {converted}g"
        else:
            # Unknown unit, assume mg if number is large
            if number > 10:
                converted = number / 1000
                conversion = f"{raw_value} (assumed mg) → {converted}g"
            else:
                converted = number
                conversion = f"{raw_value} (assumed g) → {converted}g"

        return converted, conversion

    def _normalize_gender(self, value: str) -> str:
        """Normalize gender to 'male' or 'female'"""
        value_lower = value.lower()

        if value_lower in ['male', 'm', 'man', 'gentleman', 'he', 'his', 'mr.']:
            return 'male'
        elif value_lower in ['female', 'f', 'woman', 'lady', 'she', 'her', 'mrs.', 'ms.']:
            return 'female'
        else:
            return value_lower

    def _normalize_frequency(self, value: str) -> str:
        """Normalize frequency to standard abbreviations"""
        value_upper = value.upper()

        # Already standard
        if value_upper in ['OD', 'BD', 'TDS', 'QID']:
            return value_upper

        # Map variations
        mapping = {
            'QD': 'OD',
            'BID': 'BD',
            'TID': 'TDS',
            'ONCE': 'OD',
            'TWICE': 'BD',
            'THREE TIMES': 'TDS',
            'FOUR TIMES': 'QID'
        }

        return mapping.get(value_upper, value)


# ============================================================================
# HYBRID EXTRACTOR (LLM + Fallback)
# ============================================================================

class HybridExtractor:
    """
    Combines LLM extraction with regex fallback.
    Uses fallback when LLM output is invalid or low-confidence.
    """

    def __init__(
        self,
        llm_extractor,
        confidence_threshold: float = 0.6,
        use_fallback_for_missing: bool = True
    ):
        """
        Initialize hybrid extractor.

        Args:
            llm_extractor: LLMExtractor instance
            confidence_threshold: Minimum confidence to trust LLM
            use_fallback_for_missing: Use fallback for null LLM fields
        """
        self.llm_extractor = llm_extractor
        self.fallback_extractor = FallbackExtractor()
        self.confidence_threshold = confidence_threshold
        self.use_fallback_for_missing = use_fallback_for_missing

    def extract(self, case_study_text: str) -> dict:
        """
        Extract using LLM with fallback support.

        Args:
            case_study_text: Clinical text

        Returns:
            Dictionary with extraction results and metadata
        """
        # Try LLM first
        try:
            llm_result = self.llm_extractor.extract(case_study_text)
            llm_features = llm_result["raw_features"]
            llm_failed = False
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            llm_features = None
            llm_failed = True

        # Determine if fallback is needed
        needs_fallback = self._needs_fallback(llm_features, llm_failed)

        if needs_fallback:
            # Use fallback extraction
            fallback_features = self.fallback_extractor.extract(case_study_text)

            # Merge LLM and fallback results
            if llm_features and not llm_failed:
                merged_features = self._merge_features(
                    llm_features,
                    fallback_features
                )
                method = "hybrid"
            else:
                merged_features = fallback_features
                method = "fallback_only"

            return {
                "raw_features": merged_features,
                "extraction_method": method,
                "llm_failed": llm_failed,
                "fallback_used": True,
                "provider_used": llm_result.get("provider_used") if llm_result else None
            }
        else:
            # LLM output is good enough
            return {
                "raw_features": llm_features,
                "extraction_method": "llm_only",
                "llm_failed": False,
                "fallback_used": False,
                "provider_used": llm_result["provider_used"]
            }

    def _needs_fallback(self, llm_features: Optional[RawFeatures], llm_failed: bool) -> bool:
        """Determine if fallback extraction is needed"""
        if llm_failed or llm_features is None:
            return True

        # Check overall confidence
        confidences = []
        for field_name in ['age', 'dosage', 'gender', 'route', 'frequency']:
            field = getattr(llm_features, field_name)
            if field and hasattr(field, 'confidence'):
                confidences.append(field.confidence)

        if not confidences:
            return True

        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < self.confidence_threshold:
            return True

        # Check for missing critical fields
        if self.use_fallback_for_missing:
            for field_name in ['age', 'dosage']:
                field = getattr(llm_features, field_name)
                if field is None or field.value is None:
                    return True

        return False

    def _merge_features(
        self,
        llm_features: RawFeatures,
        fallback_features: RawFeatures
    ) -> RawFeatures:
        """
        Merge LLM and fallback results, preferring higher confidence.

        Args:
            llm_features: Features from LLM
            fallback_features: Features from fallback

        Returns:
            Merged RawFeatures
        """
        merged = {}

        for field_name in ['age', 'dosage', 'gender', 'route', 'frequency']:
            llm_field = getattr(llm_features, field_name)
            fallback_field = getattr(fallback_features, field_name)

            # Choose based on confidence and value presence
            if llm_field and llm_field.value is not None:
                if fallback_field and fallback_field.value is not None:
                    # Both have values - choose higher confidence
                    if fallback_field.confidence > llm_field.confidence:
                        merged[field_name] = fallback_field
                    else:
                        merged[field_name] = llm_field
                else:
                    # Only LLM has value
                    merged[field_name] = llm_field
            elif fallback_field and fallback_field.value is not None:
                # Only fallback has value
                merged[field_name] = fallback_field
            else:
                # Neither has value - use LLM (preserves null with evidence)
                merged[field_name] = llm_field if llm_field else fallback_field

        return RawFeatures(**merged)
