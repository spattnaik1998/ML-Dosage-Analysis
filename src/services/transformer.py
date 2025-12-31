"""
Feature Transformer for converting raw LLM extraction to model input format.

This module handles the transformation of raw extracted features from LLM output
into the canonical ModelInput schema required by the ML model.
"""

import re
from typing import Optional
from src.models.schema import RawFeatures, ModelInput, ExtractedField


class FeatureTransformer:
    """
    Transforms raw extracted features into canonical model input schema.
    Handles normalization, encoding, and validation.
    """

    # Training data statistics for imputation
    MEDIAN_AGE = 55.0
    MEDIAN_DOSAGE = 0.75

    # Mapping dictionaries
    GENDER_MAPPING = {
        'male': 1, 'm': 1, 'M': 1, 'Male': 1,
        'female': 0, 'f': 0, 'F': 0, 'Female': 0
    }

    ROUTE_MAPPING = {
        'iv': ('Route_IV', 1),
        'intravenous': ('Route_IV', 1),
        'IV': ('Route_IV', 1),
        'oral': ('Route_Oral', 1),
        'PO': ('Route_Oral', 1),
        'Oral': ('Route_Oral', 1),
        'po': ('Route_Oral', 1),
        'bd': ('BD', 0),  # Both flags 0
        'BD': ('BD', 0),
    }

    FREQUENCY_MAPPING = {
        'od': 'Frequency_OD',
        'OD': 'Frequency_OD',
        'once daily': 'Frequency_OD',
        'once': 'Frequency_OD',
        'qid': 'Frequency_QID',
        'QID': 'Frequency_QID',
        'four times daily': 'Frequency_QID',
        'tds': 'Frequency_TDS',
        'TDS': 'Frequency_TDS',
        'three times daily': 'Frequency_TDS',
        'thrice daily': 'Frequency_TDS',
        'bd': 'BD',  # All flags 0
        'BD': 'BD',
        'twice daily': 'BD',
        'q12h': 'BD',  # Every 12 hours = twice daily
    }

    @staticmethod
    def parse_age(age_field: ExtractedField) -> float:
        """
        Parse age from ExtractedField.
        Examples: "60", "60 years", "60yo", 60.0

        Args:
            age_field: ExtractedField containing age information

        Returns:
            Parsed age as float
        """
        # If value is already provided and is numeric, use it
        if age_field.value is not None:
            if isinstance(age_field.value, (int, float)):
                return float(age_field.value)

        # Try to parse from raw_value
        if age_field.raw_value is None:
            return FeatureTransformer.MEDIAN_AGE

        age_str = str(age_field.raw_value)

        # Extract numeric value from string
        match = re.search(r'(\d+\.?\d*)', age_str)
        if match:
            return float(match.group(1))

        return FeatureTransformer.MEDIAN_AGE

    @staticmethod
    def parse_dosage(dosage_field: ExtractedField) -> float:
        """
        Parse dosage from ExtractedField and convert to grams.
        Examples: "0.5", "0.5g", "500mg", "0.5 grams", 0.5

        Args:
            dosage_field: ExtractedField containing dosage information

        Returns:
            Dosage in grams as float
        """
        # If value is already provided and is numeric, use it (should already be in grams)
        if dosage_field.value is not None:
            if isinstance(dosage_field.value, (int, float)):
                return float(dosage_field.value)

        # Try to parse from raw_value
        if dosage_field.raw_value is None:
            return FeatureTransformer.MEDIAN_DOSAGE

        dosage_str = str(dosage_field.raw_value).lower().strip()

        # Extract numeric value
        match = re.search(r'(\d+\.?\d*)', dosage_str)
        if not match:
            return FeatureTransformer.MEDIAN_DOSAGE

        value = float(match.group(1))

        # Convert units
        if 'mg' in dosage_str and 'mcg' not in dosage_str:
            value = value / 1000  # mg to grams
        elif 'mcg' in dosage_str or 'µg' in dosage_str:
            value = value / 1000000  # mcg to grams
        # If 'g' or 'gram' in string, value is already in grams

        return value

    @staticmethod
    def encode_gender(gender_field: ExtractedField) -> int:
        """
        Encode gender to binary (1=Male, 0=Female).

        Args:
            gender_field: ExtractedField containing gender information

        Returns:
            1 for Male, 0 for Female
        """
        # Check value first
        if gender_field.value is not None:
            gender_str = str(gender_field.value).strip().lower()
            if gender_str in ['male', 'm']:
                return 1
            elif gender_str in ['female', 'f']:
                return 0

        # Check raw_value
        if gender_field.raw_value is None:
            return 1  # Default to male

        gender_normalized = str(gender_field.raw_value).strip().lower()
        return FeatureTransformer.GENDER_MAPPING.get(gender_normalized, 1)

    @staticmethod
    def encode_route(route_field: ExtractedField) -> tuple[int, int]:
        """
        Encode route to (Route_IV, Route_Oral).

        Args:
            route_field: ExtractedField containing route information

        Returns:
            Tuple of (Route_IV, Route_Oral)
        """
        # Check value first
        if route_field.value is not None:
            route_str = str(route_field.value).strip().lower()
        elif route_field.raw_value is not None:
            route_str = str(route_field.raw_value).strip().lower()
        else:
            return (0, 0)  # BD or unlisted

        if 'iv' in route_str or 'intravenous' in route_str:
            return (1, 0)
        elif 'oral' in route_str or 'po' in route_str:
            return (0, 1)
        else:
            return (0, 0)  # BD or other

    @staticmethod
    def encode_frequency(freq_field: ExtractedField) -> tuple[int, int, int]:
        """
        Encode frequency to (Frequency_OD, Frequency_QID, Frequency_TDS).

        Args:
            freq_field: ExtractedField containing frequency information

        Returns:
            Tuple of (OD, QID, TDS)
        """
        # Check value first
        if freq_field.value is not None:
            freq_str = str(freq_field.value).strip().lower()
        elif freq_field.raw_value is not None:
            freq_str = str(freq_field.raw_value).strip().lower()
        else:
            return (0, 0, 0)  # BD (twice daily)

        # Normalize variations
        if 'od' in freq_str or 'once' in freq_str:
            return (1, 0, 0)
        elif 'qid' in freq_str or 'four times' in freq_str:
            return (0, 1, 0)
        elif 'tds' in freq_str or 'three times' in freq_str or 'thrice' in freq_str:
            return (0, 0, 1)
        else:
            return (0, 0, 0)  # BD or unlisted

    @classmethod
    def transform(cls, raw_features: RawFeatures) -> ModelInput:
        """
        Transform raw features to canonical model input.

        Args:
            raw_features: RawFeatures object from LLM extraction

        Returns:
            ModelInput object ready for model inference

        Raises:
            ValueError: If transformation results in invalid data
        """
        # Parse numeric features
        age = cls.parse_age(raw_features.age)
        dosage = cls.parse_dosage(raw_features.dosage)

        # Encode categorical features
        gender_male = cls.encode_gender(raw_features.gender)
        route_iv, route_oral = cls.encode_route(raw_features.route)
        freq_od, freq_qid, freq_tds = cls.encode_frequency(raw_features.frequency)

        # Create model input
        return ModelInput(
            Age=age,
            **{"Dosage (gram)": dosage},  # Use dict unpacking for special chars
            Gender_Male=gender_male,
            Route_IV=route_iv,
            Route_Oral=route_oral,
            Frequency_OD=freq_od,
            Frequency_QID=freq_qid,
            Frequency_TDS=freq_tds
        )

    @classmethod
    def transform_with_metadata(cls, raw_features: RawFeatures) -> dict:
        """
        Transform raw features and include transformation metadata.

        Args:
            raw_features: RawFeatures object from LLM extraction

        Returns:
            Dictionary containing model_input and transformation metadata
        """
        # Perform transformation
        model_input = cls.transform(raw_features)

        # Collect metadata about transformations
        metadata = {
            "age": {
                "original": raw_features.age.raw_value,
                "transformed": model_input.Age,
                "confidence": raw_features.age.confidence,
                "imputed": raw_features.age.value is None
            },
            "dosage": {
                "original": raw_features.dosage.raw_value,
                "transformed": model_input.Dosage_gram,
                "confidence": raw_features.dosage.confidence,
                "imputed": raw_features.dosage.value is None,
                "unit_conversion": raw_features.dosage.unit_conversion
            },
            "gender": {
                "original": raw_features.gender.raw_value,
                "transformed": "Male" if model_input.Gender_Male == 1 else "Female",
                "confidence": raw_features.gender.confidence,
                "imputed": raw_features.gender.value is None
            },
            "route": {
                "original": raw_features.route.raw_value,
                "transformed": "IV" if model_input.Route_IV == 1 else "Oral" if model_input.Route_Oral == 1 else "BD/Other",
                "confidence": raw_features.route.confidence,
                "imputed": raw_features.route.value is None
            },
            "frequency": {
                "original": raw_features.frequency.raw_value,
                "transformed": "OD" if model_input.Frequency_OD == 1 else "QID" if model_input.Frequency_QID == 1 else "TDS" if model_input.Frequency_TDS == 1 else "BD",
                "confidence": raw_features.frequency.confidence,
                "imputed": raw_features.frequency.value is None
            }
        }

        return {
            "model_input": model_input,
            "transformation_metadata": metadata
        }
