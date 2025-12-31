"""
Canonical Feature Schema for Hospital Treatment Duration Prediction Model.

This module defines the data models used throughout the ML pipeline:
- RawFeatures: LLM extraction output (intermediate representation)
- ModelInput: Canonical ML model input (validated and normalized)
- ExtractedField: Structure for individual field extraction with confidence
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from enum import Enum
import pandas as pd


# ============================================================================
# ENUMERATIONS FOR CATEGORICAL VALUES
# ============================================================================

class Gender(str, Enum):
    """Gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    M = "M"
    F = "F"


class Route(str, Enum):
    """Administration route enumeration"""
    IV = "IV"
    INTRAVENOUS = "intravenous"
    ORAL = "oral"
    PO = "PO"
    BD = "BD"  # Twice daily (when neither IV nor Oral)
    OTHER = "other"


class Frequency(str, Enum):
    """Dosing frequency enumeration"""
    OD = "OD"  # Once daily
    ONCE_DAILY = "once daily"
    BD = "BD"  # Twice daily
    TWICE_DAILY = "twice daily"
    TDS = "TDS"  # Three times daily
    THRICE_DAILY = "thrice daily"
    QID = "QID"  # Four times daily
    FOUR_TIMES_DAILY = "four times daily"


# ============================================================================
# EXTRACTED FIELD STRUCTURE (from LLM)
# ============================================================================

class ExtractedField(BaseModel):
    """
    Structure for a single extracted field from LLM.
    Includes value, raw text, evidence, and confidence score.
    """
    value: Optional[float | str] = Field(
        None,
        description="Extracted and normalized value"
    )
    raw_value: Optional[str] = Field(
        None,
        description="Original text extracted from case study"
    )
    evidence: Optional[str] = Field(
        None,
        description="Exact text span from input that supports extraction"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    unit_conversion: Optional[str] = Field(
        None,
        description="Unit conversion performed (e.g., '500mg → 0.5g')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "value": 60.0,
                "raw_value": "60 years old",
                "evidence": "Patient is 60 years old",
                "confidence": 1.0
            }
        }


# ============================================================================
# RAW FEATURE SCHEMA (LLM Extraction Output)
# ============================================================================

class RawFeatures(BaseModel):
    """
    Raw features extracted from unstructured text by LLM.
    This is the intermediate representation before transformation.
    Each field includes confidence scores and evidence.
    """
    age: ExtractedField = Field(
        ...,
        description="Patient age extraction with confidence"
    )
    dosage: ExtractedField = Field(
        ...,
        description="Medication dosage extraction with confidence"
    )
    gender: ExtractedField = Field(
        ...,
        description="Patient gender extraction with confidence"
    )
    route: ExtractedField = Field(
        ...,
        description="Administration route extraction with confidence"
    )
    frequency: ExtractedField = Field(
        ...,
        description="Dosing frequency extraction with confidence"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": {
                    "value": 60.0,
                    "raw_value": "60yo",
                    "evidence": "60yo male patient",
                    "confidence": 0.95
                },
                "dosage": {
                    "value": 0.5,
                    "raw_value": "500mg",
                    "evidence": "ceftriaxone 500mg",
                    "confidence": 1.0,
                    "unit_conversion": "500mg → 0.5g"
                },
                "gender": {
                    "value": "male",
                    "raw_value": "male",
                    "evidence": "60yo male patient",
                    "confidence": 1.0
                },
                "route": {
                    "value": "oral",
                    "raw_value": "PO",
                    "evidence": "500mg PO",
                    "confidence": 1.0
                },
                "frequency": {
                    "value": "BD",
                    "raw_value": "twice daily",
                    "evidence": "PO twice daily",
                    "confidence": 1.0
                }
            }
        }


# ============================================================================
# MODEL INPUT SCHEMA (Canonical ML Model Input)
# ============================================================================

class ModelInput(BaseModel):
    """
    Canonical schema for ML model input.
    All features must conform to this exact specification.

    Feature Encoding Rules:
    - Route: If both Route_IV=0 and Route_Oral=0, represents BD (twice daily) or unlisted route
    - Frequency: If all three (OD, QID, TDS)=0, represents BD (twice daily)
    """

    # Numeric features
    Age: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Patient age in years",
        json_schema_extra={
            "unit": "years",
            "default_on_missing": "median",
            "training_median": 55.0
        }
    )

    Dosage_gram: float = Field(
        ...,
        alias="Dosage (gram)",
        ge=0.0,
        le=10.0,
        description="Medication dosage in grams",
        json_schema_extra={
            "unit": "grams",
            "default_on_missing": "median",
            "training_median": 0.75,
            "conversion_rules": {
                "mg_to_g": "divide by 1000",
                "mcg_to_g": "divide by 1000000"
            }
        }
    )

    # Binary features - Gender
    Gender_Male: Literal[0, 1] = Field(
        ...,
        description="1 if patient is male, 0 if female",
        json_schema_extra={
            "encoding": {
                "Male": 1,
                "Female": 0
            },
            "default_on_missing": 1
        }
    )

    # Binary features - Route
    Route_IV: Literal[0, 1] = Field(
        ...,
        description="1 if intravenous administration, 0 otherwise",
        json_schema_extra={
            "encoding": {
                "IV": 1,
                "intravenous": 1,
                "other": 0
            }
        }
    )

    Route_Oral: Literal[0, 1] = Field(
        ...,
        description="1 if oral administration, 0 otherwise",
        json_schema_extra={
            "encoding": {
                "oral": 1,
                "PO": 1,
                "other": 0
            },
            "note": "If both Route_IV=0 and Route_Oral=0, route is BD (twice daily) or unlisted"
        }
    )

    # Binary features - Frequency
    Frequency_OD: Literal[0, 1] = Field(
        ...,
        description="1 if once daily (OD), 0 otherwise",
        json_schema_extra={
            "encoding": {
                "OD": 1,
                "once daily": 1,
                "other": 0
            }
        }
    )

    Frequency_QID: Literal[0, 1] = Field(
        ...,
        description="1 if four times daily (QID), 0 otherwise",
        json_schema_extra={
            "encoding": {
                "QID": 1,
                "four times daily": 1,
                "other": 0
            }
        }
    )

    Frequency_TDS: Literal[0, 1] = Field(
        ...,
        description="1 if three times daily (TDS), 0 otherwise",
        json_schema_extra={
            "encoding": {
                "TDS": 1,
                "three times daily": 1,
                "thrice daily": 1,
                "other": 0
            },
            "note": "If all frequency flags=0, frequency is BD (twice daily)"
        }
    )

    # Validators
    @field_validator('Age')
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Validate age is within reasonable range"""
        if v < 0 or v > 120:
            raise ValueError(f"Age must be between 0 and 120, got {v}")
        if v < 1:
            raise ValueError(f"Age {v} is unusually low, please verify")
        return v

    @field_validator('Dosage_gram')
    @classmethod
    def validate_dosage(cls, v: float) -> float:
        """Validate dosage is within reasonable range"""
        if v < 0 or v > 10:
            raise ValueError(f"Dosage must be between 0 and 10 grams, got {v}")
        if v == 0:
            raise ValueError("Dosage cannot be zero")
        return v

    @model_validator(mode='after')
    def validate_route_encoding(self):
        """
        Validate route encoding follows business rules.
        Warning (not error) if both routes are 0 (represents BD/unlisted route).
        """
        if self.Route_IV == 1 and self.Route_Oral == 1:
            raise ValueError(
                "Invalid route encoding: Route_IV and Route_Oral cannot both be 1. "
                "Only one route should be active."
            )
        return self

    @model_validator(mode='after')
    def validate_frequency_encoding(self):
        """
        Validate frequency encoding follows business rules.
        Multiple frequencies being 1 is invalid.
        All being 0 is valid (represents BD - twice daily).
        """
        frequency_sum = self.Frequency_OD + self.Frequency_QID + self.Frequency_TDS

        if frequency_sum > 1:
            raise ValueError(
                f"Invalid frequency encoding: Only one frequency flag can be 1, "
                f"got OD={self.Frequency_OD}, QID={self.Frequency_QID}, TDS={self.Frequency_TDS}. "
                f"If all are 0, it represents BD (twice daily)."
            )
        return self

    def to_model_array(self) -> list[float]:
        """
        Convert to ordered list matching model's expected input.
        Column order: ['Age', 'Dosage (gram)', 'Gender_Male', 'Route_IV',
                       'Route_Oral', 'Frequency_OD', 'Frequency_QID', 'Frequency_TDS']
        """
        return [
            self.Age,
            self.Dosage_gram,
            float(self.Gender_Male),
            float(self.Route_IV),
            float(self.Route_Oral),
            float(self.Frequency_OD),
            float(self.Frequency_QID),
            float(self.Frequency_TDS)
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with correct column names"""
        return pd.DataFrame([{
            'Age': self.Age,
            'Dosage (gram)': self.Dosage_gram,
            'Gender_Male': self.Gender_Male,
            'Route_IV': self.Route_IV,
            'Route_Oral': self.Route_Oral,
            'Frequency_OD': self.Frequency_OD,
            'Frequency_QID': self.Frequency_QID,
            'Frequency_TDS': self.Frequency_TDS
        }])

    class Config:
        populate_by_name = True  # Allow using alias "Dosage (gram)"
        json_schema_extra = {
            "example": {
                "Age": 60.0,
                "Dosage (gram)": 0.5,
                "Gender_Male": 1,
                "Route_IV": 0,
                "Route_Oral": 1,
                "Frequency_OD": 1,
                "Frequency_QID": 0,
                "Frequency_TDS": 0
            }
        }


# ============================================================================
# PREDICTION OUTPUT SCHEMA
# ============================================================================

class PredictionOutput(BaseModel):
    """Schema for model prediction output"""

    duration_category: Literal["<5 days", "5-10 days", ">10 days"] = Field(
        ...,
        description="Predicted treatment duration category"
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across all classes"
    )

    model_version: str = Field(
        default="1.0",
        description="Version of the model used for prediction"
    )

    inference_time_ms: Optional[float] = Field(
        None,
        description="Time taken for inference in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "duration_category": "<5 days",
                "probabilities": {
                    "<5 days": 0.72,
                    "5-10 days": 0.18,
                    ">10 days": 0.10
                },
                "model_version": "1.0",
                "inference_time_ms": 5.2
            }
        }
