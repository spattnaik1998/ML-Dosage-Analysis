"""
Test cases and examples for LLM Response Validator.

Demonstrates validation and repair of various LLM response scenarios.
"""

import json
from src.services.llm_validator import (
    LLMResponseValidator,
    ConfidenceThresholds,
    ErrorSeverity,
    ErrorCategory
)


# ============================================================================
# TEST CASES
# ============================================================================

def test_valid_response():
    """Test Case 1: Valid, well-formed response"""
    print("\n" + "="*70)
    print("TEST 1: Valid Response")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {
            "value": 62.0,
            "raw_value": "62-year-old",
            "evidence": "Patient is a 62-year-old male",
            "confidence": 1.0
        },
        "dosage": {
            "value": 1.0,
            "raw_value": "1g",
            "evidence": "ceftriaxone 1g IV",
            "confidence": 1.0,
            "unit_conversion": "1g → 1.0g"
        },
        "gender": {
            "value": "male",
            "raw_value": "male",
            "evidence": "62-year-old male",
            "confidence": 1.0
        },
        "route": {
            "value": "IV",
            "raw_value": "IV",
            "evidence": "1g IV twice daily",
            "confidence": 1.0
        },
        "frequency": {
            "value": "BD",
            "raw_value": "twice daily",
            "evidence": "IV twice daily",
            "confidence": 1.0
        }
    })

    original_text = "Patient is a 62-year-old male admitted with pneumonia. Prescribed ceftriaxone 1g IV twice daily."

    result = validator.validate_and_repair(llm_response, original_text)

    print(f"✓ Valid: {result.is_valid}")
    print(f"✓ Confidence: {result.overall_confidence:.2f}")
    print(f"✓ Repairs: {result.repair_count}")
    print(f"✓ Human Review: {result.requires_human_review}")
    print(f"✓ Errors: {len(result.errors)}")

    return result


def test_malformed_json():
    """Test Case 2: Malformed JSON with markdown wrapper"""
    print("\n" + "="*70)
    print("TEST 2: Malformed JSON (Markdown Wrapper)")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = """
    Here's the extracted data:

    ```json
    {
        "age": {"value": 45.0, "raw_value": "45yo", "evidence": "45yo F", "confidence": 0.95},
        "dosage": {"value": 0.5, "raw_value": "500mg", "evidence": "500mg PO", "confidence": 1.0},
        "gender": {"value": "female", "raw_value": "F", "evidence": "45yo F", "confidence": 1.0},
        "route": {"value": "oral", "raw_value": "PO", "evidence": "500mg PO", "confidence": 1.0},
        "frequency": {"value": "BD", "raw_value": "q12h", "evidence": "PO q12h", "confidence": 0.95}
    }
    ```

    Hope this helps!
    """

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Valid: {result.is_valid}")
    print(f"✓ Repairs: {result.repair_count}")
    print(f"✓ Repair Strategy: {[e.repair_strategy for e in result.errors if e.repair_strategy]}")
    print(f"✓ Data extracted: {result.repaired_data is not None}")

    return result


def test_missing_fields():
    """Test Case 3: Missing required fields"""
    print("\n" + "="*70)
    print("TEST 3: Missing Fields")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {
            "value": 60.0,
            "raw_value": "60",
            "evidence": "60 years old",
            "confidence": 1.0
        },
        "gender": {
            "value": "male",
            "raw_value": "male",
            "evidence": "male patient",
            "confidence": 1.0
        }
        # Missing: dosage, route, frequency
    })

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Valid: {result.is_valid}")
    print(f"✓ Repairs: {result.repair_count}")
    print(f"✓ Missing Fields Detected: {len([e for e in result.errors if e.category == ErrorCategory.MISSING_FIELD])}")
    print(f"✓ Critical Errors: {len(result.critical_errors)}")
    print(f"✓ Human Review: {result.requires_human_review}")

    for error in result.errors:
        print(f"   - {error.field_name}: {error.message}")

    return result


def test_low_confidence():
    """Test Case 4: Low confidence scores"""
    print("\n" + "="*70)
    print("TEST 4: Low Confidence Scores")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {
            "value": None,
            "raw_value": "elderly",
            "evidence": "elderly patient",
            "confidence": 0.3  # Very low
        },
        "dosage": {
            "value": 0.5,
            "raw_value": "500",
            "evidence": "500",
            "confidence": 0.5  # Below critical threshold
        },
        "gender": {
            "value": "male",
            "raw_value": "M",
            "evidence": "M",
            "confidence": 0.9
        },
        "route": {
            "value": "oral",
            "raw_value": "PO",
            "evidence": "PO",
            "confidence": 0.8
        },
        "frequency": {
            "value": "BD",
            "raw_value": "BD",
            "evidence": "BD",
            "confidence": 0.7
        }
    })

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Overall Confidence: {result.overall_confidence:.2f}")
    print(f"✓ Human Review Required: {result.requires_human_review}")
    print(f"✓ Low Confidence Warnings: {len([e for e in result.errors if e.category == ErrorCategory.LOW_CONFIDENCE])}")
    print(f"✓ Null Critical Fields: {len([e for e in result.errors if e.category == ErrorCategory.NULL_CRITICAL_FIELD])}")

    return result


def test_wrong_unit_conversion():
    """Test Case 5: Incorrect unit conversion"""
    print("\n" + "="*70)
    print("TEST 5: Wrong Unit Conversion")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {"value": 55.0, "raw_value": "55", "evidence": "55 years", "confidence": 1.0},
        "dosage": {
            "value": 500.0,  # WRONG! Should be 0.5g (LLM forgot to convert mg to g)
            "raw_value": "500mg",
            "evidence": "500mg dose",
            "confidence": 1.0,
            "unit_conversion": "500mg → 500.0g"  # Incorrect
        },
        "gender": {"value": "female", "raw_value": "F", "evidence": "F", "confidence": 1.0},
        "route": {"value": "oral", "raw_value": "oral", "evidence": "oral", "confidence": 1.0},
        "frequency": {"value": "OD", "raw_value": "OD", "evidence": "OD", "confidence": 1.0}
    })

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Conversion Errors: {len([e for e in result.errors if e.category == ErrorCategory.UNIT_CONVERSION_ERROR])}")
    print(f"✓ Original Dosage: {500.0}g")
    print(f"✓ Repaired Dosage: {result.repaired_data['dosage']['value']}g")
    print(f"✓ Auto-Fixed: {result.repair_count}")

    return result


def test_invalid_evidence():
    """Test Case 6: Evidence not in original text (hallucination)"""
    print("\n" + "="*70)
    print("TEST 6: Invalid Evidence (Hallucination)")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {
            "value": 70.0,
            "raw_value": "70",
            "evidence": "Patient is 70 years old",  # NOT in original text
            "confidence": 1.0
        },
        "dosage": {"value": 1.0, "raw_value": "1g", "evidence": "1g", "confidence": 1.0},
        "gender": {"value": "male", "raw_value": "male", "evidence": "male", "confidence": 1.0},
        "route": {"value": "IV", "raw_value": "IV", "evidence": "IV", "confidence": 1.0},
        "frequency": {"value": "BD", "raw_value": "BD", "evidence": "BD", "confidence": 1.0}
    })

    original_text = "Elderly gentleman prescribed 1g IV BD"  # Age NOT mentioned

    result = validator.validate_and_repair(llm_response, original_text)

    print(f"✓ Evidence Mismatches: {len([e for e in result.errors if e.category == ErrorCategory.EVIDENCE_MISMATCH])}")
    print(f"✓ Critical Errors: {len(result.critical_errors)}")
    print(f"✓ Human Review: {result.requires_human_review}")

    for error in result.critical_errors:
        print(f"   - {error.field_name}: {error.message}")

    return result


def test_casing_normalization():
    """Test Case 7: Incorrect casing that needs normalization"""
    print("\n" + "="*70)
    print("TEST 7: Casing Normalization")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = json.dumps({
        "age": {"value": 50.0, "raw_value": "50", "evidence": "50yo", "confidence": 1.0},
        "dosage": {"value": 0.75, "raw_value": "750mg", "evidence": "750mg", "confidence": 1.0},
        "gender": {
            "value": "Male",  # Should be lowercase
            "raw_value": "Male",
            "evidence": "Male patient",
            "confidence": 1.0
        },
        "route": {
            "value": "ORAL",  # Should be lowercase
            "raw_value": "ORAL",
            "evidence": "ORAL",
            "confidence": 1.0
        },
        "frequency": {
            "value": "bd",  # Should be uppercase
            "raw_value": "twice daily",
            "evidence": "twice daily",
            "confidence": 1.0
        }
    })

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Repairs: {result.repair_count}")
    print(f"✓ Gender: {result.repaired_data['gender']['value']}")
    print(f"✓ Route: {result.repaired_data['route']['value']}")
    print(f"✓ Frequency: {result.repaired_data['frequency']['value']}")

    return result


def test_completely_broken():
    """Test Case 8: Completely malformed response"""
    print("\n" + "="*70)
    print("TEST 8: Completely Broken Response")
    print("="*70)

    validator = LLMResponseValidator()

    llm_response = "Sorry, I couldn't extract the information from that text."

    result = validator.validate_and_repair(llm_response)

    print(f"✓ Valid: {result.is_valid}")
    print(f"✓ Human Review: {result.requires_human_review}")
    print(f"✓ Critical Errors: {len(result.critical_errors)}")

    return result


# ============================================================================
# COMPREHENSIVE EXAMPLE
# ============================================================================

def comprehensive_example():
    """Comprehensive example showing full workflow"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EXAMPLE: Full Validation Workflow")
    print("="*70)

    validator = LLMResponseValidator(
        thresholds=ConfidenceThresholds(
            ACCEPTABLE=0.7,
            CRITICAL_FIELD=0.6,
            HUMAN_REVIEW=0.4,
            REJECT=0.2
        )
    )

    # Simulate LLM response with multiple issues
    llm_response = """
    ```json
    {
        "age": {
            "value": 68.0,
            "raw_value": "68-year-old",
            "evidence": "68-year-old female",
            "confidence": 1.0
        },
        "dosage": {
            "value": 2000.0,
            "raw_value": "2g",
            "evidence": "ceftriaxone 2g",
            "confidence": 0.85,
            "unit_conversion": "2g → 2000.0g"
        },
        "gender": {
            "value": "Female",
            "raw_value": "female",
            "evidence": "68-year-old female",
            "confidence": 1.0
        },
        "route": {
            "value": "intravenous",
            "raw_value": "IV",
            "evidence": "IV ceftriaxone",
            "confidence": 1.0
        }
    }
    ```
    """

    original_text = "Patient is a 68-year-old female presenting with pneumonia. Started on IV ceftriaxone 2g once daily."

    result = validator.validate_and_repair(llm_response, original_text)

    print("\n📊 VALIDATION SUMMARY")
    print(f"   Valid: {result.is_valid}")
    print(f"   Overall Confidence: {result.overall_confidence:.2f}")
    print(f"   Repairs Made: {result.repair_count}")
    print(f"   Requires Human Review: {result.requires_human_review}")

    print("\n🔍 DETECTED ISSUES")
    for i, error in enumerate(result.errors, 1):
        print(f"\n   Issue {i}:")
        print(f"      Field: {error.field_name}")
        print(f"      Category: {error.category.value}")
        print(f"      Severity: {error.severity.value}")
        print(f"      Message: {error.message}")
        if error.repair_strategy:
            print(f"      Repair: {error.repair_strategy.value}")
        if error.repaired_value is not None:
            print(f"      Fixed To: {error.repaired_value}")

    print("\n✅ REPAIRED DATA")
    print(json.dumps(result.repaired_data, indent=2))

    print("\n📋 NEXT STEPS")
    if result.requires_human_review:
        print("   ⚠ HUMAN REVIEW REQUIRED")
        print("   Reasons:")
        for error in result.critical_errors:
            print(f"      - {error.message}")
    else:
        print("   ✓ Safe to proceed with prediction")

    return result


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LLM RESPONSE VALIDATOR - TEST SUITE")
    print("="*70)

    tests = [
        test_valid_response,
        test_malformed_json,
        test_missing_fields,
        test_low_confidence,
        test_wrong_unit_conversion,
        test_invalid_evidence,
        test_casing_normalization,
        test_completely_broken,
        comprehensive_example
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, "PASS", result))
        except Exception as e:
            results.append((test_func.__name__, "FAIL", str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)

    for test_name, status, result in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\n{passed}/{total} tests passed")
