"""
Test cases for Rule-Based Fallback Extractor.

Demonstrates regex pattern matching and confidence scoring.
"""

from src.services.fallback_extractor import FallbackExtractor, HybridExtractor


def test_age_extraction():
    """Test age extraction with various formats"""
    print("\n" + "="*70)
    print("TEST: Age Extraction")
    print("="*70)

    extractor = FallbackExtractor()

    test_cases = [
        ("62-year-old male patient", 62.0, ">= 0.9"),
        ("Patient is 45yo female", 45.0, ">= 0.9"),
        ("Age: 70 years", 70.0, ">= 0.9"),
        ("28M with acute GE", 28.0, ">= 0.8"),
        ("Elderly patient, 85 years old", 85.0, ">= 0.9"),
        ("Age 52", 52.0, ">= 0.9"),
    ]

    for text, expected_age, expected_conf in test_cases:
        result = extractor._extract_age(text)
        symbol = "✓" if result.value == expected_age else "✗"

        print(f"\n{symbol} Text: '{text}'")
        print(f"   Age: {result.value} (expected {expected_age})")
        print(f"   Confidence: {result.confidence:.2f} {expected_conf}")
        print(f"   Evidence: '{result.evidence}'")


def test_dosage_extraction():
    """Test dosage extraction with unit conversion"""
    print("\n" + "="*70)
    print("TEST: Dosage Extraction & Unit Conversion")
    print("="*70)

    extractor = FallbackExtractor()

    test_cases = [
        ("ceftriaxone 1g IV", 1.0, "1g"),
        ("ciprofloxacin 500mg PO", 0.5, "500mg"),
        ("gentamicin 80 mg", 0.08, "80 mg"),
        ("dose: 750mg", 0.75, "750mg"),
        ("prescribed 2g daily", 2.0, "2g"),
        ("dosage 250mcg", 0.00025, "250mcg"),
    ]

    for text, expected_grams, expected_raw in test_cases:
        result = extractor._extract_dosage(text)
        symbol = "✓" if result.value and abs(result.value - expected_grams) < 0.001 else "✗"

        print(f"\n{symbol} Text: '{text}'")
        print(f"   Dosage: {result.value}g (expected {expected_grams}g)")
        print(f"   Raw: '{result.raw_value}'")
        print(f"   Conversion: {result.unit_conversion}")
        print(f"   Confidence: {result.confidence:.2f}")


def test_gender_extraction():
    """Test gender extraction"""
    print("\n" + "="*70)
    print("TEST: Gender Extraction")
    print("="*70)

    extractor = FallbackExtractor()

    test_cases = [
        ("male patient with UTI", "male"),
        ("45yo F presents with", "female"),
        ("Elderly gentleman", "male"),
        ("Woman admitted for sepsis", "female"),
        ("Mr. Smith was given", "male"),
        ("She is recovering well", "female"),
    ]

    for text, expected_gender in test_cases:
        result = extractor._extract_gender(text)
        symbol = "✓" if result.value == expected_gender else "✗"

        print(f"\n{symbol} Text: '{text}'")
        print(f"   Gender: {result.value} (expected {expected_gender})")
        print(f"   Raw: '{result.raw_value}'")
        print(f"   Confidence: {result.confidence:.2f}")


def test_route_extraction():
    """Test route extraction"""
    print("\n" + "="*70)
    print("TEST: Route Extraction")
    print("="*70)

    extractor = FallbackExtractor()

    test_cases = [
        ("1g IV twice daily", "IV"),
        ("500mg PO BD", "oral"),
        ("administered orally", "oral"),
        ("given intravenously", "IV"),
        ("IM injection", "other"),
        ("by mouth", "oral"),
    ]

    for text, expected_route in test_cases:
        result = extractor._extract_route(text)
        symbol = "✓" if result.value == expected_route else "✗"

        print(f"\n{symbol} Text: '{text}'")
        print(f"   Route: {result.value} (expected {expected_route})")
        print(f"   Raw: '{result.raw_value}'")
        print(f"   Confidence: {result.confidence:.2f}")


def test_frequency_extraction():
    """Test frequency extraction"""
    print("\n" + "="*70)
    print("TEST: Frequency Extraction")
    print("="*70)

    extractor = FallbackExtractor()

    test_cases = [
        ("500mg BD", "BD"),
        ("once daily", "OD"),
        ("three times daily", "TDS"),
        ("q12h", "BD"),
        ("every 8 hours", "TDS"),
        ("QID", "QID"),
        ("2 times daily", "BD"),
    ]

    for text, expected_freq in test_cases:
        result = extractor._extract_frequency(text)
        symbol = "✓" if result.value == expected_freq else "✗"

        print(f"\n{symbol} Text: '{text}'")
        print(f"   Frequency: {result.value} (expected {expected_freq})")
        print(f"   Raw: '{result.raw_value}'")
        print(f"   Confidence: {result.confidence:.2f}")


def test_complete_extraction():
    """Test extraction of complete case study"""
    print("\n" + "="*70)
    print("TEST: Complete Case Study Extraction")
    print("="*70)

    extractor = FallbackExtractor()

    case_studies = [
        """62-year-old male admitted with suspected pneumonia.
        Prescribed ceftriaxone 1g IV twice daily for 7 days.""",

        """45yo F presents with UTI. Started on ciprofloxacin 500mg PO q12h.""",

        """Elderly gentleman, age 72, with sepsis. Commenced meropenem 1g
        intravenously three times daily.""",

        """Patient is a 28M with acute gastroenteritis.
        Rx: metronidazole 750mg IV TDS.""",
    ]

    for i, case_study in enumerate(case_studies, 1):
        print(f"\n{'─'*70}")
        print(f"Case Study {i}")
        print(f"{'─'*70}")
        print(f"Text: {case_study.strip()}")

        result = extractor.extract(case_study)

        print(f"\n📊 Extracted Features:")
        print(f"   Age: {result.age.value} (conf: {result.age.confidence:.2f})")
        print(f"   Dosage: {result.dosage.value}g (conf: {result.dosage.confidence:.2f})")
        print(f"   Gender: {result.gender.value} (conf: {result.gender.confidence:.2f})")
        print(f"   Route: {result.route.value} (conf: {result.route.confidence:.2f})")
        print(f"   Frequency: {result.frequency.value} (conf: {result.frequency.confidence:.2f})")


def test_edge_cases():
    """Test edge cases and missing data"""
    print("\n" + "="*70)
    print("TEST: Edge Cases")
    print("="*70)

    extractor = FallbackExtractor()

    edge_cases = [
        ("Patient prescribed medication", "No specific data"),
        ("Treatment ongoing", "Minimal information"),
        ("Follow-up scheduled", "No clinical details"),
        ("85 years, female, oral antibiotics", "Missing dosage/frequency"),
    ]

    for text, description in edge_cases:
        print(f"\n{'─'*70}")
        print(f"Case: {description}")
        print(f"Text: '{text}'")

        result = extractor.extract(text)

        fields_extracted = sum([
            result.age.value is not None,
            result.dosage.value is not None,
            result.gender.value is not None,
            result.route.value is not None,
            result.frequency.value is not None,
        ])

        print(f"\n   Fields extracted: {fields_extracted}/5")
        if result.age.value:
            print(f"   ✓ Age: {result.age.value}")
        if result.dosage.value:
            print(f"   ✓ Dosage: {result.dosage.value}g")
        if result.gender.value:
            print(f"   ✓ Gender: {result.gender.value}")
        if result.route.value:
            print(f"   ✓ Route: {result.route.value}")
        if result.frequency.value:
            print(f"   ✓ Frequency: {result.frequency.value}")


def test_confidence_scoring():
    """Test confidence scoring heuristics"""
    print("\n" + "="*70)
    print("TEST: Confidence Scoring Heuristics")
    print("="*70)

    extractor = FallbackExtractor()

    # Test confidence variations for age
    age_tests = [
        ("Patient is 62 years old", "Explicit, high confidence"),
        ("62yo male", "Common abbreviation, high confidence"),
        ("Age: 62", "Direct statement, high confidence"),
        ("62M", "Age with gender, medium confidence"),
        ("Elderly patient, 85", "Contextual, medium confidence"),
    ]

    print("\nAge Confidence Variations:")
    for text, description in age_tests:
        result = extractor._extract_age(text)
        print(f"   {result.confidence:.2f} - {description}")
        print(f"           Text: '{text}'")

    # Test confidence for dosage
    dosage_tests = [
        ("ceftriaxone 1g IV", "Drug + dose + route, very high"),
        ("dose: 500mg", "Explicit label, very high"),
        ("prescribed 750mg", "Action + dose, high"),
        ("500mg", "Standalone with unit, medium"),
    ]

    print("\nDosage Confidence Variations:")
    for text, description in dosage_tests:
        result = extractor._extract_dosage(text)
        print(f"   {result.confidence:.2f} - {description}")
        print(f"           Text: '{text}'")


def test_ambiguous_cases():
    """Test handling of ambiguous extractions"""
    print("\n" + "="*70)
    print("TEST: Ambiguous Cases")
    print("="*70)

    extractor = FallbackExtractor()

    ambiguous_cases = [
        ("Patient received 500 units of medication", "Units vs mg/g"),
        ("Given 2 doses today", "Count vs dosage"),
        ("125 was the reading", "Number without context"),
    ]

    for text, issue in ambiguous_cases:
        print(f"\n{'─'*70}")
        print(f"Issue: {issue}")
        print(f"Text: '{text}'")

        result = extractor.extract(text)

        print(f"\n   Dosage: {result.dosage.value} (conf: {result.dosage.confidence:.2f})")
        print(f"   Evidence: '{result.dosage.evidence}'")


def demo_hybrid_extraction():
    """Demonstrate hybrid LLM + Fallback extraction"""
    print("\n" + "="*70)
    print("DEMO: Hybrid Extraction (Concept)")
    print("="*70)

    print("""
    The HybridExtractor combines LLM and regex-based extraction:

    1. Try LLM extraction first
    2. Check if LLM output is valid and high-confidence
    3. If LLM fails or is low-confidence:
       - Use FallbackExtractor for missing/low-confidence fields
       - Merge results, preferring higher confidence values
    4. Return merged features with metadata

    Example workflow:

    LLM Output:
        age: 62 (confidence: 0.95) ✓
        dosage: None (confidence: 0.0) ✗
        gender: male (confidence: 1.0) ✓
        route: IV (confidence: 0.85) ✓
        frequency: None (confidence: 0.0) ✗

    Fallback Output:
        age: 62 (confidence: 0.90)
        dosage: 1.0g (confidence: 0.95) ✓ USED
        gender: male (confidence: 0.85)
        route: IV (confidence: 0.95)
        frequency: BD (confidence: 0.95) ✓ USED

    Merged Result:
        age: 62 (from LLM, higher confidence)
        dosage: 1.0g (from Fallback, LLM had null)
        gender: male (from LLM, higher confidence)
        route: IV (from Fallback, higher confidence)
        frequency: BD (from Fallback, LLM had null)

    Method: "hybrid" (LLM + Fallback)
    """)


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FALLBACK EXTRACTOR - TEST SUITE")
    print("="*70)

    tests = [
        test_age_extraction,
        test_dosage_extraction,
        test_gender_extraction,
        test_route_extraction,
        test_frequency_extraction,
        test_complete_extraction,
        test_edge_cases,
        test_confidence_scoring,
        test_ambiguous_cases,
        demo_hybrid_extraction,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✓ TEST SUITE COMPLETED")
    print("="*70)
