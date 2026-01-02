"""
LLM Prompts for Multi-Case Study Extraction.

Handles documents containing multiple case studies.
"""

import json
from typing import List, Dict, Any


# ============================================================================
# MULTI-CASE STUDY SYSTEM PROMPT
# ============================================================================

MULTI_CASE_SYSTEM_PROMPT = """You are a medical data extraction specialist. Your task is to identify and extract clinical information from documents that may contain ONE or MULTIPLE case studies.

## CRITICAL RULES

1. **IDENTIFY ALL CASE STUDIES**:
   - A document may contain 1 case study (single patient) OR multiple case studies (multiple patients)
   - ALWAYS return a JSON array, even for a single case study (array with 1 element)
   - Each case study typically describes a different patient with distinct demographics
   - Look for case boundaries using these indicators:
     * Explicit markers: "Case #1", "Patient ID:", "Case Study 1", numbered sections
     * Demographic changes: Different ages, genders mentioned
     * Temporal markers: "Admission:", "Patient presented...", "New case:"
     * Section breaks: Horizontal lines, blank lines, paragraph breaks
     * Sequential descriptions: "Another patient...", "Second case...", "A 45-year-old..."
   - If you find multiple sets of demographics (different ages/genders), treat as separate cases
   - If demographic information appears only once, treat as a single case

2. **JSON OBJECT FORMAT WITH CASES ARRAY**:
   - ALWAYS return a JSON object with a "cases" array
   - Format: {"cases": [case1, case2, ...]}
   - For single case: {"cases": [case1]}
   - For multiple cases: {"cases": [case1, case2, case3, ...]}
   - CRITICAL: The root must be an object, not an array, to comply with OpenAI's JSON format requirements

3. **NO HALLUCINATION**:
   - Extract ONLY information explicitly present in the text
   - If a field is not mentioned for a case, set value to null
   - Never infer, assume, or estimate missing information
   - Never merge information from different patients

4. **EXACT EVIDENCE**:
   - For each field, include the exact text span that supports your extraction
   - Evidence must be a verbatim quote from the input text
   - Include context to identify which case study it belongs to

5. **CONFIDENCE SCORING**:
   - 1.0: Explicit and unambiguous
   - 0.8-0.9: Clear but requires minor interpretation
   - 0.6-0.7: Ambiguous or requires context
   - 0.4-0.5: Uncertain, multiple interpretations
   - 0.0-0.3: Guessing or no clear evidence
   - If value is null, confidence must be 0.0

6. **DOSAGE UNIT CONVERSION**:
   - Convert all dosages to grams
   - 1000 mg = 1 g, 1,000,000 mcg (µg) = 1 g
   - Preserve original in raw_value

7. **CASE STUDY METADATA**:
   - Include case_id (extracted or auto-assigned: case_1, case_2, etc.)
   - Include page_number or section if identifiable
   - Include confidence_overall for the entire case extraction

## OUTPUT SCHEMA

Return a JSON object with a "cases" array. Each case in the array has this structure:

```json
{
  "case_id": "case_1",  // Extracted ID or auto-assigned
  "page_number": 1,  // If identifiable, else null
  "confidence_overall": 0.85,  // Aggregate confidence for this case
  "age": {
    "value": 65.0,
    "raw_value": "65-year-old",
    "evidence": "Patient is a 65-year-old male",
    "confidence": 1.0
  },
  "dosage": {
    "value": 2.0,
    "raw_value": "2 grams",
    "evidence": "Started on IV ceftriaxone 2 grams once daily",
    "confidence": 1.0,
    "unit_conversion": "2 grams → 2.0g"
  },
  "gender": {
    "value": "male",
    "raw_value": "male",
    "evidence": "65-year-old male",
    "confidence": 1.0
  },
  "route": {
    "value": "IV",
    "raw_value": "IV",
    "evidence": "Started on IV ceftriaxone",
    "confidence": 1.0
  },
  "frequency": {
    "value": "OD",
    "raw_value": "once daily",
    "evidence": "ceftriaxone 2 grams once daily",
    "confidence": 1.0
  }
}
```

## FIELD DEFINITIONS

- **age**: Patient age in years (number)
- **dosage**: Antibiotic dosage in grams (number, converted from mg/mcg if needed)
- **gender**: "male" or "female" (lowercase string)
- **route**: Administration route: "IV" (intravenous), "oral" (by mouth), "other" (any other route like IM, SC, etc.)
- **frequency**: Dosing frequency: "OD" (once daily), "BD" (twice daily), "TDS" (three times daily), "QID" (four times daily), "other" (any other frequency)

## COMMON ABBREVIATIONS

- **Age**: yo, y/o, years old
- **Gender**: M, F, male, female
- **Route**: IV, PO, oral, intravenous
- **Frequency**: OD, BD, TDS, QID, q24h, q12h, q8h, q6h

## HANDLING EDGE CASES

1. **Single Case Study**: Return {"cases": [single_case_object]}
2. **Multiple Cases**: Return {"cases": [case1, case2, case3, ...]}
3. **Ambiguous Separation**: Use best judgment based on contextual clues
4. **Missing Patient Boundaries**: Look for demographic changes, admission dates, or explicit markers
5. **Shared Information**: If information spans multiple cases (e.g., hospital-wide protocol), include in each relevant case
6. **Incomplete Cases**: Include the case with available fields; set missing fields to null

## VALIDATION

- Ensure age is between 0-120 years
- Ensure dosage is positive and in grams
- Ensure gender is "male" or "female" (lowercase)
- Ensure route is one of: "IV", "oral", "other"
- Ensure frequency is one of: "OD", "BD", "TDS", "QID", "other"
- If validation fails or uncertain, set confidence low (< 0.5)

CRITICAL: Return ONLY the JSON object with "cases" array. No markdown, no explanations.

Example minimal response:
```json
{"cases": [{"case_id": "case_1", "page_number": null, "confidence_overall": 0.9, "age": {...}, "dosage": {...}, "gender": {...}, "route": {...}, "frequency": {...}}]}
```"""


# ============================================================================
# USER PROMPT TEMPLATE
# ============================================================================

MULTI_CASE_USER_PROMPT = """Extract all case studies from the following clinical document:

{document_text}

Return a JSON object with a "cases" array containing all case studies. Format: {{"cases": [case1, case2, ...]}}
If only one case exists, return: {{"cases": [single_case]}}"""


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

MULTI_CASE_FEW_SHOT_EXAMPLES = [
    # Example 1: Single case study
    {
        "input": """A 52-year-old female patient was admitted to the emergency department with severe urinary tract infection. She presented with fever, dysuria, and flank pain. Blood cultures were positive for E. coli. The patient was started on IV ciprofloxacin 400mg twice daily. She has a history of penicillin allergy.""",
        "output": {
            "cases": [
            {
                "case_id": "case_1",
                "page_number": None,
                "confidence_overall": 0.93,
                "age": {
                    "value": 52.0,
                    "raw_value": "52-year-old",
                    "evidence": "A 52-year-old female patient",
                    "confidence": 1.0
                },
                "dosage": {
                    "value": 0.4,
                    "raw_value": "400mg",
                    "evidence": "IV ciprofloxacin 400mg twice daily",
                    "confidence": 1.0,
                    "unit_conversion": "400mg → 0.4g"
                },
                "gender": {
                    "value": "female",
                    "raw_value": "female",
                    "evidence": "52-year-old female patient",
                    "confidence": 1.0
                },
                "route": {
                    "value": "IV",
                    "raw_value": "IV",
                    "evidence": "started on IV ciprofloxacin",
                    "confidence": 1.0
                },
                "frequency": {
                    "value": "BD",
                    "raw_value": "twice daily",
                    "evidence": "400mg twice daily",
                    "confidence": 1.0
                }
            }
            ]
        }
    },
    # Example 2: Multiple case studies
    {
        "input": """CLINICAL CASE STUDIES - PNEUMONIA WARD

Case #1: Patient is a 62-year-old male admitted with severe community-acquired pneumonia. Medical history includes diabetes and COPD. Started on IV ceftriaxone 2g once daily on admission.

Case #2: 45-year-old female presenting with hospital-acquired pneumonia. No significant comorbidities. Prescribed IV piperacillin-tazobactam 4.5 grams every 6 hours.

Case #3: Elderly patient (78 years old, male) with aspiration pneumonia. Given oral levofloxacin 750mg once daily due to difficulty with IV access.""",
        "output": {
            "cases": [
            {
                "case_id": "case_1",
                "page_number": None,
                "confidence_overall": 0.95,
                "age": {
                    "value": 62.0,
                    "raw_value": "62-year-old",
                    "evidence": "Patient is a 62-year-old male",
                    "confidence": 1.0
                },
                "dosage": {
                    "value": 2.0,
                    "raw_value": "2g",
                    "evidence": "Started on IV ceftriaxone 2g once daily",
                    "confidence": 1.0,
                    "unit_conversion": "2g → 2.0g"
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
                    "evidence": "Started on IV ceftriaxone",
                    "confidence": 1.0
                },
                "frequency": {
                    "value": "OD",
                    "raw_value": "once daily",
                    "evidence": "ceftriaxone 2g once daily",
                    "confidence": 1.0
                }
            },
            {
                "case_id": "case_2",
                "page_number": None,
                "confidence_overall": 0.92,
                "age": {
                    "value": 45.0,
                    "raw_value": "45-year-old",
                    "evidence": "45-year-old female",
                    "confidence": 1.0
                },
                "dosage": {
                    "value": 4.5,
                    "raw_value": "4.5 grams",
                    "evidence": "IV piperacillin-tazobactam 4.5 grams every 6 hours",
                    "confidence": 1.0,
                    "unit_conversion": "4.5 grams → 4.5g"
                },
                "gender": {
                    "value": "female",
                    "raw_value": "female",
                    "evidence": "45-year-old female",
                    "confidence": 1.0
                },
                "route": {
                    "value": "IV",
                    "raw_value": "IV",
                    "evidence": "IV piperacillin-tazobactam",
                    "confidence": 1.0
                },
                "frequency": {
                    "value": "QID",
                    "raw_value": "every 6 hours",
                    "evidence": "4.5 grams every 6 hours",
                    "confidence": 0.9
                }
            },
            {
                "case_id": "case_3",
                "page_number": None,
                "confidence_overall": 0.88,
                "age": {
                    "value": 78.0,
                    "raw_value": "78 years old",
                    "evidence": "Elderly patient (78 years old, male)",
                    "confidence": 1.0
                },
                "dosage": {
                    "value": 0.75,
                    "raw_value": "750mg",
                    "evidence": "oral levofloxacin 750mg once daily",
                    "confidence": 1.0,
                    "unit_conversion": "750mg → 0.75g"
                },
                "gender": {
                    "value": "male",
                    "raw_value": "male",
                    "evidence": "78 years old, male",
                    "confidence": 1.0
                },
                "route": {
                    "value": "oral",
                    "raw_value": "oral",
                    "evidence": "Given oral levofloxacin",
                    "confidence": 1.0
                },
                "frequency": {
                    "value": "OD",
                    "raw_value": "once daily",
                    "evidence": "levofloxacin 750mg once daily",
                    "confidence": 1.0
                }
            }
            ]
        }
    }
]


# ============================================================================
# PROMPT BUILDER
# ============================================================================

class MultiCasePromptBuilder:
    """Build prompts for multi-case study extraction"""

    @staticmethod
    def build_openai_messages(document_text: str) -> List[Dict[str, str]]:
        """
        Build OpenAI chat completion messages.

        Args:
            document_text: Full document text

        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": MULTI_CASE_SYSTEM_PROMPT}
        ]

        # Add few-shot example
        for example in MULTI_CASE_FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": MULTI_CASE_USER_PROMPT.format(document_text=example["input"])
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(example["output"], indent=2)
            })

        # Add actual request
        messages.append({
            "role": "user",
            "content": MULTI_CASE_USER_PROMPT.format(document_text=document_text)
        })

        return messages

    @staticmethod
    def build_gemini_prompt(document_text: str) -> str:
        """
        Build Gemini prompt.

        Args:
            document_text: Full document text

        Returns:
            Complete prompt string
        """
        # Build few-shot examples
        examples_text = "\n\n---\n\n".join([
            f"INPUT:\n{ex['input']}\n\nOUTPUT:\n{json.dumps(ex['output'], indent=2)}"
            for ex in MULTI_CASE_FEW_SHOT_EXAMPLES
        ])

        return f"""{MULTI_CASE_SYSTEM_PROMPT}

## EXAMPLES

{examples_text}

---

## YOUR TASK

{MULTI_CASE_USER_PROMPT.format(document_text=document_text)}"""
