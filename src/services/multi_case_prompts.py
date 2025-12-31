"""
LLM Prompts for Multi-Case Study Extraction.

Handles documents containing multiple case studies.
"""

import json
from typing import List, Dict, Any


# ============================================================================
# MULTI-CASE STUDY SYSTEM PROMPT
# ============================================================================

MULTI_CASE_SYSTEM_PROMPT = """You are a medical data extraction specialist. Your task is to identify and extract clinical information from documents that may contain MULTIPLE case studies.

## CRITICAL RULES

1. **IDENTIFY ALL CASE STUDIES**:
   - A single document may contain 1 or more independent case studies
   - Each case study typically describes a different patient
   - Look for indicators: "Patient is...", "Case #", "Patient ID", "Admission:", section breaks
   - Return a JSON array with one object per case study

2. **JSON ARRAY FORMAT**: Return a JSON array of case studies, even if only one exists

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

Return a JSON array where each element is:

```json
{
  "case_id": "case_1",  // Extracted ID or auto-assigned
  "page_number": 1,  // If identifiable, else null
  "confidence_overall": 0.85,  // Aggregate confidence for this case
  "age": {
    "value": 65.0,
    "raw_value": "65-year-old",
    "evidence": "Patient is a 65-year-old male",
    "confidence": 1.0,
    "unit_conversion": null
  },
  "dosage": {
    "value": 2.0,
    "raw_value": "2 grams",
    "evidence": "Started on IV ceftriaxone 2 grams once daily",
    "confidence": 1.0,
    "unit_conversion": "2 grams -> 2.0 g"
  },
  "gender": {
    "value": "male",
    "raw_value": "male",
    "evidence": "65-year-old male",
    "confidence": 1.0,
    "unit_conversion": null
  },
  "route": {
    "value": "iv",
    "raw_value": "IV",
    "evidence": "Started on IV ceftriaxone",
    "confidence": 1.0,
    "unit_conversion": null
  },
  "frequency": {
    "value": "od",
    "raw_value": "once daily",
    "evidence": "ceftriaxone 2 grams once daily",
    "confidence": 1.0,
    "unit_conversion": null
  }
}
```

## FIELD DEFINITIONS

- **age**: Patient age in years (number)
- **dosage**: Antibiotic dosage in grams (number)
- **gender**: "male" or "female" (lowercase string)
- **route**: Administration route: "iv" (intravenous), "oral", "bd" (twice daily), or other (lowercase)
- **frequency**: Dosing frequency: "od" (once daily), "bd" (twice daily), "tds" (three times daily), "qid" (four times daily), or other (lowercase)

## COMMON ABBREVIATIONS

- **Age**: yo, y/o, years old
- **Gender**: M, F, male, female
- **Route**: IV, PO, oral, intravenous
- **Frequency**: OD, BD, TDS, QID, q24h, q12h, q8h, q6h

## HANDLING EDGE CASES

1. **Single Case Study**: Return array with one element
2. **Multiple Cases**: Return array with all identified cases
3. **Ambiguous Separation**: Use best judgment based on contextual clues
4. **Missing Patient Boundaries**: Look for demographic changes, admission dates, or explicit markers
5. **Shared Information**: If information spans multiple cases (e.g., hospital-wide protocol), include in each relevant case
6. **Incomplete Cases**: Include the case with available fields; set missing fields to null

## VALIDATION

- Ensure age is between 0-120
- Ensure dosage is positive
- Ensure gender is "male" or "female"
- Ensure route is one of: iv, oral, bd, other
- Ensure frequency is one of: od, bd, tds, qid, other
- If validation fails, set confidence low (< 0.5)

CRITICAL: Return ONLY the JSON array. No markdown, no explanations."""


# ============================================================================
# USER PROMPT TEMPLATE
# ============================================================================

MULTI_CASE_USER_PROMPT = """Extract all case studies from the following clinical document:

{document_text}

Return a JSON array of case studies. If only one case exists, return an array with one element."""


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

MULTI_CASE_FEW_SHOT_EXAMPLES = [
    {
        "input": """CLINICAL CASE STUDIES - PNEUMONIA WARD

Case #1: Patient is a 62-year-old male admitted with severe community-acquired pneumonia. Medical history includes diabetes and COPD. Started on IV ceftriaxone 2g once daily on admission.

Case #2: 45-year-old female presenting with hospital-acquired pneumonia. No significant comorbidities. Prescribed IV piperacillin-tazobactam 4.5 grams every 6 hours.

Case #3: Elderly patient (78 years old, male) with aspiration pneumonia. Given oral levofloxacin 750mg once daily due to difficulty with IV access.""",
        "output": [
            {
                "case_id": "case_1",
                "page_number": None,
                "confidence_overall": 0.95,
                "age": {
                    "value": 62.0,
                    "raw_value": "62-year-old",
                    "evidence": "Patient is a 62-year-old male",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "dosage": {
                    "value": 2.0,
                    "raw_value": "2g",
                    "evidence": "Started on IV ceftriaxone 2g once daily",
                    "confidence": 1.0,
                    "unit_conversion": "2g -> 2.0 g"
                },
                "gender": {
                    "value": "male",
                    "raw_value": "male",
                    "evidence": "62-year-old male",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "route": {
                    "value": "iv",
                    "raw_value": "IV",
                    "evidence": "Started on IV ceftriaxone",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "frequency": {
                    "value": "od",
                    "raw_value": "once daily",
                    "evidence": "ceftriaxone 2g once daily",
                    "confidence": 1.0,
                    "unit_conversion": None
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
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "dosage": {
                    "value": 4.5,
                    "raw_value": "4.5 grams",
                    "evidence": "IV piperacillin-tazobactam 4.5 grams every 6 hours",
                    "confidence": 1.0,
                    "unit_conversion": "4.5 grams -> 4.5 g"
                },
                "gender": {
                    "value": "female",
                    "raw_value": "female",
                    "evidence": "45-year-old female",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "route": {
                    "value": "iv",
                    "raw_value": "IV",
                    "evidence": "IV piperacillin-tazobactam",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "frequency": {
                    "value": "qid",
                    "raw_value": "every 6 hours",
                    "evidence": "4.5 grams every 6 hours",
                    "confidence": 0.9,
                    "unit_conversion": None
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
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "dosage": {
                    "value": 0.75,
                    "raw_value": "750mg",
                    "evidence": "oral levofloxacin 750mg once daily",
                    "confidence": 1.0,
                    "unit_conversion": "750mg -> 0.75 g"
                },
                "gender": {
                    "value": "male",
                    "raw_value": "male",
                    "evidence": "78 years old, male",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "route": {
                    "value": "oral",
                    "raw_value": "oral",
                    "evidence": "Given oral levofloxacin",
                    "confidence": 1.0,
                    "unit_conversion": None
                },
                "frequency": {
                    "value": "od",
                    "raw_value": "once daily",
                    "evidence": "levofloxacin 750mg once daily",
                    "confidence": 1.0,
                    "unit_conversion": None
                }
            }
        ]
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
