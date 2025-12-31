"""
LLM Extraction Prompts for Clinical Feature Extraction.

This module contains the production-grade prompts used for extracting
structured clinical features from unstructured case study text.
"""

import json
from typing import Any


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a medical data extraction specialist. Your task is to extract specific clinical information from unstructured case study text and return it in a strict JSON format.

## CRITICAL RULES

1. **JSON ONLY**: Return ONLY valid JSON. No markdown, no explanations, no preamble.

2. **NO HALLUCINATION**:
   - Extract ONLY information explicitly present in the text
   - If a field is not mentioned, set value to null
   - Never infer, assume, or estimate missing information
   - Never use default values unless explicitly stated in the text

3. **EXACT EVIDENCE**:
   - For each field, include the exact text span from the input that supports your extraction
   - Evidence must be a verbatim quote from the input text
   - If no evidence exists, set evidence to null

4. **CONFIDENCE SCORING**:
   - 1.0: Explicit and unambiguous (e.g., "Patient is 45 years old")
   - 0.8-0.9: Clear but requires minor interpretation (e.g., "45yo male")
   - 0.6-0.7: Ambiguous or requires inference from context
   - 0.4-0.5: Highly uncertain, multiple interpretations possible
   - 0.0-0.3: Guessing or no clear evidence
   - If value is null, confidence must be 0.0

5. **DOSAGE UNIT CONVERSION**:
   - Convert all dosages to grams
   - 1000 mg = 1 g
   - 1,000,000 mcg (µg) = 1 g
   - Preserve the original string in raw_value
   - If units are ambiguous, extract the number but flag low confidence

6. **PRESERVE RAW VALUES**:
   - Always include the original extracted string in raw_value
   - This allows downstream validation of your conversion/normalization

## OUTPUT SCHEMA

Return JSON matching this exact structure:

{
  "age": {
    "value": <float or null>,
    "raw_value": <string or null>,
    "evidence": <string or null>,
    "confidence": <float 0.0-1.0>
  },
  "dosage": {
    "value": <float or null (in grams)>,
    "raw_value": <string or null>,
    "evidence": <string or null>,
    "confidence": <float 0.0-1.0>,
    "unit_conversion": <string or null (e.g., "500mg → 0.5g")>
  },
  "gender": {
    "value": <"male" or "female" or null>,
    "raw_value": <string or null>,
    "evidence": <string or null>,
    "confidence": <float 0.0-1.0>
  },
  "route": {
    "value": <"IV" or "oral" or "other" or null>,
    "raw_value": <string or null>,
    "evidence": <string or null>,
    "confidence": <float 0.0-1.0>
  },
  "frequency": {
    "value": <"OD" or "BD" or "TDS" or "QID" or "other" or null>,
    "raw_value": <string or null>,
    "evidence": <string or null>,
    "confidence": <float 0.0-1.0>
  }
}

## FIELD DEFINITIONS

**age**: Patient age in years (numeric)
- Extract from: "X years old", "Xyo", "age X", etc.
- Return null if not mentioned

**dosage**: Medication dose in GRAMS (converted)
- Extract from: "Xg", "Xmg", "X grams", dose amounts
- Convert to grams using conversion rules above
- Return null if not mentioned

**gender**: Patient biological sex
- Values: "male" or "female" only
- Extract from: "male", "female", "M", "F", gender pronouns
- Return null if not mentioned

**route**: Administration route
- Values: "IV" (intravenous), "oral" (by mouth), "other" (any other route)
- Extract from: "IV", "intravenous", "oral", "PO", "intramuscular", etc.
- Return null if not mentioned

**frequency**: Dosing frequency
- Values: "OD" (once daily), "BD" (twice daily), "TDS" (three times daily), "QID" (four times daily), "other"
- Extract from: "once daily", "twice daily", "OD", "BD", "TDS", "QID", "q12h", etc.
- Return null if not mentioned

## DETERMINISTIC OUTPUT

- Process text sequentially
- Extract information in the order: age → dosage → gender → route → frequency
- Do not randomize or vary extraction logic
- Be consistent in interpretation across similar phrasings"""


# ============================================================================
# USER PROMPT TEMPLATE
# ============================================================================

USER_PROMPT_TEMPLATE = """Extract clinical information from the following case study text.

Return ONLY valid JSON with the exact schema specified in your instructions. No additional text.

CASE STUDY TEXT:
\"\"\"
{case_study_text}
\"\"\"

JSON OUTPUT:"""


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

FEW_SHOT_EXAMPLES = [
    # Example 1: Complete Information
    {
        "input": """Patient is a 62-year-old male admitted with suspected pneumonia.
Prescribed ceftriaxone 1g IV twice daily for 7 days.""",
        "output": {
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
        }
    },
    # Example 2: Unit Conversion Required
    {
        "input": """45yo F presents with UTI. Started on ciprofloxacin 500mg PO q12h.""",
        "output": {
            "age": {
                "value": 45.0,
                "raw_value": "45yo",
                "evidence": "45yo F",
                "confidence": 0.95
            },
            "dosage": {
                "value": 0.5,
                "raw_value": "500mg",
                "evidence": "ciprofloxacin 500mg",
                "confidence": 1.0,
                "unit_conversion": "500mg → 0.5g"
            },
            "gender": {
                "value": "female",
                "raw_value": "F",
                "evidence": "45yo F",
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
                "raw_value": "q12h",
                "evidence": "PO q12h",
                "confidence": 0.95
            }
        }
    },
    # Example 3: Missing Information
    {
        "input": """Elderly patient with chest infection. Prescribed oral antibiotic,
dosage as per protocol.""",
        "output": {
            "age": {
                "value": None,
                "raw_value": "Elderly",
                "evidence": "Elderly patient",
                "confidence": 0.3
            },
            "dosage": {
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0,
                "unit_conversion": None
            },
            "gender": {
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0
            },
            "route": {
                "value": "oral",
                "raw_value": "oral",
                "evidence": "Prescribed oral antibiotic",
                "confidence": 1.0
            },
            "frequency": {
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0
            }
        }
    },
    # Example 4: Abbreviations and Medical Terminology
    {
        "input": """Pt: 28M with acute GE. Rx: metronidazole 750mg IV TDS x5d.
Vitals stable.""",
        "output": {
            "age": {
                "value": 28.0,
                "raw_value": "28M",
                "evidence": "28M with acute GE",
                "confidence": 0.95
            },
            "dosage": {
                "value": 0.75,
                "raw_value": "750mg",
                "evidence": "metronidazole 750mg",
                "confidence": 1.0,
                "unit_conversion": "750mg → 0.75g"
            },
            "gender": {
                "value": "male",
                "raw_value": "M",
                "evidence": "28M with acute GE",
                "confidence": 1.0
            },
            "route": {
                "value": "IV",
                "raw_value": "IV",
                "evidence": "750mg IV TDS",
                "confidence": 1.0
            },
            "frequency": {
                "value": "TDS",
                "raw_value": "TDS",
                "evidence": "IV TDS",
                "confidence": 1.0
            }
        }
    },
    # Example 5: Ambiguous Dosage Units
    {
        "input": """72 year old woman admitted for sepsis. Started on gentamicin 80
intravenously once per day.""",
        "output": {
            "age": {
                "value": 72.0,
                "raw_value": "72 year old",
                "evidence": "72 year old woman",
                "confidence": 1.0
            },
            "dosage": {
                "value": 0.08,
                "raw_value": "80",
                "evidence": "gentamicin 80",
                "confidence": 0.7,
                "unit_conversion": "80mg (assumed) → 0.08g"
            },
            "gender": {
                "value": "female",
                "raw_value": "woman",
                "evidence": "72 year old woman",
                "confidence": 1.0
            },
            "route": {
                "value": "IV",
                "raw_value": "intravenously",
                "evidence": "80 intravenously",
                "confidence": 1.0
            },
            "frequency": {
                "value": "OD",
                "raw_value": "once per day",
                "evidence": "once per day",
                "confidence": 1.0
            }
        }
    }
]


# ============================================================================
# PROMPT BUILDER CLASS
# ============================================================================

class PromptBuilder:
    """
    Builds prompts for different LLM providers.
    Handles few-shot example formatting and message construction.
    """

    @staticmethod
    def build_messages_openai(case_study_text: str, include_examples: bool = True) -> list[dict[str, str]]:
        """
        Build messages array for OpenAI Chat Completions API.

        Args:
            case_study_text: The clinical text to extract from
            include_examples: Whether to include few-shot examples

        Returns:
            List of message dictionaries for OpenAI API
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add few-shot examples
        if include_examples:
            for example in FEW_SHOT_EXAMPLES:
                messages.append({
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        case_study_text=example["input"]
                    )
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(example["output"], indent=2)
                })

        # Add actual query
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                case_study_text=case_study_text
            )
        })

        return messages

    @staticmethod
    def build_prompt_gemini(case_study_text: str, include_examples: bool = True) -> str:
        """
        Build single prompt for Google Gemini API.

        Args:
            case_study_text: The clinical text to extract from
            include_examples: Whether to include few-shot examples

        Returns:
            Complete prompt string for Gemini API
        """
        prompt_parts = [
            "# SYSTEM INSTRUCTIONS",
            SYSTEM_PROMPT,
            ""
        ]

        # Add few-shot examples
        if include_examples:
            prompt_parts.append("# FEW-SHOT EXAMPLES")
            prompt_parts.append("")

            for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
                prompt_parts.append(f"## Example {i}")
                prompt_parts.append("")
                prompt_parts.append("INPUT:")
                prompt_parts.append(example["input"])
                prompt_parts.append("")
                prompt_parts.append("OUTPUT:")
                prompt_parts.append(json.dumps(example["output"], indent=2))
                prompt_parts.append("")

        # Add actual query
        prompt_parts.append("# YOUR TASK")
        prompt_parts.append("")
        prompt_parts.append(USER_PROMPT_TEMPLATE.format(
            case_study_text=case_study_text
        ))

        return "\n".join(prompt_parts)

    @staticmethod
    def get_example_by_index(index: int) -> dict[str, Any]:
        """Get a specific few-shot example by index"""
        if 0 <= index < len(FEW_SHOT_EXAMPLES):
            return FEW_SHOT_EXAMPLES[index]
        raise IndexError(f"Example index {index} out of range [0, {len(FEW_SHOT_EXAMPLES)-1}]")

    @staticmethod
    def get_all_examples() -> list[dict[str, Any]]:
        """Get all few-shot examples"""
        return FEW_SHOT_EXAMPLES.copy()
