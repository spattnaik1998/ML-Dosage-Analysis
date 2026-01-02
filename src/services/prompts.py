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
   - Evidence must be a verbatim quote from the input text (copy it exactly as it appears)
   - Evidence should include enough context to validate the extraction
   - If multiple mentions exist, use the clearest/most explicit one
   - If no evidence exists, set evidence to null

4. **CONFIDENCE SCORING** (CRITICAL - Be precise):
   - 1.0: Explicit and completely unambiguous
     * Example: "Patient is 45 years old" → age: 45
     * Example: "prescribed ceftriaxone 1g IV" → dosage: 1.0g, route: IV
   - 0.9-0.95: Very clear with standard abbreviations
     * Example: "45yo M" → age: 45, gender: male
     * Example: "500mg PO BD" → dosage: 0.5g, route: oral, frequency: BD
   - 0.8-0.85: Clear but requires medical knowledge/interpretation
     * Example: "gentamicin 80 IV" → dosage: 0.08g (assuming mg)
     * Example: "q12h" → frequency: BD
   - 0.6-0.7: Ambiguous or requires significant inference
     * Example: "Elderly patient" → age: null (cannot infer specific age)
     * Example: "usual dose" → dosage: null (not specific)
   - 0.4-0.5: Multiple valid interpretations possible
     * Example: "80" without units → could be 80mg, 80g, unclear
   - 0.0-0.3: Pure guessing or no real evidence
     * Only use if making educated guess from very weak signals
   - 0.0: Field not mentioned at all (value must be null)

   **IMPORTANT**: Be conservative with confidence. When in doubt, use lower confidence.

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

**age**: Patient age in years (numeric value, float or null)
- Extract from: "X years old", "Xyo", "X-year-old", "age X", "aged X"
- Accept abbreviations: "yo", "y/o", "yr", "yrs"
- Do NOT extract from vague terms: "elderly", "young adult", "middle-aged"
  * If only vague term: set value=null, confidence=0.0
  * Can include vague term in raw_value with low confidence if useful
- Return null if not mentioned or not specific enough

**dosage**: Medication dose in GRAMS (numeric value converted, float or null)
- Extract from: "Xg", "Xmg", "Xmcg", "X grams", "X milligrams"
- ALWAYS convert to grams: 1000mg = 1g, 1,000,000mcg = 1g
- Handle missing units:
  * For common antibiotics (gentamicin, vancomycin), assume mg if units missing
  * Document assumption in unit_conversion field
  * Lower confidence to 0.7-0.8 for assumed units
- If multiple doses mentioned (e.g., "500mg day 1, 250mg days 2-5"):
  * Extract the INITIAL/PRIMARY dose (500mg in this case)
  * Note in raw_value if helpful
- Return null if dosage not mentioned or too ambiguous

**gender**: Patient biological sex (string or null)
- Values: "male" or "female" ONLY (lowercase)
- Extract from:
  * Explicit: "male", "female", "man", "woman", "boy", "girl"
  * Abbreviations: "M", "F", "m", "f"
  * Pronouns: "he/him/his" → male, "she/her" → female
  * Titles: "Mr." → male, "Mrs./Ms./Miss" → female
- If both genders mentioned (e.g., in multi-patient text): Return null
- Return null if not mentioned or ambiguous

**route**: Administration route (string or null)
- Values: "IV", "oral", or "other" ONLY
- "IV" includes: IV, intravenous, intravenously, i.v.
- "oral" includes: oral, PO, by mouth, orally, per os, p.o.
- "other" includes: IM (intramuscular), SC/SQ (subcutaneous), topical, rectal, etc.
  * For "other", include actual route in raw_value (e.g., raw_value: "IM")
- Return null if not mentioned

**frequency**: Dosing frequency (string or null)
- Values: "OD", "BD", "TDS", "QID", or "other" ONLY
- "OD" (once daily): "once daily", "once a day", "once per day", "OD", "QD", "q24h", "daily"
- "BD" (twice daily): "twice daily", "twice a day", "BD", "BID", "q12h"
- "TDS" (three times daily): "three times daily", "TDS", "TID", "q8h"
- "QID" (four times daily): "four times daily", "QID", "q6h"
- "other": any other frequency (q4h, q48h, PRN, etc.)
  * For "other", include actual frequency in raw_value
- Return null if not mentioned

## COMMON EDGE CASES AND HOW TO HANDLE THEM

1. **Vague Age Terms** ("elderly", "young", "middle-aged"):
   - Set value=null, confidence=0.0
   - Can mention in raw_value but don't try to convert to number

2. **Missing Dosage Units** ("gentamicin 80", "vancomycin 1"):
   - For common antibiotics: assume mg (document in unit_conversion)
   - Reduce confidence to 0.7-0.8
   - If completely ambiguous: set null

3. **Multiple Doses** ("500mg x 3 days, then 250mg x 4 days"):
   - Extract INITIAL dose (500mg)
   - Confidence remains high if initial dose is clear

4. **Ambiguous Text** ("patient given medication"):
   - Set all unspecified fields to null
   - Don't hallucinate values

5. **Medical Abbreviations** (Rx, Pt, w/, CAP, UTI):
   - Rx = prescribed/treatment
   - Pt = patient
   - w/ = with
   - These are contextual clues, use them but don't extract as values

6. **Time-based Dosing** ("q12h", "q8h", "q6h"):
   - q12h → BD (twice daily)
   - q8h → TDS (three times daily)
   - q6h → QID (four times daily)
   - q24h → OD (once daily)

## DETERMINISTIC OUTPUT

- Process text sequentially
- Extract information in the order: age → dosage → gender → route → frequency
- Do not randomize or vary extraction logic
- Be consistent in interpretation across similar phrasings
- Same input should always produce same output"""


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
                "confidence": 0.8,
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
    },
    # Example 6: Very Sparse Information
    {
        "input": """Patient presented with respiratory infection. Treatment initiated per hospital protocol.""",
        "output": {
            "age": {
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0
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
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0
            },
            "frequency": {
                "value": None,
                "raw_value": None,
                "evidence": None,
                "confidence": 0.0
            }
        }
    },
    # Example 7: Complex Medical Abbreviations
    {
        "input": """Pt: 35F w/ CAP. Rx: azithromycin 500mg PO OD day 1, then 250mg PO OD days 2-5. Vitals stable, O2 sat 96% RA.""",
        "output": {
            "age": {
                "value": 35.0,
                "raw_value": "35F",
                "evidence": "Pt: 35F w/ CAP",
                "confidence": 0.95
            },
            "dosage": {
                "value": 0.5,
                "raw_value": "500mg",
                "evidence": "azithromycin 500mg PO OD day 1",
                "confidence": 1.0,
                "unit_conversion": "500mg → 0.5g"
            },
            "gender": {
                "value": "female",
                "raw_value": "F",
                "evidence": "35F w/ CAP",
                "confidence": 1.0
            },
            "route": {
                "value": "oral",
                "raw_value": "PO",
                "evidence": "500mg PO OD",
                "confidence": 1.0
            },
            "frequency": {
                "value": "OD",
                "raw_value": "OD",
                "evidence": "500mg PO OD day 1",
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
