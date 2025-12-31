"""
LLM Extractor Service for clinical feature extraction.

This module provides the main service for extracting structured clinical features
from unstructured text using LLM APIs (OpenAI GPT or Google Gemini).
"""

import json
import time
from typing import Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.schema import RawFeatures, ExtractedField
from src.services.prompts import PromptBuilder


class ExtractionError(Exception):
    """Base exception for extraction errors"""
    pass


class LLMAPIError(ExtractionError):
    """Exception for LLM API errors"""
    pass


class ValidationError(ExtractionError):
    """Exception for extraction validation errors"""
    pass


class LLMExtractor:
    """
    Service for extracting clinical features using LLM APIs.
    Supports OpenAI and Google Gemini with automatic retry and fallback.
    """

    def __init__(
        self,
        primary_provider: Literal["openai", "gemini"] = "openai",
        fallback_provider: Optional[Literal["openai", "gemini"]] = "gemini",
        openai_model: str = "gpt-4o",
        gemini_model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.0,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None
    ):
        """
        Initialize LLM extractor.

        Args:
            primary_provider: Primary LLM provider to use
            fallback_provider: Fallback provider if primary fails
            openai_model: OpenAI model name
            gemini_model: Gemini model name
            temperature: Sampling temperature (0.0 for deterministic)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            google_api_key: Google API key (or set GOOGLE_API_KEY env var)
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.temperature = temperature

        # Initialize API clients
        self._openai_client = None
        self._gemini_model = None

        if primary_provider == "openai" or fallback_provider == "openai":
            self._init_openai(openai_api_key)

        if primary_provider == "gemini" or fallback_provider == "gemini":
            self._init_gemini(google_api_key)

    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize OpenAI client: {e}")

    def _init_gemini(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name=self.gemini_model,
                generation_config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json"
                }
            )
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize Gemini client: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMAPIError, ConnectionError))
    )
    def _extract_with_openai(self, case_study_text: str) -> dict:
        """
        Extract features using OpenAI API with retry logic.

        Args:
            case_study_text: Clinical text to extract from

        Returns:
            Raw extraction dictionary

        Raises:
            LLMAPIError: If API call fails after retries
        """
        if not self._openai_client:
            raise LLMAPIError("OpenAI client not initialized")

        try:
            messages = PromptBuilder.build_messages_openai(case_study_text)

            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=1000,
                seed=42  # For reproducibility
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError as e:
            raise LLMAPIError(f"Failed to parse OpenAI response as JSON: {e}")
        except Exception as e:
            raise LLMAPIError(f"OpenAI API error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMAPIError, ConnectionError))
    )
    def _extract_with_gemini(self, case_study_text: str) -> dict:
        """
        Extract features using Google Gemini API with retry logic.

        Args:
            case_study_text: Clinical text to extract from

        Returns:
            Raw extraction dictionary

        Raises:
            LLMAPIError: If API call fails after retries
        """
        if not self._gemini_model:
            raise LLMAPIError("Gemini client not initialized")

        try:
            prompt = PromptBuilder.build_prompt_gemini(case_study_text)
            response = self._gemini_model.generate_content(prompt)

            return json.loads(response.text)

        except json.JSONDecodeError as e:
            raise LLMAPIError(f"Failed to parse Gemini response as JSON: {e}")
        except Exception as e:
            raise LLMAPIError(f"Gemini API error: {e}")

    def _parse_extraction_response(self, response_dict: dict) -> RawFeatures:
        """
        Parse LLM response dictionary into RawFeatures object.

        Args:
            response_dict: Dictionary from LLM API response

        Returns:
            RawFeatures object

        Raises:
            ValidationError: If response doesn't match expected schema
        """
        try:
            # Convert nested dicts to ExtractedField objects
            age = ExtractedField(**response_dict.get("age", {}))
            dosage = ExtractedField(**response_dict.get("dosage", {}))
            gender = ExtractedField(**response_dict.get("gender", {}))
            route = ExtractedField(**response_dict.get("route", {}))
            frequency = ExtractedField(**response_dict.get("frequency", {}))

            return RawFeatures(
                age=age,
                dosage=dosage,
                gender=gender,
                route=route,
                frequency=frequency
            )
        except Exception as e:
            raise ValidationError(f"Failed to parse extraction response: {e}")

    def extract(self, case_study_text: str) -> dict:
        """
        Extract clinical features from case study text.
        Automatically handles retry and fallback between providers.

        Args:
            case_study_text: Unstructured clinical text

        Returns:
            Dictionary containing:
            - raw_features: RawFeatures object
            - provider_used: Which provider was used
            - extraction_time_ms: Time taken for extraction
            - error: Error message if fallback was used

        Raises:
            ExtractionError: If both primary and fallback fail
        """
        start_time = time.time()
        error_msg = None

        # Try primary provider
        try:
            if self.primary_provider == "openai":
                response_dict = self._extract_with_openai(case_study_text)
            else:
                response_dict = self._extract_with_gemini(case_study_text)

            raw_features = self._parse_extraction_response(response_dict)
            extraction_time = (time.time() - start_time) * 1000

            return {
                "raw_features": raw_features,
                "provider_used": self.primary_provider,
                "extraction_time_ms": extraction_time,
                "error": None
            }

        except Exception as e:
            error_msg = f"Primary provider ({self.primary_provider}) failed: {str(e)}"
            print(f"WARNING: {error_msg}")

            # Try fallback provider if available
            if self.fallback_provider:
                try:
                    print(f"Attempting fallback to {self.fallback_provider}...")

                    if self.fallback_provider == "openai":
                        response_dict = self._extract_with_openai(case_study_text)
                    else:
                        response_dict = self._extract_with_gemini(case_study_text)

                    raw_features = self._parse_extraction_response(response_dict)
                    extraction_time = (time.time() - start_time) * 1000

                    return {
                        "raw_features": raw_features,
                        "provider_used": self.fallback_provider,
                        "extraction_time_ms": extraction_time,
                        "error": error_msg  # Include primary error
                    }

                except Exception as fallback_error:
                    raise ExtractionError(
                        f"Both providers failed. "
                        f"Primary: {error_msg}. "
                        f"Fallback: {str(fallback_error)}"
                    )
            else:
                raise ExtractionError(error_msg)

    def extract_batch(self, case_studies: list[str]) -> list[dict]:
        """
        Extract features from multiple case studies.

        Args:
            case_studies: List of case study texts

        Returns:
            List of extraction results (one per case study)
        """
        results = []
        for i, case_study in enumerate(case_studies):
            try:
                result = self.extract(case_study)
                result["case_study_index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "case_study_index": i,
                    "raw_features": None,
                    "provider_used": None,
                    "extraction_time_ms": 0,
                    "error": str(e)
                })
        return results


class CachedLLMExtractor(LLMExtractor):
    """
    LLM Extractor with caching support.
    Caches extraction results to avoid redundant API calls.
    """

    def __init__(self, cache_backend=None, **kwargs):
        """
        Initialize cached extractor.

        Args:
            cache_backend: Cache backend (e.g., Redis client)
            **kwargs: Arguments for parent LLMExtractor
        """
        super().__init__(**kwargs)
        self.cache = cache_backend or {}  # Default to dict cache

    def _get_cache_key(self, case_study_text: str) -> str:
        """Generate cache key from case study text"""
        import hashlib
        text_hash = hashlib.sha256(case_study_text.encode()).hexdigest()
        return f"llm_extraction:{text_hash}"

    def extract(self, case_study_text: str) -> dict:
        """
        Extract with caching support.

        Args:
            case_study_text: Clinical text to extract from

        Returns:
            Extraction result (from cache or fresh API call)
        """
        cache_key = self._get_cache_key(case_study_text)

        # Try to get from cache
        if isinstance(self.cache, dict):
            cached = self.cache.get(cache_key)
        else:
            # Assume Redis-like interface
            try:
                cached_str = self.cache.get(cache_key)
                cached = json.loads(cached_str) if cached_str else None
            except:
                cached = None

        if cached:
            # Reconstruct RawFeatures from cached dict
            cached["raw_features"] = self._parse_extraction_response(
                cached.get("raw_features_dict", {})
            )
            cached["from_cache"] = True
            return cached

        # Cache miss - perform extraction
        result = super().extract(case_study_text)

        # Store in cache
        cache_data = result.copy()
        cache_data["raw_features_dict"] = result["raw_features"].model_dump()
        cache_data["from_cache"] = False

        if isinstance(self.cache, dict):
            self.cache[cache_key] = cache_data
        else:
            # Assume Redis-like interface
            try:
                self.cache.setex(
                    cache_key,
                    86400,  # 24 hours TTL
                    json.dumps(cache_data, default=str)
                )
            except:
                pass  # Silently fail cache write

        result["from_cache"] = False
        return result
