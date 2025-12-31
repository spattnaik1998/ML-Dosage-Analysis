/**
 * Type definitions for API requests and responses
 */

export interface ExtractedFeatureDetail {
  value: any;
  raw_value: string | null;
  evidence: string | null;
  confidence: number;
  unit_conversion?: string | null;
}

export interface ValidationMessage {
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  field: string;
  message: string;
  repaired: boolean;
}

export interface PredictionResponse {
  // Final prediction
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;

  // Intermediate steps
  extracted_features: Record<string, ExtractedFeatureDetail>;
  normalized_features: Record<string, number>;
  validation_messages: ValidationMessage[];

  // Flags
  requires_human_review: boolean;
  extraction_method: string;

  // Metadata
  inference_time_ms: number;
  model_version: string;
}

export interface PredictionRequest {
  case_study_text: string;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
  stage: string;
}
