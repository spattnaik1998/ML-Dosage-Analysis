# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-01

### 🎉 Added
- **Multi-Case Study Detection**: System now correctly detects and processes multiple case studies from single documents
- **Enhanced Prompt Examples**: Added 2 new few-shot examples (sparse information, complex medical abbreviations)
- **Edge Case Handling**: Added comprehensive section for handling common edge cases (vague ages, missing units, multiple doses)
- **Model Configuration**: Added `OPENAI_MODEL` environment variable for easy model selection
- **Debug Logging**: Added detailed logging for multi-case detection (shows number of cases detected)
- **Increased Token Limit**: Raised max_tokens to 4000 for multi-case extraction

### 🔄 Changed
- **Multi-Case JSON Format**: Changed from array format `[case1, case2]` to object format `{"cases": [case1, case2]}` to comply with OpenAI's JSON object constraints
- **Prompt Structure**: Updated all multi-case prompts and examples to use new format
- **Confidence Scoring**: Enhanced confidence scoring guidelines with 6 detailed tiers and specific examples
- **Field Definitions**: Expanded field definitions with detailed extraction rules and edge case handling
- **Common Abbreviations**: Added medical abbreviation reference guide and time-based dosing conversions

### 🐛 Fixed
- **Critical Bug**: Fixed multi-case detection issue where only 1 case was detected even when document contained multiple cases
  - Root cause: `response_format={"type": "json_object"}` forced single object response
  - Solution: Updated prompt to return `{"cases": [...]}` format
  - Updated parser to extract `cases` array from response object
- **Route Field Error**: Fixed critical error in multi-case prompt where "bd" (twice daily) was incorrectly listed as a route instead of a frequency
- **Schema Inconsistency**: Removed `unit_conversion` field from non-dosage fields in multi-case prompt schema
- **Value Format Inconsistency**: Standardized route and frequency values to use consistent casing (IV, OD, BD, TDS, QID)
- **Single Case Handling**: Clarified that multi-case prompt handles both single and multiple cases

### 📝 Documentation
- **Comprehensive README**: Completely rewrote README.md with modern formatting, detailed usage instructions, and troubleshooting guide
- **Enhanced .env.example**: Added detailed comments and links to API key registration pages
- **CHANGELOG.md**: Created this changelog to track version history
- **CONTRIBUTING.md**: Added contribution guidelines for the project

### 🔧 Technical Improvements
- Better error messages with specific context
- Improved case boundary detection with 8 different indicators
- Enhanced validation with fallback handling for legacy formats
- More robust JSON parsing with multiple fallback strategies

---

## [1.0.0] - 2024-12-31

### Initial Release

#### Features
- Decision Tree classifier for treatment duration prediction (77.84% accuracy)
- LLM-based feature extraction using OpenAI GPT-4o and Google Gemini
- FastAPI backend with comprehensive validation
- React + TypeScript frontend with drag-and-drop file upload
- Support for PDF, DOCX, and TXT file parsing
- Hybrid extraction (LLM + regex fallback)
- Feature normalization pipeline
- Confidence scoring and validation
- Real-time prediction with probability scores

#### Components
- **Backend (Python)**:
  - FastAPI REST API
  - LLM extraction service (OpenAI/Gemini)
  - Fallback regex-based extractor
  - Response validator with auto-repair
  - Feature normalizer
  - ML inference engine
  - Document parser (PDF/DOCX/TXT)

- **Frontend (React)**:
  - File upload interface
  - Results visualization
  - Validation message display
  - Responsive design

- **ML Model**:
  - Decision Tree Classifier
  - 8 input features
  - 3 output classes
  - GridSearchCV optimization

---

## Version Comparison

### What's New in 2.0.0 vs 1.0.0

| Feature | 1.0.0 | 2.0.0 |
|---------|-------|-------|
| Multi-case detection | ❌ Broken | ✅ Fixed |
| Single case extraction | ✅ Working | ✅ Enhanced |
| Prompt examples | 5 examples | 7 examples |
| Edge case handling | Basic | Comprehensive |
| Model configuration | Hardcoded | Environment variable |
| Debug logging | Minimal | Detailed |
| Documentation | Basic | Comprehensive |
| Confidence scoring | 4 levels | 6 detailed tiers |

---

## Migration Guide

### Upgrading from 1.0.0 to 2.0.0

#### For API Users
No breaking changes for API endpoints. Response format remains the same. Multi-case detection now works correctly!

#### For Developers
1. **Update `.env` file**:
   ```bash
   # Add this line to specify model
   OPENAI_MODEL=gpt-4o
   ```

2. **Restart server** to load new prompts:
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

3. **Test multi-case detection** with documents containing multiple cases

#### Breaking Changes
None! All changes are backwards compatible.

---

## Future Releases

### Planned for 2.1.0
- [ ] Support for RTF and HTML documents
- [ ] Batch processing API endpoint
- [ ] Model performance monitoring dashboard
- [ ] Custom confidence threshold configuration
- [ ] Export results to CSV/JSON

### Planned for 3.0.0
- [ ] Real-time collaborative editing
- [ ] Custom model training interface
- [ ] EHR system integration (HL7, FHIR)
- [ ] Multi-language support
- [ ] Mobile application

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this changelog.

---

[2.0.0]: https://github.com/yourusername/Hospital_Details_Analysis/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yourusername/Hospital_Details_Analysis/releases/tag/v1.0.0
