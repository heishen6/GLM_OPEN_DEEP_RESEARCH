# Changelog

All notable changes to GLM_OPEN_DEEP_RESEARCH will be documented in this file.

## [1.0.0] - 2024-01-07

### Added
- 🎉 Initial release with Zhipu GLM-4.5 support
- ✨ Support for GLM-4.5 and GLM-4.5-Air models
- 🔧 Custom base URL configuration for API endpoints
- 🛡️ Error handling for structured output compatibility issues
- 📝 Comprehensive documentation in Chinese and English
- 🧪 Test script for GLM model verification (`test_glm.py`)
- 🔑 Environment variable support for Zhipu API keys

### Changed
- 📦 Modified `utils.py` to include GLM-4.5 model mappings and base_url support
- 🔄 Updated `deep_researcher.py` with try-catch blocks for structured output
- 📋 Enhanced `.env.example` with Zhipu configuration examples
- 🎨 Replaced original README with GLM-4.5 focused documentation

### Fixed
- 🐛 Fixed AssertionError in streaming responses when using GLM models
- 🔧 Resolved structured output failures with automatic fallback mechanism
- ⚡ Improved token limit handling for 128K context models

### Technical Details
- **Model Support**: Added `glm-4.5:latest` and `glm-4.5-air:latest` to model mappings
- **Token Limits**: Configured 128,000 token limit for GLM-4.5 models
- **API Integration**: Implemented base_url parameter passing through configuration chain
- **Error Handling**: Added graceful degradation for unsupported features

## [0.0.16] - Original Version
- Base implementation from LangChain's open_deep_research project
- Support for OpenAI, Anthropic, Google, and other models
- LangGraph-based research workflow
- Multi-agent research system with supervisor and researcher roles
