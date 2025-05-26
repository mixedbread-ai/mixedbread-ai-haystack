# Mixedbread AI Haystack Tests

This directory contains simplified tests for the Mixedbread AI Haystack integration.

## Test Structure

### Unit Tests
- Test initialization, configuration, and basic functionality without API calls
- Fast execution, no external dependencies
- Found in: `test_*.py` files (marked as non-integration tests)

### Integration Tests
- Test real API functionality with actual Mixedbread AI endpoints
- Require valid API key
- Essential tests only for core functionality

## Essential Integration Tests

### Document Embedder (`test_document_embedder.py`)
- `test_integration_basic_embedding`: Tests document embedding with metadata and batching

### Text Embedder (`test_text_embedder.py`)
- `test_integration_basic_text_embedding`: Tests single text embedding

### Reranker (`test_rerank.py`)
- `test_integration_basic_reranking`: Tests document reranking functionality

## Setup

1. **Set API Key:**
   ```bash
   export MXBAI_API_KEY="your-api-key-here"
   ```

2. **Optional - Custom Base URL:**
   ```bash
   export MXBAI_CUSTOM_BASE_URL="https://your-custom-endpoint.com"
   ```

## Running Tests

### Using the Test Runner Script
```bash
# Run only integration tests
python run_tests.py integration

# Run only unit tests
python run_tests.py unit

# Run all tests
python run_tests.py all

# Run specific test
python run_tests.py test_integration_basic_embedding
```

### Using pytest directly
```bash
# Run only integration tests
pytest -m integration tests/

# Run only unit tests
pytest -m "not integration" tests/

# Run specific test file
pytest tests/test_document_embedder.py::TestMixedbreadDocumentEmbedder::test_integration_basic_embedding -v
```

## Configuration

Tests use the `TestConfig` class in `test_config.py` for centralized configuration:
- API endpoints (official vs custom)
- Timeout and retry settings
- Default models
- Test parameters

Modify `test_config.py` to customize test behavior for your environment. 