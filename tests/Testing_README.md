# Test Suite

This directory contains the test suite for the ML inference system.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_utils.py` - Unit tests for utility functions (`scripts/utils.py`)
- `test_image_preprocessing.py` - Unit tests for image preprocessing (`scripts/image_preprocessing.py`)
- `test_inference_utils.py` - Unit tests for inference utilities (`scripts/inference_utils.py`)
- `test_onnx_api.py` - Tests for ONNX API endpoints
- `test_torch_api.py` - Tests for PyTorch API endpoints
- `test_coordinator_api.py` - Tests for Coordinator API endpoints

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.common.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_utils.py
pytest tests/test_onnx_api.py
```

### Run with Coverage

```bash
pytest --cov=api --cov=scripts --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run Specific Test Classes or Functions

```bash
pytest tests/test_utils.py::TestLoadConfig
pytest tests/test_utils.py::TestLoadConfig::test_load_valid_yaml
```

### Run Tests with Markers

```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m api           # Run only API tests
```

### Verbose Output

```bash
pytest -v               # Verbose output
pytest -vv              # Very verbose output
pytest -s               # Show print statements
```

## Test Categories

- **Unit Tests**: Test individual functions and utilities in isolation
- **API Tests**: Test FastAPI endpoints using TestClient (mocked dependencies)
- **Integration Tests**: Test full workflows (requires running services)

## Notes

- Some tests require model files to be present (e.g., `models/squeezenet.onnx`). These tests will be skipped if the files are not found.
- API tests use mocking to avoid requiring actual model loading, which speeds up test execution.
- For integration tests that require running Docker containers, see the main `DOCKER_README.md`.

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for test files, `test_*` for test functions
2. Use fixtures from `conftest.py` when possible
3. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
4. Mock external dependencies (model loading, HTTP requests, etc.) for unit tests
5. Use descriptive test names that explain what is being tested

