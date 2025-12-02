#!/bin/bash

# Test runner script for the ML inference system

set -e

echo "=========================================="
echo "Running Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed."
    echo "Please install dependencies: pip install -r requirements.common.txt"
    exit 1
fi

# Run tests with coverage
echo "Running tests with coverage..."
pytest --cov=api --cov=scripts --cov-report=term-missing --cov-report=html -v

echo ""
echo "=========================================="
echo "Test coverage report generated in htmlcov/"
echo "Open htmlcov/index.html in a browser to view"
echo "=========================================="

