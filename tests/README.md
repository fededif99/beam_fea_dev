# Beam FEA Test Suite

This directory contains comprehensive unit and integration tests for the beam_fea package.

## Test Files

- **test_materials.py** - Tests for material property definitions and validation
- **test_cross_sections.py** - Tests for cross-section calculations and validation
- **test_mesh.py** - Tests for mesh generation and refinement
- **test_element_matrices.py** - Tests for element stiffness and mass matrices
- **test_integration.py** - Integration tests for complete analysis workflows

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_materials.py -v
```

### Run with coverage report:
```bash
pytest tests/ --cov=beam_fea --cov-report=html
```

## Test Coverage

The test suite covers:
- ✅ Input validation for all critical parameters
- ✅ Material property calculations
- ✅ Cross-section property calculations
- ✅ Mesh generation and refinement
- ✅ Element matrix calculations
- ✅ Static analysis workflows
- ✅ Modal analysis workflows
- ✅ Boundary conditions
- ✅ Load applications

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov numpy scipy matplotlib
```
