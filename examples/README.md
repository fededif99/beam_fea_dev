# Beam FEA Examples & Templates

This directory contains usage examples and starting templates for the `beam_fea` library.

## 🚀 Getting Started

If you are new to the library, start with the templates. They are designed to be copied and modified for your specific engineering problems.

- **[template_static.py](template_static.py)**: A clean, linear workflow for solving deflection, internal forces, and stresses under external loads.
- **[template_modal.py](template_modal.py)**: A specialized template for finding natural frequencies and mode shapes (vibration analysis).
- **[template_batch.py](template_batch.py)**: A template for performing multiple analyses using parametric CSV inputs.

## 📚 Detailed Examples

These scripts demonstrate more specific or advanced use cases:

- **ex01_cantilever.py**: Simple cantilever beam verification against analytical solutions.
- **ex02_fixed_fixed_beam.py**: A beam with fixed-fixed supports under uniform distributed load (UDL).
- **ex03_vibration.py**: Analysis of a multi-span beam to find natural vibration modes.
- **ex04_combined_analysis.py**: Running both static and modal analysis on the same model.
- **ex05_composite_beam.py**: Advanced laminate analysis using Classical Laminate Theory.
- **ex06_anisotropic_coupling.py**: Capturing bend-extension coupling in asymmetric laminates.
- **ex07_angled_frame.py**: Static analysis of a 2D cranked beam/angled frame.
- **ex08_batch_analysis.py**: Performing parametric studies and generating summary reports.

## 📊 Viewing Results

Most examples generate an automated Markdown report (e.g., `static_template_results.md`). You can open these in any Markdown viewer (like VS Code's Preview) to see tabulated results and generated PNG plots.
