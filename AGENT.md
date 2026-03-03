# AGENT.md - Project Standard Operating Procedures

This document outlines the protocols and best practices for future agents (and human contributors) working on the `beam_fea` repository. **Adherence to these rules is mandatory to maintain consistency and quality.**

## 1. Codebase Onboarding & Synchronization

Before starting any task, ensure you have the latest context:

- **Always Work on Main**: Ensure you are synchronized with the `main` branch on GitHub. Check for recent updates.
- **Review Changelog**: Read `CHANGELOG.md` to understand recent changes, version jumps, and the current state of the architecture.
- **Audit README**: Consult `README.md` for the latest API reference and installation instructions.
- **Explore Examples**: Look at the `examples/` directory (e.g., `template_static.py`, `template_modal.py`) to see the latest recommended usage patterns.

## 2. Commit & Versioning Workflow

All commits must follow the `/commit` workflow protocol. Do not skip these steps.

### Update Version History

1. **Changelog**: Add a new version entry at the top of `CHANGELOG.md`. Follow the existing Semantic Versioning rules (Major.Minor.Patch).
2. **README**: Update the **Latest Version** section in `README.md` to match the new version.
3. **Automatic Sync**: Note that `beam_fea/__init__.py` and `pyproject.toml` pull the version dynamically; do not update them manually.

### Documentation Audit

- For every file changed, verify that its corresponding section in `README.md` (§3 Detailed API Reference) is accurate. Update if necessary.

### Testing

- **Run the Suite**: You MUST run `python -m pytest tests/ -v` before committing. All tests must pass.
- **Visual Tests**: For UI or plotting changes, run `scripts/visual_test_configurations.py` and verify the output in the generated image folders.

### Staging & Pushing

- `git add -A`
- `git commit -m "vX.Y.Z: <short description>"`
- `git push`

## 3. Creating Pull Requests

Follow this structure for PR descriptions to ensure clarity:

### PR Template

```markdown
## Objectives
- [What problem is this PR solving?]

## Proposed Changes
- [File A]: [Brief description of changes]
- [File B]: [Description]

## Verification
- [X] Ran full test suite (`pytest`)
- [X] Verified visual outputs (if applicable)
- [X] Audited README.md for API changes
```

- **Branch Naming**: Use descriptive prefixes: `feature/`, `fix/`, `docs/`, or `refactor/`.

## 4. Coding Suggestions & Patterns

To maintain the high performance and visual quality of the package:

- **Vectorization**: Always prefer NumPy vectorized operations (broadcasting) over loops for solver and visualizer logic.
- **Material Retrieval**: Use `get_material(name)` from the material database instead of hardcoding properties.
- **Centralized Aesthetics**: Use the `PlotStyle` dataclass for all plotting parameters to maintain a premium, consistent look.
- **Reporting**: Utilize `BeamReportGenerator` for creating automated performance/analysis summaries.
- **Minimal Dependencies**: Keep the core dependencies (`numpy`, `scipy`, `matplotlib`) up to date and avoid adding heavy new ones unless essential.

---
*Created by Antigravity (Google DeepMind).*
