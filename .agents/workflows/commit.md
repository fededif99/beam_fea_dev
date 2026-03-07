---
description: commit changes to git with a version history update
---

# Commit Workflow

Follow these steps every time you commit code changes to the repository.

## 1. Update Version History

Before committing, add a new version entry at the **top** of `CHANGELOG.md` and update the **Latest Version** section in `README.md`.

Version number rules:

- **Patch** (x.x.**Z**): bug fixes, documentation corrections, minor refactors with no API change
- **Minor** (x.**Y**.0): new features, removals, or any change to the public API (method signatures, parameters, return types)
- **Major** (**X**.0.0): breaking changes that require callers to update their code

Entry format for `CHANGELOG.md`:

```markdown
## [vX.Y.Z] - YYYY-MM-DD

- One concise bullet per meaningful change, referencing the affected file or class
- Keep each bullet to a single line
```

Update the **Latest Version** in `README.md` to match the latest entry in `CHANGELOG.md`.

> [!NOTE]
> The version in `beam_fea/__init__.py` is now dynamic and will automatically synchronize with the top entry in `CHANGELOG.md`. There is no need to update it manually.

## 2. Audit the README API Reference

For every file changed in this commit, check whether its corresponding section in `README.md` (§3 Detailed API Reference) still accurately describes the current interface. Specifically verify:

- **Method signatures** — any added, removed, or renamed parameters are reflected in the docs
- **Supported types / options** — enumerations, `element_type` values, section classes, material names, BC types, etc.
- **Behaviour descriptions** — any change in default behaviour, units, or output format is documented
- **Removed features** — any class or method that was deleted is no longer referenced in the README

If the README is out of date, update the affected section(s) before proceeding. Do **not** commit stale documentation.

## 3. Run the Test Suite

// turbo

```powershell
python -m pytest tests/ -v
```

All tests must pass before proceeding. If any fail, fix them first.

## 3. Stage and Commit

// turbo

```powershell
git add -A
```

```powershell
git commit -m "vX.Y.Z: <short imperative summary of changes>"
```

Use the new version number in the commit message subject line.

## 4. Push

// turbo

```powershell
git push
```
