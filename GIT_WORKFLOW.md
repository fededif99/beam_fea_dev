# Git Workflow — `beam_fea` Development

This document describes the dual-remote, dual-branch workflow for developing `beam_fea`
privately and publishing curated releases to the public repository.

---

## Repository Structure

| Remote | Repo | Purpose |
|--------|------|---------|
| `origin` | `beam_fea_dev` | Private development — full history, WIP commits, scratch files |
| `public` | `beam_fea` | Public release — clean history, only production-ready code |

| Branch | Purpose |
|--------|---------|
| `dev` | Active development — commit freely, internal versioning (e.g. v2.x.x) |
| `main` | Public release branch — curated commits, public versioning (e.g. v1.x.x) |

You work exclusively on the `dev` branch in `beam_fea_dev`. The `main` branch and the
public repo only ever receive deliberate, reviewed releases.

---

## One-Time Setup

Clone your private dev repo if you haven't already:

```bash
git clone https://github.com/you/beam_fea_dev.git
cd beam_fea_dev
git checkout dev
```

Add the public repo as a second remote:

```bash
git remote add public https://github.com/you/beam_fea.git
```

Verify both remotes are registered:

```bash
git remote -v
# origin   https://github.com/you/beam_fea_dev.git (fetch)
# origin   https://github.com/you/beam_fea_dev.git (push)
# public   https://github.com/you/beam_fea.git (fetch)
# public   https://github.com/you/beam_fea.git (push)
```

---

## Day-to-Day Development

All development happens on the `dev` branch:

```bash
git checkout dev          # always work here
# Make changes, run tests
git add .
git commit -m "feat: add influence lines analysis"
git push origin dev       # saves to private dev repo
```

No special rules — commit as often as you like, use WIP commits, leave debug
code, whatever works for your flow.

---

## Publishing a Release to the Public Repo

When a batch of changes is ready to go public, perform a **release merge**.

### Step 1 — Preview what has changed since the last release

```bash
git diff public/main dev --name-only
```

Review all changed files and confirm nothing unintended is included (scratch
scripts, local config, test outputs, etc.).

### Step 2 — Switch to main and bring in the new code

```bash
git checkout main

# Copy all production code from dev (NOT the version/docs files)
git checkout dev -- beam_fea/
git checkout dev -- examples/
git checkout dev -- tests/
git checkout dev -- scripts/
git checkout dev -- pyproject.toml
git checkout dev -- .gitignore
git checkout dev -- THEORY.md
git checkout dev -- .github/
git checkout dev -- GIT_WORKFLOW.md
```

> **Important**: Do NOT run `git checkout dev -- CHANGELOG.md` or
> `git checkout dev -- README.md`. Those are maintained separately on `main`
> (see Step 3).

### Step 3 — Update the public-facing version files

**Edit `CHANGELOG.md` on `main`** to add a new clean public release entry at the top:

```markdown
## [v1.x.x] - YYYY-MM-DD

- Summary of user-facing features included in this release.
```

> **Rule**: Do not copy internal dev CHANGELOG entries (e.g., v2.19.x patch notes)
> into the public CHANGELOG. Write concise, user-facing release notes instead.
> Many internal dev versions may be summarised into a single public milestone.

**Update the version line in `README.md`:**

```markdown
### Latest Version: v1.x.x *(YYYY-MM-DD)*
```

Because `_version.py` dynamically reads the version from `CHANGELOG.md`, no
further code changes are needed — `beam_fea.__version__` will automatically
return the correct public version once CHANGELOG.md is updated.

### Step 4 — Commit and push

```bash
git add -A
git commit -m "release: v1.x.x — <brief description>"
git push origin main        # save release to private repo (main branch)
git push public main        # publish to public repo
```

### Step 5 — Tag the release

```bash
git tag v1.x.x
git push origin v1.x.x     # tag in private repo
git push public v1.x.x     # tag in public repo (GitHub will create a release)
```

GitHub will automatically detect the tag so users can install a specific version:

```bash
pip install git+https://github.com/you/beam_fea.git@v1.x.x
```

### Step 6 — Return to development

```bash
git checkout dev
```

> **Rule of thumb:** Release to `public` at natural milestones — a completed
> feature set, a stability improvement batch, or when the dev branch reaches a
> stable state — not after every commit.

---

## Two Separate Version Streams

| Stream | Branch | Format | Audience |
|--------|--------|--------|---------|
| **Internal** | `dev` | `v2.x.x` | Developer only |
| **Public** | `main` | `v1.x.x` | End users |

Internal versions increment with every code change. Public versions only
increment at curated release milestones.

### Semantic Versioning for Public Releases

Follow [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`

| Increment | When |
|-----------|------|
| `PATCH` (1.0.**1**) | Bug fixes, no API changes |
| `MINOR` (1.**1**.0) | New features, backwards compatible |
| `MAJOR` (**2**.0.0) | Breaking API changes |

---

## Two Separate CHANGELOGs

| File location | Branch | Purpose |
|--------------|--------|---------|
| `CHANGELOG.md` on `dev` | `origin/dev` | Full internal dev history (every patch, refactor, WIP) |
| `CHANGELOG.md` on `main` | `origin/main` / `public/main` | Curated public release notes only |

These two files are **intentionally different** and are never merged or overwritten
with each other. When writing public release notes, summarise multiple internal
dev versions into a single user-facing milestone entry.

---

## What Belongs in Each Repo

### Always stays on `dev`, never pushed to `public`

- Scratch test scripts and notebooks
- Local paths or machine-specific config
- Work-in-progress features not ready for users
- Debug print statements
- Large output files (reports, images, CSVs)
- Internal version increments (e.g., v2.19.x granular patches)

### Push to `public` (via `main`) when ready

```
beam_fea/
├── beam_fea/           # all package source files
│   ├── __init__.py
│   ├── solver.py
│   ├── mesh.py
│   └── ...
├── examples/           # clean, documented example scripts
├── tests/              # unit test suite
├── pyproject.toml
├── README.md
├── CHANGELOG.md        # curated public release notes only
├── LICENSE
└── THEORY.md
```

---

## Handling a Mistake — Pushing Something You Didn't Mean To

If you accidentally push a file to `public` that shouldn't be there:

```bash
# Remove the file from the index (stops tracking it)
git rm --cached path/to/file.py

# Commit the removal
git commit -m "chore: remove accidental file"

# Push the fix to public
git push public main
```

> If the file contained sensitive information (credentials, keys), contact
> GitHub support — git history rewriting is needed in that case.

---

## Quick Reference

| Task | Command |
|------|---------|
| Work on development | `git checkout dev` |
| Save development work | `git push origin dev` |
| Start a release | `git checkout main` |
| Copy code from dev to main | `git checkout dev -- beam_fea/ examples/ tests/ ...` |
| Preview public diff | `git diff public/main main --name-only` |
| Publish to public repo | `git push public main` |
| Create and push release tag | `git tag v1.x.x` then `git push public main` then `git push public v1.x.x` |
| Return to development | `git checkout dev` |
| Check both remotes | `git remote -v` |
| Pull any changes from public | `git fetch public` |
