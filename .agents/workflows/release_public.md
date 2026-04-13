# Public Release Synchronization

This workflow automates the process of synchronizing production code from the `dev` branch to the public `main` branch, applying the correct public version control headers, and pushing the release out to the community.

## Context
The `beam_fea` project uses a dual-repository, dual-branch release strategy:
- **`dev` branch & remote**: Contains full, granular development history with rapidly incrementing versions (e.g., v2.19.2).
- **`main` branch & public remote**: Contains a clean, truncated history with curated, milestone-based public versions (e.g., v1.1.0).

## Instructions for Agent

Follow these steps explicitly when the user requests a public release:

### 1. Synchronize Development Branch
- Ensure your active branch is `dev`.
- Check `git status` to ensure all working tree changes are committed.
- Push the latest internal development commits: 
  ```bash
  // turbo
  git push origin dev
  ```

### 2. Determine Public Target Version
- Identify the correct target tag for the public release (e.g. `v1.2.0`). 
- If not provided by the user, parse the public `CHANGELOG.md` or ask the user for the explicit public version tag they wish to release.

### 3. Transition to Main
- Check out the public base:
  ```bash
  // turbo
  git checkout main
  ```

### 4. Fetch the Source Code
- Pull all files from `dev` directly into `main`. This completely bypasses merge histories.
  ```bash
  // turbo
  git checkout dev -- .
  ```

### 5. Curate the Documentation Versions (CRITICAL)
Because `git checkout dev -- .` brings over `dev`'s files verbatim, several files will contain incorrect internal versions. You MUST fix these before committing:
- **`CHANGELOG.md`**: Do NOT release the internal granular dev history to the public repo. Edit `CHANGELOG.md` to reflect only the curated public release milestones. You may need to revert this file to its `HEAD` state on `main` and manually add a new `v1.x.x` section at the top.
- **`README.md`**: Update the "Latest Version" block to explicitly specify your TARGET PUBLIC VERSION (e.g., `v1.2.0`) instead of the inherited `v2.x.x`. Ensure the feature summary bullets are clean and professional.

### 6. Commit, Tag, and Deploy
- Once documents are patched and staged, perform the public release:
  ```bash
  // turbo
  git add -A
  git commit -m "release: <Target Public Version> — <Short Feature Summary>"
  git tag <Target Public Version>
  ```
- Push sequentially to both the internal `origin` tracking and external `public` tracking remotes:
  ```bash
  // turbo
  git push origin main
  git push public main
  git push origin <Target Public Version>
  git push public <Target Public Version>
  ```

### 7. Return to Development State
- Return the user's active context to the ongoing development tree:
  ```bash
  // turbo
  git checkout dev
  ```
