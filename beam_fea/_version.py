def _get_version():
    """Dynamically retrieve version from metadata or CHANGELOG."""
    import re
    from pathlib import Path

    # 1. Try Parsing CHANGELOG.md (development mode)
    # This is preferred during dev to reflect changes immediately
    try:
        changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
        if changelog_path.exists():
            with open(changelog_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(r'## \[v?([\d.]+)\]', line)
                    if match:
                        return match.group(1)
    except Exception:
        pass

    # 2. Fallback: Try metadata (if installed)
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version("beam-fea")
    except (ImportError, PackageNotFoundError):
        pass

    # 3. Final Fallback
    return '1.7.3'

__version__ = _get_version()
