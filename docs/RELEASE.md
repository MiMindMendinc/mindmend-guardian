# Release Process

MindMend Guardian follows [Semantic Versioning](https://semver.org/) and documents
changes in [`CHANGELOG.md`](../CHANGELOG.md) using
[Keep a Changelog](https://keepachangelog.com/) conventions.

## Version sources

Keep these in sync when cutting a release:

- `pyproject.toml` → `[project].version`
- `guardian/__init__.py` → `__version__`
- `CHANGELOG.md` → dated section with release notes
- Git tag → `vX.Y.Z`

## Maintainer checklist

1. Ensure `main` is green in CI.
2. Move unreleased notes in `CHANGELOG.md` into a new `## [X.Y.Z] - YYYY-MM-DD` section.
3. Bump version in `pyproject.toml` and `guardian/__init__.py`.
4. Commit: `chore(release): vX.Y.Z`
5. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main --tags
   ```
6. Publish a GitHub Release using the tag. Copy the matching `CHANGELOG.md` section
   into the release description.
7. (Optional) Publish to PyPI when distribution beyond GitHub clone is desired.

## Pre-release quality gates

```bash
pip install -e ".[dev,api]"
ruff check guardian tests
ruff format --check guardian tests
pytest --cov=guardian --cov-report=term-missing
python -m guardian.demo --json
pip-audit
```

## Branch policy

- `main` is the supported development line.
- Release tags are created from `main` only.
- Hotfix releases use patch bumps (`X.Y.Z+1`) and should include CHANGELOG entries.

## License

Releases are distributed under the MIT License in [`LICENSE`](../LICENSE).
