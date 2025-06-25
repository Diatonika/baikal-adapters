# Baikal Template Repository #

## GitHub Repository Setup ##

Create the following repository GitHub Actions variables:

1. `PYTHON_VERSION`: Python version to set up in `lint` workflow environment.
2. `PYTHON_PUBLISH`: Set to `true` when project is intended to be built as python package and published to PyPI.

### Semantic Versioning ###

1. Your commits messages must
   follow [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/).
2. Package release is automated
   by [Python Semantic Release GitHub Action](https://github.com/python-semantic-release/python-semantic-release).
3. Current repository makes use
   of [Generated Release Notes GitHub action](https://github.com/marketplace/actions/release-changelog-builder).
4. When you are ready to release a stable version, set `pyproject.toml:major_on_zero` to `true` and run Semantic Release
   again. This will increment the major version to `1.0.0`.

As such, it is strongly recommended to adhere with Pull Request based workflow (`main + develop`) and Conventional
Commits Specification.

> To further customize release notes style, uncomment `configuration: changelog-config.json` line in
`.github/workflows/pipeline.yml`.

## Project Setup ##

### Tools ###

1. Install `Python 3.X`.
2. Install [Poetry](https://python-poetry.org/docs/).
3. Install [Just](https://just.systems/man/en/packages.html).

### Environment File ###

1. Copy `.env.example` file as `.env`.
2. Fill in private PyPI credentials.

### Project Initialization ###

Populate `source-dirs` variable inside `justfile` with directories for linter & formatters checks and run:

```bash
just init-dev
```

### PyCharm Plugins ###

1. [Just Plugin](https://plugins.jetbrains.com/plugin/18658-just): IDE support for `justfile`.
2. [MyPy Plugin](https://plugins.jetbrains.com/plugin/11086-mypy): IDE support for `MyPy` inspections.
3. [Ruff Plugin](https://plugins.jetbrains.com/plugin/20574-ruff): IDE support for `Ruff` inspections.

### PyCharm Setup ###

1. `CTRL + ALT + S` → `Mypy` → `Path To Config File`: `pyproject.toml`.
2. `CTRL + ALT + S` → `Tools` → `Ruff`:
    1. → `Run Ruff on Code Reformat`: `true`.
    2. → `Use Ruff Format`: `true`.
    3. → If not auto-detected, specify paths `.venv/Scripts/ruff.exe` and `pyproject.toml`.
