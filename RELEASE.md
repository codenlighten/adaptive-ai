# Releasing delta-sigma-nn to PyPI

Manual steps for the author. Requires `twine` and a PyPI account
(`https://pypi.org/account/register/`).

## One-time setup

```bash
pip install build twine
```

Create or upload your PyPI API token at https://pypi.org/manage/account/
→ "API tokens" → "Add API token". Scope it to "Entire account" for the
first upload, then optionally narrow to just the `delta-sigma-nn`
project after publishing.

Save the token (a string starting with `pypi-`) somewhere safe, then
either configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-<your-token-here>
```

…or set it as an environment variable each time:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-token-here>
```

## Release procedure

1. **Bump the version** in `pyproject.toml`.
2. **Update `CHANGELOG.md`** (create one if missing).
3. **Run tests one more time:**
   ```bash
   python -m pytest tests/ -q
   ```
4. **Clean previous builds:**
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```
5. **Build sdist + wheel:**
   ```bash
   python -m build --no-isolation
   # Outputs:
   #   dist/delta_sigma_nn-X.Y.Z.tar.gz
   #   dist/delta_sigma_nn-X.Y.Z-py3-none-any.whl
   ```
6. **Validate the wheel locally** in a fresh environment:
   ```bash
   python -m venv /tmp/dsv && /tmp/dsv/bin/pip install dist/*.whl
   /tmp/dsv/bin/python -c "from delta_sigma_nn import DeltaSigmaMLP; print('ok')"
   ```
7. **Upload to TestPyPI first** (recommended for first release):
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
   Confirm it shows up at https://test.pypi.org/project/delta-sigma-nn/
   and that `pip install -i https://test.pypi.org/simple/ delta-sigma-nn`
   works.
8. **Upload to real PyPI:**
   ```bash
   python -m twine upload dist/*
   ```
9. **Tag the release:**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
10. **Create a GitHub Release** from the tag at
    https://github.com/codenlighten/adaptive-ai/releases/new — paste
    relevant notes from `CHANGELOG.md`.

After the first publish, the next release is just steps 1, 3, 4, 5, 8, 9, 10.

## Verifying the published package

After upload:

```bash
pip install delta-sigma-nn
python -c "from delta_sigma_nn import DeltaSigmaMLP; print('it works')"
```

If the import fails, the most likely cause is an internal import
referencing something not bundled in the wheel. Check
`python -m zipfile -l dist/*.whl` against the package's actual import
graph.
