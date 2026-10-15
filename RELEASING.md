# Releasing RHIS-TS

This is a step-by-step guide for cutting a release and publishing `rhis-ts` to
[TestPyPI](https://test.pypi.org) (a sandbox) and then to
[PyPI](https://pypi.org). It assumes you have never published a package before.

## Key ideas

- **Version** lives in one place: `src/rhis_ts/__init__.py` (`__version__`).
  PyPI reads this, not the git tag.
- **A tag** is a permanent label pointing at one commit (for example `v0.1.0`).
  It is how you and GitHub mark which commit is which release.
- **A GitHub Release** is a page built on top of a tag. Publishing a Release is
  what triggers the `Publish` workflow, which uploads to PyPI.
- **Versions on PyPI are permanent.** Once `0.1.0` is on PyPI you can never
  replace it - you must bump to `0.1.1`. This is why we test on TestPyPI first.
- **Trusted Publishing** means PyPI trusts GitHub directly (via OIDC), so no
  long-lived PyPI API token is stored anywhere.

## One-time setup

### 1. TestPyPI account and token

You already have a TestPyPI account, 2FA, and an API token. Put the token in
`~/.pypirc`:

```ini
[distutils]
index-servers =
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-XXXXXXXXXXXXXXXX   # your TestPyPI token
```

> Tip: `chmod 600 ~/.pypirc`. Never commit this file.

### 2. PyPI pending Trusted Publisher

The project does not exist on PyPI yet, so register a *pending* publisher:

1. Log in to <https://pypi.org> (create the account if needed; it is separate
   from TestPyPI) and enable 2FA.
2. Go to **Account settings → Publishing → Add a pending publisher**.
3. Fill in exactly:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `rhis-ts` |
   | Owner | `MCoelho222` |
   | Repository name | `time-series` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

The first successful publish "claims" the name and turns the pending publisher
into a real one.

### 3. GitHub environment

In the GitHub repo: **Settings → Environments → New environment → `pypi`**.
The name must match the environment name used in `publish.yml` and on PyPI.
Optionally add a required reviewer so a human approves the PyPI upload.

## Cutting a release

### Step 1 - prepare the version on `master`

1. Open a pull request for your work and merge it into `master` (the default
   branch). CI (`ci.yml`) must be green.
2. Bump the version in `src/rhis_ts/__init__.py` if this is not the first
   release, and add a matching section to `CHANGELOG.md`.

### Step 2 - build and check locally

```bash
python -m build
python -m twine check dist/*
```

`twine check` must print `PASSED` for both the `.whl` and the `.tar.gz`.

> **If `python -m build` fails with `No module named build.__main__; 'build'
> is a package and cannot be directly executed`**, either the `build` tool is
> not installed in the active environment or a leftover `build/` directory in
> the repo root is shadowing the module. Fix it with:
> ```bash
> uv pip install -r requirements/requirements-dev.txt
> rm -rf build dist
> ```
> `uv build` also works and sidesteps the local-directory issue.

### Step 3 - dry run on TestPyPI

```bash
python -m twine upload -r testpypi dist/*
```

Then install it into a throwaway environment and confirm it works. The
`--extra-index-url` lets real dependencies (numpy, pandas, ...) come from PyPI
even if TestPyPI does not mirror them:

```bash
python -m venv /tmp/testpypi-venv
source /tmp/testpypi-venv/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            rhis-ts
rhis-ts
```

If something is wrong, fix it, bump the version (TestPyPI also refuses
overwrites), rebuild, and upload again. Nothing on real PyPI has changed yet.

### Step 4 - publish to PyPI via GitHub

1. On GitHub, go to **Releases → Draft a new release**.
2. Click **Choose a tag**, type `v0.1.0` (must match `v` + the package version;
   the workflow checks this), and select **Create new tag on publish**.
   The target should be your updated `master`.
3. Title it `v0.1.0`, paste the changelog notes, and click **Publish release**.

Publishing the release runs the `Publish` workflow: it builds the
distribution, tests the built wheel on Python 3.10-3.14, and uploads to PyPI
through Trusted Publishing. Watch it under the **Actions** tab.

### Step 5 - verify

```bash
pip install rhis-ts
rhis-ts
```

Your project page is now live at <https://pypi.org/project/rhis-ts/>, and the
release/tag links in `CHANGELOG.md` start working.

## If the workflow fails

- **"invalid-publisher" / OIDC error** - the pending publisher fields on PyPI
  must match `publish.yml` exactly: owner `MCoelho222`, repo `time-series`,
  workflow `publish.yml`, environment `pypi`.
- **Tag/version mismatch** - the workflow fails early if the release tag is not
  `v` + `__version__`. Fix the version or the tag and re-release with a new
  version number.
- **"File already exists"** - that version is already on PyPI. Bump the
  version; you cannot reuse it.

## Reference

- Python packaging guide: <https://packaging.python.org/en/latest/tutorials/packaging-projects/>
- Trusted Publishing: <https://docs.pypi.org/trusted-publishers/>
- `pypa/gh-action-pypi-publish`: <https://github.com/pypa/gh-action-pypi-publish>
