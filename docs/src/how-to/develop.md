# Develop and contribute

`dau-scheduler` is pure Python. Contributions are welcome under the
[Apache 2.0 license](https://github.com/dau-dev/dau-scheduler/blob/main/LICENSE);
every commit must carry a Developer Certificate of Origin sign-off (`git commit -s`), which a GitHub app checks on each pull request.

## Set up

```bash
git clone https://github.com/dau-dev/dau-scheduler.git
cd dau-scheduler
make develop
```

`make develop` installs the package in editable mode with its development
dependencies. `make` with no arguments lists the targets.

## Lint, type-check, test

```bash
make lint      # ruff, mdformat and codespell
make checks    # distribution check and ty
make test      # pytest
```

Run all three before opening a pull request. The README's Python blocks run as
a test, so an example that stops working fails the suite.

## Documentation

The pages under `docs/src` and the README are the documentation site. It is
built with [yardang](https://github.com/python-project-templates/yardang)
from `[tool.yardang]` in `pyproject.toml`, which lists the pages in order, and
published to GitHub Pages by the docs workflow after each green build on
`main`. To build it locally:

```bash
uv pip install "yardang[themes]"
yardang build
```

The site lands in `docs/html`. `make lint` formats and spell-checks the pages.

## Pull requests

Work on a fork and open a pull request against `main`; a draft is fine while
the change is in progress. Squash into logical commits, all signed. Open an
issue first for anything larger than a fix, and use the discussions page for
questions.
