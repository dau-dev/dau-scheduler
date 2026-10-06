"""Every python block in the README runs as written, in order, in one
namespace. A README that shows code that does not run is a claim the
repository cannot back."""

from __future__ import annotations

import os
import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_the_readme_examples_run_in_order(tmp_path, monkeypatch) -> None:
    blocks = BLOCK.findall(README.read_text(encoding="utf-8"))
    assert blocks, "the README has no python blocks to run"
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"README.md block {index + 1}", "exec"), namespace)  # noqa: S102 (the README is ours)
        except Exception as error:  # pragma: no cover - the message is the point
            raise AssertionError(f"README python block {index + 1} does not run as written: {error!r}\n{block}") from error
    assert "execution" in namespace and "profile" in namespace, "the blocks ran and left their results behind"
    assert os.getcwd() == str(tmp_path)
