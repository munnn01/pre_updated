from types import SimpleNamespace

import pytest
from ops.download_kernel_version_output import select_file


def test_select_file_requires_one_exact_suffix_match():
    files = [
        SimpleNamespace(file_name="outputs/a/checkpoints/preprocessor.pth"),
        SimpleNamespace(file_name="outputs/a/checkpoints/preprocessor_last.pth"),
    ]
    selected = select_file(files, "/checkpoints/preprocessor.pth")
    assert selected.file_name.endswith("/checkpoints/preprocessor.pth")
    with pytest.raises(ValueError, match="found 0"):
        select_file(files, "/missing.pth")
