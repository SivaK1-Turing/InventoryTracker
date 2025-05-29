# tests/feature2/conftest.py
import pytest
import os
import tempfile
import shutil
from typing import Optional, Dict, Any

@pytest.fixture
def isolated_filesystem():
    """Provide an isolated filesystem for tests."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield tmpdirname
        os.chdir(cwd)