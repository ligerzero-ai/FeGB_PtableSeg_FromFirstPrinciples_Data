"""Shared fixtures for FeGB_PtableSeg tests."""
import os
import pytest

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "S11-RA110-S3-32-He-19",
)


@pytest.fixture
def ddec6_output_path():
    """Path to a real VASP_DDEC_analysis.output file shipped with the repo."""
    path = os.path.join(DATA_DIR, "VASP_DDEC_analysis.output")
    if not os.path.exists(path):
        pytest.skip("Test data not available (Data/ dir not present)")
    return path


@pytest.fixture
def ddec6_directory():
    """Path to the directory containing VASP_DDEC_analysis.output."""
    if not os.path.exists(DATA_DIR):
        pytest.skip("Test data not available (Data/ dir not present)")
    return DATA_DIR
