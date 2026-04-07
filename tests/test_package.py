"""Tests for package-level imports and data loading."""
import os
import pandas as pd
import pytest


class TestPackageImports:
    def test_import_chargemol(self):
        from FeGB_PtableSeg import chargemol
        assert hasattr(chargemol, "ChargemolAnalysis")
        assert hasattr(chargemol, "parse_DDEC6_analysis_output")

    def test_import_plotters(self):
        import matplotlib
        matplotlib.use("Agg")
        from FeGB_PtableSeg import plotters
        assert hasattr(plotters, "GB_symmetries")
        assert hasattr(plotters, "custom_colors")


class TestMainDataframe:
    """Test loading the main dataset shipped with the repo."""

    @pytest.fixture
    def csv_path(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "2025_03_02_ptable_Fe_GB_df.csv.gz")
        if not os.path.exists(path):
            pytest.skip("Main CSV dataset not present")
        return path

    def test_csv_loads(self, csv_path):
        df = pd.read_csv(csv_path, nrows=5)
        assert len(df) == 5

    def test_csv_has_expected_columns(self, csv_path):
        df = pd.read_csv(csv_path, nrows=1)
        expected = ["job_name", "GB", "element", "site", "E_seg"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"
