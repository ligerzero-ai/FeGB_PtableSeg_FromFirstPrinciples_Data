"""Tests for FeGB_PtableSeg.plotters module (non-rendering checks)."""
import os
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from FeGB_PtableSeg.plotters import (
    GB_symmetries,
    custom_colors,
    gb_latex_dict,
    gb_marker_dict,
    get_element_number,
    get_element_symbol,
)


class TestElementLookups:
    def test_get_element_number_valid(self):
        assert get_element_number("Fe") == 26
        assert get_element_number("He") == 2
        assert get_element_number("C") == 6

    def test_get_element_number_invalid(self):
        result = get_element_number("Xx")
        assert np.isnan(result)

    def test_get_element_symbol_valid(self):
        assert get_element_symbol(1) == "H"
        assert get_element_symbol(26) == "Fe"

    def test_get_element_symbol_invalid(self):
        result = get_element_symbol(999)
        assert result is np.nan or (isinstance(result, float) and np.isnan(result))


class TestGBConstants:
    """Ensure the GB dictionaries are consistent with each other."""

    GB_NAMES = [
        "S3_RA110_S1_11",
        "S3_RA110_S1_12",
        "S5_RA001_S210",
        "S5_RA001_S310",
        "S9_RA110_S2_21",
        "S11_RA110_S3_32",
    ]

    def test_custom_colors_keys(self):
        for gb in self.GB_NAMES:
            assert gb in custom_colors

    def test_gb_latex_dict_keys(self):
        for gb in self.GB_NAMES:
            assert gb in gb_latex_dict

    def test_gb_marker_dict_keys(self):
        for gb in self.GB_NAMES:
            assert gb in gb_marker_dict

    def test_latex_strings_contain_sigma(self):
        for label in gb_latex_dict.values():
            assert r"\Sigma" in label


class TestGBSymmetries:
    def test_all_gbs_present(self):
        sym = GB_symmetries()
        expected = [
            "S9_RA110_S2_21",
            "S11_RA110_S3_32",
            "S3_RA110_S1_12",
            "S3_RA110_S1_11",
            "S5_RA001_S310",
            "S5_RA001_S210",
        ]
        for name in expected:
            assert name in sym.symmetry_dict_all

    def test_symmetry_values_are_lists(self):
        sym = GB_symmetries()
        for gb_name, sym_dict in sym.symmetry_dict_all.items():
            for key, value in sym_dict.items():
                assert isinstance(value, list), f"{gb_name} site {key} is not a list"

    def test_s3_1_symmetry(self):
        sym = GB_symmetries()
        sd = sym.S3_1_symmetrydict
        assert 20 in sd
        assert 36 in sd
        # Site 37 is on the GB plane, so symmetry list should have 1 element
        assert len(sd[36]) == 1

    def test_s11_symmetry(self):
        sym = GB_symmetries()
        sd = sym.S11_symmetrydict
        # Sites 21 and 22 are on the plane
        assert sd[21] == []
        assert sd[22] == []


class TestDataFiles:
    """Verify CSV data files load correctly."""

    def test_bulk_df_loads(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.join(os.path.dirname(module_path), "FeGB_PtableSeg")
        df = pd.read_csv(os.path.join(pkg_path, "bulk_df.csv"))
        assert "Z" in df.columns
        assert "element" in df.columns
        assert "energy" in df.columns
        assert len(df) > 50  # should have ~90 elements

    def test_periodic_table_csv_loads(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.join(os.path.dirname(module_path), "FeGB_PtableSeg")
        df = pd.read_csv(os.path.join(pkg_path, "periodic_table.csv"))
        assert "symbol" in df.columns
        assert "Z" in df.columns
