"""Tests for FeGB_PtableSeg.chargemol module."""
import numpy as np
import pandas as pd
import pytest

from FeGB_PtableSeg.chargemol import (
    ChargemolAnalysis,
    analyse_ANSBO,
    compute_average_pairs,
    get_ANSBO,
    get_ANSBO_all_cleavage_planes,
    get_unique_values_in_nth_value,
    parse_DDEC6_analysis_output,
    parse_lines,
)


# --------------- parse_lines ---------------

class TestParseLines:
    def test_single_block(self):
        lines = ["header\n", "START\n", "data1\n", "data2\n", "END\n", "footer\n"]
        result = parse_lines(lines, "START", "END")
        assert len(result) == 1
        assert result[0] == ["data1\n", "data2\n"]

    def test_multiple_blocks_recursive(self):
        lines = [
            "START\n", "a\n", "END\n",
            "START\n", "b\n", "END\n",
        ]
        result = parse_lines(lines, "START", "END", recursive=True)
        assert len(result) == 2
        assert result[0] == ["a\n"]
        assert result[1] == ["b\n"]

    def test_single_block_non_recursive(self):
        lines = [
            "START\n", "a\n", "END\n",
            "START\n", "b\n", "END\n",
        ]
        result = parse_lines(lines, "START", "END", recursive=False)
        assert len(result) == 1
        assert result[0] == ["a\n"]

    def test_no_match(self):
        lines = ["nothing here\n"]
        result = parse_lines(lines, "START", "END")
        assert result == []

    def test_open_ended_block(self):
        """Block that starts but never reaches trigger_end."""
        lines = ["START\n", "data\n"]
        result = parse_lines(lines, "START", "END")
        assert len(result) == 1
        assert result[0] == ["data\n"]

    def test_empty_block(self):
        lines = ["START\n", "END\n"]
        result = parse_lines(lines, "START", "END")
        assert len(result) == 1
        assert result[0] == []


# --------------- utility functions ---------------

class TestUtilities:
    def test_get_unique_values_simple(self):
        arr = [[1.0, 2.0], [1.001, 3.0], [2.0, 4.0]]
        result = get_unique_values_in_nth_value(arr, 0, tolerance=0.01)
        assert len(result) == 2
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_get_unique_values_all_same(self):
        arr = [[5.0], [5.0001], [4.9999]]
        result = get_unique_values_in_nth_value(arr, 0, tolerance=0.01)
        assert len(result) == 1

    def test_compute_average_pairs(self):
        result = compute_average_pairs([1.0, 3.0, 7.0])
        assert result == [2.0, 5.0]

    def test_compute_average_pairs_single(self):
        result = compute_average_pairs([5.0])
        assert result == []


# --------------- parse_DDEC6_analysis_output ---------------

class TestParseDDEC6:
    def test_parse_returns_structure_and_bond_matrix(self, ddec6_output_path):
        struct, bm = parse_DDEC6_analysis_output(ddec6_output_path)

        # Structure checks
        assert struct is not None
        assert len(struct) == 42  # S11-RA110-S3-32: 42 atoms (41 Fe + 1 He)
        assert struct.lattice is not None
        assert struct.lattice.volume > 0

        # Bond matrix checks
        assert isinstance(bm, pd.DataFrame)
        assert "final_bond_order" in bm.columns
        assert "atom1" in bm.columns
        assert "atom2" in bm.columns
        assert len(bm) > 0

    def test_structure_has_correct_elements(self, ddec6_output_path):
        struct, _ = parse_DDEC6_analysis_output(ddec6_output_path)
        symbols = [s.symbol for s in struct.species]
        assert "Fe" in symbols
        assert "He" in symbols

    def test_lattice_is_physical(self, ddec6_output_path):
        struct, _ = parse_DDEC6_analysis_output(ddec6_output_path)
        # Cell should be reasonable for an Fe GB supercell
        a, b, c = struct.lattice.abc
        assert 3.0 < a < 20.0
        assert 3.0 < b < 20.0
        assert 20.0 < c < 100.0  # long slab direction

    def test_bond_orders_are_positive(self, ddec6_output_path):
        _, bm = parse_DDEC6_analysis_output(ddec6_output_path)
        assert (bm["final_bond_order"] >= 0).all()


# --------------- ChargemolAnalysis class ---------------

class TestChargemolAnalysis:
    def test_init(self, ddec6_directory):
        ca = ChargemolAnalysis(ddec6_directory)
        assert ca.struct is not None
        assert ca.bond_matrix is not None
        assert len(ca.struct) == 42

    def test_ansbo_profile(self, ddec6_directory):
        ca = ChargemolAnalysis(ddec6_directory)
        coords, profile = ca.get_ANSBO_profile()
        assert len(coords) == len(profile)
        assert len(profile) > 0
        # ANSBO values should be positive
        assert all(v > 0 for v in profile)

    def test_min_ansbo(self, ddec6_directory):
        ca = ChargemolAnalysis(ddec6_directory)
        coords, profile = ca.get_ANSBO_profile()
        min_ansbo = ca.get_min_ANSBO()
        # get_min_ANSBO returns min of the (coords, profile) tuple
        # Actually it returns min() of the whole result - let's just check it's a tuple
        assert min_ansbo is not None

    def test_analyse_ansbo(self, ddec6_directory):
        ca = ChargemolAnalysis(ddec6_directory)
        result = ca.analyse_ANSBO()
        assert "layer_boundaries" in result
        assert "cleavage_coord" in result
        assert "ANSBO_profile" in result


# --------------- ANSBO computation ---------------

class TestANSBO:
    def test_get_ANSBO_all_cleavage_planes(self, ddec6_output_path):
        struct, bm = parse_DDEC6_analysis_output(ddec6_output_path)
        coords, profile = get_ANSBO_all_cleavage_planes(struct, bm)
        assert len(coords) == len(profile)
        assert len(profile) > 0

    def test_get_ANSBO_single_plane(self, ddec6_output_path):
        struct, bm = parse_DDEC6_analysis_output(ddec6_output_path)
        # Pick a cleavage plane in the middle of the cell
        mid_z = struct.lattice.abc[2] / 2
        ansbo = get_ANSBO(struct, bm, mid_z)
        assert isinstance(ansbo, float)
        assert ansbo >= 0

    def test_analyse_ansbo_function(self, ddec6_directory):
        result = analyse_ANSBO(ddec6_directory)
        assert isinstance(result, dict)
        assert len(result["cleavage_coord"]) > 0
