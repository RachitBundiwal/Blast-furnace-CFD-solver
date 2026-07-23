"""Unit tests for the mesh module."""
from __future__ import annotations

import numpy as np
import pytest

from bf2d.mesh import build_mesh
from bf2d.parameters import BFParameters2D


class TestMesh:
    @pytest.fixture
    def mesh(self):
        p = BFParameters2D(nr=20, nz=40)
        return build_mesh(p)

    def test_shape(self, mesh) -> None:
        assert mesh.r_c.shape == (20,)
        assert mesh.z_c.shape == (40,)
        assert mesh.vol.shape == (20, 40)
        assert mesh.active.shape == (20, 40)
        assert mesh.area_r.shape == (21, 40)
        assert mesh.area_z.shape == (20, 41)

    def test_faces_monotonic(self, mesh) -> None:
        assert np.all(np.diff(mesh.r_f) > 0)
        assert np.all(np.diff(mesh.z_f) > 0)

    def test_centres_inside_faces(self, mesh) -> None:
        assert np.all(mesh.r_c > mesh.r_f[:-1])
        assert np.all(mesh.r_c < mesh.r_f[1:])

    def test_volumes_positive(self, mesh) -> None:
        assert np.all(mesh.vol > 0)

    def test_axis_cells_have_smallest_radius(self, mesh) -> None:
        assert mesh.r_c[0] < mesh.r_c[-1]

    def test_active_mask_nonempty(self, mesh) -> None:
        assert mesh.active.sum() > 100

    def test_wall_flag_on_outer_cells(self, mesh) -> None:
        # For every active column at least one wall cell must exist
        for j in range(mesh.nz):
            if mesh.active[:, j].any():
                assert mesh.is_wall[:, j].any()

    def test_tuyere_band_near_bottom(self, mesh) -> None:
        # Tuyere cells must be in the lowest 1/6 of the mesh
        rows = np.where(mesh.is_tuyere)[1]
        if len(rows) > 0:
            assert rows.max() < mesh.nz // 5

    def test_stretched_mesh_distinct(self) -> None:
        p1 = BFParameters2D(nr=20, nz=40, mesh_stretch_z=1.0)
        p2 = BFParameters2D(nr=20, nz=40, mesh_stretch_z=1.1)
        m1 = build_mesh(p1)
        m2 = build_mesh(p2)
        assert not np.allclose(m1.dz, m2.dz)

    def test_wall_radius_monotonic_above_bosh(self) -> None:
        """In the stack the wall radius should not grow."""
        p = BFParameters2D(nr=20, nz=60)
        m = build_mesh(p)
        # Stack starts after hearth+bosh+belly
        stack_start = p.hearth_height + p.bosh_height + p.belly_height
        stack_rows = m.z_c >= stack_start
        r_stack = m.r_wall[stack_rows]
        if len(r_stack) > 2:
            # non-increasing
            assert np.all(np.diff(r_stack) <= 1e-9)
