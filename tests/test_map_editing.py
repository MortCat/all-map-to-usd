"""Regression checks for deliberate map changes and their exported geometry."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from pxr import Sdf, Usd, UsdPhysics

from allmaptousd.geometry.editing import MapStroke
from allmaptousd.models import BuildConfig, CanonicalMap, VehicleProxyConfig, as_points
from allmaptousd.pipeline import MapSession
from allmaptousd.web import server


def covers(boxes: np.ndarray, x: float, y: float) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)


class MapEditingTests(unittest.TestCase):
    def test_history_restores_geometry_and_new_stroke_clears_redo(self) -> None:
        state = server.AppState()
        session = state.session
        session.use(CanonicalMap())
        config = session.config
        brush = MapStroke(tool="brush", width=0.05, points=[(1.025, 1.025)])
        eraser = brush.model_copy(update={"tool": "eraser"})
        with patch.object(server, "state", state):
            def step(direction):
                request = server.EditHistoryRequest(revision=session.edit_revision, config=config)
                return server.step_map_history(direction, request)

            session.apply_edits([brush, eraser], config)
            self.assertFalse(covers(session.built.obstacle_boxes, 1.025, 1.025))
            undone = step("undo")
            self.assertTrue(covers(undone["obstacle_boxes"], 1.025, 1.025))
            self.assertTrue(undone["can_redo"])
            self.assertFalse(covers(step("redo")["obstacle_boxes"], 1.025, 1.025))
            step("undo")
            initial = step("undo")
            self.assertFalse(initial["obstacle_boxes"])
            self.assertFalse(initial["can_undo"])
            self.assertTrue(initial["can_redo"])
            session.apply_edits([MapStroke(tool="brush", width=0.05, points=[(2, 2)])], config)
            self.assertFalse(state.payload()["can_redo"])
            with self.assertRaises(HTTPException) as error:
                step("redo")
            self.assertEqual(error.exception.status_code, 422)

    def test_empty_map_edits_survive_rebuild_and_usd_export(self) -> None:
        state = server.AppState()
        config = BuildConfig()
        with patch.object(server, "state", state):
            initial = server.create_empty_map(
                server.EmptyMapRequest(width=12, height=8, resolution=0.05, config=config)
            )
            self.assertEqual(initial["bounds"], {"min_x": 0, "min_y": 0, "max_x": 12, "max_y": 8})
            self.assertEqual(initial["config"]["vehicle_proxy"]["pos_x"], 6)
            request = server.MapEditRequest(
                revision=initial["edit_revision"],
                strokes=[
                    MapStroke(tool="brush", width=0.05, points=[(1.025, 1.025)]),
                    MapStroke(tool="rectangle", width=0.05, points=[(2.025, 2.025), (3.025, 3.025)]),
                    MapStroke(tool="rectangle", width=0.05, points=[(5, 5), (7, 7)], filled=True),
                ],
                config=config,
            )
            edited = server.apply_map_edits(request)
            self.assertTrue(covers(edited["obstacle_boxes"], 1.025, 1.025))
            self.assertTrue(covers(edited["obstacle_boxes"], 2.025, 2.5))
            self.assertTrue(covers(edited["obstacle_boxes"], 2.5, 2.5))
            self.assertTrue(covers(edited["obstacle_boxes"], 6, 6))
            with self.assertRaises(HTTPException) as error:
                server.apply_map_edits(request)
            self.assertEqual(error.exception.status_code, 409)

            rebuilt = server.rebuild_map(BuildConfig(grid_resolution=0.1), edited["edit_revision"])
            self.assertEqual(rebuilt["obstacle_boxes"], edited["obstacle_boxes"])
            self.assertEqual(rebuilt["bounds"], initial["bounds"])
            outlines = server.rebuild_map(BuildConfig(fill_solid_obstacles=False), edited["edit_revision"])
            self.assertFalse(covers(outlines["obstacle_boxes"], 2.5, 2.5))
            self.assertTrue(covers(outlines["obstacle_boxes"], 6, 6))
            layer = Sdf.Layer.CreateAnonymous()
            self.assertTrue(layer.ImportFromString(state.session.export_bytes(".usda").decode()))
            stage = Usd.Stage.Open(layer)
            point_obstacle = stage.GetPrimAtPath("/World/Map/Obstacles/Wall_0000")
            self.assertTrue(point_obstacle.HasAPI(UsdPhysics.CollisionAPI))

    def test_eraser_cuts_imported_points_and_cad_segments(self) -> None:
        config = BuildConfig(dilate_cells=1)
        xs = np.arange(0.025, 5, 0.05)
        raw = CanonicalMap(
            points=as_points([(x, 1.025) for x in xs]),
            segments=as_points([[(0.025, 2.025), (4.975, 2.025)]], (2, 2)),
            segment_layers=np.array([0]),
            layer_names=["Walls"],
        )
        session = MapSession()
        session.use(raw, config=config)
        stroke = MapStroke(tool="eraser", width=0.5, points=[(2.5, 0.75), (2.5, 2.25)])
        contour = session.apply_edits([stroke], config)
        aabb = session.rebuild(config.model_copy(update={"reconstruction_mode": "aabb"}))
        for built in (contour, aabb):
            for y in (1.025, 2.025):
                self.assertTrue(covers(built.obstacle_boxes, 1, y))
                self.assertFalse(covers(built.obstacle_boxes, 2.5, y))
            self.assertLess(len(built.points), len(raw.points))
            self.assertEqual(built.layer_names, ["Walls"])
        self.assertEqual(len(raw.points), len(xs))

    def test_drawn_loop_fills_up_to_limit_without_resealing_erased_gaps(self) -> None:
        session = MapSession()
        config = BuildConfig(grid_resolution=0.1, solid_fill_max_area_m2=8.41)
        session.use(CanonicalMap(), config=config)
        boundary = [(1.05, 1.05), (4.05, 1.05), (4.05, 4.05), (1.05, 4.05), (1.05, 1.05)]
        strokes = [MapStroke(tool="line", width=0.1, points=[a, b]) for a, b in zip(boundary, boundary[1:])]
        built = session.apply_edits(strokes, config)
        self.assertTrue(covers(built.obstacle_boxes, 2, 2))
        self.assertFalse(any(np.allclose(point, [2.05, 2.05]) for point in built.drawn_points))

        smaller = session.rebuild(config.model_copy(update={"solid_fill_max_area_m2": 8.4}))
        self.assertFalse(covers(smaller.obstacle_boxes, 2, 2))
        disabled = session.rebuild(config.model_copy(update={"fill_solid_obstacles": False}))
        self.assertFalse(covers(disabled.obstacle_boxes, 2, 2))

        interior = MapStroke(tool="eraser", width=0.1, points=[(2.05, 2.05)])
        erased = session.apply_edits([interior], config)
        self.assertFalse(covers(erased.obstacle_boxes, 2.05, 2.05))
        self.assertTrue(covers(erased.obstacle_boxes, 3, 3))
        cut = MapStroke(tool="eraser", width=0.1, points=[(2.05, 1.05)])
        opened = session.apply_edits([cut], config)
        self.assertFalse(covers(opened.obstacle_boxes, 3, 3))
        self.assertTrue(covers(opened.obstacle_boxes, 1.05, 2.05))
        self.assertTrue(covers(session.step_edit_history(redo=False, config=config).obstacle_boxes, 3, 3))

    def test_drawn_line_closes_imported_cad_walls_and_retains_layer_heights(self) -> None:
        raw = CanonicalMap(
            segments=as_points([
                [(1.05, 1.05), (1.05, 4.05)],
                [(1.05, 4.05), (4.05, 4.05)],
                [(4.05, 4.05), (4.05, 1.05)],
            ], (2, 2)),
            segment_layers=np.zeros(3, dtype=int),
            layer_names=["Walls"],
        )
        config = BuildConfig(
            reconstruction_mode="aabb", grid_resolution=0.1, morph_close_cells=0,
            layer_heights={"Walls": 1.4}, solid_fill_max_area_m2=12,
        )
        session = MapSession()
        original = session.use(raw, config=config)
        self.assertFalse(covers(original.obstacle_boxes, 2.5, 2.5))

        line = MapStroke(tool="line", width=0.1, points=[(1.05, 1.05), (4.05, 1.05)])
        closed = session.apply_edits([line], config)
        self.assertTrue(covers(closed.obstacle_boxes, 2.5, 2.5))
        imported = np.asarray(closed.obstacle_box_heights) == 1.4
        self.assertTrue(covers(closed.obstacle_boxes[imported], 1.05, 3))

        excluded = session.rebuild(config.model_copy(update={"excluded_layers": ["Walls"]}))
        self.assertFalse(covers(excluded.obstacle_boxes, 2.5, 2.5))
        cut = MapStroke(tool="eraser", width=0.5, points=[(2.55, 1.05)])
        opened = session.apply_edits([cut], config)
        self.assertFalse(covers(opened.obstacle_boxes, 2.5, 2.5))
        self.assertTrue(covers(session.step_edit_history(redo=False, config=config).obstacle_boxes, 2.5, 2.5))

    def test_narrow_mouth_uses_vehicle_span_and_keeps_wide_entrance_open(self) -> None:
        config = BuildConfig(grid_resolution=0.05)
        for gap, expect_fill in ((0.20, True), (1.20, False)):
            with self.subTest(gap=gap):
                session = MapSession()
                session.use(CanonicalMap(), config=config)
                points = [
                    (2 - gap / 2, 4), (0, 4), (0, 0), (4, 0), (4, 4), (2 + gap / 2, 4),
                ]
                stroke = MapStroke(tool="brush", width=0.05, points=points)
                self.assertEqual(covers(session.apply_edits([stroke], config).obstacle_boxes, 2, 2), expect_fill)
                narrow_agv = config.model_copy(update={"vehicle_proxy": VehicleProxyConfig(width=0.10)})
                self.assertFalse(covers(session.rebuild(narrow_agv).obstacle_boxes, 2, 2))
                limited = config.model_copy(update={"solid_fill_max_area_m2": 10})
                self.assertFalse(covers(session.rebuild(limited).obstacle_boxes, 2, 2))
                if expect_fill:
                    cut = MapStroke(tool="eraser", width=0.25, points=[(2, 4)])
                    self.assertFalse(covers(session.apply_edits([cut], config).obstacle_boxes, 2, 2))

    def test_failed_edit_keeps_previous_map_and_revision(self) -> None:
        session = MapSession()
        session.use(CanonicalMap())
        session.apply_edits([MapStroke(tool="brush", width=0.05, points=[(1, 1)])], session.config)
        previous = session.built
        revision = session.edit_revision
        oversized = MapStroke(tool="line", width=0.05, points=[(0, 0), (1000, 1000)])
        with self.assertRaisesRegex(ValueError, "too many grid cells"):
            session.apply_edits([oversized], session.config)
        self.assertIs(session.built, previous)
        self.assertEqual(session.edit_revision, revision)
        self.assertTrue(covers(session.rebuild().obstacle_boxes, 1.025, 1.025))
