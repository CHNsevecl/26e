"""End-to-end recognition and UART protocol checks using a pseudo-terminal."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import unittest

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
EXE = PROJECT / "build" / "ps"


@unittest.skipUnless(EXE.exists(), "build the C++ ps executable first")
class CppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.master, self.slave = pty.openpty()
        self.serial = os.ttyname(self.slave)

    def tearDown(self):
        os.close(self.master)
        os.close(self.slave)
        self.temp.cleanup()

    def run_cpp(self, *arguments):
        return subprocess.run([str(EXE), *map(str, arguments), "--serial-device", self.serial],
                              capture_output=True, text=True, timeout=45)

    def received(self):
        data = b""
        while select.select([self.master], [], [], 0.05)[0]:
            data += os.read(self.master, 65536)
        return data

    def check_recognition(self, task, image):
        source = image if isinstance(image, Path) else self.directory / "scene.png"
        output = self.directory / "output"
        if not isinstance(image, Path):
            cv2.imwrite(str(source), image)
        result = self.run_cpp(f"--{task}", "--image", source, "--output-dir", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.received(), b"", "recognition wrote UART bytes")
        self.assertIsNotNone(cv2.imread(str(output / "a4_warped.jpg")))
        self.assertIsNotNone(cv2.imread(str(output / ("task2_machine.png" if task == "task2" else "lines_machine.png"))))
        packet = json.loads((output / "machine_targets.json").read_text())
        self.assertEqual(packet["task"], task)
        if task == "task2":
            warped = cv2.imread(str(output / "a4_warped.jpg"))
            machine = cv2.imread(str(output / "task2_machine.png"))
            rows = np.flatnonzero(np.any(machine != 0, axis=(1, 2)))
            self.assertGreater(len(rows), 0)
            self.assertLess(rows[-1], warped.shape[0] / 2 - 40,
                            "card texture extends below the A4 center line")
            self.assertTrue(all(target["target_pixels"][1] < warped.shape[0] / 2 - 40
                                for target in packet["targets"]),
                            "saved machine targets were not moved into the A4 upper half")
        result = self.run_cpp("--send-result", output / "machine_targets.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = ""
        for target in packet["targets"]:
            ox, oy = target["origin_pixels"]
            tx, ty = target["target_pixels"]
            angle = target["rotation_degrees"]
            index = target["source_index"]
            expected += f"{index} 0 x:{oy/4:.1f} y:{ox/4:.1f} r:{angle:.1f};\n"
            expected += f"{index} 1 x:{ty/4:.1f} y:{tx/4:.1f};\n"
        self.assertEqual(self.received(), (expected + "finish\n").encode())
        # Reuse the same output directory with a failed capture; old targets must disappear.
        failed = self.run_cpp(f"--{task}", "--image", self.directory / "missing.png", "--output-dir", output)
        self.assertEqual(failed.returncode, 1)
        self.assertFalse((output / "machine_targets.json").exists())
        self.assertEqual(self.received(), b"")

    def test_task1_recognition_then_manual_send(self):
        scene = np.full((1400, 1000, 3), 180, np.uint8)
        cv2.rectangle(scene, (80, 100), (920, 1288), (20, 20, 20), -1)
        cv2.rectangle(scene, (180, 280), (340, 560), (245, 245, 245), -1)
        cv2.rectangle(scene, (500, 700), (660, 980), (245, 245, 245), -1)
        self.check_recognition("task1", scene)

    def test_task2_recognition_then_manual_send(self):
        # The existing fixture is already warped. Add a desk border for A4 detection.
        warped = cv2.imread(str(PROJECT / "tests/data/card_three_uneven_light.jpeg"))
        self.assertIsNotNone(warped)
        scene = cv2.copyMakeBorder(warped, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(180, 180, 180))
        self.check_recognition("task2", scene)

    def test_sparse_spade_ace_scene_and_manual_send(self):
        # Full camera scene: includes A4 detection and edge noise introduced by
        # JPEG, rather than testing only the already-warped recognition input.
        scene = cv2.imread(str(PROJECT / "tests/data/card_four_spade_ace_scene.jpeg"))
        self.assertIsNotNone(scene)
        self.check_recognition("task2", scene)
        report = cv2.FileStorage(str(self.directory / "output/task2_result.yml"), cv2.FILE_STORAGE_READ)
        try:
            self.assertEqual(int(report.getNode("rounded_corner_prior").real()), 1)
            sources = report.getNode("source_polygons")
            self.assertEqual(sources.size(), 4)
            for index in range(sources.size()):
                source = sources.at(index)
                self.assertEqual(source.getNode("points").size(), 8,
                                 "a small edge spur became an extra cut")
                mask = cv2.imread(str(self.directory / f"output/task2_piece_{index}_mask.png"), 0)
                moments = cv2.moments(mask, binaryImage=True)
                bounds = [source.getNode("bounds").at(k).real() for k in range(4)]
                center = [source.getNode("center_gravity").at(k).real() for k in (0, 1)]
                expected = (moments["m10"] / moments["m00"] + bounds[0],
                            moments["m01"] / moments["m00"] + bounds[1])
                self.assertLess(np.linalg.norm(np.array(center) - expected), 0.25,
                                "geometry cleanup changed the physical pickup centroid")
            target = [report.getNode("target").at(k).real() for k in range(4)]
            middle = (target[0] + target[2] / 2, target[1] + target[3] / 2)
            placements = report.getNode("placements")
            centers = {int(placements.at(i).getNode("source_index").real()):
                       [placements.at(i).getNode("center_gravity").at(k).real() for k in (0, 1)]
                       for i in range(placements.size())}
            self.assertTrue(all((centers[1][k] - middle[k]) * (centers[3][k] - middle[k]) < 0
                                for k in (0, 1)), "both A indices were placed on the same side")
        finally:
            report.release()

    def test_club_six_scene_indices_pattern_and_manual_send(self):
        scene = cv2.imread(str(PROJECT / "tests/data/card_four_club_six_scene.jpeg"))
        self.assertIsNotNone(scene)
        self.check_recognition("task2", scene)
        report = cv2.FileStorage(str(self.directory / "output/task2_result.yml"), cv2.FILE_STORAGE_READ)
        try:
            self.assertEqual(int(report.getNode("corner_indices_verified").real()), 1)
            self.assertEqual(int(report.getNode("layout_accepted").real()), 1)
            self.assertGreater(report.getNode("face_pattern_match").real(), 0.7)
        finally:
            report.release()

    def test_split_straight_cut_scene_with_extended_budget_and_manual_send(self):
        # Consecutive captures: the second also has faint strokes in one index.
        for fixture in ("card_club_six_split_cut_scene.jpeg", "card_club_six_split_cut_repeat_scene.jpeg"):
            with self.subTest(fixture=fixture):
                scene = cv2.imread(str(PROJECT / "tests/data" / fixture))
                self.assertIsNotNone(scene)
                self.check_recognition("task2", scene)
                report = cv2.FileStorage(str(self.directory / "output/task2_result.yml"), cv2.FILE_STORAGE_READ)
                try:
                    self.assertEqual(report.getNode("solver_time_limit_seconds").real(), 30)
                    self.assertEqual(int(report.getNode("corner_indices_verified").real()), 1)
                    sources = report.getNode("source_polygons")
                    self.assertEqual(sources.size(), 4)
                    for index in range(sources.size()):
                        self.assertEqual(sources.at(index).getNode("points").size(), 8,
                                         "a printed cut-edge club became extra polygon edges")
                finally:
                    report.release()

    def test_continuous_club_six_capture_series(self):
        # Use camera JPEGs directly, including both false index rejections and
        # short-cut contour failures; each case verifies manual UART separation.
        fixtures = sorted((PROJECT / "tests/data/card_club_six_capture_series").glob("*.jpeg"))
        self.assertEqual(len(fixtures), 7)
        for fixture in fixtures:
            with self.subTest(capture=fixture.name):
                self.check_recognition("task2", fixture)
                report = cv2.FileStorage(str(self.directory / "output/task2_result.yml"), cv2.FILE_STORAGE_READ)
                try:
                    self.assertEqual(int(report.getNode("corner_indices_verified").real()), 1)
                    self.assertGreaterEqual(report.getNode("corner_rank_match").real(), 0.65)
                    self.assertGreaterEqual(report.getNode("corner_suit_match").real(), 0.65)
                    sources = report.getNode("source_polygons")
                    self.assertEqual(sources.size(), 4)
                    for index in range(sources.size()):
                        self.assertEqual(sources.at(index).getNode("points").size(), 8)
                finally:
                    report.release()

    def test_club_queen_consecutive_captures_and_manual_send(self):
        # The first capture originally timed out. Replaying its camera JPEG
        # also joins the Q tail to a neighboring court-frame line.
        # Captures 3/4 cover an automatic-size failure; 5/6 cover a wrongly
        # reversed unindexed quarter and the subsequent successful recapture.
        for index in (1, 2, 3, 4, 5, 6):
            with self.subTest(capture=index):
                self.check_recognition("task2", PROJECT / f"tests/data/card_club_queen_capture_{index:02d}.jpeg")
                report = cv2.FileStorage(str(self.directory / "output/task2_result.yml"), cv2.FILE_STORAGE_READ)
                try:
                    self.assertEqual(int(report.getNode("corner_indices_verified").real()), 1)
                    self.assertGreaterEqual(report.getNode("corner_rank_match").real(), 0.65)
                    self.assertGreaterEqual(report.getNode("corner_suit_match").real(), 0.65)
                    self.assertGreater(report.getNode("face_pattern_match").real(), 0.8)
                    sources = report.getNode("source_polygons")
                    self.assertEqual(sources.size(), 4)
                    if index <= 2:
                        for piece in range(sources.size()):
                            self.assertEqual(sources.at(piece).getNode("points").size(), 8)
                finally:
                    report.release()

    def test_missing_opposite_index_cannot_create_send_targets(self):
        warped = np.full((1178, 830, 3), 25, np.uint8)
        cv2.rectangle(warped, (240, 300), (480, 660), (245, 245, 245), -1)
        cv2.putText(warped, "6", (254, 355), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        for point in ((266, 380), (260, 387), (272, 387)):
            cv2.circle(warped, point, 6, (0, 0, 0), -1)
        cv2.fillConvexPoly(warped, np.array(((266, 385), (262, 397), (270, 397))), (0, 0, 0))
        scene = cv2.copyMakeBorder(warped, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(180, 180, 180))
        source = self.directory / "one_index.png"
        output = self.directory / "output"
        cv2.imwrite(str(source), scene)
        result = self.run_cpp("--task2", "--image", source, "--output-dir", output)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("角标", result.stderr)
        self.assertFalse((output / "machine_targets.json").exists())
        self.assertEqual(self.received(), b"", "unverified card wrote UART bytes")
        report = cv2.FileStorage(str(output / "task2_result.yml"), cv2.FILE_STORAGE_READ)
        try:
            self.assertEqual(int(report.getNode("recognized").real()), 1)
            self.assertEqual(int(report.getNode("corner_indices_verified").real()), 0)
        finally:
            report.release()

    def test_invalid_second_target_sends_no_partial_batch(self):
        target = {"source_index": 0, "origin_pixels": [40, 80], "target_pixels": [120, 160], "rotation_degrees": 90}
        bad_target = dict(target, source_index=1, target_pixels=[-1, 20])
        path = self.directory / "invalid.json"
        path.write_text(json.dumps({"version": 1, "ready": 1, "task": "task1", "targets": [target, bad_target]}))
        result = self.run_cpp("--send-result", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.received(), b"")


if __name__ == "__main__":
    unittest.main()
