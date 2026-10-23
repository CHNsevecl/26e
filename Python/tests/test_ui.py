"""Headless checks for UI state transitions and the exact image selection."""
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.ui import UI


class UIStateTests(unittest.TestCase):
    def setUp(self):
        self.ui = UI.__new__(UI)
        self.ui.controller = SimpleNamespace(events=queue.Queue(), busy=False,
                                             stop=Mock(), start=Mock(return_value=True),
                                             send=Mock(return_value=True))
        self.ui.root = Mock()
        self.ui.status = Mock()
        self.ui.result_path = Mock()
        self.ui.warped_panel = Mock()
        self.ui.machine_panel = Mock()
        self.ui.task1_button = Mock()
        self.ui.task2_button = Mock()
        self.ui.stop_button = Mock()
        self.ui.send_button = Mock()
        self.ui.append_log = Mock()
        self.ui.closing = False
        self.ui.state = "recognizing"
        self.ui.displayed_result = None
        self.result = SimpleNamespace(task="task2", directory=Path("/session/attempt_0002"),
                                      warped=Path("/session/attempt_0002/a4_warped.jpg"),
                                      machine=Path("/session/attempt_0002/task2_machine.png"))

    def test_success_displays_only_warp_and_current_machine_image(self):
        self.ui.controller.events.put(("ready", self.result))
        self.ui.poll()
        self.ui.warped_panel.load.assert_called_once_with(self.result.warped)
        self.ui.machine_panel.load.assert_called_once_with(self.result.machine)
        self.ui.send_button.configure.assert_called_with(state="normal")
        self.ui.controller.send.assert_not_called()

    def test_stop_discards_a_success_already_in_the_queue(self):
        self.ui.controller.events.put(("ready", self.result))
        self.ui.stop()
        self.ui.poll()
        self.ui.warped_panel.load.assert_not_called()
        self.ui.machine_panel.load.assert_not_called()
        self.ui.send_button.configure.assert_called_with(state="disabled")
        self.ui.controller.send.assert_not_called()

    def test_new_task_clears_images_and_disables_send(self):
        self.ui.state = "ready"
        self.ui.displayed_result = self.result
        self.ui.start("task1")
        self.ui.warped_panel.clear.assert_called_once_with("a4_warped.jpg")
        self.ui.machine_panel.clear.assert_called_once_with("lines_machine.png")
        self.ui.send_button.configure.assert_called_with(state="disabled")
        self.assertIsNone(self.ui.displayed_result)

    def test_send_requires_ready_and_disables_repeat_clicks(self):
        self.ui.send()
        self.ui.controller.send.assert_not_called()
        self.ui.state = "ready"
        self.ui.send()
        self.ui.send()
        self.ui.controller.send.assert_called_once()
        self.ui.send_button.configure.assert_called_with(state="disabled")
        self.ui.stop_button.configure.assert_called_with(state="disabled")


if __name__ == "__main__":
    unittest.main()
