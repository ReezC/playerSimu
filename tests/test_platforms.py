import unittest

import numpy as np

from perception.platforms import PlatformDetector, PlayerMotionTracker
from perception.world_state import Platform, Player


class PlatformPerceptionTests(unittest.TestCase):
    def test_detector_returns_top_edge_for_horizontal_grass(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        # HSV(60, 220, 180) -> BGR: deliberately inside the configured grass range.
        import cv2
        color = cv2.cvtColor(np.uint8([[[60, 220, 180]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame[100:112, 35:250] = color
        # 平台顶边下面是棕色土层；纯绿色横条不应被误判为可站立平台。
        soil = cv2.cvtColor(np.uint8([[[20, 180, 110]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame[112:140, 35:250] = soil
        found = PlatformDetector(min_width=70, roi_top=0, roi_bottom=1).detect(frame)
        self.assertEqual(len(found), 1)
        x1, x2, y, _ = found[0]
        self.assertLessEqual(x1, 35)
        self.assertGreaterEqual(x2, 250)
        self.assertLessEqual(abs(y - 100), 2)

    def test_falling_player_gets_landing_prediction(self):
        p = Player(x=100, bottom=100, w=20, found=True)
        tracker = PlayerMotionTracker()
        tracker.update(p, [], ts=1.0)
        p.x, p.bottom = 110, 120
        platform = Platform(1, 80, 240, 150)
        hint = tracker.update(p, [platform], ts=1.1)
        self.assertTrue(p.falling)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.target_platform_id, 1)


if __name__ == "__main__":
    unittest.main()
