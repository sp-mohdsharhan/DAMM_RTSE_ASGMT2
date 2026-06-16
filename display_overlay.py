"""OpenCV display helpers for the SpeedTrials2D controller."""

import cv2

from image_detection import draw_lane_curve_debug, draw_overlay
from perception_types import CurveDebug, FrontPerception, HudData, RearPerception


def show_perception(
    front_per: FrontPerception | None,
    rear_per: RearPerception | None,
    lane_offset: float | None,
    curve_dbg: CurveDebug | None,
    hud: HudData,
) -> None:
    """Render perception and lane-curve debug windows.

    Display is best-effort so UI failures do not stop the real-time tasks.
    """
    try:
        overlay = draw_overlay(front_per, rear_per, lane_offset, hud)
        cv2.imshow("Perception", overlay)

        curve_panel = draw_lane_curve_debug(curve_dbg)
        if curve_panel is not None:
            cv2.imshow("Lane Curve", curve_panel)

        cv2.waitKey(1)
    except Exception:
        pass
