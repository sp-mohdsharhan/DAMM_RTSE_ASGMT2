"""OpenCV display helpers for the SpeedTrials2D controller."""

import cv2

from image_detection import draw_lane_curve_debug, draw_overlay
from perception_types import CurveDebug, FrontPerception, HudData, RearPerception


ORB_COLOURS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
}


def _draw_detection_circle(img, info, label, colour, y_offset=0, thick=2):
    if info is None:
        return

    cx, cy, radius = info['circle']
    cy += y_offset
    cv2.circle(img, (cx, cy), radius, colour, thick)
    cv2.circle(img, (cx, cy), 3, colour, -1)
    cv2.putText(
        img,
        label,
        (max(0, cx - radius), max(14, cy - radius - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        colour,
        1,
    )


def _draw_front_camera_detections(
    front_per: FrontPerception | None,
    lane_offset: float | None,
    hud: HudData,
) -> None:
    if not front_per:
        return

    frame = front_per['frame'].copy()
    y0 = front_per.get('roi_y0', 0)
    x0 = front_per.get('roi_x0', 0)
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (x0, y0), (w - 1 - x0, h - 1), (80, 80, 80), 1)

    for orb in front_per.get('orbs', []):
        colour_name = orb.get('color', '?')
        colour = ORB_COLOURS.get(colour_name, (255, 255, 255))
        label = f"{colour_name[:1].upper()} d={orb.get('distance', 0):.0f}"
        _draw_detection_circle(frame, orb, label, colour, y0)

    nearest = front_per.get('nearest')
    if nearest is not None:
        ncx, ncy, nr = nearest['circle']
        cv2.circle(frame, (ncx, ncy + y0), nr + 5, (255, 255, 255), 2)

    _draw_detection_circle(frame, front_per.get('police'), 'POLICE', (255, 0, 0), y0, 3)

    if lane_offset is not None:
        lane_x = int(w / 2 + lane_offset * w / 2)
        cv2.line(frame, (w // 2, h - 5), (lane_x, h - 30), (255, 255, 255), 2)

    events = ','.join(hud.get('events', [])[:3])
    cv2.putText(
        frame,
        f"orbs={len(front_per.get('orbs', []))} str={hud['str']:+.2f} acc={hud['acc']:+.2f}",
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    if events:
        cv2.putText(
            frame,
            events[:58],
            (5, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
        )

    cv2.imshow("Front Camera Detection", cv2.resize(frame, (640, 480)))


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
        _draw_front_camera_detections(front_per, lane_offset, hud)

        curve_panel = draw_lane_curve_debug(curve_dbg)
        if curve_panel is not None:
            cv2.imshow("Lane Curve", curve_panel)

        cv2.waitKey(1)
    except Exception:
        pass
