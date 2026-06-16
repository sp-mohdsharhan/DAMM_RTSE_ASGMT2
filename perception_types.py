"""Shared type contracts for perception data passed between controller modules."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class DetectionInfo(TypedDict, total=False):
    """Common contour/object data returned by image_detection.py."""

    bbox: tuple[int, int, int, int]
    circle: tuple[int, int, int]
    area_frac: float
    centroid_x_norm: float
    centroid_y: float
    distance: float
    color: Literal['red', 'green', 'yellow']


class FrontPerception(TypedDict, total=False):
    """Front-camera perception bundle consumed by control and display code."""

    frame: Any
    roi_y0: int
    roi_x0: int
    road_mask: Any
    red: Optional[DetectionInfo]
    green: Optional[DetectionInfo]
    yellow: Optional[DetectionInfo]
    orbs: list[DetectionInfo]
    nearest: Optional[DetectionInfo]
    police: Optional[DetectionInfo]


class RearObjectState(TypedDict):
    """Rear-camera object state for the chasing-car challenge."""

    info: Optional[DetectionInfo]
    growing: bool


class RearPerception(TypedDict, total=False):
    """Rear-camera perception bundle consumed by control and display code."""

    frame: Any
    other_car: RearObjectState


class CurveDebug(TypedDict, total=False):
    """Lane-curve debug data returned by detect_lane_curve()."""

    warp: Any
    mask: Any
    left_pts: tuple[Any, Any]
    right_pts: tuple[Any, Any]
    left_rects: list[tuple[int, int, int, int]]
    right_rects: list[tuple[int, int, int, int]]
    left_base: Optional[int]
    right_base: Optional[int]
    curve_bias: float
    lane_center_norm: Optional[float]


class HudData(TypedDict):
    """HUD values passed to the display overlay."""

    target: float
    eff: float
    police: bool
    events: list[str]
    str: float
    acc: float


PerceptionResult = tuple[
    Optional[FrontPerception],
    Optional[RearPerception],
    Optional[float],
    Optional[CurveDebug],
    float,
    bool,
    bool,
]
