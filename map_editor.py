"""
map_editor.py — PyQt5 vector map editor for the planning system.

A QGraphicsScene shape editor for 2D maps, plus an interactive 3D authoring
mode. Shapes are stored as geometric primitives and written directly to
sim2d/demo_map.py (2D) or sim3d/demo_map3d.py (3D) — no grid-to-rectangle
merging.

Dependencies: PyQt5, matplotlib   (pip install -r requirements-map-editor.txt)

2D mode
-------
Layers:
  Obstacle  red    — rigid tissue (add_rectangular_obstacle / add_circular_obstacle)
  Tumor     yellow — injection targets (add_circular_tumor); healthy tissue (the
                     DP planner's damage objective) is every non-tumor,
                     non-obstacle cell, computed automatically via
                     World2D.compute_healthy_mask() — there is no separate
                     "protect this region" layer to draw.

Tools (toolbar + keyboard):
  V  Select / Move / Resize
  R  Draw rectangle obstacle
  C  Draw circle obstacle
  T  Draw tumor circle
  Del / Backspace  Delete selected shapes

Global shortcuts (2D mode only -- disabled in 3D mode, see below):
  S   Save map + launch main_demo.py
  Z   Undo
  Q   Quit
  Mouse-wheel   Zoom in/out

3D mode
-------
Add shapes (box/sphere/tumor) via the numeric-entry dialogs in the "Shapes"
list, then click a row to select it and click the 3D canvas to give it
keyboard focus. The selected shape is highlighted in cyan and can be
manipulated live:

  Left/Right/Up/Down    move selected shape along world x/y
  PageUp/PageDown        move selected shape along world z
  Mouse wheel            resize selected shape
  Q/W  A/S  Z/X          tilt selected box's yaw/pitch/roll (boxes only --
                         spheres/tumors are rotation-invariant)

Box rotation is real (affects collision + healthy-tissue damage geometry, not
just the preview) -- see sim3d.obstacles3d.rotation_matrix. The global 2D
shortcuts Q/S/Z are disabled while in 3D mode, since the canvas reuses those
keys for tilt; use the "Save + Run" button instead of the S shortcut there.
Key bindings/step sizes are the MOVE_STEP/ROTATE_STEP_DEG/RESIZE_FACTOR/
_KEY_ACTIONS constants near the top of this file.

Map state persisted in map_state.json / map_state_3d.json. Each save also
writes a versioned snapshot under maps/map_NNN/.
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup, QCheckBox,
    QToolBar, QAction, QActionGroup, QSplitter, QStatusBar,
    QGroupBox, QShortcut, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QGraphicsOpacityEffect,
)
from PyQt5.QtCore import (
    Qt, QRectF, QPointF, QLineF, pyqtSignal, QSettings, QPropertyAnimation, QEasingCurve,
)
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPalette, QKeySequence, QFont, QIcon,
)

import matplotlib
matplotlib.use("Qt5Agg")  # must be set before any other matplotlib import/use
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import map_editor_icons as icons
import map_editor_theme as theme

# ─────────────────────────────────────────────────────────────────────────────
# Paths & world constants
# ─────────────────────────────────────────────────────────────────────────────

_HERE          = pathlib.Path(__file__).parent
MAP_STATE_PATH = _HERE / "map_state.json"
DEMO_MAP_PATH  = _HERE / "sim2d" / "demo_map.py"
MAIN_DEMO_PATH = _HERE / "main_demo.py"
MAPS_DIR       = _HERE / "maps"

WORLD_X_MIN, WORLD_X_MAX = -5.0, 5.0
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 10.0
WORLD_X_SIZE = WORLD_X_MAX - WORLD_X_MIN
WORLD_Y_SIZE = WORLD_Y_MAX - WORLD_Y_MIN

# Scene space mirrors world space with y negated (see w2s/s2w below). Every
# shape-creation/move/resize path clamps into this box so nothing drawn in the
# editor can ever fall outside the region main_demo.py actually simulates.
SCENE_X_MIN, SCENE_X_MAX = WORLD_X_MIN, WORLD_X_MAX
SCENE_Y_MIN, SCENE_Y_MAX = -WORLD_Y_MAX, -WORLD_Y_MIN

# y nudged just clear of the y=0 boundary margin (robot radius 0.2) so the
# default start position isn't itself flagged as out-of-bounds.
DEFAULT_START_X, DEFAULT_START_Y = 0.0, 0.25

LAYER_OBSTACLE = "obstacle"
LAYER_TUMOR    = "tumor"

_LAYER_COLOR: Dict[str, QColor] = {
    LAYER_OBSTACLE: QColor("#e63946"),
    LAYER_TUMOR:    QColor("#ffd166"),
}
_LAYER_ALPHA = 140   # fill transparency

TOOL_SELECT = "select"
TOOL_RECT   = "rect"
TOOL_CIRCLE = "circle"
TOOL_TUMOR  = "tumor"

_HANDLE_HALF  = 0.13   # half-size of resize handle square (world units)
_MIN_SIZE     = 0.15   # minimum shape dimension
_PEN_WIDTH    = 0.05   # shape outline width


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers   (scene Y = −world Y so Y-axis points up on screen)
# ─────────────────────────────────────────────────────────────────────────────

def w2s(wx: float, wy: float) -> QPointF:
    """World → scene."""
    return QPointF(wx, -wy)


def s2w(sp: QPointF) -> Tuple[float, float]:
    """Scene point → (world_x, world_y)."""
    return float(sp.x()), float(-sp.y())


def _clamp_point_to_scene(p: QPointF) -> QPointF:
    return QPointF(
        min(max(p.x(), SCENE_X_MIN), SCENE_X_MAX),
        min(max(p.y(), SCENE_Y_MIN), SCENE_Y_MAX),
    )


def _clamp_rect_to_scene(r: QRectF) -> QRectF:
    """Clip a scene-space rect to the simulated region (keeps it non-degenerate
    even if the drag started/ended fully outside)."""
    return r.intersected(QRectF(SCENE_X_MIN, SCENE_Y_MIN, SCENE_X_MAX - SCENE_X_MIN, SCENE_Y_MAX - SCENE_Y_MIN))


def _max_radius_at(center: QPointF) -> float:
    """Largest radius a circle centred at `center` can have without any part
    crossing the scene bounds."""
    return max(
        _MIN_SIZE,
        min(
            center.x() - SCENE_X_MIN, SCENE_X_MAX - center.x(),
            center.y() - SCENE_Y_MIN, SCENE_Y_MAX - center.y(),
        ),
    )


def _clamp_item_pos(proposed: QPointF, local_rect: QRectF) -> QPointF:
    """Clamp a QGraphicsItem's proposed scene position so its (fixed-size)
    `local_rect` bounding box stays fully within the scene bounds."""
    return QPointF(
        min(max(proposed.x(), SCENE_X_MIN - local_rect.left()), SCENE_X_MAX - local_rect.right()),
        min(max(proposed.y(), SCENE_Y_MIN - local_rect.top()), SCENE_Y_MAX - local_rect.bottom()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class RectData:
    """Axis-aligned rectangle.  x, y = lower-left corner in world coords."""
    __slots__ = ("layer", "x", "y", "w", "h")

    def __init__(self, layer: str, x: float, y: float, w: float, h: float):
        self.layer = layer
        self.x, self.y, self.w, self.h = x, y, w, h

    def to_dict(self) -> dict:
        return {"type": "rect", "layer": self.layer,
                "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @staticmethod
    def from_dict(d: dict) -> "RectData":
        return RectData(d["layer"], d["x"], d["y"], d["w"], d["h"])


class CircleData:
    """Circle with centre (cx, cy) and radius r in world coords."""
    __slots__ = ("layer", "cx", "cy", "r")

    def __init__(self, layer: str, cx: float, cy: float, r: float):
        self.layer = layer
        self.cx, self.cy, self.r = cx, cy, r

    def to_dict(self) -> dict:
        return {"type": "circle", "layer": self.layer,
                "cx": self.cx, "cy": self.cy, "r": self.r}

    @staticmethod
    def from_dict(d: dict) -> "CircleData":
        return CircleData(d["layer"], d["cx"], d["cy"], d["r"])


ShapeData = Union[RectData, CircleData]


class MapState:
    def __init__(self) -> None:
        self.shapes:  List[ShapeData] = []
        self.start_x: float = DEFAULT_START_X
        self.start_y: float = DEFAULT_START_Y

    def to_dict(self) -> dict:
        return {
            "start":  {"x": self.start_x, "y": self.start_y},
            "shapes": [s.to_dict() for s in self.shapes],
        }

    @staticmethod
    def from_dict(d: dict) -> "MapState":
        ms = MapState()
        sx = float(d.get("start", {}).get("x", DEFAULT_START_X))
        sy = float(d.get("start", {}).get("y", DEFAULT_START_Y))
        # A start position saved under a previous world-bounds convention (or any
        # out-of-range value) would silently produce a 0-node roadmap -- fall back
        # to the default rather than loading a start point that can't work.
        margin = 0.25
        if not (
            WORLD_X_MIN + margin <= sx <= WORLD_X_MAX - margin
            and WORLD_Y_MIN + margin <= sy <= WORLD_Y_MAX - margin
        ):
            sx, sy = DEFAULT_START_X, DEFAULT_START_Y
        ms.start_x = sx
        ms.start_y = sy
        for sd in d.get("shapes", []):
            if sd["type"] == "rect":
                ms.shapes.append(RectData.from_dict(sd))
            elif sd["type"] == "circle":
                ms.shapes.append(CircleData.from_dict(sd))
        return ms

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: pathlib.Path) -> "MapState":
        try:
            return MapState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return MapState()


# ─────────────────────────────────────────────────────────────────────────────
# 3D data model + interactive canvas key bindings
# ─────────────────────────────────────────────────────────────────────────────

MAP_STATE_3D_PATH = _HERE / "map_state_3d.json"
DEMO_MAP_3D_PATH  = _HERE / "sim3d" / "demo_map3d.py"
MAIN_DEMO_3D_PATH = _HERE / "main_demo_3d.py"

DEFAULT_START_3D = (0.0, 0.0, 0.0)
# Index into sim3d.primitives3d.DIRECTIONS_3D for direction (1, 1, 1) -- matches
# main_demo_3d.py's hardcoded START_POSE default (direction_from_vector((1,1,1))).
DEFAULT_START_ORI_3D = 25
DEFAULT_BOX_SIZE_3D = 2.0
DEFAULT_RADIUS_3D   = 1.5

# Interactive 3D canvas: retune keys/step sizes here without touching the
# handler logic below. Movement/resize apply to any selected shape; rotation
# (yaw/pitch/roll) only applies to boxes (spheres/tumors are rotation-invariant).
MOVE_STEP       = 0.5    # world units per arrow/page key press
ROTATE_STEP_DEG = 15.0   # degrees per tilt key press
RESIZE_FACTOR   = 1.1    # multiplicative scale per scroll-wheel tick
ZOOM_FACTOR     = 1.25   # multiplicative camera-view scale per zoom button click
HEADING_ARROW_LENGTH = 3.0  # world units, start-heading arrow in the 3D preview

# matplotlib key name -> ("move", (dx, dy, dz)) | ("resize", factor) | ("rotate", axis, sign)
_KEY_ACTIONS = {
    "left":     ("move", (-MOVE_STEP, 0.0, 0.0)),
    "right":    ("move", (MOVE_STEP, 0.0, 0.0)),
    "up":       ("move", (0.0, MOVE_STEP, 0.0)),
    "down":     ("move", (0.0, -MOVE_STEP, 0.0)),
    "pageup":   ("move", (0.0, 0.0, MOVE_STEP)),
    "pagedown": ("move", (0.0, 0.0, -MOVE_STEP)),
    "q": ("rotate", "yaw",   -1),
    "w": ("rotate", "yaw",   +1),
    "a": ("rotate", "pitch", -1),
    "s": ("rotate", "pitch", +1),
    "z": ("rotate", "roll",  -1),
    "x": ("rotate", "roll",  +1),
}


class BoxData3D:
    """
    Box in world coords (sim3d convention: x, y, z = lower corner before
    rotation). yaw/pitch/roll (degrees) rotate the box about its own
    centroid -- see sim3d.obstacles3d.rotation_matrix for the convention.
    """
    __slots__ = ("x", "y", "z", "width", "depth", "height", "yaw_deg", "pitch_deg", "roll_deg")

    def __init__(
        self, x: float, y: float, z: float, width: float, depth: float, height: float,
        yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
    ):
        self.x, self.y, self.z = x, y, z
        self.width, self.depth, self.height = width, depth, height
        self.yaw_deg, self.pitch_deg, self.roll_deg = yaw_deg, pitch_deg, roll_deg

    def summary(self) -> str:
        rot = ""
        if self.yaw_deg or self.pitch_deg or self.roll_deg:
            rot = f"  rot=(y{self.yaw_deg:g},p{self.pitch_deg:g},r{self.roll_deg:g})"
        return (
            f"Box  pos=({self.x:g},{self.y:g},{self.z:g})  "
            f"size=({self.width:g}x{self.depth:g}x{self.height:g}){rot}"
        )

    def to_dict(self) -> dict:
        return {
            "type": "box", "x": self.x, "y": self.y, "z": self.z,
            "width": self.width, "depth": self.depth, "height": self.height,
            "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg,
        }

    @staticmethod
    def from_dict(d: dict) -> "BoxData3D":
        return BoxData3D(
            d["x"], d["y"], d["z"], d["width"], d["depth"], d["height"],
            d.get("yaw_deg", 0.0), d.get("pitch_deg", 0.0), d.get("roll_deg", 0.0),
        )


class SphereData3D:
    """Spherical obstacle: centre (cx, cy, cz) + radius."""
    __slots__ = ("cx", "cy", "cz", "r")

    def __init__(self, cx: float, cy: float, cz: float, r: float):
        self.cx, self.cy, self.cz, self.r = cx, cy, cz, r

    def summary(self) -> str:
        return f"Sphere  center=({self.cx:g},{self.cy:g},{self.cz:g})  r={self.r:g}"

    def to_dict(self) -> dict:
        return {"type": "sphere", "cx": self.cx, "cy": self.cy, "cz": self.cz, "r": self.r}

    @staticmethod
    def from_dict(d: dict) -> "SphereData3D":
        return SphereData3D(d["cx"], d["cy"], d["cz"], d["r"])


class TumorData3D:
    """Ball tumor: centre (cx, cy, cz) + radius."""
    __slots__ = ("cx", "cy", "cz", "r")

    def __init__(self, cx: float, cy: float, cz: float, r: float):
        self.cx, self.cy, self.cz, self.r = cx, cy, cz, r

    def summary(self) -> str:
        return f"Tumor  center=({self.cx:g},{self.cy:g},{self.cz:g})  r={self.r:g}"

    def to_dict(self) -> dict:
        return {"type": "tumor", "cx": self.cx, "cy": self.cy, "cz": self.cz, "r": self.r}

    @staticmethod
    def from_dict(d: dict) -> "TumorData3D":
        return TumorData3D(d["cx"], d["cy"], d["cz"], d["r"])


ShapeData3D = Union[BoxData3D, SphereData3D, TumorData3D]

_SHAPE3D_LOADERS = {"box": BoxData3D.from_dict, "sphere": SphereData3D.from_dict, "tumor": TumorData3D.from_dict}


class MapState3D:
    def __init__(self) -> None:
        self.shapes: List[ShapeData3D] = []
        self.start_x, self.start_y, self.start_z = DEFAULT_START_3D
        self.start_ori: int = DEFAULT_START_ORI_3D

    def to_dict(self) -> dict:
        return {
            "start": {
                "x": self.start_x, "y": self.start_y, "z": self.start_z,
                "ori": self.start_ori,
            },
            "shapes": [s.to_dict() for s in self.shapes],
        }

    @staticmethod
    def from_dict(d: dict) -> "MapState3D":
        ms = MapState3D()
        start = d.get("start", {})
        ms.start_x = float(start.get("x", DEFAULT_START_3D[0]))
        ms.start_y = float(start.get("y", DEFAULT_START_3D[1]))
        ms.start_z = float(start.get("z", DEFAULT_START_3D[2]))
        ori = int(start.get("ori", DEFAULT_START_ORI_3D))
        ms.start_ori = ori if 0 <= ori < 26 else DEFAULT_START_ORI_3D
        for sd in d.get("shapes", []):
            loader = _SHAPE3D_LOADERS.get(sd.get("type"))
            if loader:
                ms.shapes.append(loader(sd))
        return ms

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: pathlib.Path) -> "MapState3D":
        try:
            return MapState3D.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return MapState3D()


# ─────────────────────────────────────────────────────────────────────────────
# Resize handle
# ─────────────────────────────────────────────────────────────────────────────

class ResizeHandle(QGraphicsRectItem):
    """
    Small square child item.  Dragging it calls parent.apply_resize(role, scene_pos).
    Roles for rects: "nw" "ne" "sw" "se".  For circles: "e" (radius).
    """

    _CURSORS = {
        "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
        "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
        "e":  Qt.SizeHorCursor,
    }

    def __init__(self, role: str, parent: QGraphicsItem) -> None:
        h = _HANDLE_HALF
        super().__init__(-h, -h, 2 * h, 2 * h, parent)
        self.role      = role
        self._dragging = False
        self.setZValue(20)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor("#ffffff"), 0.02))
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setCursor(self._CURSORS.get(role, Qt.SizeAllCursor))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            # Prevent parent from moving while we resize
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self.parentItem().apply_resize(self.role, event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QBrush(QColor("#ffdd00")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QBrush(QColor("#ffffff")))
        super().hoverLeaveEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Rectangle shape item
# ─────────────────────────────────────────────────────────────────────────────

class MapRectItem(QGraphicsRectItem):
    """
    Movable + resizable rectangle.

    Position = scene centre of the rect.
    Rect in local space is always centred at the origin (initially);
    after a resize it may drift — _sync_to_data accounts for this by
    reading rect.center() in local space.
    """

    def __init__(self, data: RectData) -> None:
        super().__init__()
        self._data = data
        # Flags (in particular ItemSendsGeometryChanges) must be set before
        # _sync_from_data's setPos, or the bounds-clamping itemChange hook
        # below won't fire for a position loaded from stale/out-of-range data.
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges,
        )
        self._sync_from_data()
        self._apply_style()
        self.setAcceptHoverEvents(True)
        # Resize handles at four corners
        for role in ("nw", "ne", "sw", "se"):
            ResizeHandle(role, self)
        self._handles: List[ResizeHandle] = [
            c for c in self.childItems() if isinstance(c, ResizeHandle)
        ]
        self._reposition_handles()
        self._show_handles(False)

    # ── style ──────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        color = _LAYER_COLOR[self._data.layer]
        fill  = QColor(color)
        fill.setAlpha(_LAYER_ALPHA)
        self.setPen(QPen(color.darker(160), _PEN_WIDTH))
        self.setBrush(QBrush(fill))

    # ── data sync ─────────────────────────────────────────────────────────

    def _sync_from_data(self) -> None:
        d = self._data
        # setRect before setPos: the position-clamp itemChange hook reads
        # self.rect() to know the shape's margin, so it needs the real size
        # in place before the (possibly out-of-bounds) position is applied.
        self.setRect(-d.w / 2, -d.h / 2, d.w, d.h)
        self.setPos(d.x + d.w / 2, -(d.y + d.h / 2))

    def _sync_to_data(self) -> None:
        r   = self.rect()
        pos = self.pos()
        scene_cx = pos.x() + r.center().x()
        scene_cy = pos.y() + r.center().y()
        w = abs(float(r.width()))
        h = abs(float(r.height()))
        # world lower-left: x = scene_cx − w/2,  y = −scene_cy − h/2
        self._data.x = scene_cx - w / 2
        self._data.y = -scene_cy - h / 2
        self._data.w = w
        self._data.h = h

    # ── handles ───────────────────────────────────────────────────────────

    def _reposition_handles(self) -> None:
        r = self.rect()
        corners = {
            "nw": QPointF(r.left(),  r.top()),
            "ne": QPointF(r.right(), r.top()),
            "sw": QPointF(r.left(),  r.bottom()),
            "se": QPointF(r.right(), r.bottom()),
        }
        for h in self._handles:
            h.setPos(corners.get(h.role, QPointF()))

    def _show_handles(self, visible: bool) -> None:
        for h in self._handles:
            h.setVisible(visible)

    # ── resize ────────────────────────────────────────────────────────────

    def apply_resize(self, role: str, scene_pos: QPointF) -> None:
        # Clamp the dragged corner itself; since the opposite (fixed) corner is
        # already in-bounds, the resulting axis-aligned rect stays in-bounds too.
        local = self.mapFromScene(_clamp_point_to_scene(scene_pos))
        r = self.rect()
        if role == "nw":
            new_r = QRectF(local, r.bottomRight()).normalized()
        elif role == "ne":
            new_r = QRectF(QPointF(r.left(), local.y()),
                           QPointF(local.x(), r.bottom())).normalized()
        elif role == "sw":
            new_r = QRectF(QPointF(local.x(), r.top()),
                           QPointF(r.right(), local.y())).normalized()
        elif role == "se":
            new_r = QRectF(r.topLeft(), local).normalized()
        else:
            return
        if new_r.width() < _MIN_SIZE or new_r.height() < _MIN_SIZE:
            return
        self.setRect(new_r)
        self._reposition_handles()
        self._sync_to_data()

    # ── Qt overrides ──────────────────────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return _clamp_item_pos(value, self.rect())
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._show_handles(bool(value))
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._sync_to_data()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._sync_to_data()


# ─────────────────────────────────────────────────────────────────────────────
# Circle shape item
# ─────────────────────────────────────────────────────────────────────────────

class MapCircleItem(QGraphicsEllipseItem):
    """
    Movable + resizable circle.

    Position = scene centre.  Bounding rect in local space: (−r, −r, 2r, 2r).
    Single "e" handle controls radius.
    """

    def __init__(self, data: CircleData) -> None:
        super().__init__()
        self._data = data
        # Flags (in particular ItemSendsGeometryChanges) must be set before
        # _sync_from_data's setPos, or the bounds-clamping itemChange hook
        # below won't fire for a position loaded from stale/out-of-range data.
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges,
        )
        self._sync_from_data()
        self._apply_style()
        self.setAcceptHoverEvents(True)
        self._handle = ResizeHandle("e", self)
        self._reposition_handle()
        self._handle.setVisible(False)

    # ── style ──────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        color = _LAYER_COLOR[self._data.layer]
        fill  = QColor(color)
        fill.setAlpha(_LAYER_ALPHA)
        self.setPen(QPen(color.darker(160), _PEN_WIDTH))
        self.setBrush(QBrush(fill))

    # ── data sync ─────────────────────────────────────────────────────────

    def _sync_from_data(self) -> None:
        r = self._data.r
        # setRect before setPos: the position-clamp itemChange hook reads
        # self.rect() to know the shape's margin, so it needs the real size
        # in place before the (possibly out-of-bounds) position is applied.
        self.setRect(-r, -r, 2 * r, 2 * r)
        self.setPos(self._data.cx, -self._data.cy)

    def _sync_to_data(self) -> None:
        pos = self.pos()
        r   = self.rect()
        self._data.cx = float(pos.x())
        self._data.cy = float(-pos.y())
        self._data.r  = float(r.width() / 2)

    # ── handle ────────────────────────────────────────────────────────────

    def _reposition_handle(self) -> None:
        self._handle.setPos(self._data.r, 0.0)

    def apply_resize(self, role: str, scene_pos: QPointF) -> None:
        local  = self.mapFromScene(scene_pos)
        desired_r = max(_MIN_SIZE, math.hypot(local.x(), local.y()))
        new_r = min(desired_r, _max_radius_at(self.pos()))
        self.setRect(-new_r, -new_r, 2 * new_r, 2 * new_r)
        self._handle.setPos(new_r, 0.0)
        self._sync_to_data()

    # ── Qt overrides ──────────────────────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return _clamp_item_pos(value, self.rect())
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._handle.setVisible(bool(value))
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._sync_to_data()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._sync_to_data()


# ─────────────────────────────────────────────────────────────────────────────
# Start marker
# ─────────────────────────────────────────────────────────────────────────────

class StartMarkerItem(QGraphicsItem):
    """Draggable green crosshair marking the needle entry point."""

    def __init__(self, wx: float, wy: float) -> None:
        super().__init__()
        # Flags before setPos, so a stale/out-of-range saved start position
        # gets clamped by the itemChange hook below instead of loading as-is.
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges,
        )
        self.setPos(wx, -wy)
        self.setZValue(30)
        self.setCursor(Qt.SizeAllCursor)

    def world_pos(self) -> Tuple[float, float]:
        p = self.pos()
        return float(p.x()), float(-p.y())

    def boundingRect(self) -> QRectF:
        return QRectF(-0.45, -0.45, 0.9, 0.9)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        s = 0.35
        pen = QPen(QColor("#00ff88"), 0.07)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(-s, 0), QPointF(s, 0))
        painter.drawLine(QPointF(0, -s), QPointF(0, s))
        painter.setBrush(QBrush(QColor(0, 255, 136, 180)))
        painter.setPen(QPen(QColor("#00ff88"), 0.04))
        painter.drawEllipse(QPointF(0, 0), 0.18, 0.18)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # boundingRect is 0.45 half-width/height, comfortably clearing the
            # robot's 0.2 collision-margin radius once the marker is in-bounds.
            return _clamp_item_pos(value, self.boundingRect())
        if change == QGraphicsItem.ItemPositionHasChanged:
            scene = self.scene()
            if scene is not None:
                scene.on_start_moved()
        return super().itemChange(change, value)


class ReachabilityCircleItem(QGraphicsItem):
    """
    Non-interactive translucent circle showing the theoretical max straight-
    line reach (steps * step_size) from the start position -- a sanity-check
    overlay, not a real obstacle/tumor shape, so it's excluded from selection,
    dragging, and _rebuild_items's shape-clearing sweep.
    """

    def __init__(self, radius: float = 0.0) -> None:
        super().__init__()
        self._radius = max(0.0, float(radius))
        self.setZValue(-5)  # above the grid background, below shapes/start marker
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def set_radius(self, radius: float) -> None:
        self.prepareGeometryChange()
        self._radius = max(0.0, float(radius))
        self.update()

    def boundingRect(self) -> QRectF:
        r = self._radius
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._radius <= 0.0:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(40, 120, 255, 45)))
        painter.setPen(QPen(QColor(40, 120, 255, 140), 0.04))
        painter.drawEllipse(QPointF(0.0, 0.0), self._radius, self._radius)


# ─────────────────────────────────────────────────────────────────────────────
# Map scene
# ─────────────────────────────────────────────────────────────────────────────

class MapScene(QGraphicsScene):
    status_update = pyqtSignal(str)

    def __init__(self, map_state: MapState, parent=None) -> None:
        super().__init__(parent)
        self.setSceneRect(WORLD_X_MIN, -WORLD_Y_MAX, WORLD_X_SIZE, WORLD_Y_SIZE)
        self._ms          = map_state
        self._tool        = TOOL_SELECT
        self._draw_from:  Optional[QPointF] = None
        self._preview:    Optional[QGraphicsItem] = None
        self._drawing     = False
        self._undo_stack: List[dict] = []
        self._brush_radius: float = 0.4
        self._brush_pts:  List[QPointF] = []
        # Reachability-circle preview (radius = steps * step_size), kept in
        # sync with the Roadmap panel's Steps/Step size fields.
        self._reachability_steps = 15
        self._reachability_step_size = 0.5
        self._theme = theme.DARK
        self._font_scale = theme.FONT_SCALE_DEFAULT

        self._start = StartMarkerItem(map_state.start_x, map_state.start_y)
        self.addItem(self._start)

        self._reachability = ReachabilityCircleItem()
        self.addItem(self._reachability)
        self._update_reachability()

        for shape in map_state.shapes:
            self._add_item(shape)

    # ── tool ──────────────────────────────────────────────────────────────

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        selectable = (tool == TOOL_SELECT)
        for item in self.items():
            if isinstance(item, (MapRectItem, MapCircleItem)):
                item.setFlag(QGraphicsItem.ItemIsMovable,   selectable)
                item.setFlag(QGraphicsItem.ItemIsSelectable, selectable)
        if not selectable:
            self.clearSelection()

    # ── undo ──────────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self._undo_stack.append(deepcopy(self._ms.to_dict()))
        if len(self._undo_stack) > 80:
            self._undo_stack.pop(0)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._ms = MapState.from_dict(self._undo_stack.pop())
        self._rebuild_items()

    # ── brush ─────────────────────────────────────────────────────────────

    def set_brush_radius(self, r: float) -> None:
        self._brush_radius = max(0.05, r)

    # ── item management ───────────────────────────────────────────────────

    def _add_item(self, shape: ShapeData) -> QGraphicsItem:
        item: QGraphicsItem
        if isinstance(shape, RectData):
            item = MapRectItem(shape)
        else:
            item = MapCircleItem(shape)
        self.addItem(item)
        return item

    def _rebuild_items(self) -> None:
        for item in list(self.items()):
            if not isinstance(item, (StartMarkerItem, ReachabilityCircleItem)):
                self.removeItem(item)
        self._start.setPos(self._ms.start_x, -self._ms.start_y)
        self._update_reachability()
        for shape in self._ms.shapes:
            self._add_item(shape)

    def map_state(self) -> MapState:
        return self._ms

    def clear_all(self) -> None:
        self._push_undo()
        self._ms.shapes.clear()
        self._rebuild_items()

    def on_start_moved(self) -> None:
        wx, wy = self._start.world_pos()
        self._ms.start_x = wx
        self._ms.start_y = wy
        self._update_reachability()

    def _update_reachability(self) -> None:
        self._reachability.setPos(self._ms.start_x, -self._ms.start_y)
        self._reachability.set_radius(self._reachability_steps * self._reachability_step_size)

    def set_reachability_params(self, steps: int, step_size: float) -> None:
        """Update the reachability-circle preview radius (steps * step_size)
        from the Roadmap panel's Steps/Step size fields."""
        self._reachability_steps = int(steps)
        self._reachability_step_size = float(step_size)
        self._update_reachability()

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._tool == TOOL_SELECT:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self._push_undo()
            self._draw_from = event.scenePos()
            self._drawing   = True
            sp    = self._draw_from
            layer = _tool_to_layer(self._tool)
            color = QColor(_LAYER_COLOR[layer])
            color.setAlpha(80)
            if _is_brush_tool(self._tool):
                r  = self._brush_radius
                pi = QGraphicsEllipseItem(QRectF(sp.x() - r, sp.y() - r, 2 * r, 2 * r))
                self._brush_pts = [sp]
            elif self._tool == TOOL_RECT:
                pi = QGraphicsRectItem(QRectF(sp, sp))
            else:
                pi = QGraphicsEllipseItem(QRectF(sp, sp))
            pi.setBrush(QBrush(color))
            pi.setPen(QPen(_LAYER_COLOR[layer].darker(150), _PEN_WIDTH))
            self._preview = pi
            self.addItem(pi)
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        wx, wy = s2w(event.scenePos())
        self.status_update.emit(f"x = {wx:+.2f}   y = {wy:+.2f}")

        if self._tool == TOOL_SELECT:
            super().mouseMoveEvent(event)
            return

        if self._drawing and self._draw_from and self._preview:
            sp = event.scenePos()
            ds = self._draw_from
            if _is_brush_tool(self._tool):
                sp_c = _clamp_point_to_scene(sp)
                r = min(self._brush_radius, _max_radius_at(sp_c))
                self._preview.setRect(QRectF(sp_c.x() - r, sp_c.y() - r, 2 * r, 2 * r))
                if self._brush_pts:
                    last = self._brush_pts[-1]
                    if math.hypot(sp.x() - last.x(), sp.y() - last.y()) > self._brush_radius * 0.55:
                        self._brush_pts.append(sp)
            elif self._tool == TOOL_RECT:
                self._preview.setRect(_clamp_rect_to_scene(QRectF(ds, sp).normalized()))
            else:
                ds_c = _clamp_point_to_scene(ds)
                r = min(math.hypot(sp.x() - ds_c.x(), sp.y() - ds_c.y()), _max_radius_at(ds_c))
                self._preview.setRect(QRectF(ds_c.x() - r, ds_c.y() - r, 2 * r, 2 * r))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._tool == TOOL_SELECT:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            sp  = event.scenePos()
            ds  = self._draw_from

            if self._preview:
                self.removeItem(self._preview)
                self._preview = None

            layer = _tool_to_layer(self._tool)

            if _is_brush_tool(self._tool):
                for pt in self._brush_pts:
                    pt = _clamp_point_to_scene(pt)
                    r = min(self._brush_radius, _max_radius_at(pt))
                    cx_w, cy_w = s2w(pt)
                    shape = CircleData(layer, cx_w, cy_w, r)
                    self._ms.shapes.append(shape)
                    self._add_item(shape)
                self._brush_pts = []
            elif self._tool == TOOL_RECT:
                r = _clamp_rect_to_scene(QRectF(ds, sp).normalized())
                if r.width() >= _MIN_SIZE and r.height() >= _MIN_SIZE:
                    shape = RectData(
                        layer,
                        float(r.left()),
                        float(-r.bottom()),
                        float(r.width()),
                        float(r.height()),
                    )
                    self._ms.shapes.append(shape)
                    self._add_item(shape)
            else:
                ds = _clamp_point_to_scene(ds)
                rad = min(math.hypot(sp.x() - ds.x(), sp.y() - ds.y()), _max_radius_at(ds))
                if rad >= _MIN_SIZE:
                    cx_w, cy_w = s2w(ds)
                    shape = CircleData(layer, cx_w, cy_w, rad)
                    self._ms.shapes.append(shape)
                    self._add_item(shape)

            self._draw_from = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = [
                it for it in self.selectedItems()
                if isinstance(it, (MapRectItem, MapCircleItem))
            ]
            if selected:
                self._push_undo()
                for item in selected:
                    try:
                        self._ms.shapes.remove(item._data)
                    except ValueError:
                        pass
                    self.removeItem(item)
        else:
            super().keyPressEvent(event)

    # ── background ────────────────────────────────────────────────────────

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        t = self._theme
        painter.fillRect(rect, QColor(t.scene_bg))

        # Minor grid (0.25 units)
        minor_pen = QPen(QColor(t.grid_minor), 0)
        painter.setPen(minor_pen)
        x = math.floor(rect.left()  * 4) / 4
        while x <= rect.right():
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
            x += 0.25
        y = math.floor(rect.top() * 4) / 4
        while y <= rect.bottom():
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
            y += 0.25

        # Major grid (1 unit). Scene y = -(world y): x-lines sit directly at their
        # world x; y-lines/labels are placed at the negated world y so the grid
        # stays correct for an asymmetric y-range (e.g. 0..10), not just 0-centered.
        major_pen = QPen(QColor(t.grid_major), 0)
        painter.setPen(major_pen)
        for wx in range(int(WORLD_X_MIN), int(WORLD_X_MAX) + 1):
            painter.drawLine(QLineF(wx, rect.top(), wx, rect.bottom()))
        for wy in range(int(WORLD_Y_MIN), int(WORLD_Y_MAX) + 1):
            sy = -wy
            painter.drawLine(QLineF(rect.left(), sy, rect.right(), sy))

        # Axis labels — drawn in scene space, small font
        font = QFont("Consolas", 1)
        font.setPointSizeF(theme.GRID_LABEL_BASE_POINT_SIZE * self._font_scale)
        painter.setFont(font)
        painter.setPen(QPen(QColor(t.axis_label), 0))
        for wx in range(int(WORLD_X_MIN), int(WORLD_X_MAX) + 1):
            painter.drawText(QPointF(wx + 0.06, -WORLD_Y_MIN + 0.28), str(wx))
        for wy in range(int(WORLD_Y_MIN), int(WORLD_Y_MAX) + 1):
            if wy != int(WORLD_Y_MIN):
                painter.drawText(QPointF(0.06, float(-wy) + 0.28), str(wy))

        # World border
        border_pen = QPen(QColor(t.world_border), 0.05)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(WORLD_X_MIN, -WORLD_Y_MAX, WORLD_X_SIZE, WORLD_Y_SIZE))

    def set_theme(self, new_theme: "theme.Theme") -> None:
        self._theme = new_theme
        self.update()

    def set_font_scale(self, scale: float) -> None:
        self._font_scale = theme.clamp_scale(scale)
        self.update()


# ─────────────────────────────────────────────────────────────────────────────
# Map view
# ─────────────────────────────────────────────────────────────────────────────

class MapView(QGraphicsView):
    def __init__(self, scene: MapScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.set_theme(theme.DARK)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._fitted = False

    def set_theme(self, new_theme: "theme.Theme") -> None:
        self.setBackgroundBrush(QBrush(QColor(new_theme.scene_bg)))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._fitted = True

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


# ─────────────────────────────────────────────────────────────────────────────
# Params panel
# ─────────────────────────────────────────────────────────────────────────────

class ParamsPanel(QWidget):
    run_requested   = pyqtSignal(dict)
    brush_r_changed = pyqtSignal(float)
    roadmap_params_changed = pyqtSignal(int, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Planner ───────────────────────────────────────────────────────
        pbox = QGroupBox("Planner (--algo)")
        pb = QVBoxLayout(pbox)
        pb.setSpacing(3)
        self._planner_group = QButtonGroup(self)
        self._planner_btns: Dict[str, QRadioButton] = {}
        for i, name in enumerate(("dp", "iris", "random")):
            rb = QRadioButton(name)
            if name == "dp":
                rb.setChecked(True)
            self._planner_btns[name] = rb
            self._planner_group.addButton(rb, i)
            pb.addWidget(rb)
        root.addWidget(pbox)

        # ── Common params ────────────────────────────────────────────────
        common_box = QGroupBox("Roadmap")
        cf = QFormLayout(common_box)
        cf.setSpacing(4)
        cf.setLabelAlignment(Qt.AlignRight)

        def row(form: QFormLayout, label: str, default: str, attr: str) -> None:
            lbl  = QLabel(label)
            edit = QLineEdit(default)
            form.addRow(lbl, edit)
            setattr(self, attr, edit)

        row(cf, "Steps",     "15",  "_steps")
        row(cf, "Step size", "0.5", "_stepsz")
        row(cf, "Seed",      "0",   "_seed")
        # The 3D panel's reachability-sphere preview (radius = steps *
        # step_size) needs to stay in sync with these fields -- the sole
        # source of truth for the values actually passed to the run.
        self._steps.editingFinished.connect(self._emit_roadmap_params_changed)
        self._stepsz.editingFinished.connect(self._emit_roadmap_params_changed)

        # 2D-only: this sets --start-orientation for the 2D run. The 3D run's
        # heading is a separate control, in the 3D panel's own "Start position"
        # box (the "heading (dx,dy,dz)" dropdowns next to the 3D preview) --
        # this combo box has no effect on 3D and is disabled outside 2D mode.
        self._ori_lbl = QLabel("Start orientation (2D)")
        self._start_orientation = QComboBox()
        self._start_orientation.addItems(
            ["UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT"]
        )
        cf.addRow(self._ori_lbl, self._start_orientation)
        root.addWidget(common_box)

        # ── DP params ─────────────────────────────────────────────────────
        dp_box = QGroupBox("DP (--algo dp)")
        df = QFormLayout(dp_box)
        df.setSpacing(4)
        df.setLabelAlignment(Qt.AlignRight)
        row(df, "eps_len",  "0.5", "_eps_len")
        row(df, "eps_cov",  "0.5", "_eps_cov")
        row(df, "eps_dam",  "0.5", "_eps_dam")
        row(df, "eps_ndose", "0.5", "_eps_ndose")
        row(df, "dosage levels", "0,1,2,3", "_dosage_levels")
        row(df, "max plans/node", "", "_max_plans")
        self._depth_scaled = QCheckBox("depth-scaled eps")
        df.addRow(self._depth_scaled)
        root.addWidget(dp_box)

        # ── IRIS params ───────────────────────────────────────────────────
        iris_box = QGroupBox("IRIS (--algo iris)")
        irf = QFormLayout(iris_box)
        irf.setSpacing(4)
        irf.setLabelAlignment(Qt.AlignRight)
        row(irf, "p0", "", "_p0")
        row(irf, "epsilon0", "", "_epsilon0")
        row(irf, "max outer iters", "", "_max_outer_iters")
        row(irf, "max expansions", "", "_max_expansions")
        root.addWidget(iris_box)

        # ── Buttons ───────────────────────────────────────────────────────
        self._btn_run = QPushButton("Save + Run  [S]")
        self._btn_run.setProperty("role", "primary")
        self._btn_run.clicked.connect(self._emit_run)
        root.addWidget(self._btn_run)

        self._btn_clear = QPushButton("Clear Map")
        self._btn_clear.setProperty("role", "destructive")
        root.addWidget(self._btn_clear)

        root.addStretch()

    def connect_clear(self, slot) -> None:
        self._btn_clear.clicked.connect(slot)

    def _selected_planner(self) -> str:
        for name, btn in self._planner_btns.items():
            if btn.isChecked():
                return name
        return "dp"

    def read(self) -> dict:
        def _s(attr: str) -> str:
            return getattr(self, attr).text().strip()

        def _f(attr, default):
            try:
                txt = _s(attr)
                return float(txt) if txt else default
            except ValueError:
                return default

        def _i(attr, default):
            try:
                txt = _s(attr)
                return int(txt) if txt else default
            except ValueError:
                return default

        return {
            "planner":          self._selected_planner(),
            "steps":            _i("_steps", 15),
            "step_size":        _f("_stepsz", 0.5),
            "seed":             _i("_seed", 0),
            "start_orientation": self._start_orientation.currentText(),
            "eps_len":          _f("_eps_len", 0.5),
            "eps_cov":          _f("_eps_cov", 0.5),
            "eps_dam":          _f("_eps_dam", 0.5),
            "eps_ndose":        _f("_eps_ndose", 0.5),
            "dosage_levels":    _s("_dosage_levels"),
            "max_plans":        _s("_max_plans"),
            "depth_scaled_eps": self._depth_scaled.isChecked(),
            "p0":               _s("_p0"),
            "epsilon0":         _s("_epsilon0"),
            "max_outer_iters":  _s("_max_outer_iters"),
            "max_expansions":   _s("_max_expansions"),
        }

    def _emit_run(self) -> None:
        self.run_requested.emit(self.read())

    def _emit_roadmap_params_changed(self) -> None:
        params = self.read()
        self.roadmap_params_changed.emit(int(params["steps"]), float(params["step_size"]))


# QGroupBox styling now lives in map_editor_theme.build_app_qss (applied once,
# app-wide) -- no per-box setStyleSheet() call needed anymore.


# ─────────────────────────────────────────────────────────────────────────────
# Code generation
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = """\
from __future__ import annotations

from .world import World2D


"""


def generate_demo_map_code(ms: MapState) -> str:
    obstacles = [s for s in ms.shapes if s.layer == LAYER_OBSTACLE]
    tumors    = [s for s in ms.shapes if s.layer == LAYER_TUMOR]

    lines = [
        "def configure_demo_map(",
        "    world: World2D,",
        "    *,",
        "    start_x: float,",
        "    start_y: float,",
        "    add_obstacles: bool = True,",
        "    add_tumors: bool = True,",
        "    add_healthy: bool = True,",
        ") -> None:",
        # The map editor's canvas is a fixed (WORLD_X_MIN..WORLD_X_MAX,
        # WORLD_Y_MIN..WORLD_Y_MAX) rectangle, so every shape drawn here is
        # guaranteed to fit this viewport -- matches sim2d.viewport's
        # DEFAULT_VIEW_X/Y_LIMITS, kept explicit here so a custom map is never
        # at the mercy of that constant changing independently.
        f"    world.x_limits = ({WORLD_X_MIN}, {WORLD_X_MAX})",
        f"    world.y_limits = ({WORLD_Y_MIN}, {WORLD_Y_MAX})",
        "",
    ]

    if obstacles:
        lines.append("    if add_obstacles:")
        for s in obstacles:
            if isinstance(s, RectData):
                lines.append(
                    f"        world.add_rectangular_obstacle("
                    f"x={round(s.x,4)}, y={round(s.y,4)}, "
                    f"width={round(s.w,4)}, height={round(s.h,4)})"
                )
            elif isinstance(s, CircleData):
                lines.append(
                    f"        world.add_circular_obstacle("
                    f"center=({round(s.cx,4)}, {round(s.cy,4)}), "
                    f"radius={round(s.r,4)})"
                )
    else:
        lines.append("    pass  # no obstacles")

    if tumors:
        lines += [
            "",
            "    if not add_tumors:",
            "        return",
            "",
            "    world.reset_tumor_grid()",
            "",
        ]
        for s in tumors:
            if isinstance(s, CircleData):
                lines.append(
                    f"    world.add_circular_tumor("
                    f"center=({round(s.cx,4)}, {round(s.cy,4)}), "
                    f"radius={round(s.r,4)})"
                )
        lines += [
            "",
            "    # Healthy tissue (DP damage objective) = every non-tumor,",
            "    # non-obstacle cell -- must be computed after tumors are placed.",
            "    if add_healthy:",
            "        world.compute_healthy_mask()",
        ]

    return _HEADER + "\n".join(lines) + "\n"


_HEADER_3D = """\
from __future__ import annotations

from .world3d import World3D


"""


def generate_demo_map3d_code(ms: MapState3D) -> str:
    boxes    = [s for s in ms.shapes if isinstance(s, BoxData3D)]
    spheres  = [s for s in ms.shapes if isinstance(s, SphereData3D)]
    tumors   = [s for s in ms.shapes if isinstance(s, TumorData3D)]

    lines = [
        "def configure_demo_map_3d(",
        "    world: World3D,",
        "    *,",
        "    start_x: float,",
        "    start_y: float,",
        "    start_z: float,",
        "    add_obstacles: bool = True,",
        "    add_tumors: bool = True,",
        "    add_healthy: bool = True,",
        ") -> None:",
    ]

    if boxes or spheres:
        lines.append("    if add_obstacles:")
        for s in boxes:
            lines.append(
                f"        world.add_box_obstacle(x={round(s.x,4)}, y={round(s.y,4)}, "
                f"z={round(s.z,4)}, width={round(s.width,4)}, depth={round(s.depth,4)}, "
                f"height={round(s.height,4)}, "
                f"yaw={round(math.radians(s.yaw_deg),6)}, "
                f"pitch={round(math.radians(s.pitch_deg),6)}, "
                f"roll={round(math.radians(s.roll_deg),6)})"
            )
        for s in spheres:
            lines.append(
                f"        world.add_spherical_obstacle("
                f"center=({round(s.cx,4)}, {round(s.cy,4)}, {round(s.cz,4)}), "
                f"radius={round(s.r,4)})"
            )
    else:
        lines.append("    pass  # no obstacles")

    if tumors:
        lines += [
            "",
            "    if not add_tumors:",
            "        return",
            "",
            "    world.reset_tumor_grid()",
            "",
        ]
        for s in tumors:
            lines.append(
                f"    world.add_ball_tumor("
                f"center=({round(s.cx,4)}, {round(s.cy,4)}, {round(s.cz,4)}), "
                f"radius={round(s.r,4)})"
            )
        lines += [
            "",
            "    # Healthy tissue (DP damage objective) = every non-tumor,",
            "    # non-obstacle voxel -- must be computed after tumors are placed.",
            "    if add_healthy:",
            "        world.compute_healthy_mask()",
        ]

    return _HEADER_3D + "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# 3D panel: shape list + interactive canvas
# ─────────────────────────────────────────────────────────────────────────────

# QLineEdit/QComboBox styling now lives in map_editor_theme.build_app_qss
# (applied once, app-wide) -- no per-field setStyleSheet() call needed anymore.


class _AddShape3DDialog(QDialog):
    """Small numeric-entry dialog for one 3D shape (box / sphere / tumor)."""

    _FIELDS = {
        "box": (
            ("x", "0"), ("y", "0"), ("z", "0"), ("width", "1"), ("depth", "1"), ("height", "1"),
            ("yaw_deg", "0"), ("pitch_deg", "0"), ("roll_deg", "0"),
        ),
        "sphere": (("cx", "0"), ("cy", "0"), ("cz", "0"), ("r", "1")),
        "tumor":  (("cx", "0"), ("cy", "0"), ("cz", "0"), ("r", "2")),
    }

    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle(f"Add {kind.capitalize()}")
        form = QFormLayout(self)
        self._edits: Dict[str, QLineEdit] = {}
        for name, default in self._FIELDS[kind]:
            edit = QLineEdit(default)
            form.addRow(QLabel(name), edit)
            self._edits[name] = edit
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Dict[str, float]:
        out = {}
        for name, edit in self._edits.items():
            try:
                out[name] = float(edit.text().strip() or "0")
            except ValueError:
                out[name] = 0.0
        return out

    def shape(self) -> ShapeData3D:
        v = self.values()
        if self.kind == "box":
            return BoxData3D(
                v["x"], v["y"], v["z"], v["width"], v["depth"], v["height"],
                v["yaw_deg"], v["pitch_deg"], v["roll_deg"],
            )
        if self.kind == "sphere":
            return SphereData3D(v["cx"], v["cy"], v["cz"], v["r"])
        return TumorData3D(v["cx"], v["cy"], v["cz"], v["r"])


class Panel3D(QWidget):
    """3D map authoring: start position, a shape list, and a live interactive
    3D canvas. Click a row in the shape list to select it, then click the
    canvas to give it keyboard focus and use:

      Left/Right/Up/Down   move selected shape along x/y
      PageUp/PageDown       move selected shape along z
      Mouse wheel           resize selected shape
      Q/W  A/S  Z/X         tilt selected box's yaw/pitch/roll (boxes only)

    The "Zoom In"/"Zoom Out" buttons scale the camera view around its current
    center; matplotlib's mouse wheel is already claimed by shape-resize above,
    so camera zoom needs its own controls. The start position's heading
    (dx, dy, dz) sets the needle's initial orientation -- shown as a cyan
    arrow at the start marker -- and is what `--start-orientation` passes to
    main_demo_3d.py when you run the plan.

    See the `_KEY_ACTIONS`/`MOVE_STEP`/`ROTATE_STEP_DEG`/`RESIZE_FACTOR`
    module constants to retune bindings or step sizes.
    """

    changed = pyqtSignal()
    status_message = pyqtSignal(str)

    def __init__(self, map_state: MapState3D, parent=None) -> None:
        super().__init__(parent)
        self._ms = map_state
        self._selected_index: Optional[int] = None
        self._view_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
        # Reachability-sphere preview (radius = steps * step_size): kept in
        # sync with the Roadmap panel's Steps/Step size fields via
        # set_reachability_params, so this is just a cache of their last
        # known values, not a second source of truth.
        self._reachability_steps = 15
        self._reachability_step_size = 0.5
        self._font_scale = theme.FONT_SCALE_DEFAULT
        self._theme = theme.DARK
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Start position ───────────────────────────────────────────────
        start_box = QGroupBox("Start position")
        sf = QFormLayout(start_box)
        self._start_x = QLineEdit(str(self._ms.start_x))
        self._start_y = QLineEdit(str(self._ms.start_y))
        self._start_z = QLineEdit(str(self._ms.start_z))
        for e in (self._start_x, self._start_y, self._start_z):
            e.editingFinished.connect(self._on_start_field_edited)
        sf.addRow(QLabel("x"), self._start_x)
        sf.addRow(QLabel("y"), self._start_y)
        sf.addRow(QLabel("z"), self._start_z)

        from sim3d import DIRECTIONS_3D

        dir_row = QHBoxLayout()
        self._dir_x = QComboBox()
        self._dir_y = QComboBox()
        self._dir_z = QComboBox()
        dx0, dy0, dz0 = DIRECTIONS_3D[self._ms.start_ori]
        for combo, initial in ((self._dir_x, dx0), (self._dir_y, dy0), (self._dir_z, dz0)):
            combo.addItems(["-1", "0", "1"])
            combo.setCurrentText(str(initial))
            combo.currentIndexChanged.connect(self._on_start_field_edited)
            dir_row.addWidget(combo)
        sf.addRow(QLabel("heading (dx,dy,dz)"), dir_row)
        root.addWidget(start_box)

        # ── Shape list ────────────────────────────────────────────────────
        shapes_box = QGroupBox("Shapes (click to select for the 3D view)")
        shapes_layout = QVBoxLayout(shapes_box)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._refresh_list()
        shapes_layout.addWidget(self._list)

        add_row = QHBoxLayout()
        for kind, label in (("box", "+ Box"), ("sphere", "+ Sphere"), ("tumor", "+ Tumor")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked, k=kind: self._add_shape(k))
            add_row.addWidget(btn)
        shapes_layout.addLayout(add_row)

        del_btn = QPushButton("Delete selected")
        del_btn.setProperty("role", "destructive")
        del_btn.clicked.connect(self._delete_selected)
        shapes_layout.addWidget(del_btn)
        root.addWidget(shapes_box)

        # ── Live 3D canvas ────────────────────────────────────────────────
        canvas_box = QGroupBox("3D view")
        cv = QVBoxLayout(canvas_box)
        self._figure = Figure(figsize=(4.5, 4.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._canvas.setMinimumHeight(280)
        self._ax = self._figure.add_subplot(111, projection="3d")
        cv.addWidget(self._canvas)

        zoom_row = QHBoxLayout()
        zoom_out_btn = QPushButton("Zoom Out (−)")
        zoom_in_btn = QPushButton("Zoom In (+)")
        reset_view_btn = QPushButton("Reset view")
        zoom_out_btn.clicked.connect(lambda: self._zoom_camera(ZOOM_FACTOR))
        zoom_in_btn.clicked.connect(lambda: self._zoom_camera(1.0 / ZOOM_FACTOR))
        reset_view_btn.clicked.connect(self._reset_camera_view)
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(reset_view_btn)
        cv.addLayout(zoom_row)

        help_lbl = QLabel(
            "Click canvas first, then: Arrows move x/y, PgUp/PgDn move z, "
            "wheel resizes.\nQ/W yaw, A/S pitch, Z/X roll (selected box only)."
        )
        help_lbl.setProperty("role", "help")
        help_lbl.setWordWrap(True)
        cv.addWidget(help_lbl)
        root.addWidget(canvas_box)

        self._canvas.mpl_connect("key_press_event", self._on_key)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        root.addStretch()
        self._render()

    # ── start position ────────────────────────────────────────────────────

    def _sync_start_from_fields(self) -> None:
        from sim3d import DIRECTIONS_3D

        def _f(edit: QLineEdit, default: float) -> float:
            try:
                return float(edit.text().strip() or str(default))
            except ValueError:
                return default

        self._ms.start_x = _f(self._start_x, 0.0)
        self._ms.start_y = _f(self._start_y, 0.0)
        self._ms.start_z = _f(self._start_z, 0.0)

        direction = (
            int(self._dir_x.currentText()),
            int(self._dir_y.currentText()),
            int(self._dir_z.currentText()),
        )
        if direction == (0, 0, 0):
            # Zero vector isn't a valid heading; leave the orientation unchanged
            # rather than silently pick one for the user.
            self.status_message.emit("Heading (0,0,0) is not a direction -- pick a non-zero dx/dy/dz.")
        else:
            self._ms.start_ori = DIRECTIONS_3D.index(direction)

    def _on_start_field_edited(self) -> None:
        self._sync_start_from_fields()
        self._render()

    # ── shape list ────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for shape in self._ms.shapes:
            self._list.addItem(QListWidgetItem(shape.summary()))
        if self._selected_index is not None and 0 <= self._selected_index < self._list.count():
            self._list.setCurrentRow(self._selected_index)
        self._list.blockSignals(False)

    def _on_selection_changed(self, row: int) -> None:
        self._selected_index = row if row >= 0 else None
        self._render()

    def _add_shape(self, kind: str) -> None:
        dlg = _AddShape3DDialog(kind, self)
        if dlg.exec_() == QDialog.Accepted:
            self._ms.shapes.append(dlg.shape())
            self._selected_index = len(self._ms.shapes) - 1
            self._refresh_list()
            self._render()
            self.changed.emit()

    def _delete_selected(self) -> None:
        rows = sorted((idx.row() for idx in self._list.selectedIndexes()), reverse=True)
        for row in rows:
            del self._ms.shapes[row]
        self._selected_index = None
        self._refresh_list()
        self._render()
        self.changed.emit()

    def map_state(self) -> MapState3D:
        self._sync_start_from_fields()
        return self._ms

    # ── interactive manipulation ──────────────────────────────────────────

    def _selected_shape(self) -> Optional[ShapeData3D]:
        if self._selected_index is None:
            return None
        if not (0 <= self._selected_index < len(self._ms.shapes)):
            return None
        return self._ms.shapes[self._selected_index]

    def _move_selected(self, dx: float, dy: float, dz: float) -> None:
        shape = self._selected_shape()
        if shape is None:
            return
        if isinstance(shape, BoxData3D):
            shape.x += dx
            shape.y += dy
            shape.z += dz
        else:
            shape.cx += dx
            shape.cy += dy
            shape.cz += dz
        self._after_mutation()

    def _resize_selected(self, factor: float) -> None:
        shape = self._selected_shape()
        if shape is None:
            return
        if isinstance(shape, BoxData3D):
            # Scale about the box's own centroid so it grows/shrinks in place.
            cx, cy, cz = shape.x + shape.width / 2.0, shape.y + shape.depth / 2.0, shape.z + shape.height / 2.0
            shape.width  = max(0.1, shape.width * factor)
            shape.depth  = max(0.1, shape.depth * factor)
            shape.height = max(0.1, shape.height * factor)
            shape.x = cx - shape.width / 2.0
            shape.y = cy - shape.depth / 2.0
            shape.z = cz - shape.height / 2.0
        else:
            shape.r = max(0.1, shape.r * factor)
        self._after_mutation()

    def _rotate_selected(self, axis: str, delta_deg: float) -> None:
        shape = self._selected_shape()
        if shape is None:
            return
        if not isinstance(shape, BoxData3D):
            self.status_message.emit("Tilt only applies to boxes (spheres/tumors are rotation-invariant)")
            return
        attr = f"{axis}_deg"
        setattr(shape, attr, getattr(shape, attr) + delta_deg)
        self._after_mutation()

    def _after_mutation(self) -> None:
        if self._selected_index is not None:
            item = self._list.item(self._selected_index)
            if item is not None:
                item.setText(self._ms.shapes[self._selected_index].summary())
        self._render()
        self.changed.emit()

    # ── event handlers ─────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        action = _KEY_ACTIONS.get(event.key)
        if action is None:
            return
        if action[0] == "move":
            dx, dy, dz = action[1]
            self._move_selected(dx, dy, dz)
        elif action[0] == "rotate":
            _, axis, sign = action
            self._rotate_selected(axis, sign * ROTATE_STEP_DEG)

    def _on_scroll(self, event) -> None:
        factor = RESIZE_FACTOR if (event.step or 0) > 0 else 1.0 / RESIZE_FACTOR
        self._resize_selected(factor)

    # ── rendering ──────────────────────────────────────────────────────────

    def _build_world(self):
        from sim3d import World3D

        world = World3D()
        for shape in self._ms.shapes:
            if isinstance(shape, BoxData3D):
                world.add_box_obstacle(
                    x=shape.x, y=shape.y, z=shape.z,
                    width=shape.width, depth=shape.depth, height=shape.height,
                    yaw=math.radians(shape.yaw_deg),
                    pitch=math.radians(shape.pitch_deg),
                    roll=math.radians(shape.roll_deg),
                )
            elif isinstance(shape, SphereData3D):
                world.add_spherical_obstacle(center=(shape.cx, shape.cy, shape.cz), radius=shape.r)
            elif isinstance(shape, TumorData3D):
                world.add_ball_tumor(center=(shape.cx, shape.cy, shape.cz), radius=shape.r)
        return world

    def _zoom_camera(self, factor: float) -> None:
        """Scale the current camera view around its center (factor<1 zooms in, >1 zooms out)."""
        def _scaled(lim: Tuple[float, float]) -> Tuple[float, float]:
            lo, hi = lim
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            return (mid - half, mid + half)

        self._view_limits = (
            _scaled(self._ax.get_xlim3d()),
            _scaled(self._ax.get_ylim3d()),
            _scaled(self._ax.get_zlim3d()),
        )
        self._render()

    def _reset_camera_view(self) -> None:
        self._view_limits = None
        self._render()

    def set_reachability_params(self, steps: int, step_size: float) -> None:
        """Update the reachability-sphere preview radius (steps * step_size)
        from the Roadmap panel's Steps/Step size fields."""
        self._reachability_steps = int(steps)
        self._reachability_step_size = float(step_size)
        self._render()

    def _render(self) -> None:
        from sim3d import BoxObstacle, highlight_obstacle, orientation_to_heading, render_world_3d

        self._sync_start_from_fields()
        world = self._build_world()
        render_world_3d(
            self._ax, world,
            robot_pos=(self._ms.start_x, self._ms.start_y, self._ms.start_z),
        )

        # render_world_3d always resets the view to the full world extent
        # (ax.cla() + set_axes_equal) -- reapply any camera zoom the user set,
        # so it survives every shape edit / start-field change instead of
        # snapping back on the very next render.
        if self._view_limits is not None:
            (x0, x1), (y0, y1), (z0, z1) = self._view_limits
            self._ax.set_xlim3d(x0, x1)
            self._ax.set_ylim3d(y0, y1)
            self._ax.set_zlim3d(z0, z1)

        # Reachability preview: a translucent sphere of radius steps*step_size
        # around the start position -- the theoretical max straight-line reach,
        # so you can sanity-check whether a tumor is even in range before
        # spending minutes (or much longer) on a real DP/IRIS run.
        reach_radius = float(self._reachability_steps) * float(self._reachability_step_size)
        if reach_radius > 0.0:
            u = np.linspace(0.0, 2.0 * np.pi, 24)
            v = np.linspace(0.0, np.pi, 16)
            rx = self._ms.start_x + reach_radius * np.outer(np.cos(u), np.sin(v))
            ry = self._ms.start_y + reach_radius * np.outer(np.sin(u), np.sin(v))
            rz = self._ms.start_z + reach_radius * np.outer(np.ones_like(u), np.cos(v))
            self._ax.plot_surface(
                rx, ry, rz, color="tab:blue", alpha=0.10, linewidth=0, shade=True,
            )

        hx, hy, hz = orientation_to_heading(self._ms.start_ori)
        self._ax.quiver(
            self._ms.start_x, self._ms.start_y, self._ms.start_z,
            hx, hy, hz,
            length=HEADING_ARROW_LENGTH, color="cyan", linewidth=2.0,
            arrow_length_ratio=0.3,
        )

        shape = self._selected_shape()
        if isinstance(shape, BoxData3D):
            highlight_box = BoxObstacle(
                x=shape.x, y=shape.y, z=shape.z,
                width=shape.width, depth=shape.depth, height=shape.height,
                yaw=math.radians(shape.yaw_deg),
                pitch=math.radians(shape.pitch_deg),
                roll=math.radians(shape.roll_deg),
            )
            highlight_obstacle(self._ax, "box", highlight_box)
        elif isinstance(shape, (SphereData3D, TumorData3D)):
            highlight_obstacle(
                self._ax, "sphere", None,
                center=(shape.cx, shape.cy, shape.cz), radius=shape.r,
            )

        self._ax.set_title(
            "3D map (selected shape highlighted in cyan)" if shape is not None else "3D map",
            fontsize=theme.PANEL3D_TITLE_BASE_FONTSIZE * self._font_scale,
            color=self._theme.text_window,
        )
        # render_world_3d's ax.cla() resets matplotlib's own default (white)
        # facecolor every call -- reapply the theme's colors after, same as
        # the view-limits/zoom reapplication above.
        self._figure.set_facecolor(self._theme.bg_window)
        self._ax.set_facecolor(self._theme.bg_window)
        self._canvas.draw_idle()

    def set_font_scale(self, scale: float) -> None:
        """Rescale the matplotlib title text -- outside Qt's stylesheet
        engine, so the app-wide font-scale setting has to reach it explicitly."""
        self._font_scale = theme.clamp_scale(scale)
        self._render()

    def set_theme(self, new_theme: "theme.Theme") -> None:
        """Recolor the matplotlib figure/axes/title -- outside Qt's
        stylesheet engine, so the app-wide theme has to reach it explicitly
        (mirrors MapScene.set_theme's role for the 2D canvas)."""
        self._theme = new_theme
        self._render()


# ─────────────────────────────────────────────────────────────────────────────
# Settings dialog
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Font size + theme preferences. Changes apply live (no separate "Apply"
    step) and are persisted by the caller via QSettings, so they survive an
    app restart."""

    scale_changed = pyqtSignal(float)
    theme_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, current_scale: float, current_theme: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(300)
        # The dialog is only ever opened via MapEditorWindow._open_settings,
        # which doesn't hold a long-lived reference to it -- without this,
        # Qt's parent-child ownership keeps every past dialog instance alive
        # (invisible, still connected) for the rest of the app's lifetime.
        self.setAttribute(Qt.WA_DeleteOnClose)
        form = QFormLayout(self)

        self._scale_slider = QSlider(Qt.Horizontal)
        self._scale_slider.setMinimum(round(theme.FONT_SCALE_MIN * 100))
        self._scale_slider.setMaximum(round(theme.FONT_SCALE_MAX * 100))
        self._scale_slider.setValue(round(theme.clamp_scale(current_scale) * 100))
        self._scale_label = QLabel(f"{self._scale_slider.value()}%")
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        scale_row = QHBoxLayout()
        scale_row.addWidget(self._scale_slider)
        scale_row.addWidget(self._scale_label)
        form.addRow("Font size", scale_row)

        # Theme names come from theme.THEMES itself (not a hardcoded list),
        # so adding a third theme there automatically shows up here too.
        self._theme_names = list(theme.THEMES.keys())
        self._theme_combo = QComboBox()
        self._theme_combo.addItems([name.capitalize() for name in self._theme_names])
        current_display = current_theme.capitalize() if current_theme in self._theme_names else self._theme_names[0].capitalize()
        self._theme_combo.setCurrentText(current_display)
        self._theme_combo.currentTextChanged.connect(
            lambda text: self.theme_changed.emit(text.lower())
        )
        form.addRow("Theme", self._theme_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.clicked.connect(lambda _btn: self.accept())
        form.addRow(buttons)

    def _on_scale_changed(self, value: int) -> None:
        self._scale_label.setText(f"{value}%")
        self.scale_changed.emit(value / 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MapEditorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Map Editor")
        self.resize(1180, 820)
        self._mode = "2d"

        # Load persisted UI preferences (font size, theme) before building any
        # widgets, and apply the app-wide palette/stylesheet immediately --
        # everything constructed below picks it up automatically.
        self._settings = QSettings(QSettings.IniFormat, QSettings.UserScope,
                                    "RoboticMotionPlanning", "MapEditor")
        self._font_scale = theme.clamp_scale(
            float(self._settings.value("ui/font_scale", theme.FONT_SCALE_DEFAULT))
        )
        theme_name = str(self._settings.value("ui/theme", theme.DEFAULT_THEME_NAME))
        self._theme = theme.THEMES.get(theme_name, theme.THEMES[theme.DEFAULT_THEME_NAME])
        self._apply_theme()

        ms           = MapState.load(MAP_STATE_PATH)
        self._scene  = MapScene(ms)
        # Plain (un-animated) update: this fires on every mouse-move over the
        # canvas, far too often for the fade-in `_set_status` uses elsewhere.
        self._scene.status_update.connect(self._set_status_plain)
        self._view   = MapView(self._scene)
        # Default tool is Select: enable click-drag lasso (rubber-band) multi-select.
        self._view.setDragMode(QGraphicsView.RubberBandDrag)

        ms3d          = MapState3D.load(MAP_STATE_3D_PATH)
        self._panel3d = Panel3D(ms3d)
        self._panel3d.status_message.connect(self._set_status)

        self._params = ParamsPanel()
        self._params.run_requested.connect(self._save_and_run)
        self._params.connect_clear(self._scene.clear_all)
        self._params.brush_r_changed.connect(self._scene.set_brush_radius)
        self._params.roadmap_params_changed.connect(self._panel3d.set_reachability_params)
        self._params.roadmap_params_changed.connect(self._scene.set_reachability_params)
        initial_params = self._params.read()
        self._panel3d.set_reachability_params(initial_params["steps"], initial_params["step_size"])
        self._scene.set_reachability_params(initial_params["steps"], initial_params["step_size"])

        from PyQt5.QtWidgets import QStackedWidget

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._view)     # index 0: 2D
        self._mode_stack.addWidget(self._panel3d)  # index 1: 3D

        # ── Mode toggle ───────────────────────────────────────────────────
        mode_bar = QWidget()
        mode_bar.setProperty("role", "mode-bar")
        mode_row = QHBoxLayout(mode_bar)
        mode_row.setContentsMargins(8, 4, 8, 4)
        mode_label = QLabel("Map dimension:")
        mode_label.setProperty("role", "mode")
        mode_row.addWidget(mode_label)
        self._mode_group = QButtonGroup(self)
        for i, label in enumerate(("2D", "3D")):
            rb = QRadioButton(label)
            rb.setProperty("role", "mode")
            if i == 0:
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, lbl=label: self._on_mode_toggled(lbl, checked))
            self._mode_group.addButton(rb, i)
            mode_row.addWidget(rb)
        mode_row.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._mode_stack)
        splitter.addWidget(self._params)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 0)
        splitter.setHandleWidth(3)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(mode_bar)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._build_toolbar()

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("V = select   R = rect   C = circle   T = tumor   S = save+run   Z = undo")

        def _shortcut(key: str, fn) -> "QShortcut":
            return QShortcut(QKeySequence(key), self, fn)

        self._sc_save = _shortcut("S", self._save_and_run_default)
        self._sc_undo = _shortcut("Z", self._scene.undo)
        self._sc_quit = _shortcut("Q", self.close)
        _shortcut("V", lambda: self._set_tool(TOOL_SELECT))
        _shortcut("R", lambda: self._set_tool(TOOL_RECT))
        _shortcut("C", lambda: self._set_tool(TOOL_CIRCLE))
        _shortcut("T", lambda: self._set_tool(TOOL_TUMOR))

        # Now that the scene/view/panel3d exist, propagate the loaded theme
        # and font scale to the parts outside Qt's stylesheet engine.
        self._apply_theme()

    # ── mode toggle ───────────────────────────────────────────────────────

    def _on_mode_toggled(self, label: str, checked: bool) -> None:
        if not checked:
            return
        self._mode = "2d" if label == "2D" else "3d"
        self._mode_stack.setCurrentIndex(0 if self._mode == "2d" else 1)
        self._fade_in_current_mode_page()
        self._toolbar.setEnabled(self._mode == "2d")
        is_2d = self._mode == "2d"
        self._sc_save.setEnabled(is_2d)
        self._sc_undo.setEnabled(is_2d)
        self._sc_quit.setEnabled(is_2d)
        # The Roadmap panel's "Start orientation" combo only feeds the 2D run;
        # the 3D heading lives in the 3D panel itself. Grey it out in 3D mode
        # so it can't be mistaken for the control that moves the 3D arrow.
        self._params._start_orientation.setEnabled(is_2d)
        self._params._ori_lbl.setEnabled(is_2d)
        self._set_status(f"{label} mode")

    def _fade_in_current_mode_page(self) -> None:
        """Subtle fade-in on the 2D/3D page swap instead of an instant hard
        cut. The effect is removed once the animation finishes -- leaving a
        QGraphicsOpacityEffect attached permanently is unnecessary overhead
        for a page that isn't animating anymore.

        Guards against toggling faster than the animation's own duration: if
        a second fade starts on the same page before the first one's
        `finished` fires, the first animation's callback must not clear the
        second one's still-running effect -- so it only clears the effect if
        it's still the exact one this animation created.
        """
        page = self._mode_stack.currentWidget()
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", page)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def _clear_if_current() -> None:
            if page.graphicsEffect() is effect:
                page.setGraphicsEffect(None)

        anim.finished.connect(_clear_if_current)
        self._mode_fade_anim = anim  # keep a reference alive for the duration
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    # ── theme / settings ───────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        """Rebuild and (re)apply the app-wide palette/stylesheet from the
        current theme + font scale, and propagate to the handful of things
        that live outside Qt's stylesheet engine (the 2D scene's grid text,
        the 3D panel's matplotlib title, the toolbar icons). Safe to call
        both before child widgets exist (constructor bootstrap) and anytime
        after (settings changes) -- `hasattr` guards skip whatever isn't
        built yet."""
        app = QApplication.instance()
        if app is not None:
            app.setPalette(theme.build_palette(self._theme))
            app.setStyleSheet(theme.build_app_qss(self._theme, self._font_scale))
        if hasattr(self, "_tool_actions"):
            self._refresh_icons()
        if hasattr(self, "_scene"):
            self._scene.set_theme(self._theme)
            self._scene.set_font_scale(self._font_scale)
        if hasattr(self, "_view"):
            self._view.set_theme(self._theme)
        if hasattr(self, "_panel3d"):
            self._panel3d.set_theme(self._theme)
            self._panel3d.set_font_scale(self._font_scale)
        self._settings.setValue("ui/font_scale", self._font_scale)
        self._settings.setValue("ui/theme", self._theme.name)

    def _refresh_icons(self) -> None:
        stroke = self._theme.icon_stroke
        for tool_id, act in self._tool_actions.items():
            act.setIcon(icons.ICON_BUILDERS[tool_id](stroke))
        if hasattr(self, "_undo_action"):
            self._undo_action.setIcon(icons.icon_undo(stroke))
        if hasattr(self, "_settings_action"):
            self._settings_action.setIcon(icons.icon_settings(stroke))

    def _change_font_scale(self, factor: float) -> None:
        self._font_scale = theme.clamp_scale(self._font_scale * factor)
        self._apply_theme()

    def _set_font_scale_absolute(self, scale: float) -> None:
        self._font_scale = theme.clamp_scale(scale)
        self._apply_theme()

    def _set_theme_by_name(self, name: str) -> None:
        self._theme = theme.THEMES.get(name, self._theme)
        self._apply_theme()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self, current_scale=self._font_scale, current_theme=self._theme.name)
        dlg.scale_changed.connect(self._set_font_scale_absolute)
        dlg.theme_changed.connect(self._set_theme_by_name)
        anim = QPropertyAnimation(dlg, b"windowOpacity", dlg)
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        # Keep an explicit reference alongside the other two fade animations
        # in this file -- relying solely on dlg.exec_() blocking (which is
        # what currently keeps this alive) would break silently if this
        # dialog were ever changed to non-modal.
        self._settings_fade_anim = anim
        anim.start()
        dlg.exec_()

    # ── toolbar ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, tb)
        self._toolbar = tb

        group = QActionGroup(self)
        group.setExclusive(True)
        self._tool_actions: Dict[str, QAction] = {}

        tool_defs = [
            (TOOL_SELECT, "Select / Move / Resize  [V]"),
            (TOOL_RECT,   "Rectangle obstacle  [R]"),
            (TOOL_CIRCLE, "Circle obstacle  [C]"),
            (TOOL_TUMOR,  "Tumor circle  [T]"),
        ]
        for tool_id, tip in tool_defs:
            act = QAction(self)
            act.setToolTip(tip)
            act.setCheckable(True)
            act.triggered.connect(lambda _checked, t=tool_id: self._set_tool(t))
            group.addAction(act)
            self._tool_actions[tool_id] = act
            tb.addAction(act)

        self._tool_actions[TOOL_SELECT].setChecked(True)

        tb.addSeparator()
        self._undo_action = QAction(self)
        self._undo_action.setToolTip("Undo  [Z]")
        self._undo_action.triggered.connect(self._scene.undo)
        tb.addAction(self._undo_action)

        tb.addSeparator()
        shrink_act = QAction("A-", self)
        shrink_act.setToolTip("Decrease UI font size")
        shrink_act.triggered.connect(lambda: self._change_font_scale(1.0 / theme.FONT_SCALE_STEP))
        tb.addAction(shrink_act)
        grow_act = QAction("A+", self)
        grow_act.setToolTip("Increase UI font size")
        grow_act.triggered.connect(lambda: self._change_font_scale(theme.FONT_SCALE_STEP))
        tb.addAction(grow_act)

        tb.addSeparator()
        self._settings_action = QAction(self)
        self._settings_action.setToolTip("Settings (font size, theme)  [gear]")
        self._settings_action.triggered.connect(self._open_settings)
        tb.addAction(self._settings_action)

        self._refresh_icons()

    def _set_tool(self, tool: str) -> None:
        self._scene.set_tool(tool)
        # Rubber-band (lasso) multi-select only makes sense in Select mode --
        # other tools use click-drag to draw a shape instead.
        self._view.setDragMode(
            QGraphicsView.RubberBandDrag if tool == TOOL_SELECT else QGraphicsView.NoDrag
        )
        act = self._tool_actions.get(tool)
        if act:
            act.setChecked(True)
        names = {
            TOOL_SELECT: "Select / Move / Resize",
            TOOL_RECT:   "Draw rectangle obstacle",
            TOOL_CIRCLE: "Draw circle obstacle",
            TOOL_TUMOR:  "Draw tumor circle",
        }
        self._set_status(names.get(tool, tool))

    def _set_status(self, msg: str) -> None:
        """Discrete status message (tool switch, mode toggle, save/run) --
        gets a brief fade-in. NOT for the high-frequency mouse-coordinate
        readout (see `_set_status_plain`) -- animating on every mouse-move
        event would spawn a fresh QGraphicsOpacityEffect/QPropertyAnimation
        many times a second.

        Guards against two status updates within the animation's own 220ms
        (e.g. rapid V/R/C/T tool switching): the first animation's `finished`
        callback only clears the effect if it's still the one this specific
        animation created, so it can't clear out a second, still-running fade.
        """
        sb = self.statusBar()
        sb.showMessage(msg)
        effect = QGraphicsOpacityEffect(sb)
        sb.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", sb)
        anim.setDuration(220)
        anim.setStartValue(0.35)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def _clear_if_current() -> None:
            if sb.graphicsEffect() is effect:
                sb.setGraphicsEffect(None)

        anim.finished.connect(_clear_if_current)
        self._status_fade_anim = anim  # keep a reference alive for the duration
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    def _set_status_plain(self, msg: str) -> None:
        """Un-animated status update for high-frequency events (mouse-move
        coordinate readout) -- see `_set_status` for the animated version."""
        self.statusBar().showMessage(msg)

    # ── save + run ────────────────────────────────────────────────────────

    def _save_and_run_default(self) -> None:
        self._save_and_run(self._params.read())

    def _save_and_run(self, params: dict) -> None:
        if self._mode == "3d":
            self._save_and_run_3d(params)
        else:
            self._save_and_run_2d(params)

    def _next_snapshot_dir(self) -> Tuple[int, pathlib.Path]:
        MAPS_DIR.mkdir(exist_ok=True)
        existing = [
            int(p.name.split("_")[1])
            for p in MAPS_DIR.iterdir()
            if p.is_dir() and p.name.startswith("map_") and p.name.split("_")[1].isdigit()
        ]
        n = max(existing, default=0) + 1
        snap_dir = MAPS_DIR / f"map_{n:03d}"
        snap_dir.mkdir()
        return n, snap_dir

    def _save_and_run_2d(self, params: dict) -> None:
        ms = self._scene.map_state()
        ms.save(MAP_STATE_PATH)

        n, snap_dir = self._next_snapshot_dir()
        ms.save(snap_dir / "map_state.json")

        code = generate_demo_map_code(ms)
        DEMO_MAP_PATH.write_text(code, encoding="utf-8")
        (snap_dir / "demo_map.py").write_text(code, encoding="utf-8")

        p   = params
        cmd = [
            sys.executable, str(MAIN_DEMO_PATH),
            "--algo",      p["planner"],
            "--steps",     str(p["steps"]),
            "--step-size", str(p["step_size"]),
            "--seed",      str(p["seed"]),
            "--start-x",   str(ms.start_x),
            "--start-y",   str(ms.start_y),
            "--start-orientation", p["start_orientation"],
        ]
        if p["planner"] == "dp":
            cmd += [
                "--eps-len", str(p["eps_len"]),
                "--eps-cov", str(p["eps_cov"]),
                "--eps-dam", str(p["eps_dam"]),
                "--eps-ndose", str(p["eps_ndose"]),
            ]
            if p["dosage_levels"]:
                cmd += ["--dosage-levels", p["dosage_levels"]]
            if p["max_plans"]:
                cmd += ["--max-plans-per-node", p["max_plans"]]
            if p["depth_scaled_eps"]:
                cmd += ["--depth-scaled-eps"]
        elif p["planner"] == "iris":
            if p["p0"]:
                cmd += ["--p0", p["p0"]]
            if p["epsilon0"]:
                cmd += ["--epsilon0", p["epsilon0"]]
            if p["max_outer_iters"]:
                cmd += ["--max-outer-iters", p["max_outer_iters"]]
            if p["max_expansions"]:
                cmd += ["--max-expansions", p["max_expansions"]]

        print(f"[map_editor] Saved map_state.json -> {MAP_STATE_PATH}")
        print(f"[map_editor] Wrote demo_map.py     -> {DEMO_MAP_PATH}")
        print(f"[map_editor] Launching: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        self._set_status(f"Launched {p['planner']} | snapshot -> maps/map_{n:03d}/")

    def _save_and_run_3d(self, params: dict) -> None:
        ms = self._panel3d.map_state()
        ms.save(MAP_STATE_3D_PATH)

        n, snap_dir = self._next_snapshot_dir()
        ms.save(snap_dir / "map_state_3d.json")

        code = generate_demo_map3d_code(ms)
        DEMO_MAP_3D_PATH.write_text(code, encoding="utf-8")
        (snap_dir / "demo_map3d.py").write_text(code, encoding="utf-8")

        # main_demo_3d.py doesn't accept --dosage-levels/--depth-scaled-eps
        # (dosage levels are fixed in 3D), so those flags are omitted below.
        p   = params
        cmd = [
            sys.executable, str(MAIN_DEMO_3D_PATH),
            "--algo",      p["planner"],
            "--steps",     str(p["steps"]),
            "--step-size", str(p["step_size"]),
            "--seed",      str(p["seed"]),
            "--start-x",   str(ms.start_x),
            "--start-y",   str(ms.start_y),
            "--start-z",   str(ms.start_z),
            "--start-orientation", str(ms.start_ori),
        ]
        if p["planner"] == "dp":
            cmd += [
                "--eps-len", str(p["eps_len"]),
                "--eps-cov", str(p["eps_cov"]),
                "--eps-dam", str(p["eps_dam"]),
                "--eps-ndose", str(p["eps_ndose"]),
            ]
            if p["max_plans"]:
                cmd += ["--max-plans-per-node", p["max_plans"]]
        elif p["planner"] == "iris":
            if p["p0"]:
                cmd += ["--p0", p["p0"]]
            if p["epsilon0"]:
                cmd += ["--epsilon0", p["epsilon0"]]
            if p["max_outer_iters"]:
                cmd += ["--max-outer-iters", p["max_outer_iters"]]
            if p["max_expansions"]:
                cmd += ["--max-expansions", p["max_expansions"]]

        print(f"[map_editor] Saved map_state_3d.json -> {MAP_STATE_3D_PATH}")
        print(f"[map_editor] Wrote demo_map3d.py      -> {DEMO_MAP_3D_PATH}")
        print(f"[map_editor] Launching: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        self._set_status(f"Launched {p['planner']} (3D) | snapshot -> maps/map_{n:03d}/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_to_layer(tool: str) -> str:
    return {
        TOOL_RECT:   LAYER_OBSTACLE,
        TOOL_CIRCLE: LAYER_OBSTACLE,
        TOOL_TUMOR:  LAYER_TUMOR,
    }.get(tool, LAYER_OBSTACLE)


def _is_brush_tool(tool: str) -> bool:
    return tool in (TOOL_TUMOR,)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Palette/stylesheet themeing is applied by MapEditorWindow itself (see
    # _apply_theme), since it also needs to reapply on live settings changes.
    win = MapEditorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
