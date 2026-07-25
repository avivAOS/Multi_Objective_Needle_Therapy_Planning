"""
Toolbar icons for map_editor.py, drawn programmatically with QPainter --
no external image assets or network access needed. Each function returns a
QIcon rendered onto a transparent QPixmap at a fixed logical canvas size, in
a single `stroke` color, so every icon reads as one consistent icon set
regardless of theme (dark vs. light just pass a different stroke color).
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

_CANVAS = 32
_MARGIN = 5


def _new_painter(stroke: str, width: float = 2.2) -> tuple[QPixmap, QPainter]:
    pixmap = QPixmap(_CANVAS, _CANVAS)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(stroke), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    return pixmap, painter


def _finish(pixmap: QPixmap, painter: QPainter) -> QIcon:
    painter.end()
    return QIcon(pixmap)


def icon_select(stroke: str = "#dddddd") -> QIcon:
    """An arrow-cursor glyph -- the "Select / Move / Resize" tool."""
    pixmap, painter = _new_painter(stroke)
    m = _MARGIN
    tip = QPointF(m, m)
    body = [tip, QPointF(m, _CANVAS - m), QPointF(m + 7, _CANVAS - m - 9),
            QPointF(m + 11, _CANVAS - m - 2), QPointF(m + 14, _CANVAS - m - 6),
            QPointF(m + 9, _CANVAS - m - 12), QPointF(_CANVAS - m, m + 9)]
    for a, b in zip(body, body[1:] + [tip]):
        painter.drawLine(a, b)
    return _finish(pixmap, painter)


def icon_rect(stroke: str = "#dddddd") -> QIcon:
    """A rectangle outline -- the "Draw rectangle obstacle" tool."""
    pixmap, painter = _new_painter(stroke)
    m = _MARGIN + 1
    painter.drawRoundedRect(QRectF(m, m + 3, _CANVAS - 2 * m, _CANVAS - 2 * m - 6), 2, 2)
    return _finish(pixmap, painter)


def icon_circle(stroke: str = "#dddddd") -> QIcon:
    """A circle outline -- the "Draw circle obstacle" tool."""
    pixmap, painter = _new_painter(stroke)
    m = _MARGIN
    painter.drawEllipse(QRectF(m, m, _CANVAS - 2 * m, _CANVAS - 2 * m))
    return _finish(pixmap, painter)


def icon_tumor(stroke: str = "#dddddd") -> QIcon:
    """A filled dot inside a circle -- the "Draw tumor circle" tool."""
    pixmap, painter = _new_painter(stroke)
    m = _MARGIN
    painter.drawEllipse(QRectF(m, m, _CANVAS - 2 * m, _CANVAS - 2 * m))
    painter.setBrush(QColor(stroke))
    center = _CANVAS / 2.0
    r = 3.6
    painter.drawEllipse(QPointF(center, center), r, r)
    return _finish(pixmap, painter)


def icon_undo(stroke: str = "#dddddd") -> QIcon:
    """A counter-clockwise curved arrow -- the "Undo" action."""
    pixmap, painter = _new_painter(stroke)
    rect = QRectF(_MARGIN + 2, _MARGIN + 2, _CANVAS - 2 * _MARGIN - 4, _CANVAS - 2 * _MARGIN - 4)
    painter.drawArc(rect, 20 * 16, 300 * 16)
    # Arrowhead at the tail of the arc (~20 degrees).
    tip = QPointF(rect.center().x() + rect.width() / 2 * 0.94,
                   rect.center().y() - rect.height() / 2 * 0.34)
    painter.drawLine(tip, QPointF(tip.x() - 6, tip.y() - 2))
    painter.drawLine(tip, QPointF(tip.x() - 1, tip.y() + 6))
    return _finish(pixmap, painter)


def icon_settings(stroke: str = "#dddddd") -> QIcon:
    """A gear glyph -- opens the Settings dialog."""
    pixmap, painter = _new_painter(stroke, width=2.0)
    center = _CANVAS / 2.0
    outer_r = _CANVAS / 2.0 - _MARGIN
    inner_r = outer_r * 0.55
    painter.drawEllipse(QPointF(center, center), inner_r, inner_r)

    for i in range(8):
        angle = math.radians(i * 45.0)
        x1 = center + math.cos(angle) * inner_r
        y1 = center + math.sin(angle) * inner_r
        x2 = center + math.cos(angle) * outer_r
        y2 = center + math.sin(angle) * outer_r
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    return _finish(pixmap, painter)


# Lookup by tool id for the mutually-exclusive tool-group icons (select/
# rect/circle/tumor); the standalone undo/settings actions aren't part of
# that group but their builders live here too so every icon function has one
# shared home (see MapEditorWindow._refresh_icons for both usages).
ICON_BUILDERS = {
    "select": icon_select,
    "rect": icon_rect,
    "circle": icon_circle,
    "tumor": icon_tumor,
    "undo": icon_undo,
    "settings": icon_settings,
}
