"""
Centralized visual theme for map_editor.py: color palettes, font-scale
bounds, and QSS builders -- one source of truth instead of 20+ scattered
inline setStyleSheet() literals with hardcoded hex/px values.

Widgets opt into role-specific styling via Qt dynamic properties (e.g.
`widget.setProperty("role", "primary")`, set once at construction time
*before* the widget is first shown) rather than one-off inline stylesheets,
so `build_app_qss` below can target them with attribute selectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Theme:
    name: str
    # Backgrounds
    bg_window: str
    bg_alt: str
    bg_base: str
    bg_button: str
    bg_input: str
    bg_list: str
    bg_statusbar: str
    # Text
    text_primary: str
    text_window: str
    text_light: str
    text_muted: str
    text_dim: str
    # Chrome
    border: str
    border_hover: str
    highlight: str
    highlight_text: str
    accent_bg: str
    accent_text: str
    destructive_bg: str
    destructive_text: str
    # 2D scene canvas (drawn directly via QPainter, not QSS)
    scene_bg: str
    grid_minor: str
    grid_major: str
    axis_label: str
    world_border: str
    # Icon stroke color (map_editor_icons.py draws with this)
    icon_stroke: str


DARK = Theme(
    name="dark",
    bg_window="#0d0d1a",
    bg_alt="#111122",
    bg_base="#1a1a2e",
    bg_button="#1a1a3a",
    bg_input="#1a1a3a",
    bg_list="#12121e",
    bg_statusbar="#090910",
    text_primary="#ddeeff",
    text_window="#ddddee",
    text_light="#dddddd",
    text_muted="#aaaabb",
    text_dim="#888899",
    border="#333344",
    border_hover="#222233",
    highlight="#4444aa",
    highlight_text="#ffffff",
    accent_bg="#1a2a3a",
    accent_text="#aaccff",
    destructive_bg="#2a1a1a",
    destructive_text="#ffaaaa",
    scene_bg="#0d0d1a",
    grid_minor="#1c1c30",
    grid_major="#272744",
    axis_label="#3a3a5a",
    world_border="#445566",
    icon_stroke="#dddddd",
)

LIGHT = Theme(
    name="light",
    bg_window="#f3f5fb",
    bg_alt="#e6e9f4",
    bg_base="#ffffff",
    bg_button="#e2e6f2",
    bg_input="#ffffff",
    bg_list="#ffffff",
    bg_statusbar="#e2e6f2",
    text_primary="#1a1a2e",
    text_window="#1a1a2e",
    text_light="#222233",
    text_muted="#55597a",
    text_dim="#6b7090",
    border="#c6cbe0",
    border_hover="#d6dbee",
    highlight="#4a5fd6",
    highlight_text="#ffffff",
    accent_bg="#dde6f7",
    accent_text="#274b8f",
    destructive_bg="#f6d9d9",
    destructive_text="#9c2b2b",
    scene_bg="#f3f5fb",
    grid_minor="#dee2ef",
    grid_major="#c6cce3",
    axis_label="#8991ac",
    world_border="#7c86a3",
    icon_stroke="#2a2a3a",
)

THEMES = {"dark": DARK, "light": LIGHT}
DEFAULT_THEME_NAME = "dark"

FONT_SCALE_MIN = 0.7
FONT_SCALE_MAX = 1.6
FONT_SCALE_DEFAULT = 1.0
FONT_SCALE_STEP = 1.1

# Base px sizes at scale=1.0, named by semantic role. Every place that used
# to hardcode a literal font-size now multiplies one of these by the live
# scale via `px(role, scale)`.
_BASE_PX = {
    "label": 10,
    "small": 9,
    "toolbar_icon": 20,
    "mode_label": 11,
}

# Base sizes (not px-scaled the same way, but still centralized) for the
# things drawn outside Qt's stylesheet engine entirely.
GRID_LABEL_BASE_POINT_SIZE = 0.28  # sim2d.MapScene grid/axis labels (world units)
PANEL3D_TITLE_BASE_FONTSIZE = 9.0  # Panel3D's matplotlib ax.set_title fontsize


def px(role: str, scale: float) -> int:
    """A themed font size in pixels for the given semantic role, clamped to
    a sane floor so aggressive shrinking never produces unreadable text."""
    return max(6, round(_BASE_PX[role] * scale))


def clamp_scale(scale: float) -> float:
    return min(FONT_SCALE_MAX, max(FONT_SCALE_MIN, float(scale)))


def build_palette(theme: Theme) -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(theme.bg_window))
    pal.setColor(QPalette.WindowText, QColor(theme.text_window))
    pal.setColor(QPalette.Base, QColor(theme.bg_base))
    pal.setColor(QPalette.AlternateBase, QColor(theme.bg_alt))
    pal.setColor(QPalette.Button, QColor(theme.bg_button))
    pal.setColor(QPalette.ButtonText, QColor(theme.text_window))
    pal.setColor(QPalette.Highlight, QColor(theme.highlight))
    pal.setColor(QPalette.HighlightedText, QColor(theme.highlight_text))
    pal.setColor(QPalette.Text, QColor(theme.text_primary))
    pal.setColor(QPalette.ToolTipBase, QColor(theme.bg_alt))
    pal.setColor(QPalette.ToolTipText, QColor(theme.text_window))
    return pal


def build_app_qss(theme: Theme, scale: float = FONT_SCALE_DEFAULT) -> str:
    """
    One global stylesheet for the whole application (apply once via
    `app.setStyleSheet(...)`), replacing 20+ scattered per-widget
    `setStyleSheet()` calls. Widget-class selectors cover the common case;
    `[role="..."]` attribute selectors (matched against a dynamic Qt
    property set on specific widgets) cover the handful that need to look
    different from their class default -- e.g. the destructive Clear/Delete
    buttons, or the dimmer "help" labels.
    """
    t = theme
    label_px = px("label", scale)
    small_px = px("small", scale)
    icon_px = px("toolbar_icon", scale)
    mode_px = px("mode_label", scale)

    return f"""
    QMainWindow, QWidget {{
        background: {t.bg_window};
        color: {t.text_window};
    }}
    QLabel {{
        color: {t.text_muted};
        font-size: {label_px}px;
    }}
    QLabel[role="help"] {{
        color: {t.text_dim};
        font-size: {small_px}px;
    }}
    QLabel[role="mode"] {{
        color: {t.text_light};
        font-size: {mode_px}px;
    }}
    QLineEdit, QComboBox, QComboBox QAbstractItemView {{
        background: {t.bg_input};
        color: {t.text_primary};
        border: 1px solid {t.border};
        font-size: {label_px}px;
        padding: 1px 3px;
    }}
    QRadioButton {{
        color: {t.text_light};
        font-size: {label_px}px;
    }}
    QCheckBox {{
        color: {t.text_muted};
        font-size: {label_px}px;
    }}
    QRadioButton[role="mode"] {{
        color: {t.text_light};
        font-size: {mode_px}px;
    }}
    QGroupBox {{
        color: {t.text_muted};
        font-size: {label_px}px;
        font-weight: bold;
        border: 1px solid {t.border};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 3px;
    }}
    QPushButton {{
        background: {t.bg_button};
        color: {t.text_window};
        font-size: {label_px}px;
        padding: 5px 8px;
        border: 1px solid {t.border};
        border-radius: 4px;
    }}
    QPushButton:hover {{
        background: {t.border_hover};
    }}
    QPushButton[role="primary"] {{
        background: {t.accent_bg};
        color: {t.accent_text};
    }}
    QPushButton[role="primary"]:hover {{
        background: {t.highlight};
    }}
    QPushButton[role="destructive"] {{
        background: {t.destructive_bg};
        color: {t.destructive_text};
    }}
    QListWidget {{
        background: {t.bg_list};
        color: {t.text_light};
        font-size: {label_px}px;
        border: 1px solid {t.border};
    }}
    QToolBar {{
        background: {t.bg_alt};
        border: none;
        spacing: 6px;
        padding: 4px;
    }}
    QToolButton {{
        font-size: {icon_px}px;
        color: {t.text_light};
        padding: 6px;
        min-width: 34px;
        min-height: 34px;
        border-radius: 4px;
    }}
    QToolButton:checked {{
        background: {t.border};
    }}
    QToolButton:hover {{
        background: {t.border_hover};
    }}
    QSplitter::handle {{
        background: {t.border};
    }}
    QStatusBar {{
        color: {t.text_dim};
        font-size: {label_px}px;
        background: {t.bg_statusbar};
    }}
    QToolTip {{
        background: {t.bg_alt};
        color: {t.text_window};
        border: 1px solid {t.border};
        font-size: {small_px}px;
        padding: 3px 5px;
    }}
    QDialog {{
        background: {t.bg_window};
        color: {t.text_window};
    }}
    QWidget[role="mode-bar"] {{
        background: {t.bg_alt};
    }}
    QSlider::groove:horizontal {{
        background: {t.border};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {t.accent_text};
        width: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    """
