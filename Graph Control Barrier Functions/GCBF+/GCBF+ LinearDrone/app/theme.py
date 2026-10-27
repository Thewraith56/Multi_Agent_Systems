BG = "#14171c"
PANEL = "#1b1f27"
FG = "#d8dde6"
MUTED = "#8b94a3"
ACCENT = "#4aa3ff"
GRID = "#2a303b"
RED, GREEN, ORANGE, YELLOW = "#ff5a5f", "#3ddc97", "#ffa94d", "#ffd43b"

QSS = f"""
* {{ font-family: 'Segoe UI','Inter','Helvetica Neue',Arial; font-size: 12px; color: {FG}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {GRID}; background: {PANEL}; top: -1px; }}
QTabBar::tab {{ background: {BG}; padding: 6px 11px; border: 1px solid {GRID}; border-bottom: none; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; }}
QGroupBox {{ border: 1px solid {GRID}; border-radius: 6px; margin-top: 10px; padding: 8px 6px 6px 6px; background: {PANEL}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: #262c37; border: 1px solid {GRID}; border-radius: 5px; padding: 5px 10px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:checked {{ background: {ACCENT}; color: #08121c; }}
QPushButton#primary {{ background: {ACCENT}; color: #08121c; font-weight: 600; }}
QPushButton:disabled {{ color: #555d6b; }}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{ background: #10131a; border: 1px solid {GRID}; border-radius: 4px; padding: 3px 5px; }}
QTableWidget {{ background: #10131a; gridline-color: {GRID}; border: 1px solid {GRID}; }}
QHeaderView::section {{ background: {PANEL}; border: 1px solid {GRID}; padding: 3px; color: {MUTED}; }}
QPlainTextEdit {{ background: #0d1016; border: 1px solid {GRID}; font-family: Consolas,'DejaVu Sans Mono',monospace; font-size: 11px; }}
QProgressBar {{ border: 1px solid {GRID}; border-radius: 4px; text-align: center; background: #10131a; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QStatusBar {{ background: {PANEL}; }}
QToolBar {{ background: {PANEL}; border-bottom: 1px solid {GRID}; spacing: 6px; padding: 4px; }}
QLabel#hint {{ color: {MUTED}; }}
QSlider::groove:horizontal {{ height: 4px; background: {GRID}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px; }}
"""
