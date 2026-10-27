"""Drone GCBF+ simulator - main window."""
import os
import sys
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from gcbfplus.env import ENV
from sim.engine import ROOT, checkpoints_for
from sim.scenario import (Scenario, convert_scenario, default_area, random_scenario, random_velocities,
                          resize_scenario)
from . import theme as T
from .views import Canvas2D, Canvas3D
from .worker import SimWorker

MAX_ROWS = 300   # tables show at most this many rows (use the random generator / JSON for bigger scenes)


def dspin(lo, hi, val, step=0.05, dec=3, parent=None):
    w = QDoubleSpinBox(parent); w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step); w.setValue(val)
    w.setKeyboardTracking(False)
    return w


def ispin(lo, hi, val):
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setKeyboardTracking(False)
    return w


def hint(text):
    l = QLabel(text); l.setObjectName("hint"); l.setWordWrap(True)
    return l


def scroll(w):
    s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(w); s.setFrameShape(QScrollArea.NoFrame)
    return s


# ============================================================================================ array table
class ArrayTable(QTableWidget):
    """Editable table whose cells map straight onto numpy arrays of the scenario."""

    def __init__(self, on_edit):
        super().__init__()
        self.on_edit = on_edit
        self.cols = []   # (header, getter(sc)->array, column index or None)
        self.sc = None
        self.itemChanged.connect(self._changed)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def fill(self, sc, cols):
        self.sc, self.cols = sc, cols
        n = min(len(cols[0][1](sc)), MAX_ROWS)
        self.blockSignals(True)
        self.clear(); self.setColumnCount(len(cols)); self.setRowCount(n)
        self.setHorizontalHeaderLabels([c[0] for c in cols])
        for j, (_, get, k) in enumerate(cols):
            arr = get(sc)
            for i in range(n):
                v = arr[i] if k is None else arr[i, k]
                it = QTableWidgetItem(f"{float(v):.4g}")
                self.setItem(i, j, it)
        self.blockSignals(False)

    def _changed(self, it):
        try:
            v = float(it.text())
        except ValueError:
            return
        _, get, k = self.cols[it.column()]
        arr = get(self.sc)
        if k is None:
            arr[it.row()] = v
        else:
            arr[it.row(), k] = v
        self.on_edit()

    def selected_rows(self):
        rows = sorted({i.row() for i in self.selectionModel().selectedRows()})
        return rows


# ============================================================================================ main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone GCBF+ simulator")
        self.resize(1560, 930)
        self.mode = "2D"
        self.sc = None
        self.ready, self.dirty, self.play_after = False, True, False
        self.ckpt = None
        self._last_snap, self._next_draw, self._draw_ms = None, 0.0, 0.0
        self.worker = SimWorker()
        self.worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.configured.connect(self.on_configured)
        self.worker.start()
        self._edit_timer = QTimer(self); self._edit_timer.setSingleShot(True); self._edit_timer.timeout.connect(self._refresh_view)
        self._build_ui()
        self.switch_mode("2D", first=True)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(15)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=lambda: self.run_btn.click())
        QShortcut(QKeySequence("R"), self, activated=self.reset)

    # ------------------------------------------------------------------------------------------ UI build
    def _build_ui(self):
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["2D simulator", "3D (matplotlib)"])
        self.mode_combo.currentIndexChanged.connect(lambda i: self.switch_mode("2D" if i == 0 else "3D"))
        tb.addWidget(QLabel(" View ")); tb.addWidget(self.mode_combo)
        self.run_btn = QPushButton("▶  Run"); self.run_btn.setCheckable(True); self.run_btn.setObjectName("primary")
        self.run_btn.toggled.connect(self.toggle_run)
        step_btn = QPushButton("Step"); step_btn.clicked.connect(self.step_once)
        reset_btn = QPushButton("⟲ Reset"); reset_btn.clicked.connect(self.reset)
        apply_btn = QPushButton("Apply / recompile"); apply_btn.clicked.connect(lambda: self.apply(False))
        for w in (self.run_btn, step_btn, reset_btn, apply_btn):
            tb.addWidget(w)
        tb.addSeparator()
        tb.addWidget(QLabel(" Speed "))
        self.speed = QSlider(Qt.Horizontal); self.speed.setRange(5, 400); self.speed.setValue(33); self.speed.setFixedWidth(150)
        self.speed_lbl = QLabel("33 steps/s")
        self.speed.valueChanged.connect(lambda v: (setattr(self.worker, "sps", float(v)), self.speed_lbl.setText(f"{v} steps/s")))
        tb.addWidget(self.speed); tb.addWidget(self.speed_lbl)
        self.trails = QCheckBox("Trails"); self.trails.setChecked(True)
        self.trails.toggled.connect(lambda v: (setattr(self.c2d, "trails_on", v), setattr(self.c3d, "trails_on", v)))
        tb.addWidget(self.trails)
        self.state_lbl = QLabel("")
        tb.addSeparator(); tb.addWidget(self.state_lbl)

        # --- center: 2D / 3D canvases
        self.c2d, self.c3d = Canvas2D(), Canvas3D()
        self.c2d.edited.connect(self.on_canvas_edited)
        self.c2d.message.connect(lambda s: self.statusBar().showMessage(s, 4000))
        self.stack = QStackedWidget(); self.stack.addWidget(self.c2d); self.stack.addWidget(self.c3d)
        center = QWidget(); cl = QVBoxLayout(center); cl.setContentsMargins(4, 4, 4, 4)
        self.tool_bar = QWidget(); tl = QHBoxLayout(self.tool_bar); tl.setContentsMargins(0, 0, 0, 0)
        self.tool_group = QButtonGroup(self)
        for key, label in (("select", "✥ Select / move"), ("agent", "+ Agent (start, then target)"),
                           ("obstacle", "+ Obstacle (click, drag = radius)"), ("delete", "✕ Delete"),
                           ("obs_vel", "➚ Obstacle velocity"), ("goal_vel", "➚ Target velocity")):
            b = QPushButton(label); b.setCheckable(True); b.setProperty("tool", key)
            self.tool_group.addButton(b); tl.addWidget(b)
            if key == "select":
                b.setChecked(True)
        self.tool_group.buttonClicked.connect(lambda b: setattr(self.c2d, "tool", b.property("tool")))
        tl.addStretch(1)
        vb = QPushButton("Fit view"); vb.clicked.connect(self.c2d.reset_view); tl.addWidget(vb)
        self.hint_lbl = hint("")
        cl.addWidget(self.tool_bar); cl.addWidget(self.hint_lbl); cl.addWidget(self.stack, 1)

        # --- right: tabs
        self.tabs = QTabWidget(); self.tabs.setMinimumWidth(430); self.tabs.setMaximumWidth(560)
        self._build_scenario_tab(); self._build_tables_tab(); self._build_params_tab()
        root = QWidget(); hl = QHBoxLayout(root); hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(center, 1); hl.addWidget(self.tabs)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    # ---- Scenario tab
    def _build_scenario_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("GCBF+ simulation"); f = QFormLayout(g)
        self.env_info = hint("")
        f.addRow(QLabel("Controller"), QLabel("GCBF+ (pretrained LinearDrone checkpoint)"))
        f.addRow(self.env_info)
        lay.addWidget(g)

        g = QGroupBox("Arena and counts"); f = QFormLayout(g)
        self.area = dspin(0.5, 500, 2.0, 0.5, 2); self.n_agents = ispin(1, 20000, 8); self.n_obs = ispin(0, 20000, 8)
        self.area.valueChanged.connect(self.on_area_changed)
        f.addRow("Arena side length", self.area); f.addRow("Number of agents", self.n_agents)
        f.addRow("Number of obstacles", self.n_obs)
        hb = QHBoxLayout(); b1 = QPushButton("Randomize everything"); b1.setObjectName("primary"); b1.clicked.connect(self.randomize_all)
        b2 = QPushButton("Apply counts (keep edits)"); b2.clicked.connect(self.apply_counts); hb.addWidget(b1); hb.addWidget(b2)
        f.addRow(hb)
        lay.addWidget(g)

        g = QGroupBox("Random generation"); f = QFormLayout(g)
        self.r_min = dspin(0.005, 50, 0.075, 0.005, 3); self.r_max = dspin(0.005, 50, 0.15, 0.005, 3)
        self.os_min = dspin(0, 5, 0.0, 0.01, 3); self.os_max = dspin(0, 5, 0.1, 0.01, 3)
        self.gs_min = dspin(0, 5, 0.0, 0.01, 3); self.gs_max = dspin(0, 5, 0.05, 0.01, 3)
        self.seed = ispin(0, 10 ** 9, 0)
        rr = QHBoxLayout(); rr.addWidget(self.r_min); rr.addWidget(self.r_max); f.addRow("Obstacle radius min / max", rr)
        rr = QHBoxLayout(); rr.addWidget(self.os_min); rr.addWidget(self.os_max); f.addRow("Obstacle speed min / max", rr)
        rr = QHBoxLayout(); rr.addWidget(self.gs_min); rr.addWidget(self.gs_max); f.addRow("Target speed min / max", rr)
        f.addRow("Seed", self.seed)
        b = QPushButton("Re-roll velocities only"); b.clicked.connect(self.reroll_velocities); f.addRow(b)
        f.addRow(hint("Speeds are in arena units per second (agents are limited to 0.5 units/s). Moving obstacles and targets bounce off the arena walls. Speed 0/0 = static."))
        lay.addWidget(g)

        g = QGroupBox("Scenario file"); hb = QHBoxLayout(g)
        b = QPushButton("Load JSON"); b.clicked.connect(self.load_json); hb.addWidget(b)
        b = QPushButton("Save JSON"); b.clicked.connect(self.save_json); hb.addWidget(b)
        lay.addWidget(g); lay.addStretch(1)
        self.tabs.addTab(scroll(w), "Scenario")

    # ---- Agents / obstacles tables
    def _build_tables_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.agent_tbl = ArrayTable(self.on_table_edited); self.obs_tbl = ArrayTable(self.on_table_edited)
        g = QGroupBox("Agents - start, target and target velocity"); v = QVBoxLayout(g); v.addWidget(self.agent_tbl)
        hb = QHBoxLayout(); b = QPushButton("Add agent"); b.clicked.connect(self.add_agent); hb.addWidget(b)
        b = QPushButton("Delete selected"); b.clicked.connect(lambda: self.delete_rows("agent")); hb.addWidget(b); v.addLayout(hb)
        self.agent_vel = self._vel_form(v, "agent"); lay.addWidget(g, 1)
        g = QGroupBox("Obstacles - centre, radius and velocity"); v = QVBoxLayout(g); v.addWidget(self.obs_tbl)
        hb = QHBoxLayout(); b = QPushButton("Add obstacle"); b.clicked.connect(self.add_obstacle); hb.addWidget(b)
        b = QPushButton("Delete selected"); b.clicked.connect(lambda: self.delete_rows("obs")); hb.addWidget(b); v.addLayout(hb)
        self.obs_vel = self._vel_form(v, "obs"); lay.addWidget(g, 1)
        self.tbl_note = hint(""); lay.addWidget(self.tbl_note)
        self.tabs.addTab(w, "Agents and obstacles")

    def _vel_form(self, parent_layout, which):
        hb = QHBoxLayout()
        speed = dspin(0, 5, 0.1, 0.01, 3); head = dspin(-360, 360, 0, 5, 1); elev = dspin(-90, 90, 0, 5, 1)
        for lbl, wd in (("speed", speed), ("heading°", head), ("elev°", elev)):
            hb.addWidget(QLabel(lbl)); hb.addWidget(wd)
        b = QPushButton("Set velocity of selected rows (all if none)"); b.clicked.connect(lambda: self.set_velocity(which, speed.value(), head.value(), elev.value()))
        parent_layout.addLayout(hb); parent_layout.addWidget(b)
        return speed, head, elev, elev

    # ---- Parameter tab
    def _build_params_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Runtime / environment parameters"); f = QFormLayout(g)
        self.p_drone = dspin(0.005, 1, 0.05, 0.005, 3); self.p_comm = dspin(0.05, 20, 0.5, 0.05, 2)
        self.p_rays = ispin(4, 256, 32); self.p_dt = dspin(0.005, 0.1, 0.03, 0.005, 3)
        for k, wd in (("Drone radius", self.p_drone), ("Comm / LiDAR range", self.p_comm), ("LiDAR beams (n_rays)", self.p_rays),
                      ("Time step dt [s]", self.p_dt)):
            f.addRow(k, wd)
            wd.valueChanged.connect(self.mark_dirty)
        f.addRow(hint("The released checkpoints were trained with the defaults (radius 0.05, comm 0.5 / 1.0, dt 0.03). "
                      "Changing them is a test of generalisation, not a guarantee - collisions are always judged with "
                      "the radius set here."))
        lay.addWidget(g)
        g = QGroupBox("Scalability (large swarms)"); f = QFormLayout(g)
        self.p_nb = ispin(0, 1024, 0); self.p_ob = ispin(0, 4096, 0); self.p_chunk = ispin(0, 8192, 0)
        for k, wd in (("Max neighbours / agent (0 = exact, all)", self.p_nb), ("Max obstacles / agent LiDAR (0 = all)", self.p_ob),
                      ("LiDAR chunk size (0 = off)", self.p_chunk)):
            f.addRow(k, wd); wd.valueChanged.connect(self.mark_dirty)
        f.addRow(hint("0 = exact original graph. Above 96 agents / 24 obstacles the simulator switches on caps "
                      "(24 neighbours, 24 obstacles, chunk 32) automatically unless you set a value here - it keeps "
                      "memory and step time linear in the number of agents instead of quadratic."))
        lay.addWidget(g)
        g = QGroupBox("Loaded model (read-only)"); v = QVBoxLayout(g); self.model_info = QLabel("-"); self.model_info.setWordWrap(True)
        v.addWidget(self.model_info); v.addWidget(hint("Network architecture and learned weights are loaded from the released GCBF+ checkpoint and remain fixed during simulation."))
        lay.addWidget(g); lay.addStretch(1)
        self.tabs.addTab(scroll(w), "Parameters")

    # ------------------------------------------------------------------------------------------ mode / env
    def switch_mode(self, mode, first=False):
        self.stop_run()
        self.mode = mode
        dim = 2 if mode == "2D" else 3
        self.view = self.c2d if mode == "2D" else self.c3d
        self.stack.setCurrentWidget(self.view)
        self.tool_bar.setVisible(mode == "2D")
        self.hint_lbl.setText(
            "2D LinearDrone: click to add/move agents and obstacles. Scroll = zoom; right-click = cancel. Edit while stopped."
            if mode == "2D" else
            "3D LinearDrone (matplotlib): drag = rotate, scroll = zoom. Edit objects with the tables or random generator.")
        self.env_key_selected = "LinearDrone2D" if mode == "2D" else "LinearDrone"
        self.agent_vel[2].setEnabled(dim == 3)
        self.ckpt = max(checkpoints_for("LinearDrone", "gcbf+"), key=lambda c: c.step, default=None)
        if first or self.sc is None:
            self.area.blockSignals(True); self.area.setValue(2.0); self.area.blockSignals(False)
            self.sc = random_scenario(dim, 8, 8, 2.0, 0.05, (0.075, 0.15), obs_speed=(0, 0.1), goal_speed=(0, 0.05), seed=0)
        else:
            self.sc = convert_scenario(self.sc, dim)
        self.on_env_changed(); self.sync_widgets_from_scenario(); self.mark_dirty(); self._refresh_view()

    def env_key(self):
        return self.env_key_selected

    def on_env_changed(self, *_):
        d = ENV["LinearDrone"].PARAMS
        for wd, k in ((self.p_drone, "drone_radius"), (self.p_comm, "comm_radius"), (self.p_rays, "n_rays")):
            wd.blockSignals(True); wd.setValue(d[k]); wd.blockSignals(False)
        lo, hi = d["obs_len_range"]
        self.r_min.setValue(lo / 2); self.r_max.setValue(hi / 2)
        if self.mode == "2D":
            msg = ("Uses the released 3D LinearDrone GCBF+ policy in a planar embedding: height is fixed and vertical control is zeroed. "
                   "This is a 2D simulation view of the 3D checkpoint, not a separately trained 2D policy.")
        else:
            msg = "Uses the released 3D LinearDrone GCBF+ checkpoint associated with the paper."
        if self.ckpt is None:
            msg += " No checkpoint was found under pretrained/LinearDrone/gcbf+/models."
        self.env_info.setText(msg)
        self.update_model_info()
        self.mark_dirty()

    def update_model_info(self):
        if not self.ckpt:
            self.model_info.setText("No pretrained GCBF+ LinearDrone checkpoint found.")
            return
        actor_path = os.path.join(self.ckpt.path, "models", str(self.ckpt.step), "actor.pkl")
        self.model_info.setText(
            f"Pretrained LinearDrone GCBF+ policy\nCheckpoint step: {self.ckpt.step}\n"
            f"Actor weights: {os.path.relpath(actor_path, ROOT)}")

    def params(self):
        p = dict(drone_radius=self.p_drone.value(), comm_radius=self.p_comm.value(), n_rays=self.p_rays.value(), dt=self.p_dt.value())
        for k, wd in (("max_neighbors", self.p_nb), ("max_obs_per_agent", self.p_ob), ("lidar_chunk", self.p_chunk)):
            if wd.value():
                p[k] = wd.value()
        return p

    # ------------------------------------------------------------------------------------------ scenario ops
    def dim(self):
        return 2 if self.mode == "2D" else 3

    def sync_widgets_from_scenario(self):
        for w, v in ((self.n_agents, self.sc.n_agents), (self.n_obs, self.sc.n_obs), (self.area, self.sc.area)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        d = self.dim()
        ax = "xyz"[:d]
        self.agent_tbl.fill(self.sc, [(f"start {a}", lambda s: s.starts, i) for i, a in enumerate(ax)] +
                            [(f"goal {a}", lambda s: s.goals, i) for i, a in enumerate(ax)] +
                            [(f"v{a}", lambda s: s.goal_vel, i) for i, a in enumerate(ax)])
        self.obs_tbl.fill(self.sc, [(a, lambda s: s.obs_pos, i) for i, a in enumerate(ax)] + [("r", lambda s: s.obs_radius, None)] +
                          [(f"v{a}", lambda s: s.obs_vel, i) for i, a in enumerate(ax)])
        extra = [f"showing first {MAX_ROWS} rows of {n}" for n, nm in ((self.sc.n_agents, "agents"), (self.sc.n_obs, "obstacles")) if n > MAX_ROWS]
        self.tbl_note.setText(("Tables: " + "; ".join(extra) + " - use Randomize / JSON for bulk edits.") if extra else "")
        if self.sc.warnings:
            self.statusBar().showMessage("⚠ " + " ".join(self.sc.warnings), 8000)

    def _refresh_view(self):
        self.view.default_obs_r = float((self.r_min.value() + self.r_max.value()) / 2) if hasattr(self.view, "default_obs_r") else 0
        self.view.set_scenario(self.sc, self.p_drone.value())
        self.worker_snapshot_discard()

    def worker_snapshot_discard(self):
        self._last_snap = self.worker.latest

    def mark_dirty(self, *_):
        self.dirty = True
        self.state_lbl.setText("  edited - press Run / Apply to (re)start  ")

    def on_canvas_edited(self):
        self.sync_widgets_from_scenario(); self.mark_dirty(); self.worker_snapshot_discard()

    def on_table_edited(self):
        self.mark_dirty(); self._edit_timer.start(200)

    def on_area_changed(self, v):
        self.sc.area = float(v); self.mark_dirty(); self._edit_timer.start(200)

    def _gen_args(self):
        return dict(radius_range=(self.r_min.value(), self.r_max.value()), obs_speed=(self.os_min.value(), self.os_max.value()),
                    goal_speed=(self.gs_min.value(), self.gs_max.value()))

    def randomize_all(self):
        self.stop_run()
        g = self._gen_args()
        n = self.n_agents.value()
        self.sc = random_scenario(self.dim(), n, self.n_obs.value(), self.area.value(), self.p_drone.value(), g["radius_range"],
                                  obs_speed=g["obs_speed"], goal_speed=g["goal_speed"], seed=self.seed.value())
        self.seed.setValue(self.seed.value() + 1)
        self.sync_widgets_from_scenario(); self.mark_dirty(); self._refresh_view()

    def apply_counts(self):
        self.stop_run()
        g = self._gen_args()
        self.sc = resize_scenario(self.sc, self.n_agents.value(), self.n_obs.value(), self.p_drone.value(), g["radius_range"],
                                  g["obs_speed"], g["goal_speed"], self.seed.value())
        self.sync_widgets_from_scenario(); self.mark_dirty(); self._refresh_view()

    def reroll_velocities(self):
        self.stop_run()
        g, rng, d = self._gen_args(), np.random.default_rng(self.seed.value()), self.dim()
        self.sc.obs_vel = random_velocities(rng, self.sc.n_obs, d, g["obs_speed"])
        self.sc.goal_vel = random_velocities(rng, self.sc.n_agents, d, g["goal_speed"])
        self.seed.setValue(self.seed.value() + 1)
        self.sync_widgets_from_scenario(); self.mark_dirty(); self._refresh_view()

    def add_agent(self):
        self.stop_run(); s, A, d = self.sc, self.sc.area, self.dim()
        rng = np.random.default_rng()
        s.starts = np.vstack([s.starts, rng.uniform(0, A, d)]).astype(np.float32)
        s.goals = np.vstack([s.goals, rng.uniform(0, A, d)]).astype(np.float32)
        s.goal_vel = np.vstack([s.goal_vel, np.zeros(d)]).astype(np.float32)
        self.on_canvas_edited(); self._refresh_view()

    def add_obstacle(self):
        self.stop_run(); s, A, d = self.sc, self.sc.area, self.dim()
        s.obs_pos = np.vstack([s.obs_pos, np.full(d, A / 2)]).astype(np.float32)
        s.obs_radius = np.append(s.obs_radius, (self.r_min.value() + self.r_max.value()) / 2).astype(np.float32)
        s.obs_vel = np.vstack([s.obs_vel, np.zeros(d)]).astype(np.float32)
        self.on_canvas_edited(); self._refresh_view()

    def delete_rows(self, which):
        tbl = self.agent_tbl if which == "agent" else self.obs_tbl
        rows = tbl.selected_rows()
        if not rows:
            return
        self.stop_run(); s = self.sc
        if which == "agent":
            keep = np.setdiff1d(np.arange(s.n_agents), rows); s.starts, s.goals, s.goal_vel = s.starts[keep], s.goals[keep], s.goal_vel[keep]
        else:
            keep = np.setdiff1d(np.arange(s.n_obs), rows); s.obs_pos, s.obs_radius, s.obs_vel = s.obs_pos[keep], s.obs_radius[keep], s.obs_vel[keep]
        self.on_canvas_edited(); self._refresh_view()

    def set_velocity(self, which, speed, heading, elev):
        self.stop_run()
        tbl, arr = (self.agent_tbl, self.sc.goal_vel) if which == "agent" else (self.obs_tbl, self.sc.obs_vel)
        rows = tbl.selected_rows() or list(range(len(arr)))
        h, e = np.radians(heading), np.radians(elev)
        v = np.array([np.cos(e) * np.cos(h), np.cos(e) * np.sin(h), np.sin(e)])[:self.dim()] * speed
        arr[rows] = v
        self.on_canvas_edited(); self._refresh_view()

    def load_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load scenario", ROOT, "JSON (*.json)")
        if p:
            self.stop_run(); sc = Scenario.from_json(p)
            if sc.dim != self.dim():
                self.mode_combo.setCurrentIndex(0 if sc.dim == 2 else 1)
            self.sc = sc; self.sync_widgets_from_scenario(); self.mark_dirty(); self._refresh_view()

    def save_json(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save scenario", os.path.join(ROOT, "scenario.json"), "JSON (*.json)")
        if p:
            self.sc.to_json(p)

    # ------------------------------------------------------------------------------------------ run control
    def stop_run(self):
        self.worker.running = False
        self.play_after = False
        if self.run_btn.isChecked():
            self.run_btn.blockSignals(True); self.run_btn.setChecked(False); self.run_btn.setText("▶  Run"); self.run_btn.blockSignals(False)
        if getattr(self, "view", None) is not None:
            self.view.editable = self.view is self.c2d

    def apply(self, play):
        key, ck = self.env_key(), self.ckpt
        if ck is None:
            QMessageBox.information(self, "GCBF+ checkpoint not found",
                                    "Place the released LinearDrone GCBF+ checkpoint under "
                                    "pretrained/LinearDrone/gcbf+/models/ to run the simulation.")
            self.stop_run(); return
        self.ready, self.play_after = False, play
        self.state_lbl.setText("  compiling GCBF+ simulation ...  ")
        self.c2d.editable = False
        self.worker.configure(key, self.sc, self.params(), ck)

    def on_configured(self, ok, msg):
        if not ok:
            self.stop_run(); self.state_lbl.setText("  error  ")
            QMessageBox.warning(self, "Could not start simulation", msg)
            return
        self.ready, self.dirty = True, False
        self.state_lbl.setText("")
        self._last_snap = None
        self.view.set_scenario(self.sc, self.p_drone.value())
        if self.play_after:
            self.worker.running = True
            self.run_btn.blockSignals(True); self.run_btn.setChecked(True); self.run_btn.setText("⏸  Pause"); self.run_btn.blockSignals(False)
        self.c2d.editable = not self.play_after

    def toggle_run(self, on):
        if on:
            self.run_btn.setText("⏸  Pause")
            if self.dirty or not self.ready:
                self.apply(True)
            else:
                self.worker.running = True; self.c2d.editable = False
        else:
            self.worker.running = False; self.run_btn.setText("▶  Run"); self.c2d.editable = True

    def step_once(self):
        if self.dirty or not self.ready:
            self.apply(False); return
        self.worker.running = False; self.worker.step_once()

    def reset(self):
        if self.ready and not self.dirty:
            self.worker.reset(); self._last_snap = None
        elif self.dirty:
            self._refresh_view()

    # ------------------------------------------------------------------------------------------ frame loop
    def tick(self):
        s = self.worker.latest
        now = time.perf_counter()
        if s is None or now < self._next_draw or s is self._last_snap or self.dirty or not self.ready:
            return
        self._last_snap = s
        t0 = time.perf_counter()
        self.view.show_snapshot(s)
        dur = time.perf_counter() - t0
        self._draw_ms = 0.8 * self._draw_ms + 0.2 * dur * 1e3
        self._next_draw = now + max(0.033, 2.0 * dur)          # draw at most ~50% of the time: GUI thread keeps idle slots for input
        n = len(s["agent"])
        self.statusBar().showMessage(
            f"t = {s['t']}   |   colliding now {int(s['colliding'].sum())}/{n}   ever collided {int(s['ever_collided'].sum())}   "
            f"reached target {int(s['ever_reached'].sum())}/{n}   |   sim {self.worker.measured_sps:.0f} steps/s   draw {self._draw_ms:.0f} ms")

    # ------------------------------------------------------------------------------------------ misc
    def closeEvent(self, e):
        self.worker.shutdown()
        self.worker.wait(3000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(T.QSS)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
