"""Plot canvases. Both draw only the *newest* snapshot; the 2D view uses blitting so a frame costs ~ms."""
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import hsv_to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Signal

from . import theme as T


def agent_colors(n):
    h = (np.arange(n) * 0.618033988749895) % 1.0
    return hsv_to_rgb(np.stack([h, np.full(n, .55), np.full(n, .98)], axis=1))


class Blit:
    """Cache the static background; per frame only restore it and redraw the animated artists."""

    def __init__(self, canvas, artists, pre_draw=None):
        self.canvas, self.artists, self.pre_draw, self._bg = canvas, list(artists), pre_draw, None
        for a in self.artists:
            a.set_animated(True)
        canvas.mpl_connect("draw_event", self._on_draw)

    def _draw_animated(self):
        for a in self.artists:
            if self.pre_draw:
                self.pre_draw(a)
            self.canvas.figure.draw_artist(a)

    def _on_draw(self, _ev):
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def update(self):
        if self._bg is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._bg)
        self._draw_animated()
        self.canvas.blit(self.canvas.figure.bbox)


# ======================================================================================== 2D
class Canvas2D(FigureCanvasQTAgg):
    edited = Signal()
    message = Signal(str)

    TOOLS = ("select", "agent", "obstacle", "delete", "obs_vel", "goal_vel")

    def __init__(self):
        self.fig = Figure(facecolor=T.BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_axes([0.045, 0.045, 0.945, 0.945])
        self.sc = None
        self.drone_r = 0.05
        self.tool = "select"
        self.editable = True
        self.trails_on = True
        self.default_obs_r = 0.1
        self._drag = None
        self._pending = None
        self._hist = None
        self._last_t = -1
        self.blitter = None
        for ev, fn in (("button_press_event", self._press), ("motion_notify_event", self._motion),
                       ("button_release_event", self._release), ("scroll_event", self._scroll)):
            self.mpl_connect(ev, fn)

    # ---------------------------------------------------------------- scene construction
    def set_scenario(self, sc, drone_r):
        self.sc, self.drone_r = sc, drone_r
        self._build()

    def _build(self):
        sc, ax = self.sc, self.ax
        ax.clear()
        A = sc.area
        pad = 0.04 * A
        ax.set_xlim(-pad, A + pad); ax.set_ylim(-pad, A + pad); ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#0d1016")
        ax.grid(True, color=T.GRID, lw=0.5, alpha=.8)
        ax.tick_params(colors=T.MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(T.GRID)
        ax.add_patch(Rectangle((0, 0), A, A, fill=False, ec="#3b4352", lw=1.2, ls="--"))
        n, m = sc.n_agents, sc.n_obs
        self.cols = agent_colors(n) if n else np.zeros((0, 3))
        d = max(2 * self.drone_r, 0.007 * A)

        def ell(w, pos, fc, ec="none", alpha=1.0, lw=1.0, z=3):
            pos = np.asarray(pos, float).reshape(-1, 2)
            if len(pos) == 0:
                pos, w = np.array([[-1e4, -1e4]]), np.array([0.0])
            c = EllipseCollection(w, w, np.zeros(len(w)), units="xy", offsets=pos, offset_transform=ax.transData,
                                  facecolors=fc, edgecolors=ec, alpha=alpha, linewidths=lw, zorder=z)
            ax.add_collection(c)
            return c

        self.a_obs = ell(2 * sc.obs_radius, sc.obs_pos, T.ORANGE, "#ffd8a8", 0.45, 1.0, 2)
        self.a_link = LineCollection(np.zeros((n, 2, 2)), colors=[to_rgba(c, .28) for c in self.cols],
                                     linewidths=0.8, zorder=1); ax.add_collection(self.a_link)
        self.a_trail = LineCollection([], colors=[to_rgba(c, .45) for c in self.cols], linewidths=1.0, zorder=2)
        ax.add_collection(self.a_trail)
        self.a_ovel = LineCollection(np.zeros((m, 2, 2)), colors=T.ORANGE, linewidths=1.6, zorder=4)
        self.a_gvel = LineCollection(np.zeros((n, 2, 2)), colors=T.GREEN, linewidths=1.6, zorder=4)
        ax.add_collection(self.a_ovel); ax.add_collection(self.a_gvel)
        self.a_start = ax.scatter(*(sc.starts.T if n else ([], [])), s=28, facecolors="none",
                                  edgecolors=self.cols if n else None, linewidths=1.0, zorder=3)
        self.a_goal = ax.scatter(*(sc.goals.T if n else ([], [])), s=95, marker="*", c=self.cols if n else None,
                                 edgecolors="white", linewidths=0.4, zorder=5)
        self.a_agent = ell(np.full(max(n, 1), d), sc.starts, self.cols, "none", 1.0, 1.4, 6)
        self.a_prev, = ax.plot([], [], color=T.YELLOW, lw=1.2, ls=":", zorder=9)
        self.a_prev_pt = ax.scatter([], [], s=36, c=T.YELLOW, zorder=9)
        self._dyn = [self.a_link, self.a_trail, self.a_obs, self.a_ovel, self.a_gvel, self.a_start, self.a_goal,
                     self.a_agent, self.a_prev, self.a_prev_pt]
        self.blitter = Blit(self, self._dyn)
        self._refresh_edit_overlays()
        self._hist = None
        self._last_t = -1
        self.draw()

    def _refresh_edit_overlays(self):
        sc = self.sc
        self.a_ovel.set_segments(np.stack([sc.obs_pos, sc.obs_pos + sc.obs_vel], 1) if sc.n_obs else [])
        self.a_gvel.set_segments(np.stack([sc.goals, sc.goals + sc.goal_vel], 1) if sc.n_agents else [])
        self.a_link.set_segments(np.stack([sc.starts, sc.goals], 1) if sc.n_agents else [])
        self.a_trail.set_segments([])

    def redraw_scenario(self):
        """Edit mode: show scenario (initial) positions."""
        if self.sc is not None:
            self.set_scenario(self.sc, self.drone_r)

    # ---------------------------------------------------------------- live frames
    def show_snapshot(self, s):
        n = len(s["agent"])
        if n != self.sc.n_agents or len(s["obs"]) != self.sc.n_obs:
            return
        if self._hist is None or s["t"] < self._last_t:
            K = 40
            self._hist = np.repeat(s["agent"][None], K, 0) if n else np.zeros((K, 0, 2))
            self._hpos = 0
        if s["t"] != self._last_t and n:
            self._hist[self._hpos % len(self._hist)] = s["agent"]
            self._hpos += 1
        self._last_t = s["t"]
        ag = s["agent"]
        self.a_agent.set_offsets(ag)
        fc = self.cols.copy()
        fc[s["colliding"]] = to_rgba(T.RED)[:3]
        self.a_agent.set_facecolor(fc)
        ec = np.zeros((n, 4)); ec[s["ever_reached"]] = to_rgba(T.GREEN)
        ec[s["ever_collided"] & ~s["ever_reached"]] = to_rgba(T.RED, .8)
        self.a_agent.set_edgecolor(ec)
        self.a_goal.set_offsets(s["goal"])
        self.a_obs.set_offsets(s["obs"]) if len(s["obs"]) else None
        self.a_link.set_segments(np.stack([ag, s["goal"]], 1) if n else [])
        self.a_ovel.set_segments(np.stack([s["obs"], s["obs"] + s["obs_vel"]], 1) if len(s["obs"]) else [])
        self.a_gvel.set_segments(np.stack([s["goal"], s["goal"] + s["goal_vel"]], 1) if n else [])
        if self.trails_on and 0 < n <= 150:
            h = np.roll(self._hist, -self._hpos, axis=0)
            self.a_trail.set_segments(list(np.transpose(h, (1, 0, 2))))
        else:
            self.a_trail.set_segments([])
        self.blitter.update()

    # ---------------------------------------------------------------- mouse editing
    def _nearest(self, x, y):
        sc, tol = self.sc, 0.03 * self.sc.area
        best, hit = tol, None
        p = np.array([x, y])
        for kind, pts, rad in (("start", sc.starts, 0), ("goal", sc.goals, 0), ("obs", sc.obs_pos, sc.obs_radius)):
            if len(pts):
                d = np.linalg.norm(pts - p, axis=1) - (rad if kind == "obs" else 0)
                i = int(np.argmin(d))
                if d[i] < best:
                    best, hit = d[i], (kind, i)
        return hit

    def _press(self, ev):
        if self.sc is None or not self.editable or ev.inaxes is not self.ax or ev.xdata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)
        sc = self.sc
        if ev.button == 3:
            self._pending, self._drag = None, None
            self.a_prev_pt.set_offsets(np.zeros((0, 2))); self.a_prev.set_data([], [])
            self.blitter.update(); self.message.emit("cancelled"); return
        if ev.button != 1:
            return
        if self.tool == "select":
            h = self._nearest(x, y)
            self._drag = ("move", h) if h else None
        elif self.tool == "agent":
            if self._pending is None:
                self._pending = (x, y)
                self.a_prev_pt.set_offsets([[x, y]]); self.blitter.update()
                self.message.emit("click the TARGET location for this agent (right-click cancels)")
            else:
                sx, sy = self._pending
                self._pending = None
                sc.starts = np.vstack([sc.starts, [sx, sy]]).astype(np.float32)
                sc.goals = np.vstack([sc.goals, [x, y]]).astype(np.float32)
                sc.goal_vel = np.vstack([sc.goal_vel, np.zeros(2)]).astype(np.float32)
                self._changed()
        elif self.tool == "obstacle":
            self._drag = ("new_obs", (x, y))
        elif self.tool == "delete":
            h = self._nearest(x, y)
            if h:
                kind, i = h
                if kind == "obs":
                    keep = np.arange(sc.n_obs) != i
                    sc.obs_pos, sc.obs_radius, sc.obs_vel = sc.obs_pos[keep], sc.obs_radius[keep], sc.obs_vel[keep]
                else:
                    keep = np.arange(sc.n_agents) != i
                    sc.starts, sc.goals, sc.goal_vel = sc.starts[keep], sc.goals[keep], sc.goal_vel[keep]
                self._changed()
        elif self.tool in ("obs_vel", "goal_vel"):
            h = self._nearest(x, y)
            want = "obs" if self.tool == "obs_vel" else "goal"
            if h and h[0] == want:
                self._drag = ("vel", h)

    def _motion(self, ev):
        if self._drag is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)
        sc, (mode, h) = self.sc, self._drag
        if mode == "move":
            kind, i = h
            arr = {"start": sc.starts, "goal": sc.goals, "obs": sc.obs_pos}[kind]
            arr[i] = [x, y]
            self._refresh_edit_overlays()
            self.a_agent.set_offsets(sc.starts); self.a_start.set_offsets(sc.starts)
            self.a_goal.set_offsets(sc.goals); self.a_obs.set_offsets(sc.obs_pos)
            self.blitter.update()
        elif mode == "new_obs":
            cx, cy = h
            self.a_prev.set_data(*self._circle(cx, cy, np.hypot(x - cx, y - cy)))
            self.blitter.update()
        elif mode == "vel":
            kind, i = h
            c = sc.obs_pos[i] if kind == "obs" else sc.goals[i]
            self.a_prev.set_data([c[0], x], [c[1], y]); self.blitter.update()

    def _release(self, ev):
        if self._drag is None:
            return
        mode, h = self._drag
        self._drag = None
        sc = self.sc
        x, y = (float(ev.xdata), float(ev.ydata)) if ev.xdata is not None else (None, None)
        self.a_prev.set_data([], [])
        if mode == "new_obs":
            cx, cy = h
            r = np.hypot(x - cx, y - cy) if x is not None else 0.0
            r = float(np.clip(r if r > 0.01 else self.default_obs_r, 0.02, sc.area / 3))
            sc.obs_pos = np.vstack([sc.obs_pos, [cx, cy]]).astype(np.float32)
            sc.obs_radius = np.append(sc.obs_radius, r).astype(np.float32)
            sc.obs_vel = np.vstack([sc.obs_vel, np.zeros(2)]).astype(np.float32)
            self._changed()
        elif mode == "vel" and x is not None:
            kind, i = h
            if kind == "obs":
                sc.obs_vel[i] = [x - sc.obs_pos[i][0], y - sc.obs_pos[i][1]]     # arrow = displacement in 1 s
            else:
                sc.goal_vel[i] = [x - sc.goals[i][0], y - sc.goals[i][1]]
            self._changed()
        elif mode == "move":
            self._changed()

    def _circle(self, cx, cy, r):
        t = np.linspace(0, 2 * np.pi, 48)
        return cx + r * np.cos(t), cy + r * np.sin(t)

    def _changed(self):
        self._build()
        self.edited.emit()

    def _scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        f = 0.85 if ev.button == "up" else 1 / 0.85
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(ev.xdata - (ev.xdata - x0) * f, ev.xdata + (x1 - ev.xdata) * f)
        self.ax.set_ylim(ev.ydata - (ev.ydata - y0) * f, ev.ydata + (y1 - ev.ydata) * f)
        self.draw_idle()

    def reset_view(self):
        A = self.sc.area; pad = 0.04 * A
        self.ax.set_xlim(-pad, A + pad); self.ax.set_ylim(-pad, A + pad); self.draw_idle()


# ======================================================================================== 3D
_DEG = np.zeros((1, 2, 3))   # degenerate segment: matplotlib 3D cannot draw/autoscale empty line collections


def _unit_sphere(nlat, nlon):
    th = np.linspace(0, np.pi, nlat + 1)
    ph = np.linspace(0, 2 * np.pi, nlon + 1)
    P = np.stack([np.outer(np.sin(th), np.cos(ph)), np.outer(np.sin(th), np.sin(ph)),
                  np.outer(np.cos(th), np.ones_like(ph))], -1)
    q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)
    return q.reshape(-1, 4, 3)


class Canvas3D(FigureCanvasQTAgg):
    """Matplotlib 3D view. Editing is done via the tables on the right (matplotlib has no 3D picking)."""

    def __init__(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        self._P3, self._L3 = Poly3DCollection, Line3DCollection
        self.fig = Figure(facecolor=T.BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.sc = None
        self.drone_r = 0.05
        self.trails_on = True
        self._hist = None
        self._last_t = -1
        self.blitter = None

    def set_scenario(self, sc, drone_r):
        self.sc, self.drone_r = sc, drone_r
        ax, A, n, m = self.ax, sc.area, sc.n_agents, sc.n_obs
        ax.clear()
        ax.set_facecolor(T.BG)
        ax.set_xlim(0, A); ax.set_ylim(0, A); ax.set_zlim(0, A)
        ax.set_box_aspect((1, 1, 1), zoom=1.18)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.05, 0.06, 0.08, 1)); a.line.set_color(T.GRID)
            a._axinfo["grid"]["color"] = T.GRID
        ax.tick_params(colors=T.MUTED, labelsize=7)
        ax.set_xlabel("x", color=T.MUTED); ax.set_ylabel("y", color=T.MUTED); ax.set_zlabel("z", color=T.MUTED)
        self.cols = agent_colors(n) if n else np.zeros((0, 3))
        sz = 26 if n <= 60 else (12 if n <= 400 else 4)
        self.a_goal = ax.scatter(*(sc.goals.T if n else ([], [], [])), s=sz * 2.2, marker="*", c=self.cols if n else None,
                                 depthshade=False, edgecolors="white", linewidths=0.3)
        self.a_agent = ax.scatter(*(sc.starts.T if n else ([], [], [])), s=sz, c=self.cols if n else None,
                                  depthshade=False, edgecolors="none")
        self.a_link = self._L3(_DEG, colors=[to_rgba(c, .25) for c in self.cols] or [to_rgba('w', .25)], linewidths=0.6)
        ax.add_collection3d(self.a_link, autolim=False)
        self.a_trail = self._L3(_DEG, colors=[to_rgba(c, .5) for c in self.cols] or [to_rgba('w', .5)], linewidths=0.9)
        ax.add_collection3d(self.a_trail, autolim=False)
        self.a_ovel = self._L3(_DEG, colors=T.ORANGE, linewidths=1.5)
        ax.add_collection3d(self.a_ovel, autolim=False)
        # polygon budget: finer spheres when there are few obstacles, coarser as they multiply, markers beyond that
        self.mesh, self.obs_as_points = None, True
        for nl, no in ((6, 8), (4, 6), (3, 4), (2, 4)):
            if m * nl * no <= 900:
                self.mesh, self.obs_as_points = _unit_sphere(nl, no), False
                break
        self.a_obs = None
        if m:
            if self.obs_as_points:
                bb = ax.get_window_extent()
                px = min(bb.width, bb.height) / (A * 1.6)                    # rough pixels per arena unit
                diam_pt = 2 * sc.obs_radius * px * 72.0 / self.fig.dpi
                self.a_obs = ax.scatter(*sc.obs_pos.T, s=np.maximum(diam_pt, 3) ** 2, c=T.ORANGE, alpha=.45,
                                        depthshade=False, edgecolors="none")
            else:
                self.a_obs = self._P3(self._verts(sc.obs_pos, sc.obs_radius), facecolor=to_rgba(T.ORANGE, .38),
                                      edgecolor=to_rgba("#ffd8a8", .35), linewidths=0.3)
                ax.add_collection3d(self.a_obs, autolim=False)
        self._hist, self._last_t = None, -1
        self._set_links(sc.starts, sc.goals)
        self._set_ovel(sc.obs_pos, sc.obs_vel)
        arts = [self.a_link, self.a_trail, self.a_ovel, self.a_goal, self.a_agent] + ([self.a_obs] if self.a_obs is not None else [])
        # mplot3d artists must be re-projected with the current view matrix before being drawn on their own
        self.blitter = Blit(self, arts, pre_draw=lambda a: a.do_3d_projection())
        self.draw()

    def _verts(self, c, r):
        return (self.mesh[None] * r[:, None, None, None] + c[:, None, None, :]).reshape(-1, 4, 3)

    def _set_links(self, a, g):
        self.a_link.set_segments(np.stack([a, g], 1) if len(a) else _DEG)

    def _set_ovel(self, c, v):
        self.a_ovel.set_segments(np.stack([c, c + v], 1) if len(c) else _DEG)

    def show_scenario(self):
        self.set_scenario(self.sc, self.drone_r)

    def show_snapshot(self, s):
        sc = self.sc
        n = len(s["agent"])
        if n != sc.n_agents or len(s["obs"]) != sc.n_obs:
            return
        if self._hist is None or s["t"] < self._last_t:
            self._hist = np.repeat(s["agent"][None], 30, 0) if n else np.zeros((30, 0, 3))
            self._hpos = 0
        if s["t"] != self._last_t and n:
            self._hist[self._hpos % 30] = s["agent"]; self._hpos += 1
        self._last_t = s["t"]
        ag = s["agent"]
        if n:
            self.a_agent._offsets3d = (ag[:, 0], ag[:, 1], ag[:, 2])
            fc = self.cols.copy(); fc[s["colliding"]] = to_rgba(T.RED)[:3]
            self.a_agent.set_facecolor(fc)
            g = s["goal"]
            self.a_goal._offsets3d = (g[:, 0], g[:, 1], g[:, 2])
            self._set_links(ag, g)
            if self.trails_on and n <= 60:
                h = np.roll(self._hist, -self._hpos, axis=0)
                self.a_trail.set_segments(list(np.transpose(h, (1, 0, 2))))
            else:
                self.a_trail.set_segments(_DEG)
        if sc.n_obs:
            if self.obs_as_points:
                o = s["obs"]; self.a_obs._offsets3d = (o[:, 0], o[:, 1], o[:, 2])
            else:
                self.a_obs.set_verts(self._verts(s["obs"], s["obs_r"]))
            self._set_ovel(s["obs"], s["obs_vel"])
        self.blitter.update()
