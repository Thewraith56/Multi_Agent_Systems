"""Background thread for JAX compilation and simulation stepping."""
import queue
import time
import traceback

from PySide6.QtCore import QThread, Signal

from sim.engine import SimEngine


class SimWorker(QThread):
    status = Signal(str)
    configured = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.engine = SimEngine()
        self._cmds = queue.Queue()
        self.running = False
        self.sps = 33.0           # target simulation steps per second (33 ~ real time at dt=0.03)
        self.measured_sps = 0.0
        self.latest = None        # newest snapshot; GUI reads it and drops frames it cannot draw
        self._alive = True

    # ---- called from the GUI thread
    def configure(self, env_key, scenario, params, ckpt):
        self.running = False
        self._cmds.put(("configure", (env_key, scenario.copy(), dict(params), ckpt)))

    def reset(self):
        self._cmds.put(("reset", None))

    def step_once(self):
        self._cmds.put(("step", None))

    def shutdown(self):
        self._alive = False
        self.running = False

    # ---- worker thread
    def _handle(self, cmd, arg):
        try:
            if cmd == "configure":
                self.status.emit("Building graph & compiling JAX step (first time can take 10-60 s) ...")
                self.engine.configure(*arg)
                self.engine.step()          # warm-up: moves the one-off compile cost here
                self.latest = self.engine.reset()
                self.configured.emit(True, "ready")
            elif cmd == "reset" and self.engine.bundle is not None:
                self.latest = self.engine.reset()
            elif cmd == "step" and self.engine.bundle is not None:
                self.latest = self.engine.step()
        except Exception as e:  # noqa
            traceback.print_exc()
            self.running = False
            self.configured.emit(False, f"{type(e).__name__}: {e}")

    def run(self):
        n, t_mark = 0, time.perf_counter()
        while self._alive:
            try:
                cmd, arg = self._cmds.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                self._handle(cmd, arg)
                continue
            if self.running and self.engine.bundle is not None:
                t0 = time.perf_counter()
                try:
                    self.latest = self.engine.step()
                except Exception as e:  # noqa
                    traceback.print_exc()
                    self.running = False
                    self.configured.emit(False, f"{type(e).__name__}: {e}")
                    continue
                spent, target = time.perf_counter() - t0, 1.0 / max(self.sps, 1.0)
                if spent < target:
                    time.sleep(target - spent)
                n += 1
                now = time.perf_counter()
                if now - t_mark > 0.5:
                    self.measured_sps, n, t_mark = n / (now - t_mark), 0, now
            else:
                self.measured_sps = 0.0
                time.sleep(0.01)


