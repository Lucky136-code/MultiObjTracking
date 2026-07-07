# Real-time vehicle tracking overlay using YOLOv8 and ByteTrack
# Captures the screen, detects vehicles, and draws tracking boxes directly on top
# Supports Car, SUV, Bike, Truck, Bus with speed estimation in km/h

import tkinter as tk
from tkinter import messagebox
import threading, time
import mss
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque

# Model and detection settings
MODEL    = "yolov8n.pt"
CONF     = 0.22
CLASSES  = [2, 3, 5, 7]  # car, motorcycle, bus, truck
IMGSZ    = 416
TRAIL    = 22
CAR_W_M  = 2.0
SPD_GAIN = 0.12
SPD_WIN  = 18
DISP_MS  = 33
MAX_EX   = 0.22

# Color scheme for each vehicle type and UI elements
C = {
    "Car":"#00FF41", "SUV":"#88FFAA", "Bike":"#00CFFF",
    "Truck":"#FF9900", "Bus":"#FF4444",
    "spd":"#FFE500", "hud":"#0D0D0D", "acc":"#00FF41",
    "txt":"white",   "dim":"#555555", "bg":"black"
}


def classify(cls_id, bw, bh):
    # Map COCO class ID and bounding box shape to a readable vehicle type
    if cls_id == 3: return "Bike"
    if cls_id == 5: return "Bus"
    if cls_id == 7: return "Truck"
    # Distinguish Car vs SUV using height-to-width aspect ratio
    return "SUV" if bh / max(bw, 1) > 0.65 else "Car"


class VelPred:
    # Predicts where a vehicle box will be between detection frames
    # using its measured pixel velocity, so boxes stay on the vehicle
    def __init__(self, box, t):
        self.b = np.array(box, dtype=float)
        self.v = np.zeros(4)
        self.t = t

    def update(self, box, t):
        dt = t - self.t
        nb = np.array(box, dtype=float)
        if dt > 0.008:
            raw = (nb - self.b) / dt
            self.v = 0.45 * self.v + 0.55 * raw
        self.b, self.t = nb, t

    def get(self, t):
        dt = min(t - self.t, MAX_EX)
        return (self.b + self.v * dt).astype(int)


class State:
    # Shared memory between detection thread and rendering thread
    def __init__(self):
        self._l = threading.Lock()
        self.tracks = []
        self.fps    = 0.0

    def put(self, tracks, fps):
        with self._l:
            self.tracks = tracks
            self.fps    = fps

    def get(self):
        with self._l:
            return list(self.tracks), self.fps


class DetThread(threading.Thread):
    # Runs YOLO + ByteTrack in background, writes results to State
    def __init__(self, state: State):
        super().__init__(daemon=True)
        self.state  = state
        self._stop  = threading.Event()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL} on {self.device.upper()} with ByteTrack...")
        self.model  = YOLO(MODEL)
        self.sct    = mss.MSS()
        self.mon    = self.sct.monitors[1]
        self.trajs  = defaultdict(lambda: deque(maxlen=TRAIL))
        self.tpts   = defaultdict(lambda: deque(maxlen=TRAIL))
        self.spds   = {}

    def run(self):
        ft = deque(maxlen=25)
        while not self._stop.is_set():
            t0    = time.perf_counter()
            img   = self.sct.grab(self.mon)
            frame = np.array(img)[:, :, :3]

            # Run detection and tracking in a single call using ByteTrack
            results = self.model.track(
                frame, persist=True, verbose=False,
                conf=CONF, classes=CLASSES, imgsz=IMGSZ,
                tracker="bytetrack.yaml"
            )

            now = time.perf_counter()
            out = []

            for r in results:
                if r.boxes.id is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                ids  = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(xyxy, ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    bw = max(x2-x1, 1)
                    bh = max(y2-y1, 1)
                    vt = classify(cls, bw, bh)
                    cx = (x1+x2)//2
                    cy = y2

                    self.trajs[tid].append((cx, cy))
                    self.tpts[tid].append(now)

                    # Estimate speed using pixel displacement over time
                    spd  = 0.0
                    traj = self.trajs[tid]
                    tps  = self.tpts[tid]
                    n    = len(traj)
                    win  = min(n-1, SPD_WIN)

                    if win >= 4:
                        dt = tps[-1] - tps[-win]
                        if dt > 0.15:
                            dx  = traj[-1][0] - traj[-win][0]
                            dy  = traj[-1][1] - traj[-win][1]
                            px  = (dx*dx + dy*dy) ** 0.5
                            ppm = bw / CAR_W_M
                            spd = (px / ppm / dt) * 3.6

                    prev = self.spds.get(tid, spd)
                    # Ignore sudden jumps that are likely noise
                    if abs(spd - prev) > 60:
                        spd = prev
                    spd = prev + SPD_GAIN * (spd - prev)
                    self.spds[tid] = spd

                    out.append({
                        "id": tid, "type": vt, "speed": spd,
                        "box": (x1, y1, x2, y2),
                        "trail": list(traj), "t": now
                    })

            ft.append(time.perf_counter() - t0)
            avg = sum(ft) / len(ft)
            self.state.put(out, 1/avg if avg > 0 else 0)

    def stop(self):
        self._stop.set()
        self.sct.close()


class Overlay:
    # Transparent fullscreen tkinter window that draws tracking boxes on screen
    def __init__(self, root: tk.Tk, state: State):
        self.root    = root
        self.state   = state
        self.running = True
        self.preds   = {}

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")
        root.overrideredirect(True)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-transparentcolor", C["bg"])
        root.config(bg=C["bg"])

        self.cv = tk.Canvas(root, bg=C["bg"],
                            highlightthickness=0, cursor="none")
        self.cv.pack(fill="both", expand=True)
        root.bind("<Escape>", self._quit)
        root.bind("q",        self._quit)
        self._loop()

    def _draw_box(self, x1, y1, x2, y2, col):
        # Draw a corner-bracket style box instead of a plain rectangle
        L = min(20, max(8, int((x2-x1) * 0.18)))
        self.cv.create_rectangle(x1, y1, x2, y2, outline=col, width=1)
        corners = [
            ((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
            ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
            ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
            ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L)),
        ]
        for a, b in corners:
            self.cv.create_line(*a, *b, fill=col, width=2)

    def _loop(self):
        if not self.running:
            return
        self.cv.delete("all")
        tracks, fps = self.state.get()
        now = time.perf_counter()

        # Update or create a velocity predictor for each active track
        seen = set()
        for t in tracks:
            tid = t["id"]; seen.add(tid)
            if tid not in self.preds:
                self.preds[tid] = VelPred(t["box"], t["t"])
            else:
                self.preds[tid].update(t["box"], t["t"])
        for tid in list(self.preds):
            if tid not in seen:
                del self.preds[tid]

        info = {t["id"]: t for t in tracks}

        for tid, pred in self.preds.items():
            if tid not in info:
                continue
            t   = info[tid]
            col = C.get(t["type"], C["Car"])
            x1, y1, x2, y2 = pred.get(now)

            # Draw motion trail with a fading gradient effect
            trail = t["trail"]; n = len(trail)
            for i in range(1, n):
                f  = i / n
                g  = int(80  + f*175)
                b2 = int(140 + f*80)
                self.cv.create_line(
                    *trail[i-1], *trail[i],
                    fill=f"#00{g:02x}{b2:02x}",
                    width=max(1, int(f * 2.5))
                )

            self._draw_box(x1, y1, x2, y2, col)

            # Label showing vehicle type and speed
            spd = t["speed"]
            lbl = (f" {t['type']}  {spd:.0f} km/h "
                   if spd > 2.0 else f" {t['type']} #{tid} ")
            lw  = len(lbl)*8 + 4
            lh  = 22
            lx  = x1
            ly  = max(2, y1 - lh - 4)
            self.cv.create_rectangle(lx, ly, lx+lw, ly+lh,
                                     fill="#080808", outline=col, width=1)
            self.cv.create_text(lx+5, ly+4, anchor="nw",
                                text=lbl, fill=C["spd"],
                                font=("Consolas", 10, "bold"))

        # HUD panel showing live stats
        hx, hy, hw, hh = 16, 16, 285, 118
        self.cv.create_rectangle(hx, hy, hx+hw, hy+hh,
                                 fill=C["hud"], outline=C["acc"], width=1)
        self.cv.create_text(hx+10, hy+10, anchor="nw",
                            text="◉  VEHICLE TRACKER",
                            fill=C["acc"], font=("Consolas", 11, "bold"))
        mode = "GPU" if torch.cuda.is_available() else "CPU · ByteTrack"
        ry = hy + 37
        for k, v in [("Tracked", str(len(info))),
                     ("Detect FPS", f"{fps:.1f}"),
                     ("Engine",  mode)]:
            self.cv.create_text(hx+10, ry, anchor="nw",
                                text=f"{k:<20}{v}",
                                fill=C["txt"], font=("Consolas", 9))
            ry += 20
        self.cv.create_text(hx+10, ry+3, anchor="nw",
                            text="[Esc] to quit",
                            fill=C["dim"], font=("Consolas", 8))

        # Colour legend for vehicle types
        lx2 = hx + hw + 12
        ly2 = hy
        for lbl, col in [("Car",  C["Car"]),  ("SUV",  C["SUV"]),
                         ("Bike", C["Bike"]), ("Truck",C["Truck"]),
                         ("Bus",  C["Bus"])]:
            self.cv.create_rectangle(lx2, ly2, lx2+12, ly2+12,
                                     fill=col, outline="")
            self.cv.create_text(lx2+16, ly2, anchor="nw",
                                text=lbl, fill=C["txt"],
                                font=("Consolas", 9))
            ly2 += 20

        self.root.after(DISP_MS, self._loop)

    def _quit(self, _ = None):
        self.running = False
        self.root.destroy()


def main():
    # Show a confirmation dialog before starting
    ask = tk.Tk(); ask.withdraw()
    ok  = messagebox.askyesno(
        "Vehicle Tracker",
        "🚗  Vehicle Speed Tracker\n\n"
        "Detects: Car · SUV · Bike · Truck · Bus\n\n"
        "1. Make sure your video is fully visible\n"
        "2. Click Yes to start the overlay\n"
        "3. Press  Esc  at any time to quit"
    )
    ask.destroy()
    if not ok:
        return

    state = State()
    det   = DetThread(state)
    det.start()

    root = tk.Tk()
    Overlay(root, state)
    root.mainloop()
    det.stop()
    det.join(timeout=2)


if __name__ == "__main__":
    main()
