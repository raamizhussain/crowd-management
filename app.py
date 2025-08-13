from flask import Flask, render_template, jsonify, request
import os, cv2, time, base64, threading, queue, warnings, math, json
import numpy as np
from collections import deque

# Optional YOLO (people) — app runs even if missing
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

warnings.filterwarnings("ignore")
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
cv2.setLogLevel(0)

app = Flask(__name__)
VIDEO_FOLDER = os.path.join('static', 'videos')
os.makedirs(VIDEO_FOLDER, exist_ok=True)
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER


# =============== Utils ===============
def list_videos():
    try:
        return [f for f in os.listdir(VIDEO_FOLDER)
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
    except Exception:
        return []

def bgr_to_jpg_b64(img, quality=80):
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(enc).decode('utf-8')


# =============== Tracking & Geometry ===============
class TrackManager:
    """
    Tiny, fast nearest-centroid tracker with velocity & smoothing.
    Produces normalized top-down positions (x,z) and heading vectors.
    """
    def __init__(self, img_w, img_h):
        self.tracks = {}  # id -> dict(center, last_center, vel, last_seen, depth_raw)
        self.next_id = 1
        self.max_lost = 15
        self.img_w = img_w
        self.img_h = img_h

    def _est_depth_from_bbox_h(self, bbox_h):
        # Larger bbox height -> closer -> smaller z. Normalize z in [0,1].
        # Use a smooth clamp; typical person bbox height (close) ~ 60% of frame height
        h = max(1, bbox_h)
        ratio = h / max(1, self.img_h)
        z = 1.0 - np.clip((ratio - 0.05) / 0.55, 0.0, 1.0)  # 0.05..0.60 -> z 1..0
        return float(z)

    def update(self, detections):
        """
        detections: list of dict with keys: bbox [x1,y1,x2,y2], center [cx,cy]
        """
        now_centers = np.array([d['center'] for d in detections], dtype=np.float32) if detections else np.zeros((0,2), np.float32)
        assigned = {}
        # Build cost matrix (squared dist) vs existing tracks
        track_items = list(self.tracks.items())
        if len(track_items) and len(now_centers):
            costs = np.zeros((len(track_items), len(now_centers)), dtype=np.float32)
            for i, (tid, t) in enumerate(track_items):
                tc = np.array(t['center'], dtype=np.float32)
                dif = now_centers - tc
                costs[i] = np.sum(dif * dif, axis=1)
            # Greedy assignment
            used_rows, used_cols = set(), set()
            while True:
                min_val = None; min_r = None; min_c = None
                for r in range(costs.shape[0]):
                    if r in used_rows: continue
                    for c in range(costs.shape[1]):
                        if c in used_cols: continue
                        v = costs[r, c]
                        if (min_val is None) or (v < min_val):
                            min_val, min_r, min_c = v, r, c
                if min_val is None:
                    break
                if min_val <= (40**2):  # distance threshold (px^2)
                    used_rows.add(min_r); used_cols.add(min_c)
                    tid, t = track_items[min_r]
                    assigned[min_c] = tid
                else:
                    break

        # Update assigned and create new tracks for unassigned detections
        seen_now = set()
        for idx, det in enumerate(detections):
            cx, cy = det['center']
            x1, y1, x2, y2 = det['bbox']
            bbox_h = max(1, y2 - y1)
            if idx in assigned:
                tid = assigned[idx]
                t = self.tracks[tid]
                last_center = t['center']
                vel = (cx - last_center[0], cy - last_center[1])
                # EMA smoothing
                sm_cx = 0.7 * last_center[0] + 0.3 * cx
                sm_cy = 0.7 * last_center[1] + 0.3 * cy
                t['last_center'] = t['center']
                t['center'] = (sm_cx, sm_cy)
                t['vel'] = (0.7 * t['vel'][0] + 0.3 * vel[0], 0.7 * t['vel'][1] + 0.3 * vel[1])
                t['last_seen'] = 0
                t['depth_raw'] = self._est_depth_from_bbox_h(bbox_h)
                seen_now.add(tid)
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    'center': (float(cx), float(cy)),
                    'last_center': (float(cx), float(cy)),
                    'vel': (0.0, 0.0),
                    'last_seen': 0,
                    'depth_raw': self._est_depth_from_bbox_h(bbox_h)
                }
                seen_now.add(tid)

        # Age and remove stale tracks
        to_del = []
        for tid, t in self.tracks.items():
            if tid not in seen_now:
                t['last_seen'] += 1
            if t['last_seen'] > self.max_lost:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def export_for_minimap(self):
        """
        Returns list: {id, pos:{x,z}, dir:{dx,dz}}
        x in [0,1] left->right. z in [0,1] near(0)->far(1).
        """
        out = []
        for tid, t in self.tracks.items():
            cx, cy = t['center']
            dx, dy = t['vel']
            # Normalize lateral by image width; depth from bbox-derived z
            x_norm = float(np.clip(cx / max(1, self.img_w), 0, 1))
            z_norm = float(np.clip(t['depth_raw'], 0, 1))
            # Heading: screen space vel -> top view: right = +x, up (toward camera) ~ negative z.
            # We approximate dz from vertical velocity (dy): moving up in image -> likely farther (increase z) negative; invert sign.
            mag = math.hypot(dx, dy)
            if mag < 1e-3:
                dirx, dirz = 0.0, 0.0
            else:
                dirx = float(np.clip(dx / 20.0, -1, 1))
                dirz = float(np.clip(-dy / 20.0, -1, 1))
            out.append({'id': int(tid), 'pos': {'x': x_norm, 'z': z_norm}, 'dir': {'dx': dirx, 'dz': dirz}})
        return out


# =============== Fire Detector (HSV + Flicker) ===============
class FireHeuristic:
    """
    Lightweight fire detector that combines:
      1) HSV color gating for orange/yellow/red
      2) Motion/flicker mask (frame differencing)
      3) Contour area & solidity checks
    Returns list of fire regions (x1,y1,x2,y2) and a score in [0,1].
    """
    def __init__(self):
        self.prev_gray = None
        self.cooldown = 0

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # two ranges to capture red wraparound + orange/yellow
        lower1 = np.array([0,   80, 120], dtype=np.uint8)
        upper1 = np.array([15, 255, 255], dtype=np.uint8)
        lower2 = np.array([16,  60, 120], dtype=np.uint8)
        upper2 = np.array([45, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.medianBlur(mask, 5)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, motion = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion = cv2.medianBlur(motion, 5)

        hot = cv2.bitwise_and(mask, motion)
        hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        hot = cv2.morphologyEx(hot, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fire_boxes = []
        fire_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:  # ignore specks
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 10 or bh < 10:
                continue
            # solidity to avoid sparse noise
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull) + 1e-6
            solidity = area / hull_area
            if solidity < 0.4:
                continue
            fire_boxes.append([x, y, x+bw, y+bh])
            fire_area += area

        score = float(np.clip(fire_area / (w*h*0.08), 0, 1))  # scale vs 8% of frame
        return fire_boxes, score


# =============== People Detector ===============
class PeopleDetector:
    def __init__(self):
        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO('yolov8n.pt')  # person class 0
            except Exception:
                self.model = None

    def infer(self, frame_bgr):
        if self.model is None:
            return []  # no detections fallback
        try:
            res = self.model(frame_bgr, verbose=False)[0]
            dets = []
            for b in res.boxes:
                if int(b.cls[0]) == 0:  # person
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                    dets.append({'bbox':[x1,y1,x2,y2],'center':[cx,cy]})
            return dets
        except Exception:
            return []


# =============== Anomaly Logic (Fallen) ===============
def is_fallen(bbox):
    x1,y1,x2,y2 = bbox
    w = max(1, x2-x1); h = max(1, y2-y1)
    aspect = w / h
    # Fallen when very wide & short, and large enough area to be a person
    area_ok = (w*h) > 800
    return (aspect >= 1.65 and area_ok)


# =============== Processor & Threads ===============
class Processor:
    def __init__(self):
        self.cap = None
        self.running = False
        self.reader_t = None
        self.worker_t = None
        self.watchdog_t = None

        self.frame_q = queue.Queue(maxsize=3)
        self.detector = PeopleDetector()
        self.fire = FireHeuristic()

        self._latest = None
        self._latest_lock = threading.Lock()
        self._fps_times = deque(maxlen=30)

        self.tracker = None  # built when first frame size known

    def start(self, path):
        self.stop()
        if not os.path.exists(path):
            return {'success': False, 'error': 'Video not found'}
        self.cap = cv2.VideoCapture(path)
        if not self.cap or not self.cap.isOpened():
            return {'success': False, 'error': 'Could not open video'}

        # prime size & tracker
        ok, fr = self.cap.read()
        if not ok:
            self.cap.release(); self.cap=None
            return {'success': False, 'error': 'Could not read first frame'}
        h, w = fr.shape[:2]
        self.tracker = TrackManager(w, h)
        # rewind
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.running = True
        self.reader_t = threading.Thread(target=self._reader, daemon=True)
        self.worker_t = threading.Thread(target=self._worker, daemon=True)
        self.reader_t.start(); self.worker_t.start()
        self._start_watchdog()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 25)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {'success': True, 'fps': fps, 'frame_count': count}

    def stop(self):
        self.running = False
        try:
            if self.reader_t and self.reader_t.is_alive():
                self.reader_t.join(timeout=1)
        except Exception: pass
        try:
            if self.worker_t and self.worker_t.is_alive():
                self.worker_t.join(timeout=1)
        except Exception: pass
        if self.cap:
            try: self.cap.release()
            except Exception: pass
        self.cap = None
        with self._latest_lock:
            self._latest = None
        while not self.frame_q.empty():
            try: self.frame_q.get_nowait()
            except Exception: break

    def _start_watchdog(self):
        def wd():
            while self.running:
                time.sleep(5)
                if self.reader_t and not self.reader_t.is_alive():
                    self.reader_t = threading.Thread(target=self._reader, daemon=True); self.reader_t.start()
                if self.worker_t and not self.worker_t.is_alive():
                    self.worker_t = threading.Thread(target=self._worker, daemon=True); self.worker_t.start()
        self.watchdog_t = threading.Thread(target=wd, daemon=True); self.watchdog_t.start()

    def _reader(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                # loop for demo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            # throttle size for speed
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960.0 / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except Exception:
                pass
            time.sleep(0.01)

    def _worker(self):
        frame_id = 0
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.5)
            except Exception:
                continue

            frame_id += 1
            detections = self.detector.infer(frame)

            # Fallen detection
            fallen = []
            for d in detections:
                if is_fallen(d['bbox']):
                    fallen.append({'id': None, 'bbox': d['bbox']})  # id filled after tracking

            # Tracking + 3D-ish projection
            if self.tracker is None:
                h, w = frame.shape[:2]
                self.tracker = TrackManager(w, h)
            self.tracker.update(detections)
            tracks = self.tracker.export_for_minimap()

            # Fill fallen IDs by nearest track center
            for f in fallen:
                fx = (f['bbox'][0] + f['bbox'][2]) / 2.0
                fy = (f['bbox'][1] + f['bbox'][3]) / 2.0
                best, best_d = None, 1e9
                for t in tracks:
                    tcx = t['pos']['x'] * self.tracker.img_w
                    tcy = (1 - t['pos']['z']) * self.tracker.img_h  # not perfect but okay
                    d2 = (tcx - fx)**2 + (tcy - fy)**2
                    if d2 < best_d:
                        best_d, best = d2, t['id']
                f['id'] = int(best) if best is not None else -1

            # Fire
            fire_boxes, fire_score = self.fire.detect(frame)

            # Density (2D coarse)
            density = self._density(detections, frame.shape)

            # Annotate frame
            annotated = frame.copy()
            for d in detections:
                (x1,y1,x2,y2) = d['bbox']
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (60,220,255), 2)
            # fallen marks
            for f in fallen:
                (x1,y1,x2,y2) = f['bbox']
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(annotated, "FALLEN", (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            # fire marks
            for (x1,y1,x2,y2) in fire_boxes:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,165,255), 2)
                cv2.putText(annotated, "FIRE", (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            # HUD
            cv2.rectangle(annotated, (8,8), (360,36), (0,0,0), -1)
            hud = f"People: {len(detections)} | Density: {density['level']} | Fire: {fire_score:.2f}"
            cv2.putText(annotated, hud, (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            with self._latest_lock:
                self._latest = {
                    'annotated': annotated,
                    'timestamp': time.time(),
                    'people_count': len(detections),
                    'density': density,
                    'tracks': tracks,
                    'fallen': fallen,
                    'fire': {'regions': fire_boxes, 'score': float(fire_score)}
                }

            self._fps_times.append(time.time())

    def _density(self, detections, shape, grid=50):
        h, w = shape[:2]
        cols = max(1, w // grid)
        rows = max(1, h // grid)
        g = np.zeros((rows, cols), np.float32)
        for d in detections:
            cx, cy = d['center']
            c = min(cols-1, max(0, cx // grid))
            r = min(rows-1, max(0, cy // grid))
            g[r, c] += 1

        total = len(detections)
        ppl_per_cell = total / max(1, cols*rows)
        if total >= 15: level = 'HIGH'
        elif total >= 6: level = 'MEDIUM'
        else: level = 'LOW'
        return {'total': int(total), 'per_cell': float(round(ppl_per_cell,2)), 'level': level}

    def latest(self):
        with self._latest_lock:
            if self._latest is None: return None
            # clone and convert
            out = dict(self._latest)
            # ensure jpg b64 here to avoid double encode in route
            return out

    def fps(self):
        if len(self._fps_times) < 2: return 0.0
        dt = self._fps_times[-1] - self._fps_times[0]
        if dt <= 0: return 0.0
        return round((len(self._fps_times)-1)/dt, 1)


PROC = Processor()


# =============== Routes ===============
@app.route('/')
def home():
    return render_template('index.html', videos=list_videos())

@app.route('/start_cctv_stream', methods=['POST'])
def start_cctv_stream():
    data = request.get_json(silent=True) or {}
    name = data.get('video_name')
    if not name or name not in list_videos():
        return jsonify({'success': False, 'error': 'Invalid video selection'})
    path = os.path.join(VIDEO_FOLDER, name)
    res = PROC.start(path)
    if not res.get('success'):
        return jsonify({'success': False, 'error': res.get('error','failed')})
    return jsonify({'success': True, 'fps': res['fps'], 'frame_count': res['frame_count'], 'mode':'LIVE_SIM'})

@app.route('/get_cctv_frame')
def get_cctv_frame():
    L = PROC.latest()
    if L is None:
        return jsonify({'success': False, 'message': 'No frame yet'})
    img64 = bgr_to_jpg_b64(L['annotated'])
    if img64 is None:
        return jsonify({'success': False, 'message': 'Encode error'})
    return jsonify({
        'success': True,
        'frame_image': f"data:image/jpeg;base64,{img64}",
        'stream_data': {
            'timestamp': L['timestamp'],
            'processing_fps': PROC.fps(),
            'people_count': L['people_count'],
            'density': L['density'],
            'tracks': L['tracks'],       # [{id, pos{x,z}, dir{dx,dz}}]
        },
        'anomalies': {
            'fallen_people': L['fallen'],     # [{id, bbox}]
            'fire': L['fire']                 # {'regions': [[x1,y1,x2,y2], ...], 'score': float}
        }
    })

@app.route('/stop_cctv_stream', methods=['POST'])
def stop_cctv_stream():
    PROC.stop()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
