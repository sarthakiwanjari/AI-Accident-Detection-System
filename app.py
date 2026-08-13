"""
app.py — AccidentWatch v3.2  STABLE
======================================================
Root cause fixes:
  - Uses threading async_mode (not eventlet) → FPS works
  - Processing loop uses daemon thread correctly
  - Upload uses werkzeug stream → no memory issues
  - Video source switching is thread-safe
  - Default tab is Live Feed
======================================================
"""

import os, cv2, time, threading, logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

from detection_engine import AccidentDetector
from alerts.alert_manager import AlertManager

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "aw_v3")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

# IMPORTANT: threading mode — eventlet was blocking the loop
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Config from .env ─────────────────────────
USE_GPU     = os.getenv("USE_GPU", "True").lower() == "true"
YOLO_MODEL  = os.getenv("YOLO_MODEL", "yolov8s.pt")
PROC_WIDTH  = int(os.getenv("PROCESS_WIDTH", 640))
CONF_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
COOLDOWN    = int(os.getenv("ALERT_COOLDOWN_SECONDS", 15))
SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", 1))

DEFAULT_LOCATION = {
    "lat":  float(os.getenv("DEFAULT_LOCATION_LAT",  19.9975)),
    "lng":  float(os.getenv("DEFAULT_LOCATION_LNG",  73.7898)),
    "name": os.getenv("DEFAULT_LOCATION_NAME", "Nashik, Maharashtra, India"),
}

# ── Detector ─────────────────────────────────
detector = AccidentDetector(
    model_path     = YOLO_MODEL,
    use_gpu        = USE_GPU,
    process_width  = PROC_WIDTH,
    conf_thresh    = CONF_THRESH,
    alert_cooldown = COOLDOWN,
    skip_frames    = SKIP_FRAMES,
)
alert_mgr    = AlertManager()
incident_log = []

# ── Video state (protected by lock) ──────────
_src_lock    = threading.Lock()
_video_src   = 0          # 0 = webcam, or string path
_src_type    = "webcam"
_src_name    = "Live Camera"

_cap         = None
_cap_lock    = threading.Lock()

_frame_lock  = threading.Lock()
_live_frame  = None
_hmap_frame  = None

_fps_times   = []
_live_fps    = 0.0
TARGET_FPS   = 25


def open_capture():
    """Open/reopen video capture with current source."""
    global _cap
    with _src_lock:
        src = _video_src
    c = cv2.VideoCapture(src)
    if c.isOpened():
        c.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        fps = c.get(cv2.CAP_PROP_FPS) or 30
        detector.fps = min(float(fps), 30.0)
        logger.info(f"Opened: {src}  src_fps={fps:.0f}")
    else:
        logger.error(f"Cannot open: {src}")
    return c


# ── Accident callback ─────────────────────────
def on_accident(event):
    global incident_log

    snap_url = None
    if event.frame_snapshot is not None:
        snap_dir = Path("static/snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        fname = f"inc_{len(incident_log)+1}_{int(event.timestamp)}.jpg"
        cv2.imwrite(str(snap_dir / fname), event.frame_snapshot)
        snap_url = f"/static/snapshots/{fname}"

    loc  = event.location
    lat  = loc.get("lat",  DEFAULT_LOCATION["lat"])
    lng  = loc.get("lng",  DEFAULT_LOCATION["lng"])
    name = loc.get("name", DEFAULT_LOCATION["name"])
    maps = f"https://maps.google.com/?q={lat},{lng}"

    inc = {
        "id":                len(incident_log) + 1,
        "timestamp":         datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S"),
        "date":              datetime.fromtimestamp(event.timestamp).strftime("%d %b %Y"),
        "severity":          event.severity,
        "severity_label":    event.severity_label,
        "accident_score":    round(getattr(event, "accident_score", 0.75) * 100, 1),
        "confidence":        round(event.confidence * 100, 1),
        "vehicles_involved": len(event.vehicles_involved),
        "location_name":     name,
        "lat":               lat,
        "lng":               lng,
        "maps_url":          maps,
        "snapshot_url":      snap_url,
    }
    incident_log.append(inc)
    if len(incident_log) > 300:
        incident_log.pop(0)

    socketio.emit("accident_alert", inc)

    threading.Thread(
        target=alert_mgr.send_alert, args=(event,), daemon=True).start()
    if event.severity == 3:
        threading.Thread(
            target=alert_mgr.send_voice_call, args=(event,), daemon=True).start()

    logger.info(
        f"INCIDENT #{inc['id']} | {event.severity_label} | "
        f"score={inc['accident_score']}% | {name}\n"
        f"Maps: {maps}"
    )


detector.on_accident_detected = on_accident


# ── Main processing loop ──────────────────────
def processing_loop():
    """Runs in its own daemon thread. Never blocks Flask."""
    global _live_frame, _hmap_frame, _live_fps

    logger.info("Processing loop started")
    delay      = 1.0 / TARGET_FPS
    cap        = open_capture()
    last_src   = _video_src

    while True:
        t0 = time.perf_counter()

        # Reopen if source changed
        with _src_lock:
            current_src = _video_src
        if current_src != last_src:
            logger.info(f"Source changed → {current_src}")
            if cap: cap.release()
            cap      = open_capture()
            last_src = current_src

        ret, frame = cap.read() if cap and cap.isOpened() else (False, None)

        if not ret:
            logger.warning("Frame read failed — reopening capture")
            if cap:
                cap.release()
            time.sleep(0.3)
            cap = open_capture()
            continue

        # Run detection
        try:
            result = detector.process_frame(frame, location=DEFAULT_LOCATION)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            time.sleep(0.1)
            continue

        # Store latest frames
        with _frame_lock:
            _live_frame = result["frame"].copy()
            _hmap_frame = detector.get_heatmap_frame(frame.copy())

        # FPS calc
        now = time.perf_counter()
        _fps_times.append(now)
        if len(_fps_times) > 30: _fps_times.pop(0)
        if len(_fps_times) >= 2:
            _live_fps = round(len(_fps_times) / (_fps_times[-1] - _fps_times[0] + 1e-9), 1)
            detector.fps = _live_fps

        # Push stats every 20 frames
        if detector.frame_count % 20 == 0:
            s = result["stats"]
            s.update({"live_fps": _live_fps, "source": _src_type, "source_name": _src_name})
            try:
                socketio.emit("stats_update", s)
            except Exception:
                pass

        # Sleep to hit target FPS
        elapsed = time.perf_counter() - t0
        time.sleep(max(0, delay - elapsed))


# Start processing thread
_proc_thread = threading.Thread(target=processing_loop, daemon=True, name="DetectionLoop")
_proc_thread.start()


# ── MJPEG helpers ─────────────────────────────
def encode_jpg(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

def gen_live():
    while True:
        with _frame_lock:
            f = _live_frame
        if f is not None:
            yield encode_jpg(f)
        time.sleep(1 / TARGET_FPS)

def gen_hmap():
    while True:
        with _frame_lock:
            f = _hmap_frame
        if f is not None:
            yield encode_jpg(f)
        time.sleep(1 / 10)


# ── Routes ────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html",
                           location=DEFAULT_LOCATION,
                           device=detector.device,
                           model=YOLO_MODEL)

@app.route("/video_feed")
def video_feed():
    return Response(gen_live(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/heatmap_feed")
def heatmap_feed():
    return Response(gen_hmap(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Stream-save uploaded video file then switch source."""
    global _video_src, _src_type, _src_name, _cap

    if "video" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f.filename
    total = 0
    CHUNK = 512 * 1024  # 512 KB chunks

    try:
        with open(save_path, "wb") as out:
            while True:
                chunk = f.stream.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    size_mb = round(total / 1024 / 1024, 1)
    logger.info(f"Upload saved: {save_path}  ({size_mb} MB)")

    # Switch source
    with _src_lock:
        _video_src = str(save_path)
        _src_type  = "video"
        _src_name  = f.filename

    # Signal processing loop to reopen
    with _cap_lock:
        if _cap:
            _cap.release()
            _cap = None

    socketio.emit("source_changed", {"source": "video", "filename": f.filename})
    return jsonify({"status": "ok", "file": f.filename, "size_mb": size_mb})


@app.route("/api/use_local_path", methods=["POST"])
def use_local_path():
    """Use a video file already on the PC without uploading."""
    global _video_src, _src_type, _src_name, _cap
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path or not Path(path).exists():
        return jsonify({"error": f"File not found: {path}"}), 404

    with _src_lock:
        _video_src = path
        _src_type  = "video"
        _src_name  = Path(path).name

    with _cap_lock:
        if _cap:
            _cap.release()
            _cap = None

    socketio.emit("source_changed", {"source": "video", "filename": Path(path).name})
    logger.info(f"Local path set: {path}")
    return jsonify({"status": "ok", "file": Path(path).name})


@app.route("/api/switch_webcam")
def switch_webcam():
    global _video_src, _src_type, _src_name, _cap
    with _src_lock:
        _video_src = 0
        _src_type  = "webcam"
        _src_name  = "Live Camera"
    with _cap_lock:
        if _cap:
            _cap.release()
            _cap = None
    socketio.emit("source_changed", {"source": "webcam", "filename": "Live Camera"})
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def api_stats():
    s = detector._build_stats()
    s.update({"live_fps": _live_fps, "source": _src_type, "source_name": _src_name})
    return jsonify(s)


@app.route("/api/incidents")
def api_incidents():
    return jsonify({"incidents": incident_log[-50:][::-1], "total": len(incident_log)})


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok", "frame": detector.frame_count,
        "fps": _live_fps, "device": detector.device,
        "source": _src_type, "vehicles": len(detector.vehicles),
        "incidents": len(incident_log),
    })


@app.route("/api/test_alert")
def test_alert():
    from detection_engine import AccidentEvent
    evt = AccidentEvent(
        timestamp=time.time(), severity=2, severity_label="Moderate",
        accident_score=0.82, vehicles_involved=[101, 102],
        location=DEFAULT_LOCATION, confidence=0.91,
    )
    on_accident(evt)
    return jsonify({
        "status": "fired",
        "location": DEFAULT_LOCATION,
        "maps": f"https://maps.google.com/?q={DEFAULT_LOCATION['lat']},{DEFAULT_LOCATION['lng']}",
        "recipients": alert_mgr.recipients,
    })


@app.route("/api/clear_incidents")
def clear_incidents():
    global incident_log
    incident_log = []
    detector.total_accidents = 0
    socketio.emit("incidents_cleared", {})
    return jsonify({"status": "ok"})


@socketio.on("connect")
def on_connect():
    s = detector._build_stats()
    s.update({"live_fps": _live_fps, "source": _src_type, "source_name": _src_name})
    emit("stats_update", s)
    emit("incident_history", {"incidents": incident_log[-20:][::-1]})


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  AccidentWatch v3.2  STABLE")
    logger.info(f"  GPU     : {detector.device}")
    logger.info(f"  Model   : {YOLO_MODEL}")
    logger.info(f"  Alerts  : {alert_mgr.recipients}")
    logger.info(f"  Location: {DEFAULT_LOCATION['name']}")
    logger.info(f"  Open    : http://localhost:5000")
    logger.info("=" * 55)
    # Use threaded=True — NOT eventlet
    socketio.run(app, host="0.0.0.0", port=5000,
                 debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
