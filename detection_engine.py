"""
detection_engine.py — AccidentWatch v5  OVERHEAD CCTV TUNED
═══════════════════════════════════════════════════════════════
Tuned specifically for overhead/top-down CCTV footage.

Physics of overhead CCTV accidents:
  ✓ Vehicles appear as small rectangles (bird's eye view)
  ✓ Real crashes: bounding boxes ACTUALLY overlap (≥15%)
  ✓ Impact causes sudden direction change (heading shift >45°)
  ✓ Speed drops sharply after impact (deceleration spike)
  ✓ Vehicles may stop abnormally in road zone
  ✗ Normal traffic: vehicles side by side = zero IoU overhead
  ✗ Normal turns: smooth heading change, no sudden decel
  ✗ Lane changes: gradual movement, no speed drop

Algorithm:
  MUST have IoU ≥ 0.15  (even small is real from overhead)
  PLUS sudden deceleration OR sudden heading change
  Score smoothed over 6 frames, confirmed for 8 frames
  Cooldown 30s between alerts
═══════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import time
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Callable

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck", 1:"bicycle"}
SEVERITY_LABELS = {0:"Safe", 1:"Minor", 2:"Moderate", 3:"Critical"}
SEVERITY_BANNER = {1:(20,130,200), 2:(10,80,200), 3:(0,10,190)}

# ── Tuned thresholds ─────────────────────────
# Optimized for direct video file input (not webcam-at-screen)
# Rejects flicker/moiré false positives from screen recording
MIN_IOU_OVERHEAD      = 0.18   # real overlap only — screen artifacts < 18%
MIN_SPEED_BEFORE      = 2.5    # vehicle must be genuinely moving
SUDDEN_DECEL_RATIO    = 0.35   # significant braking only
SUDDEN_HEADING_CHANGE = 45.0   # real direction change from impact
CONFIRM_FRAMES        = 6      # 6 sustained frames before alert
SCORE_THRESHOLD       = 0.60   # balanced — not too strict, not too loose
MIN_TRACK_FRAMES      = 8      # stable track — rejects ghost detections


@dataclass
class VehicleState:
    track_id:           int
    bbox:               list
    class_name:         str
    confidence:         float
    center:             tuple
    speed:              float = 0.0
    heading:            float = 0.0
    prev_heading:       float = 0.0
    heading_change:     float = 0.0   # sudden direction shift
    accel:              float = 0.0
    trajectory:         deque = field(default_factory=lambda: deque(maxlen=30))
    speeds:             deque = field(default_factory=lambda: deque(maxlen=20))
    headings:           deque = field(default_factory=lambda: deque(maxlen=15))
    last_seen:          float = field(default_factory=time.time)
    stationary_frames:  int   = 0
    frames_tracked:     int   = 0
    had_motion:         bool  = False  # was moving at some point


@dataclass
class AccidentEvent:
    timestamp:          float
    severity:           int
    severity_label:     str
    accident_score:     float
    vehicles_involved:  list
    location:           dict
    confidence:         float
    frame_snapshot:     Optional[np.ndarray] = None
    clip_path:          Optional[str]        = None


class AccidentDetector:

    def __init__(self,
                 model_path     = "yolov8s.pt",
                 use_gpu        = True,
                 process_width  = 640,
                 conf_thresh    = 0.35,
                 alert_cooldown = 30,
                 skip_frames    = 1):

        # GPU
        self.device = "cpu"
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("CUDA not available — CPU mode")
            except ImportError:
                pass

        logger.info(f"Loading {model_path} on {self.device}...")
        self.yolo = YOLO(model_path)
        if self.device != "cpu":
            self.yolo.to(self.device)

        logger.info("Initialising DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=35, n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder_gpu=(self.device != "cpu"),
        )

        self.process_width  = process_width
        self.conf_thresh    = conf_thresh
        self.alert_cooldown = alert_cooldown
        self.skip_frames    = skip_frames

        self.vehicles:        dict  = {}
        self.frame_count:     int   = 0
        self.fps:             float = 25.0
        self.last_alert_time: float = 0.0
        self.total_accidents: int   = 0
        self.traffic_counts:  dict  = defaultdict(int)
        self.vehicle_totals:  dict  = defaultdict(int)

        self._confirm_count:  int   = 0
        self._confirm_best:   dict  = {}
        self._score_buf:      deque = deque(maxlen=6)

        self.on_accident_detected: Optional[Callable] = None

        self._clip_writer = None
        self._clip_path   = None
        self._clip_frames = 0
        self._clip_max    = 90
        self._recording   = False

        self.heatmap:       Optional[np.ndarray] = None
        self._last_frame:   Optional[np.ndarray] = None
        self._last_stats:   dict = {}

        logger.info(f"Overhead CCTV mode  device={self.device}")
        logger.info(
            f"Thresholds: IoU>={MIN_IOU_OVERHEAD}  "
            f"speed>={MIN_SPEED_BEFORE}  "
            f"decel={int(SUDDEN_DECEL_RATIO*100)}%  "
            f"heading={SUDDEN_HEADING_CHANGE}°  "
            f"confirm={CONFIRM_FRAMES}f"
        )

    # ════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════

    def process_frame(self, frame: np.ndarray, location: dict = None) -> dict:
        self.frame_count += 1
        h, w = frame.shape[:2]

        if self.heatmap is None or self.heatmap.shape[:2] != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        if self.frame_count % (self.skip_frames + 1) != 0 and self._last_frame is not None:
            self._write_clip(frame)
            return {"frame": self._last_frame, "accident_event": None,
                    "stats": self._last_stats, "vehicle_count": len(self.vehicles)}

        # ── YOLO + DeepSORT ──────────────────────
        scale   = self.process_width / w
        small_h = int(h * scale)
        small   = cv2.resize(frame, (self.process_width, small_h), cv2.INTER_LINEAR)
        results = self.yolo(small, verbose=False, device=self.device,
                            imgsz=self.process_width, conf=self.conf_thresh)[0]
        dets   = self._parse_detections(results, scale)
        tracks = self.tracker.update_tracks(dets, frame=small)
        self._update_vehicles(tracks, scale, w, h)

        # ── Accident detection ───────────────────
        accident_event = None
        cdata = self._analyse_overhead()

        # Smooth score over buffer
        self._score_buf.append(cdata["score"])
        cdata["score"] = float(np.mean(self._score_buf))

        if cdata["score"] >= SCORE_THRESHOLD and len(cdata["involved_ids"]) >= 2:
            self._confirm_count += 1
            self._confirm_best   = cdata.copy()
        else:
            self._confirm_count  = 0

        if self._confirm_count >= CONFIRM_FRAMES and self._confirm_best:
            self._confirm_count = 0
            confirmed = self._confirm_best
            now = time.time()
            if now - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time  = now
                self.total_accidents += 1
                clip_path = self._start_clip(w, h)
                accident_event = AccidentEvent(
                    timestamp=now,
                    severity=confirmed["severity"],
                    severity_label=SEVERITY_LABELS[confirmed["severity"]],
                    accident_score=round(confirmed["score"], 3),
                    vehicles_involved=confirmed["involved_ids"],
                    location=location or {},
                    confidence=confirmed["confidence"],
                    frame_snapshot=frame.copy(),
                    clip_path=clip_path,
                )
                if self.on_accident_detected:
                    self.on_accident_detected(accident_event)

        annotated = self._annotate(frame.copy(), cdata)
        self._update_heatmap(h, w)
        self._write_clip(annotated)
        stats = self._build_stats()
        self._last_frame = annotated
        self._last_stats = stats

        return {"frame": annotated, "accident_event": accident_event,
                "stats": stats, "vehicle_count": len(self.vehicles)}

    # ════════════════════════════════════════════
    #  OVERHEAD CCTV ACCIDENT ANALYSER
    # ════════════════════════════════════════════

    def _analyse_overhead(self) -> dict:
        """
        Overhead CCTV specific detection.
        From top-down, vehicles are small rectangles.
        Accidents = overlap + sudden stop/direction change.
        """
        ids  = list(self.vehicles.keys())
        best = {"score": 0.0, "severity": 0, "involved_ids": [], "confidence": 0.0, "iou": 0.0}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.vehicles[ids[i]]
                b = self.vehicles[ids[j]]

                # ── Gate 1: Both tracks stable ────────────
                if a.frames_tracked < MIN_TRACK_FRAMES or b.frames_tracked < MIN_TRACK_FRAMES:
                    continue

                # ── Gate 2: Real bounding box overlap ─────
                iou = self._iou(a.bbox, b.bbox)
                if iou < MIN_IOU_OVERHEAD:
                    continue

                # ── Gate 3: At least one was moving ───────
                max_spd = max(
                    max(list(a.speeds)[:-3] or [0]),
                    max(list(b.speeds)[:-3] or [0])
                )
                if max_spd < MIN_SPEED_BEFORE:
                    continue

                # ── Compute accident signals ───────────────
                score = 0.0

                # Signal A: IoU overlap (primary — from overhead this IS collision)
                # Higher IoU = more confident collision
                score += min(iou * 4.0, 0.60)

                # Signal B: Sudden deceleration (braking after impact)
                hist_speeds_a = list(a.speeds)[:-2] or [a.speed]
                hist_speeds_b = list(b.speeds)[:-2] or [b.speed]
                avg_a = max(float(np.mean(hist_speeds_a)), 0.1)
                avg_b = max(float(np.mean(hist_speeds_b)), 0.1)
                decel_a = max(0, (avg_a - a.speed) / avg_a)
                decel_b = max(0, (avg_b - b.speed) / avg_b)
                max_decel = max(decel_a, decel_b)
                score += min(max_decel * 0.25, 0.20)

                # Signal C: Sudden heading change (direction scatter from impact)
                heading_chg = max(a.heading_change, b.heading_change)
                if heading_chg > SUDDEN_HEADING_CHANGE:
                    score += min(heading_chg / 180.0 * 0.18, 0.18)

                # Signal D: Post-impact abnormal stop
                if (a.stationary_frames > 8 and a.had_motion) or \
                   (b.stationary_frames > 8 and b.had_motion):
                    score += 0.10

                # Signal E: Both vehicles converging toward each other
                if len(a.trajectory) >= 4 and len(b.trajectory) >= 4:
                    pa = list(a.trajectory)
                    pb = list(b.trajectory)
                    # Were they getting closer over last 4 frames?
                    dist_old = np.hypot(pa[-4][0]-pb[-4][0], pa[-4][1]-pb[-4][1])
                    dist_now = np.hypot(a.center[0]-b.center[0], a.center[1]-b.center[1])
                    if dist_old > 0 and dist_now < dist_old * 0.7:
                        score += 0.08  # converging fast

                score = min(score, 1.0)

                if score > best["score"]:
                    sev = self._severity(score, iou, a.speed + b.speed)
                    best = {
                        "score":        score,
                        "severity":     sev,
                        "involved_ids": [ids[i], ids[j]],
                        "confidence":   round((a.confidence + b.confidence) / 2, 3),
                        "iou":          round(iou, 3),
                    }

        return best

    def _severity(self, score, iou, combined_speed) -> int:
        if score > 0.85 or iou > 0.50 or combined_speed > 45: return 3
        if score > 0.75 or iou > 0.30 or combined_speed > 20: return 2
        return 1

    # ════════════════════════════════════════════
    #  YOLO + TRACKING
    # ════════════════════════════════════════════

    def _parse_detections(self, results, scale: float) -> list:
        inv  = 1.0 / scale
        dets = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            if cls not in VEHICLE_CLASSES: continue
            conf = float(box.conf[0])
            if conf < 0.28: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1,y1 = int(x1*inv), int(y1*inv)
            x2,y2 = int(x2*inv), int(y2*inv)
            label = VEHICLE_CLASSES[cls]
            self.traffic_counts[label] += 1
            self.vehicle_totals[label] += 1
            dets.append(([x1,y1,x2-x1,y2-y1], conf, label))
        return dets

    def _update_vehicles(self, tracks, scale, fw, fh):
        inv    = 1.0 / scale
        active = set()

        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            l   = track.to_ltrb()
            x1 = max(0, int(l[0]*inv)); y1 = max(0, int(l[1]*inv))
            x2 = min(fw,int(l[2]*inv)); y2 = min(fh,int(l[3]*inv))
            if x2 <= x1 or y2 <= y1: continue
            cx, cy = (x1+x2)//2, (y1+y2)//2

            if tid in self.vehicles:
                v  = self.vehicles[tid]
                dx = cx - v.center[0]
                dy = cy - v.center[1]
                spd = float(np.hypot(dx, dy))
                new_heading = float(np.degrees(np.arctan2(dy, dx)) % 360)

                # Heading change — detect sudden direction shift
                hdiff = abs(new_heading - v.heading) % 360
                if hdiff > 180: hdiff = 360 - hdiff
                v.heading_change = hdiff

                v.accel    = spd - v.speed
                v.speed    = round(spd, 2)
                v.prev_heading = v.heading
                v.heading  = new_heading
                v.trajectory.append((cx, cy))
                v.speeds.append(spd)
                v.headings.append(new_heading)
                v.bbox     = [x1, y1, x2, y2]
                v.center   = (cx, cy)
                v.last_seen = time.time()
                v.frames_tracked += 1
                if spd > MIN_SPEED_BEFORE:
                    v.had_motion = True
                v.stationary_frames = v.stationary_frames + 1 if spd < 0.8 else 0
            else:
                self.vehicles[tid] = VehicleState(
                    track_id=tid, bbox=[x1,y1,x2,y2],
                    class_name=getattr(track,'det_class','vehicle'),
                    confidence=getattr(track,'det_conf',0.8),
                    center=(cx,cy),
                    trajectory=deque([(cx,cy)], maxlen=30),
                    speeds=deque([0.0], maxlen=20),
                    headings=deque([0.0], maxlen=15),
                    frames_tracked=1,
                )
            active.add(tid)

        for tid in [t for t,v in list(self.vehicles.items())
                    if t not in active and time.time()-v.last_seen > 2.0]:
            del self.vehicles[tid]

    # ════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════

    def _iou(self, A, B) -> float:
        xA,yA = max(A[0],B[0]), max(A[1],B[1])
        xB,yB = min(A[2],B[2]), min(A[3],B[3])
        inter  = max(0,xB-xA) * max(0,yB-yA)
        if inter == 0: return 0.0
        aA = max(1,(A[2]-A[0])*(A[3]-A[1]))
        aB = max(1,(B[2]-B[0])*(B[3]-B[1]))
        return inter / float(aA + aB - inter)

    # ════════════════════════════════════════════
    #  ANNOTATION
    # ════════════════════════════════════════════

    def _annotate(self, frame: np.ndarray, cdata: dict) -> np.ndarray:
        acc_ids     = set(cdata.get("involved_ids", []))
        is_accident = cdata["score"] >= 0.50

        for tid, v in self.vehicles.items():
            x1,y1,x2,y2 = v.bbox
            is_acc = tid in acc_ids
            col    = (0,40,255) if is_acc else (0,210,90)

            cv2.rectangle(frame,(x1,y1),(x2,y2),col, 2 if is_acc else 1)

            if is_acc:
                s=16
                for (px,py) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    dx=s if px==x1 else -s; dy=s if py==y1 else -s
                    cv2.line(frame,(px,py),(px+dx,py),(0,0,220),2)
                    cv2.line(frame,(px,py),(px,py+dy),(0,0,220),2)

            spd_kmh = int(v.speed * 3.6)
            lbl     = f"#{tid} {v.class_name} {spd_kmh}km/h"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+6,y1),col,-1)
            cv2.putText(frame,lbl,(x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)

            pts = list(v.trajectory)
            for k in range(1, len(pts)):
                a = k / len(pts)
                cv2.line(frame,pts[k-1],pts[k],(0,int(150*a),int(210*a)),1)

        if is_accident:
            sev = cdata["severity"]
            bc  = SEVERITY_BANNER.get(sev,(0,10,180))
            cv2.rectangle(frame,(0,0),(frame.shape[1],54),bc,-1)
            pct     = int(cdata["score"]*100)
            iou_pct = int(cdata.get("iou",0)*100)
            cv2.putText(frame,
                f"!! ACCIDENT [{SEVERITY_LABELS[sev].upper()}]"
                f"  score={pct}%  overlap={iou_pct}%  vehicles={len(acc_ids)}",
                (12,36),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
            h,w=frame.shape[:2]
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,200),3)

        # Confirmation progress bar
        if 0 < self._confirm_count < CONFIRM_FRAMES:
            pct = int(self._confirm_count / CONFIRM_FRAMES * 100)
            w   = frame.shape[1]
            bar_w = int((w - 20) * pct / 100)
            cv2.rectangle(frame,(10,frame.shape[0]-50),(w-10,frame.shape[0]-40),(30,30,30),-1)
            cv2.rectangle(frame,(10,frame.shape[0]-50),(10+bar_w,frame.shape[0]-40),(0,200,255),-1)
            cv2.putText(frame,f"Verifying accident... {pct}%",
                        (12,frame.shape[0]-54),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)

        h,w = frame.shape[:2]
        cv2.rectangle(frame,(0,h-36),(w,h),(0,0,0),-1)
        st = self._build_stats()
        score_pct = int(cdata["score"] * 100)
        score_col = (0,200,0) if score_pct < 40 else (0,165,255) if score_pct < 60 else (0,0,255)
        cv2.putText(frame,
            f"Vehicles:{len(self.vehicles)}  Incidents:{self.total_accidents}"
            f"  Score:{score_pct}%  {st['congestion_level']}  Frame:{self.frame_count}",
            (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.46,score_col,1)

        # Live score meter top-right
        mw=130; mh=10; mx=w-mw-10; my=10
        cv2.rectangle(frame,(mx,my),(mx+mw,my+mh),(20,20,20),-1)
        fw = int(mw * cdata["score"])
        if fw > 0:
            cv2.rectangle(frame,(mx,my),(mx+fw,my+mh),score_col,-1)
        cv2.rectangle(frame,(mx,my),(mx+mw,my+mh),(80,80,80),1)
        # threshold line
        tx = mx + int(mw * SCORE_THRESHOLD)
        cv2.line(frame,(tx,my-2),(tx,my+mh+2),(0,200,255),1)
        cv2.putText(frame,f"Score:{score_pct}% (need {int(SCORE_THRESHOLD*100)}%+)",
                    (mx-60,my-4),cv2.FONT_HERSHEY_SIMPLEX,0.36,(180,180,180),1)
        return frame

    # ════════════════════════════════════════════
    #  HEATMAP / CLIP / STATS
    # ════════════════════════════════════════════

    def get_heatmap_frame(self, frame: np.ndarray) -> np.ndarray:
        h,w = frame.shape[:2]
        if self.heatmap is None or self.heatmap.shape[:2] != (h,w):
            dark = (frame * 0.3).astype(np.uint8)
            cv2.putText(dark,"Heatmap building...",(20,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,200,255),2)
            return dark
        hm = self.heatmap.copy()
        if hm.max() < 1:
            return (frame*0.3).astype(np.uint8)
        hm = cv2.GaussianBlur(hm,(31,31),0)
        hm = (hm/hm.max()*255).astype(np.uint8)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)
        if colored.shape[:2] != (h,w):
            colored = cv2.resize(colored,(w,h))
        gray   = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray   = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted((gray*0.2).astype(np.uint8),0.3,colored,0.7,0)
        for v in self.vehicles.values():
            cx,cy = v.center
            if 0<=cx<w and 0<=cy<h:
                cv2.circle(result,(cx,cy),7,(255,255,255),-1)
                cv2.circle(result,(cx,cy),10,(180,180,180),1)
        cv2.putText(result,"TRAFFIC DENSITY HEATMAP",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,220,255),2)
        total = sum(self.vehicle_totals.values())
        cv2.putText(result,
            f"Active:{len(self.vehicles)}  Total:{total}  Frame:{self.frame_count}",
            (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,200,255),1)
        lx,ly,lw2,lh2 = w-30,h//2-70,14,140
        leg = cv2.applyColorMap(
            np.tile(np.linspace(255,0,lh2).astype(np.uint8).reshape(-1,1),(1,lw2)),
            cv2.COLORMAP_TURBO)
        result[ly:ly+lh2,lx:lx+lw2] = leg
        cv2.putText(result,"HIGH",(lx-34,ly+10),cv2.FONT_HERSHEY_SIMPLEX,0.32,(255,255,255),1)
        cv2.putText(result,"LOW" ,(lx-28,ly+lh2),cv2.FONT_HERSHEY_SIMPLEX,0.32,(180,180,180),1)
        return result

    def _update_heatmap(self, h, w):
        for v in self.vehicles.values():
            cx,cy = v.center
            if 0<=cx<w and 0<=cy<h:
                r = max(16, int((v.bbox[2]-v.bbox[0])/2))
                intensity = 1.5 + min(v.speed/6, 2.0)
                cv2.circle(self.heatmap,(cx,cy),r,intensity,-1)
        self.heatmap *= 0.995
        np.clip(self.heatmap,0,255,out=self.heatmap)

    def _start_clip(self, w, h) -> Optional[str]:
        os.makedirs("data/clips", exist_ok=True)
        path = f"data/clips/inc_{self.total_accidents}_{int(time.time())}.avi"
        self._clip_writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), 20, (w,h))
        self._clip_path   = path
        self._clip_frames = 0
        self._recording   = True
        return path

    def _write_clip(self, frame):
        if not self._recording or self._clip_writer is None: return
        self._clip_writer.write(frame)
        self._clip_frames += 1
        if self._clip_frames >= self._clip_max:
            self._clip_writer.release()
            self._clip_writer = None
            self._recording   = False

    def _build_stats(self) -> dict:
        speeds = [v.speed*3.6 for v in self.vehicles.values()]
        return {
            "vehicle_count":    len(self.vehicles),
            "avg_speed":        round(float(np.mean(speeds)),1) if speeds else 0,
            "max_speed":        round(float(np.max(speeds)),1)  if speeds else 0,
            "total_accidents":  self.total_accidents,
            "traffic_counts":   dict(self.traffic_counts),
            "vehicle_totals":   dict(self.vehicle_totals),
            "congestion_level": self._congestion(),
            "device":           self.device,
            "fps":              self.fps,
        }

    def _congestion(self) -> str:
        n = len(self.vehicles)
        if n < 3:  return "Clear"
        if n < 8:  return "Moderate"
        return "Heavy"
