# app.py -- All-in-one CCTV 3D density detector + Flask server
from flask import Flask, render_template, jsonify, request
import cv2
import os
import base64
import time
import warnings
import threading
import queue
import numpy as np
import math
from ultralytics import YOLO
from fire_model import FireClassifier  # Add this to use your trained fire model


os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cv2.setLogLevel(0)
warnings.filterwarnings("ignore")

# Redirect stderr to suppress codec errors
import sys
from contextlib import redirect_stderr
from io import StringIO

app = Flask(__name__)

# Configuration
VIDEO_FOLDER = 'static/videos'
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Global variables
current_video = None
video_list = []

class CameraModel:
    # world ground plane is y=0; camera at (0, camera_height, 0)
    # image coords: u to right, v down. Camera coords used: x right, y up, z forward.
    def __init__(self, image_width, image_height, fx=None, fy=None, cx=None, cy=None,
                 hfov_deg=None, vfov_deg=None, camera_height=2.5, camera_pitch_deg=10.0):
        self.width = image_width
        self.height = image_height
        self.cx = float(cx) if cx is not None else float(image_width) / 2.0
        self.cy = float(cy) if cy is not None else float(image_height) / 2.0
        self.camera_height = float(camera_height)
        self.pitch = math.radians(camera_pitch_deg)
        # estimate fx/fy from FOV if not provided
        if fx is None:
            hfov_deg = 60.0 if hfov_deg is None else hfov_deg
            self.fx = (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        else:
            self.fx = float(fx)
        if fy is None:
            if vfov_deg is None:
                vfov_deg = (hfov_deg * (image_height / image_width)) if hfov_deg is not None else 45.0
            self.fy = (image_height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        else:
            self.fy = float(fy)
        # rotation camera -> world: rotate about X by -pitch (camera pitched down)
        cp = math.cos(-self.pitch); sp = math.sin(-self.pitch)
        self.R_cam2world = np.array([[1.0, 0.0, 0.0],
                                     [0.0, cp, -sp],
                                     [0.0, sp, cp]], dtype=float)
        self.R_world2cam = self.R_cam2world.T
        self.camera_pos = np.array([0.0, float(self.camera_height), 0.0], dtype=float)

    def pixel_to_cam_dir(self, u, v):
        # direction in image coords (y down)
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0], dtype=float)

    def cam_dir_to_world_dir(self, d_cam):
        # convert image y-down -> camera y-up then rotate to world frame
        d_cam_vec = np.array([d_cam[0], -d_cam[1], d_cam[2]], dtype=float)
        d_world = self.R_cam2world.dot(d_cam_vec)
        n = np.linalg.norm(d_world)
        if n == 0:
            return d_world
        return d_world / n

    def intersect_ground_from_pixel(self, u, v):
        # cast ray from camera through (u,v), intersect ground plane y=0
        d_cam = self.pixel_to_cam_dir(u, v)
        d_world = self.cam_dir_to_world_dir(d_cam)
        dy = d_world[1]
        if abs(dy) < 1e-6:
            return None
        t = - self.camera_pos[1] / dy
        if t <= 0:
            return None
        P_world = self.camera_pos + t * d_world
        P_world[1] = 0.0
        return P_world  # (x,0,z)

    def pixel_and_depth_to_world(self, u, v, Z):
        # back-project pixel with known Z (meters) into world coords
        if Z is None or Z <= 0:
            return None
        d_cam = self.pixel_to_cam_dir(u, v)
        P_cam = np.array([d_cam[0] * Z, -d_cam[1] * Z, d_cam[2] * Z], dtype=float)
        P_world = self.R_cam2world.dot(P_cam) + self.camera_pos
        return P_world

    def estimate_depth_from_bbox_height(self, bbox_h_pixels, real_height_m=1.7):
        if bbox_h_pixels <= 0:
            return None
        return (real_height_m * self.fy) / float(bbox_h_pixels)

    def world_to_image(self, P_world):
        # project world point into image (u,v). Returns None if behind camera.
        P_cam = self.R_world2cam.dot(P_world - self.camera_pos)
        Xc, Yc, Zc = P_cam[0], P_cam[1], P_cam[2]
        if Zc <= 1e-6:
            return None
        v = (-Yc * self.fy) / Zc + self.cy
        u = (Xc * self.fx) / Zc + self.cx
        return np.array([u, v], dtype=float)


class CCTVStyleDetector:
    """
    Replacement detector: returns 3D world positions per detection (if possible)
    and computes a ground-grid density (people/m^2).
    """
    def __init__(self, camera_params=None, average_person_height_m=1.7):
        print("📹 Initializing CCTV-Style Detection + 3D Density Analysis...")
        try:
            self.model = YOLO('yolov8n.pt')
            self.model.overrides.update({
                'verbose': False,
                'conf': 0.1,
                'iou': 0.6,
                'max_det': 50,
                'device': 'cpu',
                'imgsz': 416
            })
        except Exception as e:
            print(f"❌ YOLO init error (continuing without model): {e}")
            self.model = None

        self.average_person_height_m = float(average_person_height_m)
        self.camera = None
        self.camera_params = camera_params or {}
        print("✅ CCTV Detection + 3D Density Analysis Ready!")

    def ensure_camera_model(self, frame_shape):
        h, w = frame_shape[:2]
        if self.camera is None:
            # Prepare params but filter out keys CameraModel doesn't accept
            params = self.camera_params.copy()
            params.setdefault('image_width', w)
            params.setdefault('image_height', h)
            params.setdefault('camera_height', params.get('camera_height', 2.5))
            params.setdefault('camera_pitch_deg', params.get('camera_pitch_deg', 10.0))
            params.setdefault('hfov_deg', params.get('hfov_deg', 60.0))

            # Allowed keys for CameraModel.__init__
            allowed = {'image_width', 'image_height', 'fx', 'fy', 'cx', 'cy',
                       'hfov_deg', 'vfov_deg', 'camera_height', 'camera_pitch_deg'}
            filtered = {k: v for k, v in params.items() if k in allowed}

            self.camera = CameraModel(**filtered)

    def detect_people_cctv(self, frame):
        # returns detections same as earlier but without 3D. 3D gets computed separately.
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                frame,
                verbose=False,
                conf=0.1,
                iou=0.6,
                max_det=50,
                imgsz=416,
                device='cpu'
            )
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if class_id == 0 and confidence > 0.3:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2.0
                            center_y = (y1 + y2) / 2.0
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': [int(center_x), int(center_y)],
                                'confidence': confidence,
                                'id': len(detections) + 1
                            })
            return detections
        except Exception as e:
            print(f"[DETECT ERROR] {e}")
            return []

    def compute_3d_positions(self, detections, frame_shape):
        # Add detection['position_3d'] = {'x', 'y', 'z'} where possible
        self.ensure_camera_model(frame_shape)
        positions_3d = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            foot_u = (x1 + x2) / 2.0
            foot_v = float(y2)
            P_world = self.camera.intersect_ground_from_pixel(foot_u, foot_v) if self.camera else None
            used_method = 'ground_intersection'
            if P_world is None and self.camera is not None:
                bbox_h = max(1.0, y2 - y1)
                Z = self.camera.estimate_depth_from_bbox_height(bbox_h, real_height_m=self.average_person_height_m)
                if Z is not None:
                    P_world = self.camera.pixel_and_depth_to_world(foot_u, foot_v, Z)
                    used_method = 'depth_from_height'
            if P_world is None:
                det['position_3d'] = None
                det['position_method'] = None
                continue
            det['position_3d'] = {'x': float(P_world[0]), 'y': float(P_world[1]), 'z': float(P_world[2])}
            det['position_method'] = used_method
            positions_3d.append((P_world[0], P_world[2]))
        return positions_3d

    def _calculate_grid_density(self, detections, width, height, grid_size=50):
        # a simple 2D fallback grid if 3D grid isn't available
        cols = max(1, width // grid_size)
        rows = max(1, height // grid_size)
        grid = np.zeros((rows, cols), dtype=np.int32)
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            col = int(cx // grid_size)
            row = int(cy // grid_size)
            col = np.clip(col, 0, cols - 1)
            row = np.clip(row, 0, rows - 1)
            grid[row, col] += 1
        return grid

    def calculate_crowd_density(self, detections, frame_shape):
        # Compute 3D positions and then a ground-grid density (people per m^2)
        height, width = frame_shape[:2]
        positions = self.compute_3d_positions(detections, frame_shape)
        total_people = len(detections)
        if len(positions) == 0:
            # fallback: approximate pixel->area mapping (user should calibrate)
            frame_area_pixels = width * height
            frame_area_m2 = frame_area_pixels / (100.0 * 100.0)
            people_per_m2 = total_people / frame_area_m2 if frame_area_m2 > 0 else 0
            if total_people == 0:
                density_level = "EMPTY"; density_color = (128,128,128)
            elif total_people <= 5:
                density_level = "LOW"; density_color = (0,255,0)
            elif total_people <= 15:
                density_level = "MEDIUM"; density_color = (0,165,255)
            elif total_people <= 25:
                density_level = "HIGH"; density_color = (0,100,255)
            else:
                density_level = "CRITICAL"; density_color = (0,0,255)
            grid_density = self._calculate_grid_density(detections, width, height, grid_size=50)
            return {
                'total_people': total_people,
                'density_level': density_level,
                'density_color': density_color,
                'people_per_m2': round(people_per_m2, 2),
                'grid_density': grid_density,
                'grid_size': 50,
                'ground_grid': None
            }
        xs = [p[0] for p in positions]; zs = [p[1] for p in positions]
        padding = 1.0
        min_x, max_x = min(xs) - padding, max(xs) + padding
        min_z, max_z = min(zs) - padding, max(zs) + padding
        grid_size_m = self.camera_params.get('grid_size_m', 1.0)
        cols = max(1, int(math.ceil((max_x - min_x) / grid_size_m)))
        rows = max(1, int(math.ceil((max_z - min_z) / grid_size_m)))
        grid = np.zeros((rows, cols), dtype=np.int32)
        for x,z in positions:
            col = int((x - min_x) / grid_size_m)
            row = int((z - min_z) / grid_size_m)
            col = np.clip(col, 0, cols - 1)
            row = np.clip(row, 0, rows - 1)
            grid[row, col] += 1
        area_m2 = max((max_x - min_x) * (max_z - min_z), 1e-3)
        people_per_m2 = total_people / area_m2 if area_m2 > 0 else 0
        if total_people == 0:
            density_level = "EMPTY"; density_color = (128,128,128)
        elif people_per_m2 <= 0.5:
            density_level = "LOW"; density_color = (0,255,0)
        elif people_per_m2 <= 1.5:
            density_level = "MEDIUM"; density_color = (0,165,255)
        elif people_per_m2 <= 3.0:
            density_level = "HIGH"; density_color = (0,100,255)
        else:
            density_level = "CRITICAL"; density_color = (0,0,255)
        return {
            'total_people': total_people,
            'density_level': density_level,
            'density_color': density_color,
            'people_per_m2': round(people_per_m2, 2),
            'grid_density': grid,
            'grid_size_m': grid_size_m,
            'ground_grid': {
                'min_x': min_x, 'max_x': max_x,
                'min_z': min_z, 'max_z': max_z,
                'rows': rows, 'cols': cols
            }
        }

    def draw_density_overlay(self, frame, density_data):
        overlay = frame.copy()
        ground_grid = density_data.get('ground_grid', None)
        if ground_grid is None or self.camera is None:
            # fallback: draw 2D image-space grid
            grid_density = density_data['grid_density']
            grid_size = density_data.get('grid_size', 50)
            max_density = grid_density.max() if grid_density.size > 0 and grid_density.max() > 0 else 1
            for row in range(grid_density.shape[0]):
                for col in range(grid_density.shape[1]):
                    if grid_density[row, col] > 0:
                        intensity = float(grid_density[row, col]) / float(max_density)
                        color = (0, int(255 * (1 - intensity)), int(255 * intensity))
                        x1 = int(col * grid_size)
                        y1 = int(row * grid_size)
                        x2 = int(min(x1 + grid_size, frame.shape[1]))
                        y2 = int(min(y1 + grid_size, frame.shape[0]))
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(frame, 1 - 0.3, overlay, 0.3, 0, overlay)
            return overlay
        # project each ground cell back into image
        grid = density_data['grid_density']
        rows = ground_grid['rows']; cols = ground_grid['cols']
        min_x = ground_grid['min_x']; min_z = ground_grid['min_z']
        grid_size = density_data.get('grid_size_m', 1.0)
        max_density = grid.max() if grid.size > 0 and grid.max() > 0 else 1
        for r in range(rows):
            for c in range(cols):
                cnt = int(grid[r, c])
                if cnt <= 0:
                    continue
                x0 = min_x + c * grid_size
                z0 = min_z + r * grid_size
                corners_world = [
                    np.array([x0, 0.0, z0]),
                    np.array([x0 + grid_size, 0.0, z0]),
                    np.array([x0 + grid_size, 0.0, z0 + grid_size]),
                    np.array([x0, 0.0, z0 + grid_size])
                ]
                pts_img = []
                for P in corners_world:
                    uv = self.camera.world_to_image(P)
                    if uv is None:
                        pts_img = []
                        break
                    pts_img.append((int(uv[0]), int(uv[1])))
                if len(pts_img) != 4:
                    continue
                intensity = float(cnt) / float(max_density)
                if intensity <= 0.3:
                    color = (0, 255, 0)
                elif intensity <= 0.6:
                    color = (0, 200, 255)
                elif intensity <= 0.9:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
                pts = np.array(pts_img, dtype=np.int32)
                cv2.fillConvexPoly(overlay, pts, color)
        cv2.addWeighted(frame, 1 - 0.35, overlay, 0.35, 0, overlay)
        return overlay

    def draw_anomalies(self, frame, anomalies):
        overlay = frame.copy()

        # Draw fallen people
        for fallen in anomalies.get('fallen_people', []):
            x1, y1, x2, y2 = fallen['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"FALLEN P{fallen['id']}"
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw fire zones
        for visual in anomalies.get('visual_anomalies', []):
            if visual['type'] == 'Fire':
                # Draw each fire region
                for region in visual.get('fire_regions', []):
                    x1, y1, x2, y2 = region['bbox']
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    
                    # Add fire label with confidence
                    confidence = visual.get('confidence', 0)
                    label = f"FIRE {confidence:.0%}"
                    cv2.putText(overlay, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return overlay

    def draw_cctv_style_with_density(self, frame, detections, density_data, frame_count=0, anomalies=None):
        frame_with_density = self.draw_density_overlay(frame, density_data)
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            person_id = detection['id']
            box_color = density_data['density_color']
            box_thickness = 4
            cv2.rectangle(frame_with_density, (x1, y1), (x2, y2), box_color, box_thickness)
            corner_length = 20; corner_thickness = 5
            cv2.line(frame_with_density, (x1, y1), (x1 + corner_length, y1), box_color, corner_thickness)
            cv2.line(frame_with_density, (x1, y1), (x1, y1 + corner_length), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y1), (x2 - corner_length, y1), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y1), (x2, y1 + corner_length), box_color, corner_thickness)
            cv2.line(frame_with_density, (x1, y2), (x1 + corner_length, y2), box_color, corner_thickness)
            cv2.line(frame_with_density, (x1, y2), (x1, y2 - corner_length), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y2), (x2 - corner_length, y2), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y2), (x2, y2 - corner_length), box_color, corner_thickness)
            label = f"P{person_id:02d} [{confidence:.0%}]"
            font = cv2.FONT_HERSHEY_DUPLEX; font_scale = 0.6; font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_bg_top = y1 - text_height - 15; label_bg_bottom = y1 - 5
            cv2.rectangle(frame_with_density, (x1, label_bg_top), (x1 + text_width + 10, label_bg_bottom), (0,0,0), -1)
            cv2.putText(frame_with_density, label, (x1 + 5, y1 - 10), font, font_scale, (255,255,255), font_thickness)
            pos = detection.get('position_3d', None)
            if pos is not None and self.camera is not None:
                uv = self.camera.world_to_image(np.array([pos['x'], pos['y'], pos['z']]))
                if uv is not None:
                    cv2.circle(frame_with_density, (int(uv[0]), int(uv[1])), 6, (255,0,0), -1)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        density_level = density_data['density_level']
        total_people = density_data['total_people']
        people_per_m2 = density_data['people_per_m2']
        info_text = f"LIVE | Frame: {frame_count:06d} | People: {total_people:02d} | Density: {density_level} | {timestamp}"
        cv2.rectangle(frame_with_density, (10, 10), (980, 50), (0, 0, 0), -1)
        cv2.putText(frame_with_density, info_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, density_data['density_color'], 2)
        density_text = f"Density: {people_per_m2}/m² | Level: {density_level} | 3D Grid: {'ON' if density_data.get('ground_grid') else 'OFF'}"
        cv2.rectangle(frame_with_density, (10, 55), (760, 85), (0,0,0), -1)
        cv2.putText(frame_with_density, density_text, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, density_data['density_color'], 2)
        
        # Draw anomalies if any
        if anomalies:
            frame_with_density = self.draw_anomalies(frame_with_density, anomalies)

        return frame_with_density


class AnomalyDetector:
    def __init__(self):
        print("🧠 Initializing Enhanced Anomaly Detector...")
        self.prev_frame = None
        self.prev_fire_mask = None
        self.fire_detected = False
        self.smoke_detected = False
        # Load the FireClassifier (robust loader inside)
        self.fire_model = FireClassifier(model_path='models/flame_detector.pth')
        print("✅ Anomaly Detector Ready!")

    def detect_fallen_person(self, detections, frame_shape):
        """
        Improved fallen-person heuristic using width/height ratio, size thresholds,
        detection confidence and vertical location (closer to bottom => more likely lying on ground).
        """
        fallen = []
        frame_height = frame_shape[0] if frame_shape is not None else 720

        # Tunable thresholds
        min_area = 2000
        min_width = 60
        min_height = 30
        min_confidence = 0.5
        fallen_aspect_ratio = 1.6  # width / height > this suggests lying horizontally

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            area = width * height
            confidence = det.get('confidence', 0.0)

            # basic filters
            if area < min_area or width < min_width or height < min_height:
                continue
            if confidence < min_confidence:
                continue

            aspect_ratio = width / float(height + 1e-6)

            # Fallen if significantly wider than tall and located reasonably low in the frame
            if aspect_ratio >= fallen_aspect_ratio and y1 > frame_height * 0.25:
                reason = f"AR={aspect_ratio:.2f}, area={area}, conf={confidence:.2f}"
                fallen.append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'confidence': confidence,
                    'reason': f'Fallen person detected ({reason})'
                })

        return fallen

    def _mask_to_regions(self, mask, min_area_pixels=500):
        """Converts a binary mask to list of bbox dicts with area."""
        regions = []
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area_pixels:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({'bbox': [int(x), int(y), int(x + w), int(y + h)], 'area': float(area)})
        except Exception as e:
            print(f"[AnomalyDetector] contour extraction failed: {e}")
        return regions

    def detect_smoke_or_fire(self, frame):
        """
        Use both the trained classifier (if loaded) and HSV color segmentation to
        detect likely flame regions. Returns a list of visual anomaly dicts.
        """
        anomalies = []
        model_confidence = 0.0
        try:
            model_confidence = float(self.fire_model.predict(frame)) if self.fire_model else 0.0
        except Exception as e:
            print(f"[AnomalyDetector] Fire model predict error: {e}")
            model_confidence = 0.0

        # Color segmentation (HSV) to get candidate regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([0, 80, 120])
        upper_orange = np.array([35, 255, 255])
        lower_red2 = np.array([160, 80, 120])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        color_area = float(mask.sum()) / 255.0
        total_pixels = float(frame.shape[0] * frame.shape[1])
        color_ratio = color_area / (total_pixels + 1e-9)
        # scale color ratio into pseudo-confidence
        color_confidence = float(min(1.0, color_ratio * 6.0))

        final_confidence = max(model_confidence, color_confidence)

        regions = self._mask_to_regions(mask, min_area_pixels=max(200, int(total_pixels * 0.0005)))

        threshold = 0.35  # tunable
        if final_confidence > threshold:
            # If color-based regions found, use them; otherwise fallback to whole-frame region
            if len(regions) == 0:
                regions = [{
                    'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                    'area': float(frame.shape[0] * frame.shape[1])
                }]
            anomalies.append({
                'type': 'Fire',
                'confidence': final_confidence,
                'fire_regions': regions
            })
            self.fire_detected = True
            print(f"🔥 FIRE DETECTED conf={final_confidence:.2f}, regions={len(regions)}")
        else:
            self.fire_detected = False

        # (Optional) Smoke detection could be added (e.g., using lighter grayscale thresholds or a trained model)
        return anomalies

    def detect_anomalies(self, frame, detections):
        anomalies = {
            'fallen_people': self.detect_fallen_person(detections, frame.shape),
            'visual_anomalies': self.detect_smoke_or_fire(frame)
        }
        self.prev_frame = frame.copy()
        return anomalies

class CCTVStreamProcessor:
    def __init__(self):
        camera_params = {
            'image_width': 1280,
            'image_height': 720,
            'hfov_deg': 65.0,         # or supply fx/fy directly
            'camera_height': 3.0,    # meters (height of camera above ground)
            'camera_pitch_deg': 12.0,# degrees (positive = camera pointing down)
            'grid_size_m': 1.0       # meter grid for density (kept in detector params only)
        }
        self.detector = CCTVStyleDetector(camera_params=camera_params, average_person_height_m=1.75)
        self.anomaly_detector = AnomalyDetector()
        self.is_streaming = False
        self.current_cap = None
        self.frame_queue = queue.Queue(maxsize=5)  # Small buffer for real-time
        self.result_queue = queue.Queue(maxsize=5)
        self.detection_thread = None
        self.capture_thread = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

    def start_watchdog(self):
        """Watchdog thread to monitor and restart crashed streams"""
        def watchdog():
            while self.is_streaming:
                time.sleep(5)  # Check every 5 seconds

                # Check if threads are alive
                capture_alive = self.capture_thread and self.capture_thread.is_alive()
                process_alive = self.detection_thread and self.detection_thread.is_alive()

                if not capture_alive and self.is_streaming:
                    print("🚨 Capture thread died, restarting...")
                    try:
                        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
                        self.capture_thread.start()
                        print("✅ Capture thread restarted")
                    except Exception as e:
                        print(f"❌ Failed to restart capture thread: {e}")

                if not process_alive and self.is_streaming:
                    print("🚨 Processing thread died, restarting...")
                    try:
                        self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
                        self.detection_thread.start()
                        print("✅ Processing thread restarted")
                    except Exception as e:
                        print(f"❌ Failed to restart processing thread: {e}")

        # Start watchdog thread
        self.watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        self.watchdog_thread.start()

    def start_stream(self, video_path):
        """Start CCTV-style streaming with enhanced video switching"""
        try:
            print(f"🎥 DEBUG: Loading video from: {video_path}")

            # ENHANCED: Complete cleanup of any existing stream
            if self.is_streaming or self.current_cap:
                print("🧹 Cleaning up previous video session...")
                self.stop_stream()
                for _ in range(3):
                    self._clear_queues()
                    time.sleep(0.1)
                import gc
                gc.collect()
                time.sleep(1.0)
                print("✅ Previous session cleaned up")

            self.current_cap = None
            self.is_streaming = False

            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}

            file_size = os.path.getsize(video_path)
            print(f"📁 Video file size: {file_size} bytes")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🎬 Opening video (attempt {attempt + 1}/{max_retries})...")
                    self.current_cap = cv2.VideoCapture(video_path)
                    if self.current_cap.isOpened():
                        print("✅ Video opened successfully")
                        break
                    else:
                        print(f"❌ Video open failed (attempt {attempt + 1})")
                        if self.current_cap:
                            self.current_cap.release()
                            self.current_cap = None
                        if attempt < max_retries - 1:
                            time.sleep(0.3)
                except Exception as open_error:
                    print(f"❌ Video open exception (attempt {attempt + 1}): {open_error}")
                    if self.current_cap:
                        self.current_cap.release()
                        self.current_cap = None
                    if attempt < max_retries - 1:
                        time.sleep(0.3)

            if not self.current_cap or not self.current_cap.isOpened():
                error_msg = f"Could not open video after {max_retries} attempts: {video_path}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}

            ret, test_frame = self.current_cap.read()
            if not ret:
                error_msg = f"Could not read first frame from: {video_path}"
                print(f"❌ {error_msg}")
                self.current_cap.release()
                self.current_cap = None
                return {'success': False, 'error': error_msg}

            print(f"✅ First frame read successfully: {test_frame.shape}")

            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = self.current_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"📊 Video properties: {fps} FPS, {frame_count} frames")

            self._clear_queues()
            self.is_streaming = True

            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            print("✅ Capture thread started")

            self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.detection_thread.start()
            print("✅ Detection thread started")

            self.start_watchdog()
            print("✅ Watchdog started")

            return {
                'success': True,
                'fps': fps if fps > 0 else 25,
                'frame_count': frame_count if frame_count > 0 else 1000
            }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            if hasattr(self, 'current_cap') and self.current_cap:
                self.current_cap.release()
                self.current_cap = None
            self.is_streaming = False
            return {'success': False, 'error': error_msg}

    def _capture_frames(self):
        """Enhanced frame capture with crash protection"""
        frame_counter = 0
        consecutive_errors = 0
        max_consecutive_errors = 20

        while self.is_streaming and self.current_cap:
            try:
                ret, frame = self.current_cap.read()

                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        print("🔄 Multiple read failures, restarting video...")
                        try:
                            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            consecutive_errors = 0
                            time.sleep(0.1)
                            continue
                        except Exception as restart_error:
                            print(f"❌ Restart failed: {restart_error}")
                            break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0

                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame.copy(), frame_counter), timeout=1)
                    else:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put((frame.copy(), frame_counter), timeout=1)
                        except (queue.Empty, queue.Full):
                            pass
                except Exception as queue_error:
                    print(f"⚠️ Queue error (non-fatal): {queue_error}")
                    continue

                frame_counter += 1
                time.sleep(0.066)  # ~30 FPS

            except Exception as e:
                consecutive_errors += 1
                print(f"⚠️ Capture error #{consecutive_errors}: {e}")
                if consecutive_errors > max_consecutive_errors:
                    print("❌ Too many errors, stopping capture thread")
                    self.is_streaming = False
                    break
                time.sleep(0.1)
                continue

        print("🔚 Capture thread ended")

    def _process_frames(self):
        """Enhanced frame processing with crash protection"""
        consecutive_errors = 0
        max_consecutive_errors = 20

        while self.is_streaming:
            try:
                frame, frame_count = self.frame_queue.get(timeout=0.5)
                consecutive_errors = 0

                try:
                    detections = self.detector.detect_people_cctv(frame)
                except Exception as detect_error:
                    print(f"⚠️ Detection error (using empty detections): {detect_error}")
                    detections = []

                try:
                    density_data = self.detector.calculate_crowd_density(detections, frame.shape)
                except Exception as density_error:
                    print(f"⚠️ Density error (using default): {density_error}")
                    density_data = {
                        'total_people': len(detections),
                        'density_level': 'UNKNOWN',
                        'density_color': (128, 128, 128),
                        'people_per_m2': 0,
                        'grid_density': np.zeros((1, 1)),
                        'grid_size': 50
                    }

                try:
                    anomalies = self.anomaly_detector.detect_anomalies(frame, detections)
                    current_time = time.time()
                    if not hasattr(self, 'last_alert_time'):
                        self.last_alert_time = 0

                    if current_time - self.last_alert_time > 2.0:  # Only log every 2 seconds
                        if anomalies.get('fallen_people'):
                            print(f"🚨 ALERT: {len(anomalies['fallen_people'])} fallen person(s) detected!")
                        
                        for visual in anomalies.get('visual_anomalies', []):
                            print(f"🔥 ALERT: {visual['type']} detected")
                        
                        if anomalies.get('fallen_people') or anomalies.get('visual_anomalies'):
                            self.last_alert_time = current_time
                except Exception as anomaly_error:
                    print(f"⚠️ Anomaly detection error: {anomaly_error}")
                    anomalies = {}

                try:
                    annotated_frame = self.detector.draw_cctv_style_with_density(
                        frame, detections, density_data, frame_count, anomalies=anomalies
                    )
                except Exception as draw_error:
                    print(f"⚠️ Drawing error (using original frame): {draw_error}")
                    annotated_frame = frame

                result_data = {
                    'frame': annotated_frame,
                    'detections': detections,
                    'density_data': density_data,
                    'frame_count': frame_count,
                    'people_count': len(detections),
                    'timestamp': time.time(),
                    'anomalies': anomalies
                }

                try:
                    if not self.result_queue.full():
                        self.result_queue.put(result_data, timeout=1)
                    else:
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put(result_data, timeout=1)
                        except (queue.Empty, queue.Full):
                            pass
                except Exception as result_error:
                    print(f"⚠️ Result queue error: {result_error}")

                self._update_fps()

            except queue.Empty:
                continue

            except Exception as e:
                consecutive_errors += 1
                print(f"⚠️ Processing error #{consecutive_errors}: {e}")
                if consecutive_errors > max_consecutive_errors:
                    print("❌ Too many processing errors, stopping processing thread")
                    self.is_streaming = False
                    break
                time.sleep(0.1)
                continue

        print("🔚 Processing thread ended")

    def get_latest_frame(self):
        """Get the latest processed frame (CCTV-style)"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def get_fps(self):
        """Get current processing FPS"""
        return self.current_fps

    def _clear_queues(self):
        """Clear all queues"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def stop_stream(self):
        """Enhanced CCTV streaming stop with proper cleanup"""
        print("🛑 Stopping CCTV stream...")

        # Signal threads to stop
        self.is_streaming = False

        # Wait for threads to finish gracefully
        if self.capture_thread and self.capture_thread.is_alive():
            print("⏳ Waiting for capture thread to stop...")
            self.capture_thread.join(timeout=2)
            if self.capture_thread.is_alive():
                print("⚠️ Capture thread didn't stop gracefully")

        if self.detection_thread and self.detection_thread.is_alive():
            print("⏳ Waiting for detection thread to stop...")
            self.detection_thread.join(timeout=2)
            if self.detection_thread.is_alive():
                print("⚠️ Detection thread didn't stop gracefully")

        # Enhanced video capture cleanup
        if self.current_cap:
            print("🧹 Cleaning up video capture...")
            try:
                # Force release and cleanup
                self.current_cap.release()
                self.current_cap = None

                # Give a moment for internal cleanup
                time.sleep(0.2)

                # Force garbage collection of video resources
                import gc
                gc.collect()

            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

        # Clear all queues
        self._clear_queues()

        print("✅ CCTV stream stopped and cleaned up")

# Initialize global CCTV processor
cctv_processor = CCTVStreamProcessor()

def get_video_list():
    """Get list of available videos"""
    videos = []
    if os.path.exists(VIDEO_FOLDER):
        for file in os.listdir(VIDEO_FOLDER):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(file)
    return videos

def frame_to_base64_optimized(frame):
    """Optimized frame encoding for streaming"""
    try:
        # High quality for CCTV clarity
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)

        if result:
            return base64.b64encode(encoded_img).decode()
        return None
    except:
        return None

@app.route('/')
def index():
    global video_list
    video_list = get_video_list()
    return render_template('index.html', videos=video_list)

@app.route('/start_cctv_stream', methods=['POST'])
def start_cctv_stream():
    """Start CCTV-style streaming"""
    global current_video

    data = request.get_json()
    selected_video = data.get('video_name')

    if selected_video and selected_video in get_video_list():
        current_video = selected_video
        video_path = os.path.join(VIDEO_FOLDER, selected_video)

        # Start CCTV streaming
        result = cctv_processor.start_stream(video_path)

        if result['success']:
            return jsonify({
                'success': True,
                'message': f'CCTV Stream Started: {selected_video}',
                'stream_info': {
                    'name': selected_video,
                    'fps': result.get('fps', 30),
                    'frame_count': result.get('frame_count', 0),
                    'mode': 'LIVE_CCTV_SIMULATION'
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to start CCTV stream'})
    else:
        return jsonify({'success': False, 'message': 'Invalid video selection'})

@app.route('/get_cctv_frame', methods=['GET'])
def get_cctv_frame():
    """Get latest CCTV frame with density analysis - JSON fixed"""
    if not current_video:
        return jsonify({'success': False, 'message': 'No CCTV stream active'})

    try:
        # Get latest processed frame
        frame_data = cctv_processor.get_latest_frame()

        if frame_data is None:
            return jsonify({'success': False, 'message': 'No frame available'})

        # Convert frame to base64
        frame_base64 = frame_to_base64_optimized(frame_data['frame'])

        if frame_base64 is None:
            return jsonify({'success': False, 'message': 'Frame encoding error'})

        print(f"✅ Frame processed successfully: {len(frame_base64)} bytes")

        # Convert numpy arrays to lists for JSON serialization
        density_data = frame_data.get('density_data', {})
        if 'grid_density' in density_data and hasattr(density_data['grid_density'], 'tolist'):
            density_data['grid_density'] = density_data['grid_density'].tolist()

        # Convert detection centers to regular lists
        detections = frame_data['detections']
        for detection in detections:
            if 'center' in detection and hasattr(detection['center'], 'tolist'):
                detection['center'] = detection['center'].tolist()

        return jsonify({
            'success': True,
            'frame_image': f"data:image/jpeg;base64,{frame_base64}",
            'stream_data': {
                'frame_count': frame_data['frame_count'],
                'people_count': frame_data['people_count'],
                'processing_fps': cctv_processor.get_fps(),
                'timestamp': frame_data['timestamp'],
                'detections': detections,
                'density_data': density_data
            },
            'anomalies': frame_data.get('anomalies', {})
        })

    except Exception as e:
        print(f"❌ get_cctv_frame error: {e}")
        return jsonify({'success': False, 'message': f'Stream error: {str(e)}'})


@app.route('/stop_cctv_stream', methods=['POST'])
def stop_cctv_stream():
    """Stop CCTV streaming"""
    try:
        cctv_processor.stop_stream()
        return jsonify({'success': True, 'message': 'CCTV stream stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Stop error: {str(e)}'})

@app.route('/get_stream_status', methods=['GET'])
def get_stream_status():
    """Get current stream status"""
    return jsonify({
        'active': cctv_processor.is_streaming,
        'current_video': current_video,
        'fps': cctv_processor.get_fps(),
        'available_videos': get_video_list()
    })

# Keep backward compatibility with old endpoints
@app.route('/select_video', methods=['POST'])
def select_video():
    """Legacy endpoint - redirect to CCTV stream"""
    return start_cctv_stream()

@app.route('/get_current_video')
def get_current_video():
    """Get currently selected video info"""
    return jsonify({
        'current_video': current_video,
        'available_videos': get_video_list()
    })

if __name__ == '__main__':
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    print("📹 CCTV-Style Real-Time AI Safety System!")
    print("🧈 Butter-smooth streaming detection!")
    print("📦 Extra-wide detection boxes!")
    print(f"📁 Videos folder: {os.path.abspath(VIDEO_FOLDER)}")
    print("🌐 Open http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
