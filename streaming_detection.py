import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque
import queue

class CCTVStyleDetector:
    def __init__(self):
        """Initialize CCTV-style streaming detector"""
        print("📹 Initializing CCTV-Style Detection System...")
        try:
            # Use optimized model
            self.model = YOLO('yolov8n.pt')
            
            # CCTV optimizations
            self.model.overrides.update({
                'verbose': False,
                'conf': 0.3,
                'iou': 0.6,
                'max_det': 50,
                'device': 'cpu',
                'imgsz': 416  # Balanced size for CCTV quality
            })
            
            # Warm up
            dummy = np.zeros((416, 416, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            
            print("✅ CCTV Detection System Ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model = None
    
    def detect_people_cctv(self, frame):
        """CCTV-style people detection"""
        if self.model is None:
            return []
        
        try:
            results = self.model.predict(
                frame,
                verbose=False,
                conf=0.3,
                iou=0.6,
                max_det=50,
                imgsz=416,
                device='cpu'
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        if len(box.cls) > 0 and len(box.conf) > 0:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if class_id == 0 and confidence > 0.3:  # People only
                                x1, y1, x2, y2 = box.xyxy[0]
                                
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'id': len(detections) + 1
                                })
            
            return detections
            
        except Exception as e:
            return []
    
    def draw_cctv_style(self, frame, detections, frame_count=0):
        """Draw CCTV-style detection boxes with wider borders"""
        overlay = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            person_id = detection['id']
            
            # WIDER detection box with professional CCTV styling
            box_thickness = 3  # Much thicker box
            
            # Main detection rectangle - bright green for visibility
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            
            # Corner markers for professional look
            corner_length = 15
            corner_thickness = 4
            
            # Top-left corner
            cv2.line(overlay, (x1, y1), (x1 + corner_length, y1), (0, 255, 0), corner_thickness)
            cv2.line(overlay, (x1, y1), (x1, y1 + corner_length), (0, 255, 0), corner_thickness)
            
            # Top-right corner
            cv2.line(overlay, (x2, y1), (x2 - corner_length, y1), (0, 255, 0), corner_thickness)
            cv2.line(overlay, (x2, y1), (x2, y1 + corner_length), (0, 255, 0), corner_thickness)
            
            # Bottom-left corner
            cv2.line(overlay, (x1, y2), (x1 + corner_length, y2), (0, 255, 0), corner_thickness)
            cv2.line(overlay, (x1, y2), (x1, y2 - corner_length), (0, 255, 0), corner_thickness)
            
            # Bottom-right corner
            cv2.line(overlay, (x2, y2), (x2 - corner_length, y2), (0, 255, 0), corner_thickness)
            cv2.line(overlay, (x2, y2), (x2, y2 - corner_length), (0, 255, 0), corner_thickness)
            
            # Professional label background
            label = f"PERSON-{person_id:02d} [{confidence:.0%}]"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Label background (dark with transparency)
            label_bg_top = y1 - text_height - 15
            label_bg_bottom = y1 - 5
            cv2.rectangle(overlay, (x1, label_bg_top), (x1 + text_width + 10, label_bg_bottom), 
                         (0, 0, 0), -1)
            
            # Semi-transparent overlay
            cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, overlay)
            
            # Label text - bright white
            cv2.putText(overlay, label, (x1 + 5, y1 - 10), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        # CCTV-style timestamp and info overlay
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"LIVE | Frame: {frame_count:06d} | Detected: {len(detections):02d} | {timestamp}"
        
        # Info background
        cv2.rectangle(overlay, (10, 10), (750, 50), (0, 0, 0), -1)
        cv2.putText(overlay, info_text, (15, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
        
        return overlay

class CCTVStreamProcessor:
    def __init__(self):
        self.detector = CCTVStyleDetector()
        self.is_streaming = False
        self.current_cap = None
        self.frame_queue = queue.Queue(maxsize=5)  # Small buffer for real-time
        self.result_queue = queue.Queue(maxsize=5)
        self.detection_thread = None
        self.capture_thread = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def start_stream(self, video_path):
        """Start CCTV-style streaming"""
        try:
            if self.current_cap:
                self.current_cap.release()
            
            self.current_cap = cv2.VideoCapture(video_path)
            
            # CCTV capture settings for smooth streaming
            self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.current_cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Clear queues
            self._clear_queues()
            
            # Start streaming threads
            self.is_streaming = True
            
            # Capture thread - continuously reads frames
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Detection thread - continuously processes frames
            self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.detection_thread.start()
            
            return {
                'success': True,
                'fps': self.current_cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            return {'success': False}
    
    def _capture_frames(self):
        """Continuously capture frames (like CCTV)"""
        frame_counter = 0
        
        while self.is_streaming and self.current_cap:
            ret, frame = self.current_cap.read()
            
            if not ret:
                # Loop video for continuous demo
                self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize for optimal performance
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Add to queue (drop old frames if full - real-time behavior)
            if not self.frame_queue.full():
                self.frame_queue.put((frame, frame_counter))
            else:
                # Remove old frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, frame_counter))
                except queue.Empty:
                    pass
            
            frame_counter += 1
            time.sleep(0.01)  # Small delay to prevent overwhelming
    
    def _process_frames(self):
        """Continuously process frames for detection"""
        while self.is_streaming:
            try:
                # Get frame from queue
                frame, frame_count = self.frame_queue.get(timeout=0.1)
                
                # Detect people
                detections = self.detector.detect_people_cctv(frame)
                
                # Draw CCTV-style annotations
                annotated_frame = self.detector.draw_cctv_style(frame, detections, frame_count)
                
                # Add to result queue
                result_data = {
                    'frame': annotated_frame,
                    'detections': detections,
                    'frame_count': frame_count,
                    'people_count': len(detections),
                    'timestamp': time.time()
                }
                
                if not self.result_queue.full():
                    self.result_queue.put(result_data)
                else:
                    # Remove old result and add new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result_data)
                    except queue.Empty:
                        pass
                
                # Update FPS counter
                self._update_fps()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
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
        """Stop CCTV streaming"""
        self.is_streaming = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.detection_thread:
            self.detection_thread.join(timeout=1)
        
        if self.current_cap:
            self.current_cap.release()
        
        self._clear_queues()
    
    def cleanup(self):
        """Clean up all resources"""
        self.stop_stream()
