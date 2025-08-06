import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import threading
from collections import deque
import torch

class UltraFastPeopleDetector:
    def __init__(self):
        """Initialize ultra-fast detector with maximum optimizations"""
        print("🚀 Initializing Ultra-Fast YOLOv8...")
        try:
            # Use the fastest possible model
            self.model = YOLO('yolov8n.pt')  # Nano - fastest version
            
            # Maximum speed optimizations
            self.model.overrides.update({
                'verbose': False,
                'save': False,
                'save_txt': False,
                'save_conf': False,
                'save_crop': False,
                'show': False,
                'conf': 0.25,  # Lower confidence for speed
                'iou': 0.7,    # Higher IoU to reduce NMS time
                'max_det': 30, # Limit max detections
                'agnostic_nms': False,
                'augment': False,
                'visualize': False,
                'device': 'cpu'  # Force CPU for consistency
            })
            
            # Warm up the model
            dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
            
            print("✅ Ultra-Fast YOLOv8 ready!")
        except Exception as e:
            print(f"❌ Error loading YOLOv8: {e}")
            self.model = None
    
    def detect_people_ultra_fast(self, frame):
        """
        Ultra-fast detection with aggressive optimizations
        """
        if self.model is None:
            return []
        
        try:
            # Run detection with maximum speed settings
            results = self.model.predict(
                frame,
                verbose=False,
                stream=False,
                conf=0.25,
                iou=0.7,
                max_det=30,
                device='cpu',
                half=False,  # Disable half precision for CPU
                imgsz=320    # Very small image size for speed
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
                            
                            # Only people with minimal confidence
                            if class_id == 0 and confidence > 0.25:
                                x1, y1, x2, y2 = box.xyxy[0]
                                
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence
                                })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_minimal(self, frame, detections, frame_count=0):
        """
        Ultra-minimal drawing for maximum speed
        """
        # Draw directly on input frame (no copy)
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Simple green rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Minimal label
            cv2.putText(frame, str(i+1), (x1, y1-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Quick stats overlay
        text = f"F:{frame_count} P:{len(detections)}"
        cv2.putText(frame, text, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

class UltraFastVideoProcessor:
    def __init__(self):
        self.detector = UltraFastPeopleDetector()
        self.current_cap = None
        self.frame_cache = {}
        self.detection_cache = {}
        self.processing_size = (320, 240)  # Very small processing size
        
    def prepare_video_ultra_fast(self, video_path):
        """Prepare video with ultra-fast settings"""
        try:
            if self.current_cap:
                self.current_cap.release()
                
            self.current_cap = cv2.VideoCapture(video_path)
            
            # Ultra-fast capture settings
            self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.current_cap.set(cv2.CAP_PROP_FPS, 30)  # Force 30 FPS
            
            fps = self.current_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'ready': True
            }
        except Exception as e:
            print(f"Error preparing video: {e}")
            return {'ready': False}
    
    def process_frame_ultra_fast(self, frame_number, force_detect=True):
        """
        Ultra-fast frame processing
        """
        try:
            if not self.current_cap:
                return None, {}
            
            # Check cache first
            cache_key = frame_number
            if not force_detect and cache_key in self.frame_cache:
                return self.frame_cache[cache_key]
            
            # Jump to frame
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.current_cap.read()
            
            if not ret:
                return None, {}
            
            # Aggressive resizing for speed
            frame_small = cv2.resize(frame, self.processing_size)
            
            # Detect on small frame
            start_time = time.perf_counter()
            detections = self.detector.detect_people_ultra_fast(frame_small)
            detection_time = time.perf_counter() - start_time
            
            # Scale detections back to original frame size
            if detections:
                h_orig, w_orig = frame.shape[:2]
                h_small, w_small = frame_small.shape[:2]
                scale_x = w_orig / w_small
                scale_y = h_orig / h_small
                
                for detection in detections:
                    bbox = detection['bbox']
                    bbox[0] = int(bbox[0] * scale_x)  # x1
                    bbox[1] = int(bbox[1] * scale_y)  # y1
                    bbox[2] = int(bbox[2] * scale_x)  # x2
                    bbox[3] = int(bbox[3] * scale_y)  # y2
            
            # Draw on original frame
            annotated_frame = self.detector.draw_minimal(frame, detections, frame_number)
            
            detection_data = {
                'frame': frame_number,
                'people_count': len(detections),
                'detections': detections,
                'processing_time': detection_time,
                'fps': 1.0 / detection_time if detection_time > 0 else 0
            }
            
            # Cache result (limit cache size)
            if len(self.frame_cache) < 50:
                self.frame_cache[cache_key] = (annotated_frame.copy(), detection_data)
            
            return annotated_frame, detection_data
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return None, {}
    
    def cleanup(self):
        """Clean up resources"""
        if self.current_cap:
            self.current_cap.release()
        self.frame_cache.clear()
        self.detection_cache.clear()
