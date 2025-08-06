import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import threading
from collections import deque

class FastPeopleDetector:
    def __init__(self):
        """Initialize the optimized people detector"""
        print("🚀 Initializing Fast YOLOv8 model...")
        try:
            # Use nano model for speed, with optimized settings
            self.model = YOLO('yolov8n.pt')
            
            # Optimize model for speed
            self.model.overrides['verbose'] = False
            self.model.overrides['save'] = False
            
            print("✅ Fast YOLOv8 model loaded!")
        except Exception as e:
            print(f"❌ Error loading YOLOv8: {e}")
            self.model = None
    
    def detect_people_fast(self, frame):
        """
        Fast people detection with optimization
        """
        if self.model is None:
            return []
        
        try:
            # Run detection with speed optimizations
            results = self.model(
                frame, 
                verbose=False,
                conf=0.4,  # Lower confidence for speed
                iou=0.5,   # NMS threshold
                max_det=50,  # Limit detections
                device='cpu'  # Force CPU for consistency
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Only people (class 0) with decent confidence
                        if class_id == 0 and confidence > 0.4:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': 'person'
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections_fast(self, frame, detections, frame_count=0):
        """
        Fast drawing with minimal operations
        """
        # Create overlay instead of copying entire frame
        overlay = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Simple label
            label = f"P{i+1}"
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add quick stats
        stats = f"Frame: {frame_count} | People: {len(detections)}"
        cv2.putText(overlay, stats, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay

class RealTimeVideoProcessor:
    def __init__(self):
        self.detector = FastPeopleDetector()
        self.is_processing = False
        self.current_cap = None
        self.frame_cache = deque(maxlen=30)  # Cache last 30 frames
        self.detection_cache = {}
        
    def prepare_video(self, video_path):
        """Prepare video for real-time processing"""
        try:
            if self.current_cap:
                self.current_cap.release()
                
            self.current_cap = cv2.VideoCapture(video_path)
            
            # Optimize capture settings
            self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get video info
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
    
    def process_frame_realtime(self, frame_number, skip_detection=False):
        """
        Process frame optimized for real-time performance
        """
        try:
            if not self.current_cap:
                return None, {}
            
            # Jump to frame
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.current_cap.read()
            
            if not ret:
                return None, {}
            
            # Resize for speed (smaller = faster)
            height, width = frame.shape[:2]
            if width > 640:  # Process at max 640px width
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Skip detection every few frames for speed
            if skip_detection or frame_number in self.detection_cache:
                if frame_number in self.detection_cache:
                    detections = self.detection_cache[frame_number]
                else:
                    detections = []
            else:
                # Detect people
                start_time = time.time()
                detections = self.detector.detect_people_fast(frame)
                detection_time = time.time() - start_time
                
                # Cache result
                self.detection_cache[frame_number] = detections
                
                # Limit cache size
                if len(self.detection_cache) > 100:
                    # Remove oldest entries
                    keys = list(self.detection_cache.keys())
                    for key in keys[:50]:
                        del self.detection_cache[key]
            
            # Draw detections
            annotated_frame = self.detector.draw_detections_fast(frame, detections, frame_number)
            
            detection_data = {
                'frame': frame_number,
                'people_count': len(detections),
                'detections': detections,
                'processed_at': time.time()
            }
            
            return annotated_frame, detection_data
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return None, {}
    
    def cleanup(self):
        """Clean up resources"""
        if self.current_cap:
            self.current_cap.release()
        self.detection_cache.clear()
        self.frame_cache.clear()
