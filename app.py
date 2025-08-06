from flask import Flask, render_template, jsonify, request
import cv2
import os
import base64
import time
import warnings
import threading
import queue
import numpy as np
from ultralytics import YOLO
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

class CCTVStyleDetector:
    def __init__(self):
        """Initialize CCTV-style streaming detector with density analysis"""
        print("📹 Initializing CCTV-Style Detection + Density Analysis...")
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
                'imgsz': 416
            })
            
            # Warm up
            dummy = np.zeros((416, 416, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            
            print("✅ CCTV Detection + Density Analysis Ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model = None
    
    def detect_people_cctv(self, frame):
        """CCTV-style people detection (FIXED)"""
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
                    # DON'T convert to numpy here - keep as YOLO Boxes object
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0 and confidence > 0.3:  # People only
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Convert to numpy HERE
                            
                            # Calculate center point for density analysis
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
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

    
    def calculate_crowd_density(self, detections, frame_shape):
        """Calculate crowd density metrics"""
        height, width = frame_shape[:2]
        total_people = len(detections)
        
        # Calculate density per square meter (approximate)
        # Assuming 1 pixel = 1cm (adjust based on your camera setup)
        frame_area_pixels = width * height
        frame_area_m2 = frame_area_pixels / (100 * 100)  # Convert to square meters
        
        if frame_area_m2 > 0:
            people_per_m2 = total_people / frame_area_m2
        else:
            people_per_m2 = 0
        
        # Determine density level
        if total_people == 0:
            density_level = "EMPTY"
            density_color = (128, 128, 128)  # Gray
        elif total_people <= 5:
            density_level = "LOW"
            density_color = (0, 255, 0)  # Green
        elif total_people <= 15:
            density_level = "MEDIUM"
            density_color = (0, 165, 255)  # Orange
        elif total_people <= 25:
            density_level = "HIGH"
            density_color = (0, 100, 255)  # Red-Orange
        else:
            density_level = "CRITICAL"
            density_color = (0, 0, 255)  # Red
        
        # Calculate grid-based density for heatmap
        grid_size = 50  # 50x50 pixel grid cells
        grid_density = self._calculate_grid_density(detections, width, height, grid_size)
        
        return {
            'total_people': total_people,
            'density_level': density_level,
            'density_color': density_color,
            'people_per_m2': round(people_per_m2, 2),
            'grid_density': grid_density,
            'grid_size': grid_size
        }
    
    def _calculate_grid_density(self, detections, width, height, grid_size):
        """Calculate density for each grid cell"""
        grid_cols = width // grid_size
        grid_rows = height // grid_size
        grid_density = np.zeros((grid_rows, grid_cols))
        
        for detection in detections:
            center_x, center_y = detection['center']
            grid_x = min(center_x // grid_size, grid_cols - 1)
            grid_y = min(center_y // grid_size, grid_rows - 1)
            
            if 0 <= grid_x < grid_cols and 0 <= grid_y < grid_rows:
                grid_density[grid_y, grid_x] += 1
        
        return grid_density
    
    def draw_density_overlay(self, frame, density_data):
        """Draw density heatmap overlay"""
        overlay = frame.copy()
        grid_density = density_data['grid_density']
        grid_size = density_data['grid_size']
        
        # Create heatmap overlay - FIXED
        max_density = grid_density.max() if grid_density.size > 0 and grid_density.max() > 0 else 1
        
        for row in range(grid_density.shape[0]):
            for col in range(grid_density.shape[1]):
                if grid_density[row, col] > 0:
                    # Calculate heat intensity
                    intensity = grid_density[row, col] / max_density
                    
                    # Create heat color (blue to red gradient)
                    if intensity <= 0.3:
                        color = (255, int(255 * intensity * 3), 0)  # Blue to cyan
                    elif intensity <= 0.6:
                        color = (int(255 * (1 - (intensity - 0.3) * 3)), 255, 0)  # Cyan to green
                    elif intensity <= 0.8:
                        color = (0, 255, int(255 * (intensity - 0.6) * 5))  # Green to yellow
                    else:
                        color = (0, int(255 * (1 - (intensity - 0.8) * 5)), 255)  # Yellow to red
                    
                    # Draw heat square
                    x1 = col * grid_size
                    y1 = row * grid_size
                    x2 = x1 + grid_size
                    y2 = y1 + grid_size
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Blend with original frame
        alpha = 0.3  # Transparency
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, overlay)
        
        return overlay

    
    def draw_cctv_style_with_density(self, frame, detections, density_data, frame_count=0):
        """Draw CCTV-style detection with density visualization"""
        # First draw density heatmap
        frame_with_density = self.draw_density_overlay(frame, density_data)
        
        # Then draw detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            person_id = detection['id']
            
            # Color detection box based on density level
            box_color = density_data['density_color']
            box_thickness = 4
            
            # Main detection rectangle
            cv2.rectangle(frame_with_density, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Corner markers
            corner_length = 20
            corner_thickness = 5
            
            # Top-left corner
            cv2.line(frame_with_density, (x1, y1), (x1 + corner_length, y1), box_color, corner_thickness)
            cv2.line(frame_with_density, (x1, y1), (x1, y1 + corner_length), box_color, corner_thickness)
            
            # Top-right corner
            cv2.line(frame_with_density, (x2, y1), (x2 - corner_length, y1), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y1), (x2, y1 + corner_length), box_color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(frame_with_density, (x1, y2), (x1 + corner_length, y2), box_color, corner_thickness)
            cv2.line(frame_with_density, (x1, y2), (x1, y2 - corner_length), box_color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(frame_with_density, (x2, y2), (x2 - corner_length, y2), box_color, corner_thickness)
            cv2.line(frame_with_density, (x2, y2), (x2, y2 - corner_length), box_color, corner_thickness)
            
            # Professional label
            label = f"P{person_id:02d} [{confidence:.0%}]"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Label background
            label_bg_top = y1 - text_height - 15
            label_bg_bottom = y1 - 5
            cv2.rectangle(frame_with_density, (x1, label_bg_top), (x1 + text_width + 10, label_bg_bottom), 
                         (0, 0, 0), -1)
            
            # Label text
            cv2.putText(frame_with_density, label, (x1 + 5, y1 - 10), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        # Enhanced CCTV info overlay with density information
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        density_level = density_data['density_level']
        total_people = density_data['total_people']
        people_per_m2 = density_data['people_per_m2']
        
        # Main info bar
        info_text = f"LIVE | Frame: {frame_count:06d} | People: {total_people:02d} | Density: {density_level} | {timestamp}"
        cv2.rectangle(frame_with_density, (10, 10), (900, 50), (0, 0, 0), -1)
        cv2.putText(frame_with_density, info_text, (15, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, density_data['density_color'], 2)
        
        # Density details bar
        density_text = f"Density: {people_per_m2}/m² | Level: {density_level} | Grid Analysis: ACTIVE"
        cv2.rectangle(frame_with_density, (10, 55), (700, 85), (0, 0, 0), -1)
        cv2.putText(frame_with_density, density_text, (15, 75), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, density_data['density_color'], 2)
        
        # Density legend
        legend_y = 100
        legend_colors = [
            ("EMPTY", (128, 128, 128)),
            ("LOW", (0, 255, 0)),
            ("MEDIUM", (0, 165, 255)),
            ("HIGH", (0, 100, 255)),
            ("CRITICAL", (0, 0, 255))
        ]
        
        for i, (level, color) in enumerate(legend_colors):
            x = 15 + i * 120
            cv2.rectangle(frame_with_density, (x, legend_y), (x + 15, legend_y + 15), color, -1)
            cv2.putText(frame_with_density, level, (x + 20, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame_with_density


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
                
                # Extra cleanup delay for codec resources
                time.sleep(0.5)
                print("✅ Previous session cleaned up")
            
            # Ensure we start fresh
            self.current_cap = None
            self.is_streaming = False
            
            # Check if file exists
            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Check file size
            file_size = os.path.getsize(video_path)
            print(f"📁 Video file size: {file_size} bytes")
            
            # Try to open video with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🎬 Opening video (attempt {attempt + 1}/{max_retries})...")
                    self.current_cap = cv2.VideoCapture(video_path)
                    
                    # Check if video opened
                    if self.current_cap.isOpened():
                        print("✅ Video opened successfully")
                        break
                    else:
                        print(f"❌ Video open failed (attempt {attempt + 1})")
                        if self.current_cap:
                            self.current_cap.release()
                            self.current_cap = None
                        
                        if attempt < max_retries - 1:
                            time.sleep(0.3)  # Wait before retry
                            
                except Exception as open_error:
                    print(f"❌ Video open exception (attempt {attempt + 1}): {open_error}")
                    if self.current_cap:
                        self.current_cap.release()
                        self.current_cap = None
                    
                    if attempt < max_retries - 1:
                        time.sleep(0.3)
            
            # Final check if video is ready
            if not self.current_cap or not self.current_cap.isOpened():
                error_msg = f"Could not open video after {max_retries} attempts: {video_path}"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Test first frame
            ret, test_frame = self.current_cap.read()
            if not ret:
                error_msg = f"Could not read first frame from: {video_path}"
                print(f"❌ {error_msg}")
                self.current_cap.release()
                self.current_cap = None
                return {'success': False, 'error': error_msg}
            
            print(f"✅ First frame read successfully: {test_frame.shape}")
            
            # Reset and get properties
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = self.current_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📊 Video properties: {fps} FPS, {frame_count} frames")
            
            # Clear queues and start fresh
            self._clear_queues()
            self.is_streaming = True
            
            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            print("✅ Capture thread started")
            
            self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.detection_thread.start()
            print("✅ Detection thread started")
            
            # Start watchdog
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
            
            # Cleanup on error
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
                            # Restart video from beginning
                            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            consecutive_errors = 0
                            time.sleep(0.1)
                            continue
                        except Exception as restart_error:
                            print(f"❌ Restart failed: {restart_error}")
                            break
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Resize for optimal performance
                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Safe queue operation
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame.copy(), frame_counter), timeout=1)
                    else:
                        # Remove old frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put((frame.copy(), frame_counter), timeout=1)
                        except (queue.Empty, queue.Full):
                            pass
                except Exception as queue_error:
                    print(f"⚠️ Queue error (non-fatal): {queue_error}")
                    continue
                
                frame_counter += 1
                time.sleep(0.033)  # ~30 FPS
                
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
                # Get frame from queue with timeout
                frame, frame_count = self.frame_queue.get(timeout=0.5)
                
                # Reset error counter on successful queue get
                consecutive_errors = 0
                
                # Safely detect people
                try:
                    detections = self.detector.detect_people_cctv(frame)
                except Exception as detect_error:
                    print(f"⚠️ Detection error (using empty detections): {detect_error}")
                    detections = []
                
                # Safely calculate density
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
                
                # Safely draw annotations
                try:
                    annotated_frame = self.detector.draw_cctv_style_with_density(
                        frame, detections, density_data, frame_count
                    )
                except Exception as draw_error:
                    print(f"⚠️ Drawing error (using original frame): {draw_error}")
                    annotated_frame = frame
                
                # Safe result queue operation
                result_data = {
                    'frame': annotated_frame,
                    'detections': detections,
                    'density_data': density_data,
                    'frame_count': frame_count,
                    'people_count': len(detections),
                    'timestamp': time.time()
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
                
                # Update FPS counter
                self._update_fps()
                
            except queue.Empty:
                # Normal timeout, continue
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
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
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
        if 'grid_density' in density_data:
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
            }
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
