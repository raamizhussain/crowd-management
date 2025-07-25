import cv2
import numpy as np
from ultralytics import YOLO
import colorsys
import threading
import time
from queue import Queue, Empty
import torch
from collections import deque
import math
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDenseCrowdAnalyzer:
    def __init__(self, gpu_acceleration=True, target_fps=30):
        # Performance settings
        self.target_fps = target_fps
        self.frame_skip = 2
        self.detection_resolution = (416, 320)
        self.display_resolution = (640, 480)
        
        # Initialize enhanced YOLO models
        self.det_model = self._initialize_optimized_yolo(gpu_acceleration)
        self.head_model = None  # Will be initialized if needed
        
        # Threading and queues for async processing
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_active = False
        self.processing_thread = None
        
        # Frame management
        self.frame_count = 0
        self.cached_detections = None
        self.cached_results = None
        self.last_detection_time = 0
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=10)
        
        # Dense crowd detection parameters
        self.density_thresholds = {
            'sparse': 5,
            'medium': 15,
            'dense': 25,
            'extreme': 40
        }
        
        # Adaptive detection parameters
        self.detection_params = {
            'sparse': {'conf': 0.25, 'iou': 0.45, 'max_det': 100, 'imgsz': 416},
            'medium': {'conf': 0.18, 'iou': 0.35, 'max_det': 200, 'imgsz': 640},
            'dense': {'conf': 0.12, 'iou': 0.25, 'max_det': 400, 'imgsz': 640},
            'extreme': {'conf': 0.08, 'iou': 0.15, 'max_det': 600, 'imgsz': 832}
        }
        
        # Advanced ground plane detection parameters
        self.camera_height_estimate = 3.0
        self.average_person_height = 1.7
        self.camera_tilt_angle = 0.0
        self.focal_length_estimate = 500.0
        
        # Ground plane and calibration
        self.ground_plane_normal = None
        self.ground_plane_point = None
        self.vanishing_point = None
        self.horizon_line = None
        self.camera_matrix = None
        self.ground_homography = None
        
        # Dynamic grid parameters
        self.base_grid_size = 1.0
        self.min_grid_size = 0.5
        self.max_grid_size = 2.0
        self.adaptive_grid_cells = None
        self.grid_depth_levels = 8
        self.grid_width_cells = 12
        
        # Ground detection history and validation
        self.detection_history = deque(maxlen=50)
        self.bbox_history = deque(maxlen=30)
        self.ground_plane_stability = 0.0
        self.calibration_confidence = 0.0
        self.ground_detection_samples = []
        
        # Multi-scale detection tracking
        self.detection_scales = [416, 640, 832]
        self.current_density_level = "sparse"
        self.density_history = deque(maxlen=10)
        
        # Visualization and analysis
        self.density_colors = self._generate_enhanced_colormap()
        self.crowd_threshold = 5
        self.ground_mask = None
        self.ground_roi_detailed = None
        
        # State tracking
        self.calibrated = False
        self.ground_plane_detected = False
        self.advanced_calibration_complete = False
        self.multi_scale_enabled = False
        
        # Enhanced error recovery
        self.detection_failures = 0
        self.max_detection_failures = 5
        
        # Preallocated arrays
        self._preallocate_enhanced_arrays()

    def _initialize_optimized_yolo(self, gpu_acceleration):
        """Initialize YOLO with dense crowd optimizations"""
        try:
            model = YOLO("yolov8n.pt")
            
            if gpu_acceleration and torch.cuda.is_available():
                model.to('cuda')
                try:
                    model.model = model.model.half()
                    logger.info("GPU acceleration enabled with FP16 precision for dense crowd detection")
                except Exception as e:
                    logger.warning(f"FP16 precision failed, using FP32: {e}")
                    logger.info("GPU acceleration enabled with FP32 precision")
            else:
                model.to('cpu')
                logger.info("Using CPU inference")
            
            # Warm up the model with different scales
            for scale in [320, 416, 640]:
                dummy_frame = np.zeros((scale, scale, 3), dtype=np.uint8)
                try:
                    model.predict(dummy_frame, verbose=False, imgsz=scale, conf=0.1)
                except Exception as e:
                    logger.warning(f"Warmup failed for scale {scale}: {e}")
            
            logger.info("Model warmed up for multi-scale dense crowd detection")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def _preallocate_enhanced_arrays(self):
        """Preallocate arrays for enhanced processing"""
        self.temp_frame = np.zeros((*self.detection_resolution, 3), dtype=np.uint8)
        self.display_frame = np.zeros((*self.display_resolution, 3), dtype=np.uint8)
        self.overlay = np.zeros((*self.display_resolution, 3), dtype=np.uint8)
        self.edge_map = np.zeros(self.display_resolution[::-1], dtype=np.uint8)
        self.gradient_x = np.zeros(self.display_resolution[::-1], dtype=np.float32)
        self.gradient_y = np.zeros(self.display_resolution[::-1], dtype=np.float32)
        
        # Multi-scale processing arrays
        self.multi_scale_results = []
        self.merged_detections = None

    def _generate_enhanced_colormap(self):
        """Generate enhanced color map for density visualization"""
        colors = []
        max_density = 20  # Increased for dense crowds
        for i in range(max_density + 1):
            if i == 0:
                colors.append((30, 30, 30))
            else:
                normalized = i / max_density
                if normalized <= 0.15:
                    # Deep Blue to Blue
                    factor = normalized / 0.15
                    r, g, b = int(20 * factor), int(100 * factor), 255
                elif normalized <= 0.3:
                    # Blue to Cyan
                    factor = (normalized - 0.15) / 0.15
                    r, g, b = int(20 + 80 * factor), int(100 + 155 * factor), 255
                elif normalized <= 0.45:
                    # Cyan to Green
                    factor = (normalized - 0.3) / 0.15
                    r, g, b = int(100 - 100 * factor), 255, int(255 - 255 * factor)
                elif normalized <= 0.6:
                    # Green to Yellow
                    factor = (normalized - 0.45) / 0.15
                    r, g, b = int(255 * factor), 255, 0
                elif normalized <= 0.75:
                    # Yellow to Orange
                    factor = (normalized - 0.6) / 0.15
                    r, g, b = 255, int(255 - 128 * factor), 0
                elif normalized <= 0.9:
                    # Orange to Red
                    factor = (normalized - 0.75) / 0.15
                    r, g, b = 255, int(127 - 127 * factor), 0
                else:
                    # Red to Dark Red
                    factor = (normalized - 0.9) / 0.1
                    r, g, b = int(255 - 100 * factor), 0, int(50 * factor)
                
                colors.append((b, g, r))  # BGR format for OpenCV
        return colors

    def start_async_processing(self):
        """Start asynchronous frame processing thread"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._async_process_frames, daemon=True)
            self.processing_thread.start()
            logger.info("Async processing started")

    def stop_async_processing(self):
        """Stop asynchronous processing"""
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Async processing stopped")

    def _async_process_frames(self):
        """Enhanced asynchronous frame processing worker"""
        while self.processing_active:
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=0.01)
                    if frame_data is not None:
                        frame, frame_number = frame_data
                        result = self._process_frame_internal_enhanced(frame, frame_number)
                        
                        try:
                            self.result_queue.put_nowait((result, frame_number))
                        except:
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait((result, frame_number))
                            except:
                                pass
                else:
                    time.sleep(0.001)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Enhanced processing error: {e}")

    def estimate_crowd_density_level(self, detections):
        """Enhanced crowd density estimation"""
        if detections is None or len(detections) == 0:
            return "sparse"
        
        num_people = len(detections)
        
        # Calculate area-based density if ground plane is detected
        if self.advanced_calibration_complete and self.adaptive_grid_cells is not None:
            total_area = self.grid_width_cells * self.grid_depth_levels * (self.base_grid_size ** 2)
            people_per_sqm = num_people / total_area if total_area > 0 else 0
            
            if people_per_sqm > 3.0:
                return "extreme"
            elif people_per_sqm > 2.0:
                return "dense"
            elif people_per_sqm > 1.0:
                return "medium"
            else:
                return "sparse"
        else:
            # Fallback to count-based estimation
            if num_people >= self.density_thresholds['extreme']:
                return "extreme"
            elif num_people >= self.density_thresholds['dense']:
                return "dense"
            elif num_people >= self.density_thresholds['medium']:
                return "medium"
            else:
                return "sparse"

    def detect_vanishing_point_enhanced(self, frame):
        """Enhanced vanishing point detection with better line filtering"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced edge detection with multiple methods
            edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Detect lines with multiple parameter sets
            lines_sets = []
            hough_params = [
                {'threshold': 80, 'minLineLength': 50, 'maxLineGap': 10},
                {'threshold': 60, 'minLineLength': 30, 'maxLineGap': 15},
                {'threshold': 40, 'minLineLength': 25, 'maxLineGap': 20}
            ]
            
            for params in hough_params:
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, **params)
                if lines is not None:
                    lines_sets.extend(lines)
            
            if len(lines_sets) < 15:
                return None
            
            # Filter lines by angle (remove horizontal lines, keep perspective lines)
            filtered_lines = []
            for line in lines_sets:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                # Keep lines that are not too horizontal (perspective lines)
                if abs(angle) > 15 and abs(angle) < 165:
                    filtered_lines.append(line[0])
            
            if len(filtered_lines) < 10:
                return None
            
            # Calculate intersection points with improved filtering
            intersections = []
            h, w = frame.shape[:2]
            
            for i in range(len(filtered_lines)):
                for j in range(i + 1, min(i + 50, len(filtered_lines))):  # Limit combinations for performance
                    line1 = filtered_lines[i]
                    line2 = filtered_lines[j]
                    
                    intersection = self._line_intersection_enhanced(line1, line2)
                    if intersection is not None:
                        x, y = intersection
                        # Expanded bounds for vanishing point detection
                        if -w <= x <= 2*w and -h <= y <= 2*h:
                            intersections.append([x, y])
            
            if len(intersections) < 8:
                return None
            
            # Enhanced clustering with density-based approach
            intersections = np.array(intersections)
            
            # Use adaptive epsilon based on image size
            eps = min(w, h) * 0.08
            clustering = DBSCAN(eps=eps, min_samples=4).fit(intersections)
            
            labels = clustering.labels_
            unique_labels = set(labels)
            if len(unique_labels) <= 1 or -1 not in unique_labels:
                return None
            
            # Find the largest cluster (excluding noise)
            valid_labels = [label for label in unique_labels if label != -1]
            if not valid_labels:
                return None
            
            largest_cluster_label = max(valid_labels, key=lambda x: np.sum(labels == x))
            cluster_points = intersections[labels == largest_cluster_label]
            
            if len(cluster_points) < 4:
                return None
            
            # Use median instead of mean for robustness
            vanishing_point = np.median(cluster_points, axis=0)
            
            return vanishing_point
            
        except Exception as e:
            logger.error(f"Enhanced vanishing point detection error: {e}")
            return None

    def _line_intersection_enhanced(self, line1, line2):
        """Enhanced line intersection with better numerical stability"""
        try:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Check if lines are parallel
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-8:
                return None
            
            # Calculate intersection using more stable formula
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            # Check if intersection is within reasonable bounds
            if -2.0 <= t <= 3.0 and -2.0 <= u <= 3.0:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return (x, y)
            
            return None
        except Exception as e:
            logger.debug(f"Line intersection error: {e}")
            return None

    def multi_scale_detection(self, frame, density_level):
        """Multi-scale detection strategy for dense crowds"""
        try:
            if density_level in ["sparse", "medium"]:
                # Single scale for less dense crowds
                return self._single_scale_detection(frame, density_level)
            
            # Multi-scale detection for dense crowds
            all_detections = []
            detection_weights = []
            
            # Use multiple scales with different parameters
            scales_config = {
                640: {'conf': 0.15, 'iou': 0.3},
                832: {'conf': 0.12, 'iou': 0.25},
                1024: {'conf': 0.10, 'iou': 0.2} if density_level == "extreme" else None
            }
            
            original_h, original_w = frame.shape[:2]
            
            for scale, config in scales_config.items():
                if config is None:
                    continue
                    
                try:
                    # Resize frame to target scale
                    scale_factor = scale / max(original_w, original_h)
                    new_w = int(original_w * scale_factor)
                    new_h = int(original_h * scale_factor)
                    
                    scaled_frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Run detection
                    results = self.det_model.predict(
                        scaled_frame,
                        conf=config['conf'],
                        iou=config['iou'],
                        max_det=800,
                        verbose=False,
                        imgsz=scale
                    )
                    
                    if results[0].boxes is not None and results[0].boxes.xyxy is not None:
                        detections = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        
                        # Scale detections back to original resolution
                        detections[:, [0, 2]] *= (original_w / new_w)
                        detections[:, [1, 3]] *= (original_h / new_h)
                        
                        # Add confidence scores
                        detections_with_conf = np.column_stack([detections, confidences])
                        all_detections.append(detections_with_conf)
                        detection_weights.append(len(detections))
                        
                except Exception as e:
                    logger.warning(f"Error in scale {scale}: {e}")
                    continue
            
            if not all_detections:
                return None
            
            # Merge and filter multi-scale detections
            merged_detections = self._merge_multi_scale_detections(all_detections, detection_weights)
            return merged_detections
            
        except Exception as e:
            logger.error(f"Multi-scale detection error: {e}")
            return self._single_scale_detection(frame, density_level)

    def _single_scale_detection(self, frame, density_level):
        """Single scale detection with adaptive parameters"""
        try:
            params = self.detection_params.get(density_level, self.detection_params['sparse'])
            
            # Resize frame if needed
            target_size = params['imgsz']
            if max(frame.shape[:2]) != target_size:
                scale_factor = target_size / max(frame.shape[:2])
                new_w = int(frame.shape[1] * scale_factor)
                new_h = int(frame.shape[0] * scale_factor)
                detection_frame = cv2.resize(frame, (new_w, new_h))
            else:
                detection_frame = frame
                scale_factor = 1.0
            
            # Run detection with error recovery
            try:
                results = self.det_model.predict(
                    detection_frame,
                    conf=params['conf'],
                    iou=params['iou'],
                    max_det=params['max_det'],
                    verbose=False,
                    imgsz=params['imgsz']
                )
                
                # Reset failure counter on success
                self.detection_failures = 0
                
                if results[0].boxes is not None and results[0].boxes.xyxy is not None:
                    detections = results[0].boxes.xyxy.cpu().numpy()
                    
                    # Scale back to original resolution if needed
                    if scale_factor != 1.0:
                        detections /= scale_factor
                    
                    return detections
                
            except Exception as e:
                self.detection_failures += 1
                logger.warning(f"Detection failed (attempt {self.detection_failures}): {e}")
                
                if self.detection_failures >= self.max_detection_failures:
                    logger.error("Max detection failures reached, using cached results")
                    return self.cached_detections
            
            return None
            
        except Exception as e:
            logger.error(f"Single scale detection error: {e}")
            return None

    def _merge_multi_scale_detections(self, all_detections, weights):
        """Merge detections from multiple scales with weighted NMS"""
        try:
            if not all_detections:
                return None
            
            # Combine all detections
            combined_detections = []
            for i, detections in enumerate(all_detections):
                # Add scale weight to confidence
                weight = min(1.0, weights[i] / max(weights)) if weights else 1.0
                detections_weighted = detections.copy()
                detections_weighted[:, 4] *= weight  # Multiply confidence by weight
                combined_detections.append(detections_weighted)
            
            if not combined_detections:
                return None
            
            merged = np.vstack(combined_detections)
            
            # Apply weighted NMS
            final_detections = self._weighted_nms(merged, iou_threshold=0.3)
            
            return final_detections[:, :4] if final_detections is not None else None
            
        except Exception as e:
            logger.error(f"Multi-scale merge error: {e}")
            return None

    def _weighted_nms(self, detections, iou_threshold=0.3):
        """Weighted Non-Maximum Suppression for multi-scale detections"""
        try:
            if len(detections) == 0:
                return None
            
            # Sort by confidence
            indices = np.argsort(detections[:, 4])[::-1]
            detections = detections[indices]
            
            keep = []
            while len(detections) > 0:
                # Take the detection with highest confidence
                current = detections[0]
                keep.append(current)
                
                if len(detections) == 1:
                    break
                
                # Calculate IoU with remaining detections
                ious = self._calculate_iou_batch(current[:4], detections[1:, :4])
                
                # Keep detections with IoU below threshold
                mask = ious < iou_threshold
                detections = detections[1:][mask]
            
            return np.array(keep) if keep else None
            
        except Exception as e:
            logger.error(f"Weighted NMS error: {e}")
            return detections

    def _calculate_iou_batch(self, box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        try:
            x1, y1, x2, y2 = box
            xx1 = np.maximum(x1, boxes[:, 0])
            yy1 = np.maximum(y1, boxes[:, 1])
            xx2 = np.minimum(x2, boxes[:, 2])
            yy2 = np.minimum(y2, boxes[:, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union = area1 + area2 - intersection
            
            return intersection / np.maximum(union, 1e-8)
        except Exception as e:
            logger.debug(f"IoU calculation error: {e}")
            return np.zeros(len(boxes))

    def detect_ground_plane_advanced_enhanced(self, frame, detections):
        """Enhanced ground plane detection with dense crowd handling"""
        if detections is None or len(detections) == 0:
            return False
        
        try:
            h, w = frame.shape[:2]
            
            # Extract and validate detection points with enhanced filtering
            person_points = []
            person_heights = []
            
            for detection in detections:
                x1, y1, x2, y2 = detection
                foot_x = (x1 + x2) / 2
                foot_y = y2
                person_height = y2 - y1
                person_width = x2 - x1
                
                # Enhanced validation for dense crowds
                if (0 <= foot_x < w and 0 <= foot_y < h and 
                    person_height > 15 and person_width > 8 and
                    person_height < h * 0.8 and person_width < w * 0.5):
                    person_points.append([foot_x, foot_y])
                    person_heights.append(person_height)
            
            if len(person_points) < 2:
                return False
            
            # Store detection data with enhanced capacity
            self.detection_history.append(person_points.copy())
            self.bbox_history.append(person_heights.copy())
            
            # Need more samples for dense crowds
            min_samples = 8 if self.current_density_level in ["dense", "extreme"] else 5
            if len(self.detection_history) < min_samples:
                return False
            
            # Enhanced vanishing point detection
            self.vanishing_point = self.detect_vanishing_point_enhanced(frame)
            
            # Estimate camera parameters
            self.estimate_camera_parameters_enhanced(frame, detections)
            
            # Enhanced ground plane estimation using RANSAC
            all_points = []
            all_heights = []
            for hist_points, hist_heights in zip(list(self.detection_history), list(self.bbox_history)):
                all_points.extend(hist_points)
                all_heights.extend(hist_heights)
            
            if len(all_points) < min_samples * 2:
                return False
            
            ground_plane_points = self._estimate_ground_plane_ransac_enhanced(all_points, all_heights)
            
            if ground_plane_points is None:
                return False
            
            # Calculate ground plane parameters
            self._calculate_ground_plane_geometry_enhanced(ground_plane_points, all_heights)
            
            # Create advanced ground ROI
            self._create_advanced_ground_roi_enhanced(frame, ground_plane_points)
            
            # Calibrate perspective transformation
            success = self._calibrate_advanced_homography_enhanced(frame, all_points, all_heights)
            
            if success:
                self.ground_plane_detected = True
                self.advanced_calibration_complete = True
                self.calibration_confidence = self._calculate_calibration_confidence_enhanced()
                logger.info(f"Enhanced ground plane detected with confidence: {self.calibration_confidence:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Enhanced ground plane detection error: {e}")
            return False

    def estimate_camera_parameters_enhanced(self, frame, detections):
        """Enhanced camera parameter estimation with dense crowd considerations"""
        try:
            h, w = frame.shape[:2]
            
            # Enhanced focal length estimation
            if self.vanishing_point is not None:
                vp_x, vp_y = self.vanishing_point
                cx, cy = w/2, h/2
                
                # More robust focal length estimation
                focal_estimate_1 = math.sqrt((vp_x - cx)**2 + (vp_y - cy)**2)
                
                # Alternative estimation using person heights
                if len(detections) > 0:
                    heights = [(det[3] - det[1]) for det in detections]
                    median_height = np.median(heights)
                    # Assume person should be around 100-200 pixels at mid-distance
                    focal_estimate_2 = (median_height * w) / (2 * self.average_person_height)
                    
                    # Weighted average of estimates
                    self.focal_length_estimate = (focal_estimate_1 * 0.7 + focal_estimate_2 * 0.3)
                else:
                    self.focal_length_estimate = focal_estimate_1
            else:
                # Fallback estimation
                self.focal_length_estimate = w * 0.8
            
            # Enhanced camera matrix
            self.camera_matrix = np.array([
                [self.focal_length_estimate, 0, w/2],
                [0, self.focal_length_estimate, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Enhanced camera tilt estimation
            if self.vanishing_point is not None:
                self.camera_tilt_angle = math.atan2(self.vanishing_point[1] - h/2, 
                                                   self.focal_length_estimate)
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced camera parameter estimation error: {e}")
            return False

    def _estimate_ground_plane_ransac_enhanced(self, points, heights):
        """Enhanced RANSAC ground plane estimation for dense crowds"""
        try:
            if len(points) < 6:
                return None
            
            points_array = np.array(points)
            heights_array = np.array(heights)
            
            # Enhanced outlier filtering for dense crowds
            height_median = np.median(heights_array)
            height_mad = np.median(np.abs(heights_array - height_median))  # Median Absolute Deviation
            
            # Use MAD for more robust outlier detection
            threshold = 2.5 * height_mad if height_mad > 0 else height_median * 0.3
            valid_indices = np.abs(heights_array - height_median) < threshold
            
            filtered_points = points_array[valid_indices]
            if len(filtered_points) < 6:
                return None
            
            # Enhanced RANSAC with adaptive parameters
            max_iterations = 200 if self.current_density_level in ["dense", "extreme"] else 100
            best_inliers = []
            best_score = 0
            
            for iteration in range(max_iterations):
                # Sample more points for better plane estimation in dense crowds
                sample_size = min(5, len(filtered_points))
                sample_indices = np.random.choice(len(filtered_points), sample_size, replace=False)
                sample_points = filtered_points[sample_indices]
                
                # Fit plane using median for robustness
                plane_y = np.median(sample_points[:, 1])
                
                # Adaptive inlier threshold based on density
                if self.current_density_level in ["dense", "extreme"]:
                    distance_threshold = 25  # More lenient for dense crowds
                else:
                    distance_threshold = 15  # Stricter for sparse crowds
                
                # Count inliers
                inliers = []
                for i, point in enumerate(filtered_points):
                    distance = abs(point[1] - plane_y)
                    if distance < distance_threshold:
                        inliers.append(i)
                
                if len(inliers) > best_score:
                    best_score = len(inliers)
                    best_inliers = inliers.copy()
            
            if len(best_inliers) < max(6, len(filtered_points) * 0.3):
                return None
            
            return filtered_points[best_inliers]
            
        except Exception as e:
            logger.error(f"Enhanced RANSAC ground plane estimation error: {e}")
            return None

    def _calculate_ground_plane_geometry_enhanced(self, ground_points, heights):
        """Enhanced ground plane geometry calculation"""
        try:
            # Enhanced ground plane normal calculation
            y_coords = ground_points[:, 1]
            x_coords = ground_points[:, 0]
            
            # Use robust statistics
            ground_level = np.median(y_coords)
            ground_variance = np.var(y_coords)
            
            # Store enhanced ground plane parameters
            self.ground_plane_point = np.array([np.median(x_coords), ground_level, 0])
            self.ground_plane_normal = np.array([0, 1, 0])  # Horizontal assumption
            
            # Enhanced scale estimation for dense crowds
            if len(heights) > 0:
                # Use multiple robust statistics
                median_height = np.median(heights)
                height_75th = np.percentile(heights, 75)
                height_25th = np.percentile(heights, 25)
                
                # Use 75th percentile for dense crowds (less affected by partial occlusions)
                representative_height = height_75th if self.current_density_level in ["dense", "extreme"] else median_height
                
                # Enhanced pixel-to-meter ratio with perspective correction
                base_ratio = self.average_person_height / representative_height
                
                # Apply perspective correction based on position in frame
                if hasattr(self, 'vanishing_point') and self.vanishing_point is not None:
                    # Adjust ratio based on distance from vanishing point
                    avg_distance_from_vp = np.mean([
                        math.sqrt((pt[0] - self.vanishing_point[0])**2 + (pt[1] - self.vanishing_point[1])**2)
                        for pt in ground_points
                    ])
                    frame_diagonal = math.sqrt(640**2 + 480**2)
                    perspective_factor = 1.0 + (avg_distance_from_vp / frame_diagonal) * 0.5
                    self.pixel_to_meter_ratio = base_ratio * perspective_factor
                else:
                    self.pixel_to_meter_ratio = base_ratio
            else:
                self.pixel_to_meter_ratio = 0.003  # Enhanced fallback
            
            # Store ground plane stability metric
            self.ground_plane_stability = 1.0 / (1.0 + ground_variance / 100.0)
            
        except Exception as e:
            logger.error(f"Enhanced ground plane geometry calculation error: {e}")

    def _create_advanced_ground_roi_enhanced(self, frame, ground_points):
        """Enhanced ground ROI creation for dense crowds"""
        try:
            h, w = frame.shape[:2]
            
            # Calculate enhanced ground bounds
            x_coords = ground_points[:, 0]
            y_coords = ground_points[:, 1]
            
            # Use robust statistics for bounds
            x_median = np.median(x_coords)
            y_median = np.median(y_coords)
            x_mad = np.median(np.abs(x_coords - x_median))
            y_mad = np.median(np.abs(y_coords - y_median))
            
            # Expand bounds more aggressively for dense crowds
            expansion_factor = 1.5 if self.current_density_level in ["dense", "extreme"] else 1.2
            
            x_min = max(0, x_median - x_mad * expansion_factor * 2)
            x_max = min(w-1, x_median + x_mad * expansion_factor * 2)
            y_min = max(0, y_median - y_mad * expansion_factor * 1.5)
            y_max = min(h-1, y_median + y_mad * expansion_factor * 2)
            
            # Enhanced perspective-aware trapezoid creation
            if self.vanishing_point is not None:
                vp_x, vp_y = self.vanishing_point
                
                # Create more accurate perspective trapezoid
                # Top edge (closer to vanishing point)
                top_y = max(0, int(y_min - 30))
                
                # Calculate perspective-aware top width
                vp_distance = math.sqrt((vp_x - x_median)**2 + (vp_y - y_median)**2)
                frame_diagonal = math.sqrt(w**2 + h**2)
                perspective_ratio = min(0.8, max(0.4, vp_distance / frame_diagonal))
                
                top_width_factor = perspective_ratio
                top_left_x = max(0, int(vp_x - (vp_x - x_min) * top_width_factor))
                top_right_x = min(w-1, int(vp_x + (x_max - vp_x) * top_width_factor))
                
                # Bottom edge (further from vanishing point)
                bottom_y = min(h-1, int(y_max + 30))
                bottom_left_x = max(0, int(x_min - 30))
                bottom_right_x = min(w-1, int(x_max + 30))
                
                self.ground_roi_detailed = np.array([
                    [top_left_x, top_y],
                    [top_right_x, top_y],
                    [bottom_right_x, bottom_y],
                    [bottom_left_x, bottom_y]
                ], dtype=np.int32)
            else:
                # Enhanced rectangular ROI
                roi_x_min = max(0, int(x_min - 40))
                roi_x_max = min(w-1, int(x_max + 40))
                roi_y_min = max(0, int(y_min - 30))
                roi_y_max = min(h-1, int(y_max + 30))
                
                self.ground_roi_detailed = np.array([
                    [roi_x_min, roi_y_min],
                    [roi_x_max, roi_y_min],
                    [roi_x_max, roi_y_max],
                    [roi_x_min, roi_y_max]
                ], dtype=np.int32)
            
            # Create enhanced ground mask
            self.ground_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(self.ground_mask, [self.ground_roi_detailed], 255)
            
        except Exception as e:
            logger.error(f"Enhanced ground ROI creation error: {e}")

    def _calibrate_advanced_homography_enhanced(self, frame, all_points, all_heights):
        """Enhanced homography calibration with dense crowd adaptations"""
        try:
            if self.ground_roi_detailed is None:
                return False
            
            h, w = frame.shape[:2]
            
            # Source points (image coordinates)
            src_pts = self.ground_roi_detailed.astype(np.float32)
            
            # Enhanced real-world dimension calculation
            roi_height_pixels = np.max(src_pts[:, 1]) - np.min(src_pts[:, 1])
            roi_width_pixels = np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])
            
            # Advanced scale estimation with perspective correction
            if hasattr(self, 'pixel_to_meter_ratio') and self.pixel_to_meter_ratio > 0:
                estimated_width_meters = roi_width_pixels * self.pixel_to_meter_ratio
                
                # Enhanced depth estimation with perspective correction
                if self.vanishing_point is not None:
                    # Use vanishing point for better depth estimation
                    vp_y = self.vanishing_point[1]
                    roi_center_y = (np.max(src_pts[:, 1]) + np.min(src_pts[:, 1])) / 2
                    perspective_factor = 1.0 + abs(roi_center_y - vp_y) / h
                    estimated_depth_meters = roi_height_pixels * self.pixel_to_meter_ratio * perspective_factor
                else:
                    estimated_depth_meters = roi_height_pixels * self.pixel_to_meter_ratio * 1.3
            else:
                # Enhanced fallback estimation
                estimated_width_meters = roi_width_pixels * 0.008
                estimated_depth_meters = roi_height_pixels * 0.012
            
            # Enhanced adaptive grid sizing for dense crowds
            total_area = estimated_width_meters * estimated_depth_meters
            expected_people = len(all_points) if all_points else 1
            
            if total_area > 0:
                people_per_sqm = expected_people / total_area
                
                # Density-adaptive grid sizing
                if people_per_sqm > 3.0:  # Extreme density
                    self.base_grid_size = 0.6
                elif people_per_sqm > 2.0:  # High density
                    self.base_grid_size = 0.8
                elif people_per_sqm > 1.0:  # Medium density
                    self.base_grid_size = 1.0
                else:  # Low density
                    self.base_grid_size = 1.3
            
            # Calculate enhanced grid dimensions
            self.grid_width_cells = max(6, min(18, int(estimated_width_meters / self.base_grid_size)))
            self.grid_depth_levels = max(4, min(15, int(estimated_depth_meters / self.base_grid_size)))
            
            # Destination points with enhanced scaling
            dst_width = self.grid_width_cells * self.base_grid_size
            dst_height = self.grid_depth_levels * self.base_grid_size
            
            dst_pts = np.array([
                [0, 0],
                [dst_width, 0],
                [dst_width, dst_height],
                [0, dst_height]
            ], dtype=np.float32)
            
            # Compute enhanced homography with validation
            self.ground_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Validate homography quality
            if self._validate_homography_quality(src_pts, dst_pts):
                # Initialize enhanced adaptive grid
                self.adaptive_grid_cells = np.zeros((self.grid_depth_levels, self.grid_width_cells), dtype=int)
                self.calibrated = True
                
                logger.info(f"Enhanced calibration complete:")
                logger.info(f"  Density Level: {self.current_density_level}")
                logger.info(f"  Grid: {self.grid_width_cells}x{self.grid_depth_levels}")
                logger.info(f"  Area: {dst_width:.1f}m x {dst_height:.1f}m")
                logger.info(f"  Cell size: {self.base_grid_size:.1f}m")
                logger.info(f"  People density: {expected_people/total_area:.1f} people/m²")
                
                return True
            else:
                logger.warning("Homography validation failed")
                return False
            
        except Exception as e:
            logger.error(f"Enhanced homography calibration error: {e}")
            return False

    def _validate_homography_quality(self, src_pts, dst_pts):
        """Validate homography transformation quality"""
        try:
            # Check if homography is reasonable
            if self.ground_homography is None:
                return False
            
            # Test homography on corner points
            test_pts = src_pts.reshape(-1, 1, 2)
            transformed_pts = cv2.perspectiveTransform(test_pts, self.ground_homography)
            transformed_pts = transformed_pts.reshape(-1, 2)
            
            # Check if transformation preserves relative positions
            dst_pts_check = dst_pts.reshape(-1, 2)
            distances = np.linalg.norm(transformed_pts - dst_pts_check, axis=1)
            
            # If transformation is too far off, it's invalid
            if np.mean(distances) > np.mean(dst_pts_check) * 0.1:
                return False
            
            # Check for reasonable aspect ratios
            homography_det = np.linalg.det(self.ground_homography[:2, :2])
            if abs(homography_det) < 1e-6 or abs(homography_det) > 1e6:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Homography validation error: {e}")
            return False

    def _calculate_calibration_confidence_enhanced(self):
        """Enhanced calibration confidence calculation"""
        try:
            confidence = 0.0
            
            # Factor 1: Detection sample quality (30%)
            sample_count = len(self.detection_history)
            if sample_count >= 30:
                confidence += 0.30
            elif sample_count >= 15:
                confidence += 0.25
            elif sample_count >= 8:
                confidence += 0.15
            
            # Factor 2: Vanishing point detection (25%)
            if self.vanishing_point is not None:
                confidence += 0.25
            
            # Factor 3: Ground plane stability (20%)
            if hasattr(self, 'ground_plane_stability'):
                confidence += 0.20 * self.ground_plane_stability
            
            # Factor 4: Pixel-to-meter ratio validity (15%)
            if (hasattr(self, 'pixel_to_meter_ratio') and 
                0.001 < self.pixel_to_meter_ratio < 0.02):
                confidence += 0.15
            
            # Factor 5: ROI coverage and quality (10%)
            if self.ground_roi_detailed is not None:
                roi_area = cv2.contourArea(self.ground_roi_detailed)
                frame_area = 640 * 480
                coverage_ratio = roi_area / frame_area
                if 0.1 < coverage_ratio < 0.8:
                    confidence += 0.10
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.debug(f"Calibration confidence calculation error: {e}")
            return 0.0

    def _process_frame_internal_enhanced(self, frame, frame_number):
        """Enhanced internal frame processing with dense crowd handling"""
        start_time = time.time()
        
        # Resize for detection
        if frame.shape[:2] != self.display_resolution[::-1]:
            frame = cv2.resize(frame, self.display_resolution)
        
        # Run enhanced detection
        detections = None
        if frame_number % self.frame_skip == 0:
            try:
                # First, get initial density estimate
                initial_detections = self._single_scale_detection(frame, "medium")
                initial_density = self.estimate_crowd_density_level(initial_detections)
                
                # Update density history for stability
                self.density_history.append(initial_density)
                if len(self.density_history) > 5:
                    # Use most common density level from recent history
                    density_counts = {}
                    for d in list(self.density_history)[-5:]:
                        density_counts[d] = density_counts.get(d, 0) + 1
                    self.current_density_level = max(density_counts, key=density_counts.get)
                else:
                    self.current_density_level = initial_density
                
                # Use appropriate detection strategy based on density
                if self.current_density_level in ["dense", "extreme"]:
                    detections = self.multi_scale_detection(frame, self.current_density_level)
                else:
                    detections = self._single_scale_detection(frame, self.current_density_level)
                
                # Apply post-processing for dense crowds
                if detections is not None and self.current_density_level in ["dense", "extreme"]:
                    detections = self._post_process_dense_detections(detections, frame)
                
                self.cached_detections = detections
                self.last_detection_time = time.time()
                
            except Exception as e:
                logger.error(f"Enhanced detection error: {e}")
                detections = self.cached_detections
        else:
            detections = self.cached_detections

        # Enhanced ground plane detection
        if not self.advanced_calibration_complete and frame_number % 3 == 0:
            self.detect_ground_plane_advanced_enhanced(frame, detections)

        # Enhanced density estimation
        density_level = self.estimate_crowd_density_level(detections)
        
        # Update enhanced grid
        if self.advanced_calibration_complete and detections is not None:
            self._update_adaptive_grid_enhanced(detections)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return {
            'detections': detections,
            'density_level': density_level,
            'current_density_level': self.current_density_level,
            'processing_time': processing_time,
            'ground_detected': self.ground_plane_detected,
            'calibration_confidence': self.calibration_confidence,
            'vanishing_point': self.vanishing_point,
            'multi_scale_used': self.current_density_level in ["dense", "extreme"],
            'detection_count': len(detections) if detections is not None else 0
        }

    def _post_process_dense_detections(self, detections, frame):
        """Post-process detections for dense crowd scenarios"""
        try:
            if detections is None or len(detections) == 0:
                return detections
            
            # Filter detections by size consistency
            heights = detections[:, 3] - detections[:, 1]
            widths = detections[:, 2] - detections[:, 0]
            
            # Remove extremely small or large detections
            height_median = np.median(heights)
            width_median = np.median(widths)
            
            # More lenient filtering for dense crowds
            valid_height = (heights > height_median * 0.3) & (heights < height_median * 3.0)
            valid_width = (widths > width_median * 0.3) & (widths < width_median * 3.0)
            valid_aspect = (heights / np.maximum(widths, 1)) > 0.8  # Ensure person-like aspect ratio
            
            valid_mask = valid_height & valid_width & valid_aspect
            filtered_detections = detections[valid_mask]
            
            # Apply density-based spatial filtering
            if len(filtered_detections) > 20:  # Only for very dense crowds
                filtered_detections = self._spatial_density_filter(filtered_detections, frame)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Dense crowd post-processing error: {e}")
            return detections

    def _spatial_density_filter(self, detections, frame):
        """Apply spatial density filtering for very dense crowds"""
        try:
            if len(detections) < 10:
                return detections
            
            h, w = frame.shape[:2]
            
            # Create spatial grid for density analysis
            grid_size = 50  # 50x50 pixel cells
            grid_h = h // grid_size + 1
            grid_w = w // grid_size + 1
            density_grid = np.zeros((grid_h, grid_w))
            
            # Count detections per grid cell
            for detection in detections:
                x1, y1, x2, y2 = detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                grid_x = min(grid_w - 1, center_x // grid_size)
                grid_y = min(grid_h - 1, center_y // grid_size)
                density_grid[grid_y, grid_x] += 1
            
            # Filter out detections from overly dense cells
            max_density_per_cell = 8
            filtered_detections = []
            
            for detection in detections:
                x1, y1, x2, y2 = detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                grid_x = min(grid_w - 1, center_x // grid_size)
                grid_y = min(grid_h - 1, center_y // grid_size)
                
                if density_grid[grid_y, grid_x] <= max_density_per_cell:
                    filtered_detections.append(detection)
                elif random.random() < 0.3:  # Keep 30% of detections from dense cells
                    filtered_detections.append(detection)
            
            return np.array(filtered_detections) if filtered_detections else detections
            
        except Exception as e:
            logger.debug(f"Spatial density filter error: {e}")
            return detections

    def _update_adaptive_grid_enhanced(self, detections):
        """Enhanced adaptive grid update with dense crowd handling"""
        if self.adaptive_grid_cells is None or self.ground_homography is None:
            return
            
        self.adaptive_grid_cells.fill(0)
        
        # Enhanced position tracking for dense crowds
        for detection in detections:
            x1, y1, x2, y2 = detection
            
            # Use bottom center for foot position
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            try:
                # Transform to ground plane coordinates
                ground_point = cv2.perspectiveTransform(
                    np.array([[[foot_x, foot_y]]], dtype=np.float32), 
                    self.ground_homography
                )[0][0]
                
                # Calculate grid cell indices with bounds checking
                col = int(ground_point[0] / self.base_grid_size)
                row = int(ground_point[1] / self.base_grid_size)
                
                # Enhanced bounds checking
                col = max(0, min(self.grid_width_cells - 1, col))
                row = max(0, min(self.grid_depth_levels - 1, row))
                
                # Increment count with saturation to prevent overflow in dense areas
                current_count = self.adaptive_grid_cells[row, col]
                if current_count < 25:  # Saturation limit for very dense areas
                    self.adaptive_grid_cells[row, col] += 1
                
            except Exception as e:
                logger.debug(f"Grid update error for detection: {e}")
                continue

    def process_frame_realtime(self, frame):
        """Enhanced main real-time processing function"""
        frame_start = time.time()
        self.frame_count += 1
        
        try:
            if frame.shape[:2] != self.display_resolution[::-1]:
                frame = cv2.resize(frame, self.display_resolution)
            
            frame_data = (frame.copy(), self.frame_count)
            try:
                self.frame_queue.put_nowait(frame_data)
            except:
                pass
            
            result_data = None
            try:
                while not self.result_queue.empty():
                    result_data = self.result_queue.get_nowait()
            except:
                pass
            
            if result_data:
                self.cached_results = result_data[0]
            
            output_frame = self._draw_enhanced_visualization_complete(frame, self.cached_results)
            
            frame_time = time.time() - frame_start
            self.fps_counter.append(1.0 / max(frame_time, 0.001))
            
            return output_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

    def _draw_enhanced_visualization_complete(self, frame, results):
        """Complete enhanced visualization with all features"""
        if results is None:
            return frame
        
        detections = results.get('detections')
        density_level = results.get('density_level', 'sparse')
        current_density = results.get('current_density_level', 'sparse')
        vanishing_point = results.get('vanishing_point')
        multi_scale_used = results.get('multi_scale_used', False)
        detection_count = results.get('detection_count', 0)
        
        # Draw detections with density-based coloring
        if detections is not None:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                
                # Color based on current density level
                if current_density == "extreme":
                    color = (0, 0, 255)  # Red
                elif current_density == "dense":
                    color = (0, 100, 255)  # Orange
                elif current_density == "medium":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                
                # Thicker boxes for better visibility in dense crowds
                thickness = 3 if current_density in ["dense", "extreme"] else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw vanishing point with enhanced visualization
        if vanishing_point is not None:
            vp_x, vp_y = map(int, vanishing_point)
            if 0 <= vp_x < frame.shape[1] and 0 <= vp_y < frame.shape[0]:
                cv2.circle(frame, (vp_x, vp_y), 12, (255, 0, 255), -1)
                cv2.circle(frame, (vp_x, vp_y), 16, (255, 255, 255), 2)
                cv2.putText(frame, "VP", (vp_x + 20, vp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw ground ROI with confidence-based styling
        if self.ground_roi_detailed is not None:
            confidence_color = (0, int(255 * self.calibration_confidence), int(255 * (1 - self.calibration_confidence)))
            thickness = max(2, int(4 * self.calibration_confidence))
            cv2.polylines(frame, [self.ground_roi_detailed], True, confidence_color, thickness)
        
        # Draw enhanced adaptive grid
        if self.advanced_calibration_complete and self.adaptive_grid_cells is not None:
            self._draw_adaptive_grid_enhanced_complete(frame, current_density)
        
        # Draw comprehensive statistics
        self._draw_enhanced_stats_complete(frame, results)
        
        # Draw density legend
        self._draw_density_legend_enhanced(frame)
        
        return frame

    def _draw_adaptive_grid_enhanced_complete(self, frame, density_level):
        """Complete enhanced adaptive grid visualization with transparent backgrounds"""
        if self.ground_homography is None or self.adaptive_grid_cells is None:
            return
        
        h, w = frame.shape[:2]
        
        for row in range(self.grid_depth_levels):
            for col in range(self.grid_width_cells):
                count = self.adaptive_grid_cells[row, col]
                
                if count == 0:
                    continue
                
                # Calculate ground plane coordinates
                x1_ground = col * self.base_grid_size
                y1_ground = row * self.base_grid_size
                x2_ground = (col + 1) * self.base_grid_size
                y2_ground = (row + 1) * self.base_grid_size
                
                corners_ground = np.array([
                    [[x1_ground, y1_ground]],
                    [[x2_ground, y1_ground]], 
                    [[x2_ground, y2_ground]],
                    [[x1_ground, y2_ground]]
                ], dtype=np.float32)
                
                try:
                    # Transform to image coordinates
                    corners_img = cv2.perspectiveTransform(
                        corners_ground, 
                        np.linalg.inv(self.ground_homography)
                    )
                    corners_img = corners_img.reshape(-1, 2).astype(int)
                    
                    # Enhanced bounds checking
                    if (np.all(corners_img >= [-50, -50]) and 
                        np.all(corners_img[:, 0] < w + 50) and 
                        np.all(corners_img[:, 1] < h + 50)):
                        
                        # Get enhanced density color
                        color_idx = min(count, len(self.density_colors) - 1)
                        outline_color = self.density_colors[color_idx]
                        
                        # Enhanced line thickness based on density
                        if density_level in ["dense", "extreme"]:
                            line_thickness = 4
                        else:
                            line_thickness = 3
                        
                        # Draw colored outline (transparent background)
                        cv2.polylines(frame, [corners_img], True, outline_color, line_thickness)
                        
                        # Enhanced count text with better visibility
                        if cv2.contourArea(corners_img) > 40:
                            center = np.mean(corners_img, axis=0).astype(int)
                            
                            text = str(count)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            
                            # Adaptive font size based on cell size
                            cell_area = cv2.contourArea(corners_img)
                            if cell_area > 200:
                                font_scale = 0.8
                                font_thickness = 3
                            elif cell_area > 100:
                                font_scale = 0.6
                                font_thickness = 2
                            else:
                                font_scale = 0.4
                                font_thickness = 2
                            
                            (text_width, text_height), baseline = cv2.getTextSize(
                                text, font, font_scale, font_thickness
                            )
                            
                            text_x = center[0] - text_width // 2
                            text_y = center[1] + text_height // 2
                            
                            # Enhanced semi-transparent background
                            bg_padding = 4
                            overlay = frame.copy()
                            cv2.rectangle(overlay, 
                                        (text_x - bg_padding, text_y - text_height - bg_padding),
                                        (text_x + text_width + bg_padding, text_y + baseline + bg_padding),
                                        (0, 0, 0), -1)
                            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
                            
                            # Draw enhanced text
                            cv2.putText(frame, text, (text_x, text_y), 
                                      font, font_scale, (255, 255, 255), font_thickness)
                            
                except Exception as e:
                    logger.debug(f"Grid drawing error: {e}")
                    continue

    def _draw_enhanced_stats_complete(self, frame, results):
        """Complete enhanced statistics visualization"""
        if results is None:
            return
        
        detections = results.get('detections')
        density_level = results.get('density_level', 'sparse')
        current_density = results.get('current_density_level', 'sparse')
        processing_time = results.get('processing_time', 0)
        calibration_confidence = results.get('calibration_confidence', 0)
        multi_scale_used = results.get('multi_scale_used', False)
        detection_count = results.get('detection_count', 0)
        
        current_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Enhanced color scheme based on density
        if current_density == "extreme":
            level_color = (0, 0, 255)  # Red
        elif current_density == "dense":
            level_color = (0, 100, 255)  # Orange  
        elif current_density == "medium":
            level_color = (0, 255, 255)  # Yellow
        else:
            level_color = (0, 255, 0)  # Green
        
        total_people = len(detections) if detections is not None else 0
        
        # Enhanced comprehensive stats
        stats = [
            f"FPS: {current_fps:.1f}",
            f"People: {total_people}",
            f"Density: {current_density.upper()}",
            f"Processing: {processing_time*1000:.1f}ms",
            f"Calibration: {calibration_confidence:.1%}",
            f"Grid: {self.grid_width_cells}x{self.grid_depth_levels}" if self.advanced_calibration_complete else "Grid: Detecting...",
            f"Cell Size: {self.base_grid_size:.1f}m" if hasattr(self, 'base_grid_size') else "Cell Size: --",
            f"Multi-Scale: {'ON' if multi_scale_used else 'OFF'}",
            f"VP Detected: {'✓' if self.vanishing_point is not None else '✗'}"
        ]
        
        # Enhanced background with dynamic sizing
        stats_height = len(stats) * 22 + 20
        stats_width = 320
        cv2.rectangle(frame, (5, 5), (stats_width, stats_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (stats_width, stats_height), level_color, 3)
        
        # Enhanced text rendering
        for i, stat in enumerate(stats):
            y_pos = 28 + i * 22
            
            # Highlight important stats
            if i == 2:  # Density level
                color = level_color
                font_thickness = 2
            elif i == 4:  # Calibration confidence
                confidence_color = (0, int(255 * calibration_confidence), int(255 * (1 - calibration_confidence)))
                color = confidence_color
                font_thickness = 1
            else:
                color = (255, 255, 255)
                font_thickness = 1
            
            cv2.putText(frame, stat, (12, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, font_thickness)

    def _draw_density_legend_enhanced(self, frame):
        """Enhanced density legend with more detailed information"""
        h, w = frame.shape[:2]
        legend_width = 30
        legend_height = 200
        start_x = w - 60
        start_y = 60
        
        # Enhanced legend background
        cv2.rectangle(frame, (start_x - 8, start_y - 25), 
                     (start_x + legend_width + 50, start_y + legend_height + 15), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (start_x - 8, start_y - 25), 
                     (start_x + legend_width + 50, start_y + legend_height + 15), 
                     (255, 255, 255), 2)
        
        # Enhanced title
        cv2.putText(frame, "Density", (start_x - 5, start_y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Enhanced color gradient with labels
        segment_height = legend_height // min(len(self.density_colors), 15)
        density_labels = ["0", "1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14", "15+"]
        
        for i in range(min(len(self.density_colors), 15)):
            y1 = start_y + i * segment_height
            y2 = y1 + segment_height
            
            color = self.density_colors[i]
            cv2.rectangle(frame, (start_x, y1), (start_x + legend_width, y2), color, -1)
            
            # Enhanced labels
            if i % 2 == 0 and i < len(density_labels):
                label = density_labels[i]
                cv2.putText(frame, label, (start_x + legend_width + 5, y1 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def get_enhanced_performance_stats(self):
        """Get comprehensive enhanced performance statistics"""
        current_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        avg_processing_time = np.mean(list(self.processing_times)) if self.processing_times else 0
        
        return {
            'fps': current_fps,
            'processing_time_ms': avg_processing_time * 1000,
            'ground_detected': self.ground_plane_detected,
            'advanced_calibration': self.advanced_calibration_complete,
            'calibration_confidence': self.calibration_confidence,
            'current_density_level': self.current_density_level,
            'grid_dimensions': f"{self.grid_width_cells}x{self.grid_depth_levels}" if self.advanced_calibration_complete else "N/A",
            'cell_size': f"{self.base_grid_size:.1f}m" if hasattr(self, 'base_grid_size') else "N/A",
            'vanishing_point_detected': self.vanishing_point is not None,
            'multi_scale_enabled': self.current_density_level in ["dense", "extreme"],
            'pixel_to_meter_ratio': getattr(self, 'pixel_to_meter_ratio', 0),
            'ground_plane_stability': getattr(self, 'ground_plane_stability', 0),
            'queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'detection_scales_available': self.detection_scales,
            'density_history': list(self.density_history),
            'detection_failures': self.detection_failures
        }

# For backward compatibility
RealtimeCrowdAnalyzer = EnhancedDenseCrowdAnalyzer

# Enhanced main function
def main():
    analyzer = EnhancedDenseCrowdAnalyzer(gpu_acceleration=True)
    
    cap = cv2.VideoCapture("videos/crowd3.mp4")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    analyzer.start_async_processing()
    
    print("🚀 ENHANCED DENSE CROWD DETECTION ANALYZER")
    print("=" * 60)
    print("ADVANCED FEATURES:")
    print("✓ Multi-scale detection for dense crowds")
    print("✓ Density-adaptive processing parameters")
    print("✓ Enhanced vanishing point detection")
    print("✓ RANSAC-based robust ground plane estimation")
    print("✓ Perspective-aware dynamic grid sizing")
    print("✓ Dense crowd post-processing filters")
    print("✓ Real-time density level adaptation")
    print("✓ Enhanced visualization and statistics")
    print("✓ Comprehensive error handling and recovery")
    print("=" * 60)
    print("CONTROLS:")
    print("- 'q': Quit")
    print("- 's': Save frame")
    print("- 'p': Detailed performance stats")
    print("- 'r': Reset calibration")
    print("- 'd': Toggle density info")
    print("=" * 60)
    
    show_density_info = False
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = analyzer.process_frame_realtime(frame)
            
            # Show density information overlay
            if show_density_info:
                density_text = f"Current: {analyzer.current_density_level.upper()}"
                cv2.putText(processed_frame, density_text, (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Enhanced Dense Crowd Analysis", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"enhanced_dense_crowd_analysis_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                logger.info(f"Enhanced frame saved as {filename}")
            elif key == ord('p'):
                stats = analyzer.get_enhanced_performance_stats()
                print("\n" + "=" * 70)
                print("ENHANCED PERFORMANCE STATISTICS")
                print("=" * 70)
                for key, value in stats.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                print("=" * 70)
            elif key == ord('r'):
                # Reset enhanced calibration
                analyzer.ground_plane_detected = False
                analyzer.advanced_calibration_complete = False
                analyzer.calibrated = False
                analyzer.detection_history.clear()
                analyzer.bbox_history.clear()
                analyzer.density_history.clear()
                analyzer.vanishing_point = None
                analyzer.ground_homography = None
                analyzer.adaptive_grid_cells = None
                analyzer.ground_roi_detailed = None
                analyzer.ground_mask = None
                analyzer.calibration_confidence = 0.0
                analyzer.current_density_level = "sparse"
                analyzer.detection_failures = 0
                logger.info("🔄 Enhanced calibration reset - recalibrating with dense crowd features...")
            elif key == ord('d'):
                show_density_info = not show_density_info
                status = "ON" if show_density_info else "OFF"
                logger.info(f"Density info overlay: {status}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        analyzer.stop_async_processing()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application terminated")

if __name__ == "__main__":
    main()
