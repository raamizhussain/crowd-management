import cv2
import numpy as np
from ultralytics import YOLO
import colorsys

class DynamicGroundCrowdAnalyzer:
    def __init__(self):
        self.det_model = YOLO("yolov8n.pt")
        self.grid_size = 1.0  # Reduced to 1 meter per grid cell for finer resolution
        self.calibrated = False
        self.H = None  # Homography matrix for ground plane transformation
        self.frame_size = (640, 480)
        self.density_colors = self._generate_density_colormap()
        self.crowd_threshold = 8
        
        # Initialize grid dimensions with default values for better coverage
        self.grid_rows = 8
        self.grid_cols = 12
        
        # Ground detection parameters
        self.ground_plane_detected = False
        self.ground_mask = None
        self.ground_contours = None
        self.adaptive_grid = None
        self.ground_roi = None
        
        # Detection history for ground plane estimation
        self.detection_history = []
        self.max_history = 30
        
        # Store detection bounding boxes for scale estimation
        self.bbox_history = []
        self.max_bbox_history = 10 # Keep a shorter history for bbox
        
    def _generate_density_colormap(self):
        """Generate color map for density visualization"""
        colors = []
        max_density = 15
        for i in range(max_density + 1):
            if i == 0:
                colors.append((50, 50, 50))  # Dark gray for empty cells
            else:
                # Blue (low) -> Cyan -> Green -> Yellow -> Orange -> Red (high)
                hue = max(0, 240 - (i / max_density) * 270) / 360
                saturation = min(1.0, 0.6 + (i / max_density) * 0.4)
                value = min(1.0, 0.5 + (i / max_density) * 0.5)
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def detect_ground_plane_from_people(self, frame, detections):
        """Detect ground plane using people's feet positions - fully automatic, covers full scene"""
        if detections is None or len(detections) == 0:
            return False
            
        h, w = frame.shape[:2]
        
        # Extract feet positions (bottom center of detections) and store bboxes
        feet_points = []
        current_frame_bboxes = []
        for detection in detections:
            x1, y1, x2, y2 = detection
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)  # Bottom of detection
            if 0 <= foot_x < w and 0 <= foot_y < h:
                feet_points.append([foot_x, foot_y])
                current_frame_bboxes.append(detection)
        
        if len(feet_points) < 2:  # Reduced threshold for faster detection
            return False
            
        # Add to detection history
        self.detection_history.append(feet_points)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
            
        # Add bboxes to history (for scale estimation)
        self.bbox_history.append(current_frame_bboxes)
        if len(self.bbox_history) > self.max_bbox_history:
            self.bbox_history.pop(0)
            
        # Collect all historical feet points
        all_feet_points = []
        for hist_points in self.detection_history:
            all_feet_points.extend(hist_points)
            
        if len(all_feet_points) < 5:  # Reduced threshold
            return False
            
        # Use custom RANSAC-like algorithm to find the dominant ground region
        feet_array = np.array(all_feet_points)
        
        try:
            # Robust ground plane estimation covering full scene
            y_coords = feet_array[:, 1]
            
            # Find ground level automatically - use median for robustness
            ground_y_median = np.median(y_coords)
            ground_y_std = np.std(y_coords)
            
            # Define ground range more liberally to include more area
            ground_y_min = ground_y_median - ground_y_std
            ground_y_max = h - 1  # Always extend to bottom
            
            # Filter points within the ground level range
            ground_mask = (y_coords >= ground_y_min) & (y_coords <= ground_y_max)
            ground_points = feet_array[ground_mask]
            
            if len(ground_points) < 3:
                return False
            
            # Create ground region covering full scene width
            self._create_adaptive_ground_region(frame, ground_points)
            return True
            
        except Exception as e:
            print(f"Ground plane detection failed: {e}")
            return False

    def _create_adaptive_ground_region(self, frame, feet_points):
        """Create adaptive ground region based on detected feet positions - covers full scene"""
        h, w = frame.shape[:2]
        
        # Expand detection area to cover the full scene width
        # Use feet points to determine the vertical ground boundaries but expand horizontally
        min_y = int(np.min(feet_points[:, 1]) * 0.95)  # Start slightly above highest detection
        max_y = h - 1  # Full height to bottom
        
        # Always use full width for better coverage
        min_x = 0
        max_x = w - 1
        
        # Clamp vertical bounds
        min_y = max(int(h * 0.2), min_y)  # Don't go too high up (reserve 20% for sky/background)
        
        # Create ground ROI polygon covering full width
        # Perspective-aware trapezoid shape but using full scene width
        top_width_factor = 0.8  # Less narrow at top for better coverage
        center_x = w // 2
        top_width = int(w * top_width_factor)
        
        self.ground_roi = np.array([
            [center_x - top_width // 2, min_y],  # Top-left
            [center_x + top_width // 2, min_y],  # Top-right  
            [max_x, max_y],                      # Bottom-right (full width)
            [min_x, max_y]                       # Bottom-left (full width)
        ], dtype=np.int32)
        
        # Create ground mask
        self.ground_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.ground_mask, [self.ground_roi], 255)
        
        # Calibrate homography for this ground region
        self._calibrate_adaptive_homography(frame)

    def _calibrate_adaptive_homography(self, frame):
        """Calibrate homography matrix based on detected ground region - fully automatic and dynamically scaled"""
        if self.ground_roi is None:
            return False
            
        h, w = frame.shape[:2]
        
        # Use the detected ground ROI as source points
        src_pts = self.ground_roi.astype(np.float32)
        
        # --- Advanced Logic: Estimate Real-World Scale from People's Heights ---
        # This aims to replace fixed scale factors (0.03, 0.025) with a more dynamic estimate.
        
        # Collect all historical bounding boxes
        all_bboxes = []
        for hist_bboxes in self.bbox_history:
            all_bboxes.extend(hist_bboxes)
            
        if len(all_bboxes) < 5: # Need a reasonable number of bboxes for estimation
            # Fallback to previous estimation if not enough bboxes
            print("Not enough bboxes for dynamic scale, falling back to heuristic.")
            roi_height_px = np.max(self.ground_roi[:, 1]) - np.min(self.ground_roi[:, 1])
            roi_width_px = w
            
            estimated_depth_m = roi_height_px * 0.03   # Original heuristic factor
            estimated_width_m = roi_width_px * 0.025   # Original heuristic factor
            
        else:
            # Try to find a robust average person height in pixels on the ground
            person_heights_px = []
            
            # Focus on people relatively close to the camera (larger pixel height)
            # or those well within the detected ground ROI
            for bbox in all_bboxes:
                x1, y1, x2, y2 = bbox
                
                # Check if the person's feet are on the detected ground plane
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                if self.ground_mask is not None and 0 <= foot_x < w and 0 <= foot_y < h:
                    if self.ground_mask[foot_y, foot_x] > 0:
                        person_heights_px.append(y2 - y1)
            
            if len(person_heights_px) > 0:
                # Use a robust average (median) of person heights in pixels
                avg_person_height_px = np.median(person_heights_px)
                
                # Assume an average real-world person height (e.g., 1.7 meters)
                # This is the crucial real-world anchor for scale.
                average_human_height_m = 1.7 
                
                # Calculate pixels per meter based on average person
                # Be careful with division by zero
                pixels_per_meter = avg_person_height_px / average_human_height_m if average_human_height_m > 0 else 0
                
                if pixels_per_meter > 1.0: # Ensure a reasonable scale factor (at least 1 pixel per meter)
                    # Now use this pixels_per_meter to convert ROI pixel dimensions to meters
                    roi_height_px = np.max(self.ground_roi[:, 1]) - np.min(self.ground_roi[:, 1])
                    roi_width_px = w # Full width of the frame is often mapped directly

                    estimated_depth_m = roi_height_px / pixels_per_meter
                    estimated_width_m = roi_width_px / pixels_per_meter
                    
                    print(f"Dynamic Scale Est: Avg Person Height PX={avg_person_height_px:.2f}, Pixels/Meter={pixels_per_meter:.2f}")
                else:
                    # Fallback if dynamic estimation yields unreasonable results
                    print("Dynamic scale too small, falling back to heuristic.")
                    roi_height_px = np.max(self.ground_roi[:, 1]) - np.min(self.ground_roi[:, 1])
                    roi_width_px = w
                    estimated_depth_m = roi_height_px * 0.03
                    estimated_width_m = roi_width_px * 0.025
            else:
                # Fallback if no valid person heights found
                print("No valid person heights for dynamic scale, falling back to heuristic.")
                roi_height_px = np.max(self.ground_roi[:, 1]) - np.min(self.ground_roi[:, 1])
                roi_width_px = w
                estimated_depth_m = roi_height_px * 0.03
                estimated_width_m = roi_width_px * 0.025
        
        # Calculate grid dimensions for full coverage using the estimated real-world dimensions
        # Use max to ensure a minimum grid size, and min to cap for performance
        self.grid_cols = max(8, min(20, int(estimated_width_m / self.grid_size)))
        self.grid_rows = max(5, min(15, int(estimated_depth_m / self.grid_size)))
        
        # Ensure minimum dimensions if estimation leads to tiny grids
        if self.grid_cols < 4: self.grid_cols = 4
        if self.grid_rows < 3: self.grid_rows = 3
        
        # Recalculate ground_width and ground_height based on final grid dimensions
        ground_width = self.grid_cols * self.grid_size
        ground_height = self.grid_rows * self.grid_size
        
        # Destination points in real-world coordinates (0,0) to (ground_width, ground_height)
        dst_pts = np.array([
            [0, 0],                           # Top-left
            [ground_width, 0],                # Top-right
            [ground_width, ground_height],    # Bottom-right
            [0, ground_height]                # Bottom-left
        ], dtype=np.float32)
        
        try:
            self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.calibrated = True
            self.ground_plane_detected = True
            
            # Initialize adaptive grid
            self.adaptive_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
            
            print(f"Full-scene ground plane calibrated: {self.grid_cols}x{self.grid_rows} grid")
            print(f"Estimated ground area: {ground_width:.1f}m x {ground_height:.1f}m")
            return True
            
        except Exception as e:
            print(f"Adaptive calibration failed: {e}")
            return False

    def enhance_ground_detection_with_edges(self, frame):
        """Use edge detection to refine ground plane boundaries"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find horizontal lines (potential ground boundaries)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Find ground boundary candidates
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and self.ground_roi is not None:
            # Filter contours that are within the ground ROI
            ground_boundaries = []
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    # Check if contour intersects with ground ROI
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    intersection = cv2.bitwise_and(mask, self.ground_mask)
                    if np.sum(intersection) > 0:
                        ground_boundaries.append(contour)
            
            self.ground_contours = ground_boundaries

    def image_to_ground_plane(self, img_points):
        """Transform image coordinates to ground plane coordinates"""
        if self.H is None:
            return np.array(img_points, dtype=np.float32).reshape(-1, 2)
            
        try:
            img_points = np.array(img_points, dtype=np.float32).reshape(-1, 1, 2)
            ground_points = cv2.perspectiveTransform(img_points, self.H)
            return ground_points.reshape(-1, 2)
        except Exception as e:
            return np.array(img_points, dtype=np.float32).reshape(-1, 2)

    def ground_to_image_plane(self, ground_points):
        """Transform ground plane coordinates back to image coordinates"""
        if self.H is None:
            return np.array(ground_points, dtype=np.float32).reshape(-1, 2)
            
        try:
            ground_points = np.array(ground_points, dtype=np.float32).reshape(-1, 1, 2)
            img_points = cv2.perspectiveTransform(ground_points, np.linalg.inv(self.H))
            return img_points.reshape(-1, 2)
        except Exception as e:
            return np.array(ground_points, dtype=np.float32).reshape(-1, 2)

    def get_adaptive_grid_cell(self, ground_x, ground_y):
        """Convert ground plane coordinates to adaptive grid cell indices"""
        col = int(ground_x / self.grid_size)
        row = int(ground_y / self.grid_size)
        
        # Clamp to valid grid bounds
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        
        return row, col

    def detect_crowd_density_level(self, detections):
        """Determine crowd density level"""
        if detections is None or len(detections) == 0:
            return "sparse"
        
        num_people = len(detections)
        
        # Use ground area for density calculation if available
        if self.ground_plane_detected and self.grid_cols and self.grid_rows:
            ground_area = self.grid_cols * self.grid_rows * (self.grid_size ** 2)
            density = num_people / ground_area
            
            if density > 2.0:
                return "dense_heads"
            elif density > 1.0:
                return "dense_mixed"
            else:
                return "sparse"
        else:
            # Fallback to original method (less accurate without ground plane)
            frame_area = self.frame_size[0] * self.frame_size[1]
            if len(detections) > 0:
                # Calculate average area of detections. This is a crude proxy.
                avg_area = np.mean([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in detections])
                relative_size = avg_area / frame_area
                
                # These thresholds might need adjustment based on typical detection sizes in your videos
                if relative_size < 0.01 and num_people > self.crowd_threshold:
                    return "dense_heads"
                elif num_people > self.crowd_threshold * 1.5:
                    return "dense_mixed"
                else:
                    return "sparse"
        
        return "sparse"

    def filter_detections_by_ground(self, detections):
        """Filter detections to only include those on the detected ground plane"""
        if detections is None or self.ground_mask is None:
            return detections
        
        filtered_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            
            h, w = self.ground_mask.shape
            if 0 <= foot_x < w and 0 <= foot_y < h:
                if self.ground_mask[foot_y, foot_x] > 0:
                    filtered_detections.append(detection)
        
        return np.array(filtered_detections) if filtered_detections else None

    def get_ground_position_from_detection(self, detection):
        """Get ground position from detection box"""
        x1, y1, x2, y2 = detection
        ground_x = (x1 + x2) / 2  # Center horizontally
        ground_y = y2             # Bottom of detection box
        return [ground_x, ground_y]

    def estimate_occluded_people_on_ground(self, detections, density_level):
        """Estimate occluded people and map their positions to ground plane"""
        if density_level == "sparse" or detections is None:
            return []
        
        estimated_ground_positions = []
        
        for detection in detections:
            ground_pos_px = self.get_ground_position_from_detection(detection)
            ground_coords_m = self.image_to_ground_plane([ground_pos_px])[0] # Convert to meters
            
            # Adaptive occlusion factor based on detected ground area (if calibrated)
            if self.ground_plane_detected and self.grid_cols and self.grid_rows:
                # Base occlusion factor, higher for dense_heads
                base_occlusion = 1.5 if density_level == "dense_heads" else 1.2
                
                # Add randomness to spread out estimated positions
                num_additional = int(base_occlusion) - 1 # How many additional people to estimate for this detection
                
                # Add more people for higher density levels, up to a limit
                if density_level == "dense_heads":
                     num_additional += np.random.randint(0, 3) # Add 0-2 more for very dense
                elif density_level == "dense_mixed":
                     num_additional += np.random.randint(0, 1) # Add 0-1 more for mixed
                
                # Ensure at least 1 person accounted for from original detection
                num_to_estimate_total = max(1, num_additional + 1) # Total count for this original detection
                
                # Generate new estimated positions
                for i in range(num_to_estimate_total - 1): # -1 because the original detection already accounts for one
                    # Offset within a reasonable range (e.g., within 0.5m radius)
                    offset_x = np.random.normal(0, self.grid_size * 0.4) # std dev of 0.4m
                    offset_y = np.random.normal(self.grid_size * 0.2, self.grid_size * 0.3) # Offset slightly backward (positive y in ground plane)
                    
                    est_ground_x = ground_coords_m[0] + offset_x
                    est_ground_y = ground_coords_m[1] + offset_y
                    
                    # Clamp to estimated ground bounds
                    est_ground_x = max(0, min(self.grid_cols * self.grid_size, est_ground_x))
                    est_ground_y = max(0, min(self.grid_rows * self.grid_size, est_ground_y))
                    estimated_ground_positions.append([est_ground_x, est_ground_y])
            else:
                # Fallback for when ground plane is not detected
                # This estimation will be less accurate as it's not mapped to a real-world scale
                if density_level == "dense_heads":
                    num_additional = 2
                elif density_level == "dense_mixed":
                    num_additional = 1
                else:
                    num_additional = 0
                
                # For fallback, just add multiple points at the detected foot position
                # This is a crude approximation as no real-world spread can be done
                for _ in range(num_additional):
                    estimated_ground_positions.append(ground_coords_m) # Add multiple at same spot
        
        return estimated_ground_positions

    def update_adaptive_density_counts(self, detections, estimated_ground_positions):
        """Update adaptive grid counts based on ground plane positions"""
        if not self.grid_rows or not self.grid_cols or self.adaptive_grid is None:
            # Reinitialize if grid dimensions changed or grid is None
            if self.ground_plane_detected:
                self.adaptive_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
            else:
                return # Cannot update if ground plane not detected

        # Reset grid counts for the current frame
        self.adaptive_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        if detections is None:
            return
        
        # Count detected people
        for detection in detections:
            ground_pos_px = self.get_ground_position_from_detection(detection)
            ground_coords_m = self.image_to_ground_plane([ground_pos_px])[0]
            row, col = self.get_adaptive_grid_cell(ground_coords_m[0], ground_coords_m[1])
            self.adaptive_grid[row, col] += 1
        
        # Add estimated occluded people
        for ground_pos_m in estimated_ground_positions:
            row, col = self.get_adaptive_grid_cell(ground_pos_m[0], ground_pos_m[1])
            self.adaptive_grid[row, col] += 1

    def draw_adaptive_ground_grid(self, frame, density_level):
        """Draw adaptive ground plane grid cells projected onto the image"""
        if not self.ground_plane_detected or self.H is None or self.adaptive_grid is None:
            return frame
        
        overlay = frame.copy()
        
        # Draw each adaptive grid cell
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Define ground plane coordinates for this cell
                x1_ground = col * self.grid_size
                y1_ground = row * self.grid_size
                x2_ground = (col + 1) * self.grid_size
                y2_ground = (row + 1) * self.grid_size
                
                ground_corners = np.array([
                    [x1_ground, y1_ground],
                    [x2_ground, y1_ground], 
                    [x2_ground, y2_ground],
                    [x1_ground, y1_ground] # This was originally a typo, ensuring a closed polyline
                ], dtype=np.float32)

                # Corrected corner order for proper polygon filling, closing the shape
                ground_corners_for_poly = np.array([
                    [x1_ground, y1_ground],
                    [x2_ground, y1_ground],
                    [x2_ground, y2_ground],
                    [x1_ground, y2_ground]
                ], dtype=np.float32)
                
                # Transform to image coordinates
                img_corners = self.ground_to_image_plane(ground_corners_for_poly)
                img_corners = np.array(img_corners, dtype=np.float32).astype(int)
                
                # Check if corners are within image bounds
                h, w = frame.shape[:2]
                valid_corners = []
                for corner in img_corners:
                    if 0 <= corner[0] < w and 0 <= corner[1] < h:
                        valid_corners.append(corner)
                
                # Only draw if at least 3 points are valid for a polygon
                if len(valid_corners) >= 3:
                    count = self.adaptive_grid[row, col]
                    
                    # Get color based on density
                    color_idx = min(count, len(self.density_colors) - 1)
                    color = self.density_colors[color_idx]
                    
                    # Fill grid cell with density color
                    if count > 0: # Only fill if there are people
                        cv2.fillPoly(overlay, [np.array(valid_corners)], color)
                    
                    # Draw grid lines
                    cv2.polylines(frame, [np.array(valid_corners)], True, (255, 255, 255), 1)
                    
                    # Add count text if the cell is visible enough and has people
                    if count > 0 and len(valid_corners) == 4: # Ensure it's a quad
                        # Calculate center more robustly for potentially distorted quadrilaterals
                        center_x = int(np.mean([p[0] for p in valid_corners]))
                        center_y = int(np.mean([p[1] for p in valid_corners]))
                        
                        # Only draw text if the cell is large enough
                        if cv2.contourArea(np.array(valid_corners)) > 50: # Adjust threshold as needed
                             cv2.putText(frame, str(count), (center_x - 10, center_y + 5), # Adjust position
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay
        alpha = 0.4 if density_level in ["dense_heads", "dense_mixed"] else 0.3
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        return frame

    def draw_ground_detection_overlay(self, frame):
        """Draw ground detection visualization"""
        if self.ground_roi is not None:
            # Draw ground ROI
            cv2.polylines(frame, [self.ground_roi], True, (0, 255, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.ground_roi], (0, 255, 255))
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        if self.ground_contours is not None:
            # Draw detected ground boundaries
            cv2.drawContours(frame, self.ground_contours, -1, (255, 0, 255), 2)
        
        return frame

    def draw_detections_with_ground_projection(self, frame, detections, estimated_ground_positions, density_level):
        """Draw detections and their ground projections"""
        if detections is not None:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection)
                
                # Draw detection box
                color = (0, 255, 0) if density_level == "sparse" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ground projection
                ground_pos_px = self.get_ground_position_from_detection(detection)
                
                # Only project if homography is available
                if self.H is not None:
                    ground_coords_m = self.image_to_ground_plane([ground_pos_px])[0]
                    projected_img_pos = self.ground_to_image_plane([ground_coords_m])[0]
                    
                    # Ensure projected_img_pos is numpy array and convert to int
                    projected_img_pos = np.array(projected_img_pos, dtype=np.float32).astype(int)
                    
                    bottom_center = ((x1 + x2) // 2, y2)
                    
                    # Check if projected position is valid
                    h, w = frame.shape[:2]
                    if 0 <= projected_img_pos[0] < w and 0 <= projected_img_pos[1] < h:
                        cv2.line(frame, bottom_center, tuple(projected_img_pos), color, 2)
                        cv2.circle(frame, tuple(projected_img_pos), 4, color, -1)
        
        # Draw estimated people
        if density_level in ["dense_heads", "dense_mixed"]:
            for ground_pos_m in estimated_ground_positions:
                # Only project if homography is available
                if self.H is not None:
                    img_pos = self.ground_to_image_plane([ground_pos_m])[0]
                    img_pos = np.array(img_pos, dtype=np.float32).astype(int)
                    
                    h, w = frame.shape[:2]
                    if 0 <= img_pos[0] < w and 0 <= img_pos[1] < h:
                        cv2.circle(frame, tuple(img_pos), 3, (255, 100, 100), -1)
                        cv2.circle(frame, tuple(img_pos), 6, (255, 150, 150), 1)
        
        return frame

    def draw_statistics(self, frame, density_level, total_detected, total_estimated):
        """Draw comprehensive statistics"""
        total_people = total_detected + total_estimated
        
        max_density = 0
        occupied_cells = 0
        total_cells = 0
        total_area_sqm = 0.0
        people_per_sqm = 0.0

        if self.adaptive_grid is not None and self.ground_plane_detected:
            max_density = np.max(self.adaptive_grid) if self.adaptive_grid.size > 0 else 0
            occupied_cells = np.count_nonzero(self.adaptive_grid)
            total_cells = self.grid_rows * self.grid_cols
            total_area_sqm = total_cells * (self.grid_size ** 2)
            people_per_sqm = total_people / total_area_sqm if total_area_sqm > 0 else 0
            
        
        # Determine crowd level
        if density_level == "sparse":
            crowd_level = "Low Density"
            level_color = (0, 255, 0)
        elif density_level == "dense_mixed":
            crowd_level = "Medium Density" 
            level_color = (0, 255, 255)
        else:
            crowd_level = "High Density"
            level_color = (0, 100, 255)
        
        # Statistics text
        ground_status = "Detected" if self.ground_plane_detected else "Estimating..."
        stats_text = [
            f"Ground: {ground_status}",
            f"Crowd Level: {crowd_level}",
            f"Detected: {total_detected}",
            f"Estimated: {total_estimated}", 
            f"Total People: {total_people}",
            f"Density: {people_per_sqm:.2f} people/m²",
            f"Grid: {self.grid_cols}x{self.grid_rows}",
            f"Area: {total_area_sqm:.1f} m²"
        ]
        
        # Background rectangle
        rect_height = len(stats_text) * 22 + 20
        cv2.rectangle(frame, (10, 10), (380, rect_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (380, rect_height), level_color, 2)
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 22
            color = level_color if i == 1 else (255, 255, 255)
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

    def draw_density_legend(self, frame):
        """Draw color legend for density levels"""
        h, w = frame.shape[:2]
        legend_width = 25
        legend_height = 180
        start_x = w - 50
        start_y = 50
        
        # Draw legend background
        cv2.rectangle(frame, (start_x - 5, start_y - 15), 
                     (start_x + legend_width + 35, start_y + legend_height + 5), 
                     (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "People/Cell", (start_x - 10, start_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw color gradient
        segment_height = legend_height // len(self.density_colors)
        for i, color in enumerate(self.density_colors):
            y1 = start_y + i * segment_height
            y2 = y1 + segment_height
            cv2.rectangle(frame, (start_x, y1), (start_x + legend_width, y2), color, -1)
            
            # Add density labels
            if i % 3 == 0 or i == len(self.density_colors) - 1:
                cv2.putText(frame, str(i), (start_x + legend_width + 5, y1 + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame

    def process_frame(self, frame):
        """Main processing pipeline with dynamic ground detection"""
        # Resize frame for consistent processing
        frame = cv2.resize(frame, self.frame_size)
        
        # Detect people
        # Added imgsz to control input size for YOLO model, matching frame_size
        results = self.det_model.track(frame, classes=0, conf=0.1, verbose=False, imgsz=self.frame_size[0]) 
        
        # Extract detections
        detections = None
        if results[0].boxes is not None and results[0].boxes.xyxy is not None:
            detections = results[0].boxes.xyxy.cpu().numpy()
        
        # Detect ground plane from people positions
        if not self.ground_plane_detected:
            self.detect_ground_plane_from_people(frame, detections)
            
        # Enhance ground detection with edge information
        if self.ground_plane_detected:
            self.enhance_ground_detection_with_edges(frame)
        
        # Filter detections to only those on ground
        if self.ground_plane_detected:
            detections = self.filter_detections_by_ground(detections)
        
        # Determine crowd density level
        density_level = self.detect_crowd_density_level(detections)
        
        # Estimate occluded people
        estimated_ground_positions = []
        if detections is not None and len(detections) > 0 and self.ground_plane_detected:
            estimated_ground_positions = self.estimate_occluded_people_on_ground(detections, density_level)
        
        # Update density counts
        if self.ground_plane_detected:
            self.update_adaptive_density_counts(detections, estimated_ground_positions)
        
        # Draw ground detection overlay (optional - can be toggled)
        # frame = self.draw_ground_detection_overlay(frame)
        
        # Draw adaptive ground plane grid
        if self.ground_plane_detected:
            frame = self.draw_adaptive_ground_grid(frame, density_level)
        
        # Draw detections with ground projections
        frame = self.draw_detections_with_ground_projection(frame, detections, estimated_ground_positions, density_level)
        
        # Add statistics and legend
        total_detected = len(detections) if detections is not None else 0
        total_estimated = len(estimated_ground_positions)
        
        frame = self.draw_statistics(frame, density_level, total_detected, total_estimated)
        frame = self.draw_density_legend(frame)
        
        return frame

def main():
    analyzer = DynamicGroundCrowdAnalyzer()
    
    # Use webcam (0) or video file
    cap = cv2.VideoCapture("videos/crowd3.mp4")  # Change to 0 for webcam
    
    # Set video properties (these might be overridden by camera defaults or video resolution)
    # It's better to ensure frame_size in analyzer matches what you expect
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Dynamic Ground Detection Crowd Analyzer")
    print("======================================")
    print("Features:")
    print("- Automatic ground plane detection from people positions")
    print("- Adaptive grid generation based on detected ground area")
    print("- Perspective-aware ground region mapping")
    print("- RANSAC-based ground plane estimation")
    print("- Edge-enhanced ground boundary detection")
    print("- Dynamic real-world scale estimation based on average person height")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame") 
    print("- Press 'r' to reset ground detection")
    print("- Press 'g' to toggle ground overlay visualization")
    print("- Press 'h' for help")
    
    show_ground_overlay = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = analyzer.process_frame(frame)
        
        # Show ground detection overlay if toggled
        if show_ground_overlay and analyzer.ground_plane_detected:
            processed_frame = analyzer.draw_ground_detection_overlay(processed_frame)
        
        cv2.imshow("Dynamic Ground Detection Crowd Analysis", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("dynamic_ground_crowd_density.jpg", processed_frame)
            print("Frame saved!")
        elif key == ord('r'):
            analyzer.ground_plane_detected = False
            analyzer.calibrated = False
            analyzer.detection_history = []
            analyzer.bbox_history = [] # Reset bbox history
            analyzer.ground_mask = None
            analyzer.ground_roi = None
            analyzer.adaptive_grid = None
            # Reset grid dimensions to better defaults for full coverage
            analyzer.grid_rows = 8
            analyzer.grid_cols = 12
            print("Ground detection reset...")
        elif key == ord('g'):
            show_ground_overlay = not show_ground_overlay
            status = "ON" if show_ground_overlay else "OFF"
            print(f"Ground overlay visualization: {status}")
        elif key == ord('h'):
            print("\nHelp:")
            print("- The system automatically detects the ground plane from people positions")
            print("- It uses RANSAC algorithm to find the dominant ground surface")
            print("- Grid adapts to the detected ground area size and shape")
            print("- Edge detection helps refine ground boundaries")
            print("- Dynamic scaling attempts to use average person pixel height to estimate real-world scale.")
            print("- Green boxes: sparse crowd, Yellow boxes: dense crowd")
            print("- Grid cells show people count with color-coded density")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
