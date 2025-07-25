import cv2
import time
import logging
from crowd_analyzer import EnhancedDenseCrowdAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDenseCrowdApp:
    def __init__(self):
        try:
            self.analyzer = EnhancedDenseCrowdAnalyzer(gpu_acceleration=True)
            self.video_sources = {
                "webcam": 0,
                "crowd_demo_1": "videos/crowd6.mp4",
                "crowd_demo_2": "videos/crowd7.mp4", 
                "crowd_demo_3": "videos/crowd8.mp4",
                "rtsp_stream": "rtsp://camera_ip:554/stream"
            }
            self.current_source = None
            self.cap = None
            self.is_running = False
            
            # Performance monitoring
            self.total_frames_processed = 0
            self.start_time = None
            
            logger.info("Enhanced Dense Crowd App initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
        
    def list_video_sources(self):
        """List available video sources"""
        print("Available video sources:")
        for i, (name, source) in enumerate(self.video_sources.items()):
            print(f"{i+1}. {name}: {source}")
    
    def select_video_source(self, selection):
        """Select video source with validation"""
        try:
            sources = list(self.video_sources.values())
            if 0 < selection <= len(sources):
                self.current_source = sources[selection-1]
                source_name = list(self.video_sources.keys())[selection-1]
                logger.info(f"Selected video source: {source_name}")
                return True
            else:
                logger.warning(f"Invalid selection: {selection}")
                return False
        except Exception as e:
            logger.error(f"Error selecting video source: {e}")
            return False
    
    def validate_video_source(self):
        """Validate the selected video source"""
        if self.current_source is None:
            return False
        
        try:
            # Test video capture
            test_cap = cv2.VideoCapture(self.current_source)
            if not test_cap.isOpened():
                logger.error(f"Cannot open video source: {self.current_source}")
                test_cap.release()
                return False
            
            # Test reading a frame
            ret, frame = test_cap.read()
            test_cap.release()
            
            if not ret:
                logger.error(f"Cannot read from video source: {self.current_source}")
                return False
            
            logger.info(f"Video source validation successful: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Video source validation failed: {e}")
            return False
    
    def start_processing(self):
        """Start enhanced dense crowd processing with comprehensive error handling"""
        if self.current_source is None:
            logger.error("No video source selected!")
            print("❌ No video source selected!")
            return False
        
        # Validate video source before starting
        if not self.validate_video_source():
            print("❌ Video source validation failed!")
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.current_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video capture: {self.current_source}")
                print("❌ Failed to open video source!")
                return False
            
            # Optimize capture settings
            if isinstance(self.current_source, int):  # Webcam
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start async processing
            self.analyzer.start_async_processing()
            self.is_running = True
            self.start_time = time.time()
            
            print("🚀 ENHANCED DENSE CROWD ANALYSIS STARTED!")
            print("=" * 65)
            print("BREAKTHROUGH FEATURES:")
            print("🎯 Multi-scale detection for extreme density crowds")
            print("🧠 AI-powered density-adaptive processing")
            print("📐 Advanced perspective-aware ground calibration")
            print("🔍 Enhanced vanishing point detection algorithm")
            print("🎲 RANSAC robust ground plane estimation")
            print("⚡ Real-time density level adaptation")
            print("🎨 Dynamic grid sizing based on crowd physics")
            print("📊 Comprehensive crowd analytics")
            print("=" * 65)
            print("ENHANCED CONTROLS:")
            print("- 'q': Quit analysis")
            print("- 's': Save enhanced frame")
            print("- 'p': Comprehensive performance statistics")
            print("- 'r': Reset advanced calibration")
            print("- 'd': Toggle density level overlay")
            print("- 'c': Show calibration confidence")
            print("- 'm': Toggle multi-scale detection info")
            print("- 'v': Show vanishing point detection status")
            print("=" * 65)
            
            # Start the main processing loop
            self._enhanced_processing_loop()
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            print(f"❌ Processing error: {e}")
            return False
        finally:
            self.cleanup()
    
    def _enhanced_processing_loop(self):
        """Enhanced processing loop with advanced monitoring"""
        calibration_shown = False
        density_overlay = False
        multi_scale_info = False
        vp_info = False
        frame_count = 0
        last_stats_time = time.time()
        
        try:
            while self.cap.isOpened() and self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame, ending processing")
                    break
                
                frame_count += 1
                self.total_frames_processed += 1
                
                # Process frame with enhanced features
                processed_frame = self.analyzer.process_frame_realtime(frame)
                
                # Show calibration completion notification
                if (self.analyzer.advanced_calibration_complete and 
                    not calibration_shown and 
                    self.analyzer.calibration_confidence > 0.6):
                    print(f"\n🎉 ENHANCED CALIBRATION COMPLETE! 🎉")
                    print(f"   Confidence: {self.analyzer.calibration_confidence:.1%}")
                    print(f"   Density Level: {self.analyzer.current_density_level.upper()}")
                    print(f"   Grid: {self.analyzer.grid_width_cells}x{self.analyzer.grid_depth_levels}")
                    print(f"   Cell Size: {self.analyzer.base_grid_size:.1f}m")
                    print(f"   Vanishing Point: {'✓ DETECTED' if self.analyzer.vanishing_point is not None else '✗ Not Found'}")
                    print(f"   Multi-Scale: {'ACTIVE' if self.analyzer.current_density_level in ['dense', 'extreme'] else 'Standby'}")
                    calibration_shown = True
                
                # Enhanced overlay information
                if density_overlay:
                    density_text = f"DENSITY: {self.analyzer.current_density_level.upper()}"
                    detection_count = len(self.analyzer.cached_detections) if self.analyzer.cached_detections is not None else 0
                    count_text = f"PEOPLE: {detection_count}"
                    
                    cv2.putText(processed_frame, density_text, (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(processed_frame, count_text, (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if multi_scale_info:
                    ms_status = "MULTI-SCALE: ACTIVE" if self.analyzer.current_density_level in ["dense", "extreme"] else "MULTI-SCALE: STANDBY"
                    cv2.putText(processed_frame, ms_status, (320, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                
                if vp_info:
                    vp_status = "VP: DETECTED" if self.analyzer.vanishing_point is not None else "VP: SEARCHING..."
                    cv2.putText(processed_frame, vp_status, (320, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Display enhanced frame
                cv2.imshow("Enhanced Dense Crowd Analysis System", processed_frame)
                
                # Periodic automatic stats (every 30 seconds)
                current_time = time.time()
                if current_time - last_stats_time > 30:
                    self._show_periodic_stats()
                    last_stats_time = current_time
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"enhanced_dense_crowd_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Enhanced frame saved: {filename}")
                elif key == ord('p'):
                    self._show_comprehensive_stats()
                elif key == ord('r'):
                    self._reset_enhanced_analyzer()
                    calibration_shown = False
                elif key == ord('d'):
                    density_overlay = not density_overlay
                    status = "ON" if density_overlay else "OFF"
                    print(f"🎯 Density overlay: {status}")
                elif key == ord('c'):
                    confidence = self.analyzer.calibration_confidence
                    print(f"🎯 Calibration Confidence: {confidence:.1%} {'(Excellent)' if confidence > 0.8 else '(Good)' if confidence > 0.6 else '(Fair)' if confidence > 0.4 else '(Poor)'}")
                elif key == ord('m'):
                    multi_scale_info = not multi_scale_info
                    status = "ON" if multi_scale_info else "OFF"
                    print(f"⚡ Multi-scale info: {status}")
                elif key == ord('v'):
                    vp_info = not vp_info
                    status = "ON" if vp_info else "OFF"
                    print(f"📐 Vanishing point info: {status}")
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            self.is_running = False
    
    def _show_comprehensive_stats(self):
        """Show comprehensive enhanced statistics"""
        try:
            stats = self.analyzer.get_enhanced_performance_stats()
            print("\n" + "🔥" * 25 + " ENHANCED PERFORMANCE ANALYSIS " + "🔥" * 25)
            print("┌" + "─" * 80 + "┐")
            
            # Core Performance
            print("│ CORE PERFORMANCE" + " " * 48 + "│")
            print("├" + "─" * 80 + "┤")
            print(f"│ Frame Rate: {stats['fps']:.1f} FPS" + " " * (68 - len(f"Frame Rate: {stats['fps']:.1f} FPS")) + "│")
            print(f"│ Processing Time: {stats['processing_time_ms']:.1f} ms" + " " * (62 - len(f"Processing Time: {stats['processing_time_ms']:.1f} ms")) + "│")
            
            # Advanced Calibration
            print("├" + "─" * 80 + "┤")
            print("│ ADVANCED CALIBRATION" + " " * 44 + "│")
            print("├" + "─" * 80 + "┤")
            print(f"│ Ground Detection: {'✓ ACTIVE' if stats['ground_detected'] else '✗ INACTIVE'}" + " " * (58 - len(f"Ground Detection: {'✓ ACTIVE' if stats['ground_detected'] else '✗ INACTIVE'}")) + "│")
            print(f"│ Advanced Calibration: {'✓ COMPLETE' if stats['advanced_calibration'] else '✗ IN PROGRESS'}" + " " * (52 - len(f"Advanced Calibration: {'✓ COMPLETE' if stats['advanced_calibration'] else '✗ IN PROGRESS'}")) + "│")
            print(f"│ Calibration Confidence: {stats['calibration_confidence']:.1%}" + " " * (56 - len(f"Calibration Confidence: {stats['calibration_confidence']:.1%}")) + "│")
            print(f"│ Vanishing Point: {'✓ DETECTED' if stats['vanishing_point_detected'] else '✗ NOT FOUND'}" + " " * (56 - len(f"Vanishing Point: {'✓ DETECTED' if stats['vanishing_point_detected'] else '✗ NOT FOUND'}")) + "│")
            
            # Dense Crowd Features
            print("├" + "─" * 80 + "┤")
            print("│ DENSE CROWD FEATURES" + " " * 44 + "│")
            print("├" + "─" * 80 + "┤")
            print(f"│ Current Density: {stats['current_density_level'].upper()}" + " " * (64 - len(f"Current Density: {stats['current_density_level'].upper()}")) + "│")
            print(f"│ Multi-Scale Detection: {'✓ ACTIVE' if stats['multi_scale_enabled'] else '✗ STANDBY'}" + " " * (52 - len(f"Multi-Scale Detection: {'✓ ACTIVE' if stats['multi_scale_enabled'] else '✗ STANDBY'}")) + "│")
            print(f"│ Available Scales: {', '.join(map(str, stats['detection_scales_available']))}" + " " * (62 - len(f"Available Scales: {', '.join(map(str, stats['detection_scales_available']))}")) + "│")
            
            # Grid Analysis
            print("├" + "─" * 80 + "┤")
            print("│ GRID ANALYSIS" + " " * 51 + "│")
            print("├" + "─" * 80 + "┤")
            print(f"│ Grid Dimensions: {stats['grid_dimensions']}" + " " * (64 - len(f"Grid Dimensions: {stats['grid_dimensions']}")) + "│")
            print(f"│ Cell Size: {stats['cell_size']}" + " " * (70 - len(f"Cell Size: {stats['cell_size']}")) + "│")
            print(f"│ Pixel to Meter Ratio: {stats['pixel_to_meter_ratio']:.4f}" + " " * (56 - len(f"Pixel to Meter Ratio: {stats['pixel_to_meter_ratio']:.4f}")) + "│")
            print(f"│ Ground Plane Stability: {stats['ground_plane_stability']:.3f}" + " " * (55 - len(f"Ground Plane Stability: {stats['ground_plane_stability']:.3f}")) + "│")
            
            # System Status
            print("├" + "─" * 80 + "┤")
            print("│ SYSTEM STATUS" + " " * 51 + "│")
            print("├" + "─" * 80 + "┤")
            print(f"│ Frame Queue: {stats['queue_size']}" + " " * (68 - len(f"Frame Queue: {stats['queue_size']}")) + "│")
            print(f"│ Result Queue: {stats['result_queue_size']}" + " " * (66 - len(f"Result Queue: {stats['result_queue_size']}")) + "│")
            
            print("└" + "─" * 80 + "┘")
            print("🔥" * 88)
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
            print("❌ Error displaying statistics")
    
    def _show_periodic_stats(self):
        """Show periodic automatic statistics"""
        try:
            stats = self.analyzer.get_enhanced_performance_stats()
            current_time = time.strftime("%H:%M:%S")
            
            print(f"\n⏰ [{current_time}] AUTO-STATS:")
            print(f"   FPS: {stats['fps']:.1f} | Density: {stats['current_density_level'].upper()} | People: {len(self.analyzer.cached_detections) if self.analyzer.cached_detections is not None else 0}")
            print(f"   Calibration: {stats['calibration_confidence']:.1%} | Multi-Scale: {'ON' if stats['multi_scale_enabled'] else 'OFF'}")
        except Exception as e:
            logger.error(f"Error showing periodic stats: {e}")
    
    def _reset_enhanced_analyzer(self):
        """Reset enhanced analyzer state"""
        try:
            self.analyzer.ground_plane_detected = False
            self.analyzer.advanced_calibration_complete = False
            self.analyzer.calibrated = False
            self.analyzer.detection_history.clear()
            self.analyzer.bbox_history.clear()
            self.analyzer.density_history.clear()
            self.analyzer.vanishing_point = None
            self.analyzer.ground_homography = None
            self.analyzer.adaptive_grid_cells = None
            self.analyzer.ground_roi_detailed = None
            self.analyzer.ground_mask = None
            self.analyzer.calibration_confidence = 0.0
            self.analyzer.current_density_level = "sparse"
            self.analyzer.ground_plane_stability = 0.0
            print("🔄 Enhanced system reset - Initializing advanced calibration with dense crowd features...")
        except Exception as e:
            logger.error(f"Error resetting analyzer: {e}")
    
    def cleanup(self):
        """Enhanced cleanup"""
        try:
            if hasattr(self, 'analyzer') and self.analyzer:
                self.analyzer.stop_async_processing()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Log final statistics
            if self.start_time and self.total_frames_processed > 0:
                total_time = time.time() - self.start_time
                avg_fps = self.total_frames_processed / total_time
                logger.info(f"Session complete: {self.total_frames_processed} frames, {avg_fps:.1f} avg FPS")
            
            logger.info("✅ Enhanced cleanup completed")
            print("✅ Enhanced cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Enhanced main application entry point"""
    try:
        app = EnhancedDenseCrowdApp()
        
        print("🌟" * 30)
        print("🚀 ENHANCED DENSE CROWD ANALYSIS SYSTEM 🚀")
        print("🌟" * 30)
        print()
        print("🔥 REVOLUTIONARY FEATURES:")
        print("🎯 Multi-Scale Detection Technology")
        print("🧠 Density-Adaptive AI Processing")
        print("📐 Advanced Perspective Calibration")
        print("🔍 Enhanced Vanishing Point Detection")
        print("🎲 RANSAC Robust Ground Estimation")
        print("⚡ Real-Time Density Adaptation")
        print("🎨 Dynamic Physics-Based Grid Sizing")
        print("📊 Comprehensive Crowd Analytics")
        print("🏃‍♂️ Optimized for Dense Crowd Scenarios")
        print()
        print("=" * 60)
        
        # List and select video source
        app.list_video_sources()
        
        while True:
            try:
                selection = int(input(f"\nSelect video source (1-{len(app.video_sources)}): "))
                if app.select_video_source(selection):
                    break
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Please enter a valid number!")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
        
        # Start processing
        app.start_processing()
        
    except Exception as e:
        logger.error(f"Critical application error: {e}")
        print(f"❌ Critical error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")

if __name__ == "__main__":
    main()
