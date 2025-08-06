// CCTV-Style Butter-Smooth Real-Time System
document.addEventListener('DOMContentLoaded', function() {
    console.log('📹 CCTV-Style Real-Time System Initialized');
    
    // DOM elements
    const videoDropdown = document.getElementById('videoDropdown');
    const selectVideoBtn = document.getElementById('selectVideoBtn');
    const videoInfo = document.getElementById('videoInfo');
    const systemStatus = document.getElementById('systemStatus');
    const detectionCard = document.getElementById('detectionCard');
    const processFrameBtn = document.getElementById('processFrameBtn');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const detectionResults = document.getElementById('detectionResults');
    const processedFrame = document.getElementById('processedFrame');
    const peopleCount = document.getElementById('peopleCount');
    
    // CCTV State
    let isStreaming = false;
    let streamInterval = null;
    let fpsCounter = 0;
    let lastFpsTime = Date.now();
    let currentDisplayFps = 0;
    
    // Performance tracking
    let frameUpdateTimes = [];
    
    // Event listeners
    selectVideoBtn.addEventListener('click', startCCTVStream);
    playBtn.addEventListener('click', startCCTVViewing);
    pauseBtn.addEventListener('click', stopCCTVViewing);
    
    // Hide slider and frame controls (not needed for CCTV streaming)
    const frameSlider = document.getElementById('frameSlider');
    const frameControls = frameSlider?.parentElement;
    if (frameControls) {
        frameControls.style.display = 'none';
    }
    
    videoDropdown.addEventListener('change', function() {
        if (this.value) {
            selectVideoBtn.disabled = false;
            selectVideoBtn.textContent = '📹 Start CCTV Stream';
        }
        stopCCTVViewing();
        detectionCard.style.display = 'none';
    });
    
    // Initialize
    updateSystemStatus('📹 Ready for CCTV-style streaming!', 'pending');
    
    async function startCCTVStream() {
        const selectedVideo = videoDropdown.value;
        if (!selectedVideo) {
            showToast('Please select a video for CCTV simulation!', 'error');
            return;
        }
        
        selectVideoBtn.disabled = true;
        selectVideoBtn.textContent = '📹 Starting Stream...';
        updateSystemStatus('🚀 Initializing CCTV stream...', 'processing');
        
        try {
            const response = await fetch('/start_cctv_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_name: selectedVideo })
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayStreamInfo(result.stream_info);
                setupCCTVInterface();
                
                updateSystemStatus(`📹 CCTV Stream Active: ${selectedVideo}`, 'active');
                showToast(result.message, 'success');
                
                // Auto-start viewing
                setTimeout(startCCTVViewing, 500);
            } else {
                // Show detailed error message
                const errorMsg = result.error || result.message || 'Unknown error';
                showToast(`Error: ${errorMsg}`, 'error');
                updateSystemStatus(`❌ Failed to start CCTV stream: ${errorMsg}`, 'pending');
                console.error('CCTV Stream Error:', result);
            }
        } catch (error) {
            console.error('Stream start error:', error);
            showToast(`Network error: ${error.message}`, 'error');
            updateSystemStatus('❌ Network error', 'pending');
        } finally {
            selectVideoBtn.disabled = false;
            selectVideoBtn.textContent = '📹 Start CCTV Stream';
        }
    }

    
    function displayStreamInfo(info) {
        document.getElementById('videoName').textContent = info.name;
        document.getElementById('videoDuration').textContent = info.mode;
        document.getElementById('videoResolution').textContent = 'CCTV Quality';
        document.getElementById('videoFPS').textContent = `${info.fps} FPS`;
        videoInfo.style.display = 'block';
    }
    
    function setupCCTVInterface() {
        detectionCard.style.display = 'block';
        detectionResults.style.display = 'none';
        
        // Update button labels for CCTV
        processFrameBtn.style.display = 'none';  // No single frame processing in CCTV
        playBtn.textContent = '📹 Start Live View';
        pauseBtn.textContent = '⏸️ Stop Live View';
        
        // Hide frame controls (not needed for streaming)
        if (frameControls) {
            frameControls.style.display = 'none';
        }
    }
    
    function startCCTVViewing() {
        if (isStreaming) return;
        
        isStreaming = true;
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'inline-block';
        
        updateSystemStatus('📹 LIVE CCTV Feed - Butter-smooth detection!', 'processing');
        
        detectionResults.style.display = 'block';
        
        // Start butter-smooth frame fetching
        streamInterval = setInterval(async () => {
            await fetchLatestFrame();
        }, 33);  // ~30 FPS display rate
        
        // Start FPS counter
        fpsCounter = 0;
        lastFpsTime = Date.now();
    }
    
    function stopCCTVViewing() {
        isStreaming = false;
        playBtn.style.display = 'inline-block';
        pauseBtn.style.display = 'none';
        
        if (streamInterval) {
            clearInterval(streamInterval);
            streamInterval = null;
        }
        
        // Enhanced: Stop the backend stream when stopping viewing
        fetch('/stop_cctv_stream', { method: 'POST' })
            .then(response => response.json())
            .then(result => {
                console.log('Stream stopped:', result.message);
            })
            .catch(error => {
                console.log('Stop stream error (non-fatal):', error);
            });
        
        updateSystemStatus('📹 CCTV feed stopped', 'active');
    }

    
    async function fetchLatestFrame() {
        if (!isStreaming) return;
        
        try {
            const startTime = performance.now();
            
            const response = await fetch('/get_cctv_frame');
            const result = await response.json();
            
            if (result.success) {
                // Update display smoothly
                processedFrame.src = result.frame_image;
                processedFrame.style.display = 'block';
                peopleCount.textContent = result.stream_data.people_count;
                
                // Update density information
                const densityData = result.stream_data.density_data;
                if (densityData) {
                    // Update people count with density level
                    peopleCount.textContent = `${result.stream_data.people_count} (${densityData.density_level})`;
                    
                    // Change people count color based on density
                    const densityColors = {
                        'EMPTY': '#808080',
                        'LOW': '#00FF00',
                        'MEDIUM': '#FFA500',
                        'HIGH': '#FF6400',
                        'CRITICAL': '#FF0000'
                    };
                    peopleCount.style.color = densityColors[densityData.density_level] || '#48bb78';
                }
                
                // Track display FPS
                const endTime = performance.now();
                frameUpdateTimes.push(endTime - startTime);
                if (frameUpdateTimes.length > 10) {
                    frameUpdateTimes.shift();
                }
                
                // Update FPS counter
                fpsCounter++;
                const now = Date.now();
                if (now - lastFpsTime >= 1000) {
                    currentDisplayFps = fpsCounter;
                    fpsCounter = 0;
                    lastFpsTime = now;
                }
                
                // Enhanced status with density info
                const avgUpdateTime = frameUpdateTimes.reduce((a, b) => a + b, 0) / frameUpdateTimes.length;
                const uiFps = Math.round(1000 / avgUpdateTime);
                const processingFps = result.stream_data.processing_fps;
                const densityLevel = densityData ? densityData.density_level : 'UNKNOWN';
                
                updateSystemStatus(
                    `📹 LIVE | Frame: ${result.stream_data.frame_count} | People: ${result.stream_data.people_count} | Density: ${densityLevel} | Processing: ${processingFps} FPS | Display: ${currentDisplayFps} FPS`,
                    'active'
                );
                
            } else if (result.message !== 'No frame available') {
                console.warn('Frame fetch issue:', result.message);
            }
            
        } catch (error) {
            console.error('Frame fetch error:', error);
            // Don't stop streaming for network hiccups
        }
    }

    
    function updateSystemStatus(message, status) {
        systemStatus.innerHTML = `
            <div class="status-item">
                <span class="status-dot ${status}"></span>
                <span>${message}</span>
            </div>
        `;
    }
    
    function showToast(message, type) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        setTimeout(() => toast.classList.remove('show'), 2000);
    }
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', async () => {
        if (isStreaming) {
            stopCCTVViewing();
            try {
                await fetch('/stop_cctv_stream', { method: 'POST' });
            } catch (e) {
                console.log('Cleanup error:', e);
            }
        }
    });
});
