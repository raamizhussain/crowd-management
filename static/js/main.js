// Live UI glue — pulls frames + metadata and draws the mini-map every tick.
document.addEventListener('DOMContentLoaded', () => {
  const videoDropdown  = document.getElementById('videoDropdown');
  const selectBtn      = document.getElementById('selectVideoBtn');
  const playBtn        = document.getElementById('playBtn');
  const pauseBtn       = document.getElementById('pauseBtn');

  const processedFrame = document.getElementById('processedFrame');
  const detectionCard  = document.getElementById('detectionCard');
  const anomalyCard    = document.getElementById('anomalyCard');
  const videoInfo      = document.getElementById('videoInfo');

  const peopleCount    = document.getElementById('peopleCount');
  const procFps        = document.getElementById('procFps');
  const densityLevel   = document.getElementById('densityLevel');
  const densityValue   = document.getElementById('densityValue');

  const streamState    = document.getElementById('streamState');
  const lastFrame      = document.getElementById('lastFrame');
  const lastDetections = document.getElementById('lastDetections');
  const positions3D    = document.getElementById('positions3D');

  const fallenCount    = document.getElementById('fallenCount');
  const fireRisk       = document.getElementById('fireRisk');
  const fireRegions    = document.getElementById('fireRegions');
  const anomalyDetails = document.getElementById('anomalyDetails');

  const miniCanvas     = document.getElementById('miniMapCanvas');
  const toggleMini     = document.getElementById('toggleMini');
  const ctxMini        = miniCanvas.getContext('2d');

  let isViewing = false;
  let pollTimer = null;
  let miniVisible = true;

  selectBtn.addEventListener('click', startStream);
  playBtn.addEventListener('click', startViewing);
  pauseBtn.addEventListener('click', stopViewing);
  toggleMini.addEventListener('click', () => {
    miniVisible = !miniVisible;
    miniCanvas.style.display = miniVisible ? 'block' : 'none';
  });

  window.addEventListener('beforeunload', () => {
    if (isViewing) fetch('/stop_cctv_stream', { method: 'POST' }).catch(()=>{});
  });

  function startStream() {
    const name = videoDropdown.value;
    if (!name) {
      toast('Select a video first', 'error');
      return;
    }
    selectBtn.disabled = true; selectBtn.textContent = 'Loading…';
    fetch('/start_cctv_stream', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ video_name: name })
    }).then(r=>r.json())
      .then(data=>{
        if (!data.success) { toast(data.error || 'Failed to start', 'error'); return; }
        document.getElementById('videoName').textContent = name;
        document.getElementById('videoMode').textContent = data.mode || '-';
        document.getElementById('videoFPS').textContent = data.fps || '-';
        document.getElementById('videoFrameCount').textContent = data.frame_count || '-';
        videoInfo.style.display = 'block';
        detectionCard.style.display = 'block';
        anomalyCard.style.display = 'block';
        toast('Video loaded', 'success');
        setTimeout(startViewing, 250);
      })
      .catch(()=>toast('Network error', 'error'))
      .finally(()=>{ selectBtn.disabled=false; selectBtn.textContent='Load Video'; });
  }

  function startViewing() {
    if (isViewing) return;
    isViewing = true;
    playBtn.style.display = 'none';
    pauseBtn.style.display = 'inline-block';
    streamState.textContent = 'Live';
    streamState.classList.add('success');
    // ~20Hz UI refresh
    pollTimer = setInterval(fetchTick, 50);
  }

  function stopViewing() {
    if (!isViewing) return;
    isViewing = false;
    clearInterval(pollTimer);
    pollTimer = null;
    playBtn.style.display = 'inline-block';
    pauseBtn.style.display = 'none';
    streamState.textContent = 'Stopped';
    streamState.classList.remove('success');
    fetch('/stop_cctv_stream', { method:'POST' }).catch(()=>{});
  }

  function fetchTick() {
    fetch('/get_cctv_frame')
      .then(r=>r.json())
      .then(data=>{
        if (!data.success) return;

        // Frame
        processedFrame.src = data.frame_image;
        processedFrame.style.display = 'block';

        // Stats
        const sd = data.stream_data || {};
        peopleCount.textContent = sd.people_count ?? 0;
        procFps.textContent = `${sd.processing_fps ?? 0} FPS`;
        densityLevel.textContent = (sd.density && sd.density.level) || '-';
        densityValue.textContent = (sd.density && sd.density.per_cell) ?? 0;
        lastFrame.textContent = new Date((sd.timestamp || Date.now())*1000).toLocaleTimeString();
        lastDetections.textContent = sd.people_count ?? 0;

        // Anomalies
        const an = data.anomalies || {};
        const fallen = an.fallen_people || [];
        const fire = an.fire || {score:0, regions:[]};
        fallenCount.textContent = fallen.length;
        fireRisk.textContent = Number(fire.score || 0).toFixed(2);
        fireRegions.textContent = (fire.regions || []).length;

        anomalyDetails.innerHTML = '';
        fallen.forEach(f=>{
          const div = document.createElement('div');
          div.className = 'anomaly-item fallen';
          div.innerHTML = `<div class="anomaly-title">Fallen person</div>
                           <div class="anomaly-desc">Track ID: ${f.id}</div>`;
          anomalyDetails.appendChild(div);
        });
        (fire.regions || []).forEach((b, i)=>{
          const div = document.createElement('div');
          div.className = 'anomaly-item fire';
          div.innerHTML = `<div class="anomaly-title">Fire region #${i+1}</div>
                           <div class="anomaly-desc">Box: [${b.join(', ')}]</div>`;
          anomalyDetails.appendChild(div);
        });

        // Live mini-map from tracks
        drawMini(sd.tracks || [], fallen, fire.regions || []);
        positions3D.textContent = `${(sd.tracks||[]).length} tracked`;
      })
      .catch(()=>{ /* ignore transient */ });
  }

  function drawMini(tracks, fallenList = [], fireBoxes = []) {
    if (!miniVisible) return;
    const w = miniCanvas.width, h = miniCanvas.height;
    ctxMini.clearRect(0, 0, w, h);

    // Background
    ctxMini.fillStyle = '#0b1220';
    ctxMini.fillRect(0, 0, w, h);

    // Grid
    ctxMini.globalAlpha = 0.3;
    ctxMini.strokeStyle = '#94a3b8';
    for (let i = 1; i < 6; i++) {
        const x = (i / 6) * w;
        const y = (i / 6) * h;
        ctxMini.beginPath(); ctxMini.moveTo(x, 0); ctxMini.lineTo(x, h); ctxMini.stroke();
        ctxMini.beginPath(); ctxMini.moveTo(0, y); ctxMini.lineTo(w, y); ctxMini.stroke();
    }
    ctxMini.globalAlpha = 1;

    // 🔥 Fire regions
    fireBoxes.forEach(box => {
        const [x1, y1, x2, y2] = box;
        const fx = ((x1 + x2) / 2) / 960;
        const fz = 1.0 - ((y1 + y2) / 2) / 540;
        const px = fx * w, py = fz * h;
        ctxMini.beginPath();
        ctxMini.arc(px, py, 10, 0, Math.PI * 2);
        ctxMini.fillStyle = 'rgba(255, 85, 0, 0.6)';
        ctxMini.fill();
        ctxMini.strokeStyle = '#ff4500';
        ctxMini.stroke();
    });

    // ❌ Fallen markers
    fallenList.forEach(f => {
        const fx = (f.bbox[0] + f.bbox[2]) / 2 / 960;
        const fz = 1.0 - (f.bbox[1] + f.bbox[3]) / 2 / 540;
        const px = fx * w, py = fz * h;
        ctxMini.strokeStyle = '#ef4444';
        ctxMini.lineWidth = 2;
        ctxMini.beginPath();
        ctxMini.moveTo(px - 6, py - 6);
        ctxMini.lineTo(px + 6, py + 6);
        ctxMini.moveTo(px + 6, py - 6);
        ctxMini.lineTo(px - 6, py + 6);
        ctxMini.stroke();
    });

    // 🧍 Tracks
    tracks.forEach(t => {
        const px = t.pos.x * w;
        const py = t.pos.z * h;
        const ax = px + t.dir.dx * 20;
        const ay = py + t.dir.dz * 20;

        ctxMini.beginPath();
        ctxMini.arc(px, py, 5, 0, Math.PI * 2);
        ctxMini.fillStyle = '#38bdf8';
        ctxMini.fill();
        ctxMini.strokeStyle = '#0ea5e9';
        ctxMini.stroke();

        ctxMini.beginPath();
        ctxMini.moveTo(px, py);
        ctxMini.lineTo(ax, ay);
        ctxMini.strokeStyle = '#f59e0b';
        ctxMini.stroke();

        ctxMini.fillStyle = '#fff';
        ctxMini.font = '11px sans-serif';
        ctxMini.fillText(`#${t.id}`, px + 8, py - 8);
    });
    }


  function toast(msg, type='info') {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = `toast ${type}`;
    el.style.display = 'block';
    setTimeout(()=>{ el.style.display = 'none'; }, 1800);
  }
});
