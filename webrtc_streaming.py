import asyncio
import websockets
import json
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import av
import cv2
from crowd_analyzer import RealtimeCrowdAnalyzer


class WebRTCCrowdStream:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.pc = None
        self.websocket = None
        
    async def start_webrtc_server(self, host="localhost", port=8080):
        """Start WebRTC signaling server"""
        async def handle_client(websocket, path):
            self.websocket = websocket
            async for message in websocket:
                data = json.loads(message)
                await self.handle_signaling(data)
        
        start_server = websockets.serve(handle_client, host, port)
        print(f"WebRTC server started on ws://{host}:{port}")
        await start_server
    
    async def handle_signaling(self, data):
        """Handle WebRTC signaling"""
        if data["type"] == "offer":
            await self.handle_offer(data)
        elif data["type"] == "ice-candidate":
            await self.handle_ice_candidate(data)
    
    async def handle_offer(self, data):
        """Handle WebRTC offer"""
        self.pc = RTCPeerConnection()
        
        # Add video track
        video_track = CrowdAnalysisVideoTrack(self.analyzer)
        self.pc.addTrack(video_track)
        
        # Set remote description
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        )
        
        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        # Send answer
        await self.websocket.send(json.dumps({
            "type": "answer",
            "sdp": self.pc.localDescription.sdp
        }))

class CrowdAnalysisVideoTrack(VideoStreamTrack):
    """Custom video track for crowd analysis"""
    
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.cap = cv2.VideoCapture(0)  # Or video file
        
    async def recv(self):
        """Receive and process video frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Process frame with crowd analysis
        processed_frame = self.analyzer.process_frame_realtime(frame)
        
        # Convert to av.VideoFrame
        av_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        av_frame.pts = self.next_timestamp()
        av_frame.time_base = self.time_base
        
        return av_frame
