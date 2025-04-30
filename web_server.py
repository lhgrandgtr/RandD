from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import threading
import queue
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Global queues for frame and thoughts
frame_queue = queue.Queue(maxsize=1)
thoughts_queue = queue.Queue(maxsize=10)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        try:
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            yield f"data: data:image/jpeg;base64,{frame_bytes}\n\n"
        except queue.Empty:
            continue

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='text/event-stream')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send any queued thoughts
    while not thoughts_queue.empty():
        thought = thoughts_queue.get()
        socketio.emit('thought', {'data': thought})

def update_frame(frame):
    """Update the current frame"""
    try:
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass

def update_thoughts(thought):
    """Update agent thoughts"""
    try:
        thoughts_queue.put_nowait(thought)
        socketio.emit('thought', {'data': thought})
    except queue.Full:
        pass

def start_server():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    start_server()