import os
from dotenv import load_dotenv

load_dotenv()

FRAME_PATH = './frames/frame.png'
BLUETOOTH_PORT = '/dev/rfcomm0'
SERVER_URL = 'http://localhost:5000'
BAUD_RATE = 115200
VIDEO_URL = "http://192.168.56.181:8080/video"  # Updated to include /video endpoint
FRAME_INTERVAL = 5
NAV_SYS_PRMPT = (
    'You are a navigation system for a toy car. You can control the robot using the following commands:'
    'forward, backward, left, right, stop. '
    'You can also control the speed'
    'You should navigate the car to the given destination from the image as fast as possible.'
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", '')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", 'AIzaSyBcWirSU7KEHvpfO7y4uJP0plkl5QV1Tdk')