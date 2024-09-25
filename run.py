import os
import sys

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_dir)

from app import app, socketio

if __name__ == '__main__':
    print("Starting Tetris AI Application")
    print("Access the application at http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)