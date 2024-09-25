from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from tetris_game import TetrisGame
from tetris_ai import TetrisAI
import numpy as np
import threading
import time

app = Flask(__name__, static_folder='../frontend')
socketio = SocketIO(app, cors_allowed_origins="*")

game = TetrisGame()
ai = TetrisAI(game, learning_rate=0.001, max_epochs=10000, batch_size=320, memory_size=10000)

training = False
training_thread = None

def train_ai():
    global training
    print("Training thread started")
    while training:
        try:
            done = ai.train_step()
            socketio.emit('game_update', game.get_game_state())
            socketio.emit('training_update', ai.get_training_info())
            if done:
                game.reset()
            socketio.sleep(0.1)  # Use socketio.sleep instead of time.sleep
        except Exception as e:
            print(f"Error in training: {e}")
            training = False
    print("Training thread stopped")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('set_hyperparameters')
def set_hyperparameters(data):
    learning_rate = float(data['learning_rate'])
    batch_size = int(data['batch_size'])
    max_epochs = int(data['max_epochs'])
    ai.set_hyperparameters(learning_rate, batch_size, max_epochs)
    emit('hyperparameters_updated', {'message': 'Hyperparameters updated successfully'})

@socketio.on('start_training')
def start_training():
    global training, training_thread
    print("Received start_training event")
    if not training:
        game.reset()
        ai.reset()
        training = True
        training_thread = socketio.start_background_task(train_ai)
        emit('training_started', {'message': 'Training started'})
    else:
        emit('training_started', {'message': 'Training already in progress'})

@socketio.on('stop_training')
def stop_training():
    global training
    print("Received stop_training event")
    training = False
    emit('training_stopped', {'message': 'Training stopped'})

if __name__ == '__main__':
    print("Starting Flask-SocketIO server")
    socketio.run(app, debug=True)