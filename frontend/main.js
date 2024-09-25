class TetrisVisualizer {
    constructor(socket) {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.cellSize = 30;
        this.colors = [
            '#000000', '#00FFFF', '#FFFF00', '#800080',
            '#FFA500', '#0000FF', '#00FF00', '#FF0000'
        ];
        this.socket = socket;
        this.setupSocketListeners();
    }

    setupSocketListeners() {
        this.socket.on('game_update', (gameState) => {
            console.log('Received game update:', gameState);
            this.drawBoard(gameState.board, gameState.current_piece);
        });

        this.socket.on('training_update', (trainingInfo) => {
            console.log('Received training update:', trainingInfo);
            this.updateProgressInfo(trainingInfo);
        });

        this.socket.on('training_started', (data) => {
            console.log('Training started:', data.message);
        });

        this.socket.on('training_stopped', (data) => {
            console.log('Training stopped:', data.message);
        });
    }

    drawBoard(board, currentPiece) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for (let y = 0; y < 20; y++) {
            for (let x = 0; x < 10; x++) {
                this.drawCell(x, y, this.colors[board[y][x]]);
            }
        }
        if (currentPiece) {
            for (let y = 0; y < currentPiece.shape.length; y++) {
                for (let x = 0; x < currentPiece.shape[y].length; x++) {
                    if (currentPiece.shape[y][x]) {
                        this.drawCell(currentPiece.x + x, currentPiece.y + y, this.colors[currentPiece.shape[y][x]]);
                    }
                }
            }
        }
    }

    drawCell(x, y, color) {
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x * this.cellSize, y * this.cellSize, this.cellSize - 1, this.cellSize - 1);
    }

    startTraining() {
        console.log('Emitting start_training event');
        this.socket.emit('start_training');
    }

    stopTraining() {
        console.log('Emitting stop_training event');
        this.socket.emit('stop_training');
    }

    updateProgressInfo(data) {
        document.getElementById('episode-info').textContent = `Episode: ${data.episode}`;
        document.getElementById('score-info').textContent = `Score: ${data.score}`;
        document.getElementById('loss-info').textContent = `Loss: ${data.loss !== null ? data.loss.toFixed(4) : 'N/A'}`;
        document.getElementById('epoch-info').textContent = `Epoch: ${data.epoch}/${data.max_epochs}`;
        document.getElementById('learning-rate-info').textContent = `Learning Rate: ${data.learning_rate.toFixed(6)}`;
        document.getElementById('loss-rate-info').textContent = `Loss Rate: ${data.loss_rate.toFixed(6)}`;
        document.getElementById('epsilon-info').textContent = `Epsilon: ${data.epsilon.toFixed(4)}`;
    }

    updateHyperparameters() {
        const learningRate = document.getElementById('learning-rate').value;
        const batchSize = document.getElementById('batch-size').value;
        const maxEpochs = document.getElementById('max-epochs').value;

        console.log('Emitting set_hyperparameters event');
        this.socket.emit('set_hyperparameters', {
            learning_rate: learningRate,
            batch_size: batchSize,
            max_epochs: maxEpochs
        });
    }
}

window.addEventListener('socketReady', function() {
    console.log('socketReady event received');
    const visualizer = new TetrisVisualizer(window.tetrisSocket);

    document.getElementById('startTraining').addEventListener('click', () => {
        console.log('Start Training button clicked');
        visualizer.startTraining();
    });

    document.getElementById('stopTraining').addEventListener('click', () => {
        console.log('Stop Training button clicked');
        visualizer.stopTraining();
    });

    document.getElementById('updateHyperparameters').addEventListener('click', () => {
        console.log('Update Hyperparameters button clicked');
        visualizer.updateHyperparameters();
    });
});