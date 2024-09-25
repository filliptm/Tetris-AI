import numpy as np

class TetrisGame:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.pieces = [
            [[1, 1, 1, 1]],           # I piece
            [[2, 2], [2, 2]],         # O piece
            [[3, 3, 3], [0, 3, 0]],   # T piece
            [[4, 4, 4], [4, 0, 0]],   # L piece
            [[5, 5, 5], [0, 0, 5]],   # J piece
            [[0, 6, 6], [6, 6, 0]],   # S piece
            [[7, 7, 0], [0, 7, 7]]    # Z piece
        ]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.current_piece = self.new_piece()
        self.score = 0
        self.game_over = False
        self.previous_score = 0

    def new_piece(self):
        piece = np.array(self.pieces[np.random.randint(len(self.pieces))])
        x = np.random.randint(self.width - piece.shape[1] + 1)
        return {'shape': piece, 'x': x, 'y': 0}

    def move(self, direction):
        if direction == 'left':
            self.current_piece['x'] = max(0, self.current_piece['x'] - 1)
        elif direction == 'right':
            self.current_piece['x'] = min(self.width - self.current_piece['shape'].shape[1], self.current_piece['x'] + 1)
        elif direction == 'down':
            self.current_piece['y'] += 1
            if not self.is_valid_position():
                self.current_piece['y'] -= 1
                self.lock_piece()
        elif direction == 'rotate':
            self.rotate_piece()

    def rotate_piece(self):
        original_shape = self.current_piece['shape']
        self.current_piece['shape'] = np.rot90(self.current_piece['shape'], k=-1)
        if not self.is_valid_position():
            self.current_piece['shape'] = original_shape

    def is_valid_position(self):
        piece = self.current_piece['shape']
        x, y = self.current_piece['x'], self.current_piece['y']
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i][j]:
                    if (y + i >= self.height or x + j < 0 or x + j >= self.width or
                        (y + i >= 0 and self.board[y + i][x + j])):
                        return False
        return True

    def lock_piece(self):
        piece = self.current_piece['shape']
        x, y = self.current_piece['x'], self.current_piece['y']
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i][j]:
                    if y + i < 0:
                        self.game_over = True
                        return
                    self.board[y + i][x + j] = piece[i][j]
        self.clear_lines()
        self.current_piece = self.new_piece()

    def clear_lines(self):
        lines_cleared = 0
        for i in range(self.height):
            if np.all(self.board[i]):
                self.board = np.vstack((np.zeros((1, self.width)), self.board[:i], self.board[i+1:]))
                lines_cleared += 1
        self.score += lines_cleared ** 2 * 100

    def get_state(self):
        state = self.board.copy()
        piece = self.current_piece['shape']
        x, y = self.current_piece['x'], self.current_piece['y']
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i][j] and 0 <= y+i < self.height and 0 <= x+j < self.width:
                    state[y+i][x+j] = piece[i][j]
        return state.flatten()

    def step(self, action):
        self.move(action)
        self.move('down')
        reward = self.score - self.previous_score
        self.previous_score = self.score

        if self.is_game_over():
            self.game_over = True
            reward -= 100  # Penalty for losing

        return self.get_state(), reward, self.game_over

    def is_game_over(self):
        # Check if any part of the new piece would be above the top of the board
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell and self.current_piece['y'] + y < 0:
                    return True
        return False

    def get_game_state(self):
        return {
            'board': self.board.tolist(),
            'current_piece': {
                'shape': self.current_piece['shape'].tolist(),
                'x': int(self.current_piece['x']),
                'y': int(self.current_piece['y'])
            },
            'score': int(self.score),
            'game_over': bool(self.game_over)
        }