import numpy as np


class TetrisGame:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.pieces = [
            [[1, 1, 1, 1]],  # I piece
            [[2, 2], [2, 2]],  # O piece
            [[3, 3, 3], [0, 3, 0]],  # T piece
            [[4, 4, 4], [4, 0, 0]],  # L piece
            [[5, 5, 5], [0, 0, 5]],  # J piece
            [[0, 6, 6], [6, 6, 0]],  # S piece
            [[7, 7, 0], [0, 7, 7]]  # Z piece
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
        y = 0  # Start at the top of the board
        return {'shape': piece, 'x': x, 'y': y}

    def is_game_over(self):
        return not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y'])

    def move(self, direction):
        original_x, original_y = self.current_piece['x'], self.current_piece['y']
        if direction == 'left':
            self.current_piece['x'] -= 1
        elif direction == 'right':
            self.current_piece['x'] += 1
        elif direction == 'down':
            self.current_piece['y'] += 1
        elif direction == 'rotate':
            rotated_shape = np.rot90(self.current_piece['shape'], k=-1)
            if self.is_valid_position(rotated_shape, self.current_piece['x'], self.current_piece['y']):
                self.current_piece['shape'] = rotated_shape
            return  # Don't need to check for locking after rotation

        if not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.current_piece['x'], self.current_piece['y'] = original_x, original_y
            if direction == 'down':
                self.lock_piece()
                self.current_piece = self.new_piece()
                if self.is_game_over():
                    self.game_over = True

    def is_valid_position(self, piece, x, y):
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
                    self.board[y + i][x + j] = piece[i][j]
        self.clear_lines()

    def clear_lines(self):
        lines_cleared = 0
        for y in range(self.height - 1, -1, -1):
            if np.all(self.board[y]):
                self.board = np.vstack((np.zeros((1, self.width)), self.board[:y], self.board[y + 1:]))
                lines_cleared += 1
        self.score += lines_cleared ** 2 * 100

    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True

        self.move(action)
        self.move('down')  # Always move down after each action
        reward = self.score - self.previous_score
        self.previous_score = self.score

        if self.game_over:
            reward -= 100  # Penalty for losing

        return self.get_state(), reward, self.game_over

    def get_state(self):
        state = self.board.copy()
        piece = self.current_piece['shape']
        x, y = self.current_piece['x'], self.current_piece['y']
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i][j] and 0 <= y + i < self.height and 0 <= x + j < self.width:
                    state[y + i][x + j] = piece[i][j]
        return state.flatten()

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