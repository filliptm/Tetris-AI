import tensorflow as tf
import numpy as np
from collections import deque
import random


class TetrisAI:
    def __init__(self, game, learning_rate=0.001, max_epochs=10000, batch_size=32, memory_size=1000000):
        self.game = game
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.99  # Increased discount rate for long-term rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.current_episode = 0
        self.total_episodes = 0
        self.current_score = 0
        self.current_loss = None
        self.epoch = 0
        self.loss_rate = 0

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.game.width * self.game.height,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self):
        state = self.game.get_state()
        action = self.act(state)
        next_state, reward, done = self.game.step(['left', 'right', 'down', 'rotate'][action])
        self.remember(state, action, reward, next_state, done)
        self.current_score = self.game.score
        self.epoch += 1
        self.replay()

        if done:
            self.current_episode += 1
            self.game.reset()
            if self.current_episode % 10 == 0:  # Update target network periodically
                self.update_target_model()

        if self.current_episode > 0:
            self.loss_rate = self.current_loss / self.current_episode if self.current_loss else 0
        else:
            self.loss_rate = 0

        return done

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        self.current_loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_hyperparameters(self, learning_rate, batch_size, max_epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        self.target_model.compile(optimizer=optimizer, loss='mse')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def get_training_info(self):
        return {
            "episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "score": self.current_score,
            "loss": float(self.current_loss) if self.current_loss is not None else None,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "learning_rate": float(self.learning_rate),
            "loss_rate": float(self.loss_rate),
            "epsilon": float(self.epsilon)
        }

    def reset(self):
        self.current_episode = 0
        self.epoch = 0
        self.current_score = 0
        self.current_loss = None
        self.loss_rate = 0
        self.memory.clear()
