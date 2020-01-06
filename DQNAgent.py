import time
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Lambda, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import Sequential
import tensorflow as tf
from ModifiedTensorBoard import ModifiedTensorBoard
from experience_replay import EpisodeBuffer

MODEL_NAME = "custom-model"
IM_WIDTH = 256
IM_HEIGHT = 90
REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE
UPDATE_TARGET_EVERY = 5
DISCOUNT = 0.99


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        # Load model - optional
        # self.model.load_weights('model_initial_training_data_01.h5')

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Standard DQN - No HRS
        #self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Adepted DQN - With HRS
        self.replay_memory = EpisodeBuffer()

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):

        base_model = Sequential()
        base_model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
        base_model.add(BatchNormalization())
        base_model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        base_model.add(BatchNormalization())
        base_model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        base_model.add(BatchNormalization())
        base_model.add(Dropout(0.5, noise_shape=None, seed=None))
        base_model.add(Flatten())
        base_model.add(Dense(100, activation='elu'))
        base_model.add(BatchNormalization())
        base_model.add(Dense(10, activation='elu'))
        base_model.add(BatchNormalization())
        base_model.add(Dense(3, activation='linear'))
        base_model.summary()
        model = Model(inputs=base_model.input, outputs=base_model.outputs)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        #transition = (current_state, action, reward, new_state, done)

        # Standard DQN - No HRS
        #self.replay_memory.append(transition)

        # Adepted DQN - With HRS
        self.replay_memory.memorize(transition)
        
       

    def sample_batch(self, batch_size):
        return self.replay_memory.sample_batch(batch_size)

    def train(self):
        # Standard DQN - No HRS
        # if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        #     return

        # Adepted DQN - With HRS
        if self.replay_memory.count < MIN_REPLAY_MEMORY_SIZE:
            return

        try:
            # Standard DQN - No HRS
            #minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

            # Adepted DQN - With HRS
            minibatch = self.sample_batch(MINIBATCH_SIZE)
            
            current_states = np.array([transition[0] for transition in minibatch])
            with self.graph.as_default():
                current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

            new_current_states = np.array([transition[3] for transition in minibatch])
            with self.graph.as_default():
                future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

            X = []
            y = []

            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(current_state)
                y.append(current_qs)

            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step

            with self.graph.as_default():
                self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                               callbacks=[self.tensorboard] if log_this_step else None)

            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
        except:
            print('Exception handler - Error training model')

    def get_qs(self, state):
        try:
            return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
        except Exception as e:
            print('Exception handler - Error model predictions')
            print(state)
            return 2

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)