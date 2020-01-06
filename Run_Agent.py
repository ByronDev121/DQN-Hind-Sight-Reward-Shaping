import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from main import MEMORY_FRACTION
from AirSim_Gym import Gym


MODEL_PATH = 'models/custom-model___157.00max__142.30avg__130.00min__9960.model'
IM_HEIGHT = 90
IM_WIDTH = 256

COMPUTE_AUTONOMY = True

if __name__ == '__main__':

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)
    model.summary()

    # Create environment
    env = Gym()

    # For agent speed measurements - keeps last 20 frametimes
    fps_counter = deque(maxlen=20)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.ones((1, IM_HEIGHT, IM_WIDTH, 3)))

    ep_number = 0
    total_ep_time = 0
    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        done = False

        ep_number += 1
        ep_time = time.time()
        if COMPUTE_AUTONOMY:
            if ep_number > 5:
                break

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                total_ep_time += time.time() - ep_time
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

    if COMPUTE_AUTONOMY:
        autonomy_percentage = (1 - (5/total_ep_time)) * 100
        print(autonomy_percentage)
