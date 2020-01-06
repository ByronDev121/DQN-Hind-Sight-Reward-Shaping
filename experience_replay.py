from collections import deque
import random

class EpisodeBuffer(object):
    def __init__(self, buffer_size=3000):
        self.episode_buffer = deque()
        self.shaped_experience_buffer = deque()
        self.terminal_reward = None
        self.buffer_size = buffer_size
        self.count = 0

    def reset_episode_buffer(self):
        self.episode_buffer = deque()
        self.terminal_reward = None

    def memorize(self, transition):
        # transition = (current_state, action, reward, new_state,done)
        self.episode_buffer.append(transition)
        if transition[4]:
            self.terminal_reward = transition[2]
            self.shape_terminal_reward()
        self.count += 1

    def sample_batch(self, batch_size):
        batch = random.sample(self.shaped_experience_buffer, batch_size)
        return batch

    def shape_terminal_reward(self):
        s = 0
        for i, transition in reversed(list(enumerate(self.episode_buffer))):
            if -10 < s < 0:
                reward = transition[2] + ((2.71828 * self.terminal_reward) / 2.71828 ** (1 - (0.2 * s)))
            elif s < -9:
                reward = transition[2]
            else:
                reward = self.terminal_reward
            shaped_experience = (transition[0], transition[1], reward, transition[3], transition[4])

            if self.count < self.buffer_size:
                self.shaped_experience_buffer.append(shaped_experience)

            else:
                self.shaped_experience_buffer.popleft()
                self.shaped_experience_buffer.append(shaped_experience)

            s = s - 1
            # print(i)
            # print(reward)

        self.reset_episode_buffer()

