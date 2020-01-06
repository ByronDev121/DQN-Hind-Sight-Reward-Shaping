import numpy as np
from collections import deque

class ImageUtils:
    def __init__(self):
        self.stacked_frames = None

    def crop(self, image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[54:144, :]/255 # remove the sky and the car front

    def stack_frames(self, frame, is_new_episode):

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros((90, 256), dtype=np.int) for i in range(3)], maxlen=3)

            # Because we're in a new episode, copy the same frame 4x
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

    def preprocess(self, image, is_new_episode):
        """
        Combine all preprocess functions into one
        """
        image = self.crop(image)
        image = self.stack_frames(image, is_new_episode)
        return image

