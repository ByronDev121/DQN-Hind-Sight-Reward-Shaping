import time
import cv2
import numpy as np
import airsim
from utils import ImageUtils

DEFAULT_FPS = 8
utils = ImageUtils()

class Gym:

    def __init__(self, show_cam=True):
        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()

        self.episode_start = time.time()
        self.SHOW_CAM = show_cam

        self.fps = DEFAULT_FPS
        self.period = 1/DEFAULT_FPS
        self.previous_action_time = None
        self.step_num = 0

        self.previous_distance = None


    def reset(self):
        self.client.reset()
        time.sleep(0.1)
        img = self.get_image(True)
        self.episode_start = time.time()
        return img

    def get_image(self, is_new_episode):
        try:
            # RGB image
            # image = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            # image1d = np.fromstring(image.image_data_uint8, dtype=np.uint8)
            # image_rgb = image1d.reshape(image.height, image.width, 3)
            # frame = utils.preprocess(image_rgb, is_new_episode)

            # Depth image
            depth = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])[0]
            depth = np.array(depth.image_data_float, dtype=np.float32)
            depth = depth.reshape(144, 256)
            frame = np.array(depth * 255, dtype=np.uint8)
            frame = utils.preprocess(frame, is_new_episode)

            if self.SHOW_CAM:
                cv2.imshow("", frame)
                cv2.waitKey(1)

            return frame
        except Exception as e:
            self.client.reset()

    def step(self, action):

        # 0 - accelerate (0-1) - Constant
        self.car_controls.throttle = 0.65

        # Discrete steering control:
        # 1 - turn right (0 to 1)
        if action == 0:
            self.car_controls.steering = 0.5
        # 2 - dont't turn (0)
        elif action == 1:
            self.car_controls.steering = 0
        # 3 - turn left (0 to -1)
        elif action == 2:
            self.car_controls.steering = -0.5

        self.client.setCarControls(self.car_controls)

        current_img = self.get_image(False)

        collision_info = self.client.simGetCollisionInfo()

        distance_info = self.client.getDistanceSensorData()

        reward = 0

        if distance_info.distance < 20:
            reward = reward - 10
            if self.previous_distance is not None:
                if distance_info.distance < self.previous_distance:
                    reward = reward - 10
        self.previous_distance = distance_info.distance

        if collision_info.has_collided:
            done = True
            reward = reward - 50
            if (time.time() - self.episode_start) < 0.5:
                reward = reward + 50

        else:
            done = False
            reward = 1

            if(time.time() - self.episode_start) > 90:
                done = True

        self.regulate_fps()

        return current_img, reward, done, None

    def regulate_fps(self):
        now = time.time()
        if self.previous_action_time:
            delta = now - self.previous_action_time
            if delta < self.period:
                time.sleep(delta)
            else:
                fps = 1. / delta
                if self.step_num > 5 and fps < self.fps / 2:
                    print('Step %r took %rs - target is %rs', self.step_num, delta, 1 / self.fps)
        self.previous_action_time = now
