import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import pandas as pd
from collections import deque
import tensorflow as tf
#from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import tensorflow.keras.backend as backend
from threading import Thread
from tqdm import tqdm
tf.config.run_functions_eagerly(True)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 3_000 #5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 50
MODEL_NAME = "DQL_TF2_TEST_REPLAY3000_LIMIT_3X64C_HUBER" #Xception"

MIN_REWARD = -1

EPISODES = 6000 #100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.997 ## 0.9975 99975
MIN_EPSILON = 0.002

AGGREGATE_STATS_EVERY = 10 


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.set_model(model) # agrega esta línea

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        # call parent method to set model first
        super().set_model(model)

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                if name in ['batch', 'size']:
                    continue
                tf.summary.scalar(name, value, step=self.step)
        self.writer.flush()

    def _get_writer(self):
        if not hasattr(self, '_writer') or self._writer is None:
            self._writer = tf.summary.create_file_writer(self.log_dir)
            self._writer.set_as_default()

        return self._writer

    def _write_logs(self, logs, index):
        with self._get_writer().as_default():
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                tf.summary.scalar(name, value, step=index)
        self._get_writer().flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 0.25
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        if self.model_3.has_attribute('color'):
            self.model_3.set_attribute('color', '255, 255,0')

    def reset(self):
        self.collision_hist = []
        self.lane_invaded = False
        self.actor_list = []
        cond = False
        while not cond:
            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                cond = True
            except RuntimeError as e:
                print("Wrong spawn point")
            
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(3)
        
        lane_inv_sensor = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_inv_sensor = self.world.spawn_actor(lane_inv_sensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lane_inv_sensor)
        self.lane_inv_sensor.listen(lambda event: self.process_lane_data(event))

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_hist.append(intensity)
    
    def process_lane_data(self, event):
        print("LANE INVASION")
        self.lane_invaded = True

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        done = False
        reward = 0
        
        if len(self.collision_hist) != 0:
            max_collision = max(self.collision_hist)
            # Apply sigmoid to collision in order to limit  between -1 and 1 extreme collisions
            sigmoid_max_collision = 1/(1 + np.exp(-max_collision))
            print(sigmoid_max_collision)
            done = True
            reward = -(1+sigmoid_max_collision)   
            
        elif self.lane_invaded:
            done = True
            reward = -1
            
        elif 30 < kmh < 60:
            # Si esta en el rango entre 30 y 60 le damos refuerzo positivo, no le penalizamos para que no intente acabar cuanto antes para maximizar (chocadose)
            reward = 1-((abs(kmh-40))/100)

        elif abs(kmh-40)/100<=1:
            # Si no esta en el rango le penalizamos segun lo lejos que este de la velocidad deseada
            reward = -1*(abs(kmh-40)/100)
        else:
            # Si la velocidad fuera superior a 100, el reward negativo seria mayor que 1, forzamos que no lo sea de esta manera
            reward=-1
        #Tener en cuenta las aceleraciones de los momentos anteriores para que no vaya dando acelerones y frenazos, y conduzca mas smooth?

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_target_model()
        self.target_model.set_weights(self.model.get_weights())
        #self.cnn_model.predict(np.array([[0,0]])) # warmup

        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        #self.speed_history = deque(maxlen=10)
        # NOTE: CAMBIAR EL LOG DIR SI FUERA NECESARIO AL HACER CAMBIOS DE DQL_TF2: logs/dql_tf2_CORREGIDO/{MODEL_NAME}-{int(time.time())}
        self.tensorboard = ModifiedTensorBoard(self.model, log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0


        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model= Sequential()
        model.add(BatchNormalization(input_shape=(IM_HEIGHT, IM_WIDTH,3)))
        model.add(Conv2D(64,3, activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())
        
        model.add(GlobalAveragePooling2D())
        model.add(Dense(3, activation='linear'))
        huber_loss = Huber(delta=1.0)
        model.compile(loss=huber_loss, optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", huber_loss, "mse"])
        return model
        
    def create_target_model(self):
        model= Sequential()
        model.add(BatchNormalization(input_shape=(IM_HEIGHT, IM_WIDTH,3)))
        model.add(Conv2D(64,3, activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())
        
        model.add(GlobalAveragePooling2D())
        model.add(Dense(3, activation='linear'))
        huber_loss = Huber(delta=1.0)
        model.compile(loss=huber_loss, optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", huber_loss, "mse"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        # PREDICTION_BATCH_SIZE IS USED TO ALLOCATE MEMORY TO GPU IN ORDER TO MAKE THE PREDICTIONS, IF ITS VALUE IS 1, ONLY 1 FRAME WILL BE PROCESSED AT A TIME IN GPU
        # HIGHER VALUES COULD LEAD TO EXCESS OF ALLOCATION IN RAM/GPU MEMORY
        with tf.device('/CPU:0'):
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with tf.device('/CPU:0'):
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
            self.last_logged_episode = self.tensorboard.step
            
        with tf.device('/CPU:0'):
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=1, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        with tf.device('/CPU:0'):
            return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with tf.device('/CPU:0'):
            self.model.fit(X,y, verbose=1, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    monitor_data  = {'Avg_Reward':[], 'Max_Reward': [], 'Min_Reward':[], 'Epsilon':[], 'Learning Rate (Both models)':[], 'Avg_episode_time':[]}
    FPS = 60
    # For stats
    ep_rewards = [] #ep_rewards = [-200]
    episode_elapsed_times = []
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
            env.collision_hist = []

            # Update tensorboard step every episode
            agent.tensorboard.step = episode


            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    qs = agent.get_qs(current_state)
                    action = np.argmax(qs)
                    print(qs)
                else:
                    # Get random action
                    #print("Random action")
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                new_state, reward, done, _ = env.step(action)
                time.sleep(0.25)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
                if done:
                    break
            episode_end = time.time()
            elapsed_time = episode_end - episode_start
            episode_elapsed_times.append(elapsed_time)
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            print("Reward: ", episode_reward)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                avg_episode_elapsed_time = sum(episode_elapsed_times[-AGGREGATE_STATS_EVERY:])/len(episode_elapsed_times[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                monitor_data['Avg_Reward'].append(average_reward)
                monitor_data['Max_Reward'].append(max_reward)
                monitor_data['Min_Reward'].append(min_reward)
                monitor_data['Epsilon'].append(epsilon)
                monitor_data['Learning Rate (Both models)'].append(backend.eval(agent.model.optimizer.lr))
                monitor_data['Avg_episode_time'].append(avg_episode_elapsed_time)
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                df = pd.DataFrame(data = monitor_data)
                df.to_csv(f'/home/xeo/Desktop/salidas_rl/RL_TF2/results/results_{MODEL_NAME}_lr_0.0001_ep_decay_{EPSILON_DECAY}_discount_{DISCOUNT}_UPDATE_CADA_50.csv')
                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.h5')
                    
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    df = pd.DataFrame(data = monitor_data)
    df.to_csv(f'/home/xeo/Desktop/salidas_rl/RL_TF2/results/results_{MODEL_NAME}_lr_0.0001_ep_decay_{EPSILON_DECAY}_discount_{DISCOUNT}_UPDATE_CADA_50.csv')
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.hdf5')
