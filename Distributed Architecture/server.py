import glob
import re
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
from tensorflow.keras.utils import custom_object_scope
import tensorflow.keras.backend as backend
from threading import Thread
from tqdm import tqdm
tf.config.run_functions_eagerly(True)

# NOTE: IMPORTANTE, AHORA FUNCIONA CON TENSORFLOW2, PERO NO CON GPU, PUEDES VERLO CON LO DEL WITH (PERO AHORA SI ACTUALIZA LA TARGET NETWORK, CREO, PROBAR MAÑANA)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SERVER_UPDATE_FREC = 500
TOTAL_AGENTS = 2
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 50
MODEL_NAME = "DQL_TF2_REPLAY5000_LANEINV_BRAKE_THRSMTH_3X64C_32D_MSE" 

MIN_REWARD = -1

EPISODES = 6000 #100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.997 ## 0.9975 99975
MIN_EPSILON = 0.002

AGGREGATE_STATS_EVERY = 50 
        
class Server:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.index = 0
        if not os.path.isdir(f'server/{MODEL_NAME}/{self.index}'):
            os.makedirs(f'server/{MODEL_NAME}/{self.index}')    
        print(f'server/{MODEL_NAME}/{self.index}')
        self.model.save(f'server/{MODEL_NAME}/{self.index}/main.hdf5')
        print("main server model saved")
        self.target_model.save(f'server/{MODEL_NAME}/{self.index}/target.hdf5')
        print("target server model saved")
        self.target_update_counter = 0
    
    def create_model(self):
        huber_loss = Huber(delta=1.0)
        model= Sequential()
        model.add(BatchNormalization(input_shape=(IM_HEIGHT, IM_WIDTH,3)))
        model.add(Conv2D(64,8, activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64,4, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64,3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", "mse"])
        return model

def mean_weights (models_list, server_index):
    main_pattern = re.compile(r"main_\d+\.hdf5")
    target_pattern = re.compile(r"target_\d+\.hdf5")
    main_models_weights = []
    target_models_weights = []
    
    for elem in models_list:
         if os.path.isfile(os.path.join(f'agents/{MODEL_NAME}/models/{server_index}', elem)):
         
             if main_pattern.match(elem):
                 agent_main_model = tf.keras.models.load_model(f'agents/{MODEL_NAME}/models/{server_index}/{elem}')
                 weights = agent_main_model.weights
                 main_models_weights.append(weights)
                 
             elif target_pattern.match(elem):
                 agent_target_model = tf.keras.models.load_model(f'agents/{MODEL_NAME}/models/{server_index}/{elem}')
                 weights = agent_target_model.weights
                 target_models_weights.append(weights)
                 
             else:
                 pass
    
    new_main_server_weights = np.mean(main_models_weights, axis=0)      
    new_target_server_weights = np.mean(target_models_weights, axis=0) 
    return new_main_server_weights, new_target_server_weights

        
if __name__ == '__main__':
    num_agents = TOTAL_AGENTS
    monitor_data  = {'Avg_Reward':[], 'Max_Reward': [], 'Min_Reward':[], 'Epsilon':[], 'Learning Rate (Both models)':[], 'Avg_episode_time':[]}
    FPS = 60
    # For stats
    ep_rewards = [] #ep_rewards = [-200]
    episode_elapsed_times = []
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    threads = []

    # Memory fraction, used mostly when trai8ning multiple agents
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create agent and environment
    server = Server()
    
    # SI LA FRECUENCIA DE UPDATE DEL SERVER ES CADA 500 EPOCHS Y EN TOTAL SON 6000, SE ACTUALIZARA 6000/500 VECES (ES DECIR, 12)
    for i in range(0, EPISODES//SERVER_UPDATE_FREC):
        
        # MIENTRAS NO SE HAYA CREADO EL DIRECTORIO PARA ESTE INDEX ESPERAMOS A QUE UNO DE LOS AGENTES LO CREE
        while not os.path.isdir(f'agents/{MODEL_NAME}/models/{server.index}'):
            print("Esperando a que un agente cree el directorio para este index: ", server.index)
            time.sleep(10)
        
        # OBTENEMOS LOS FICHEROS DENTRO DEL DIRECTORIO
        agent_models_list = os.listdir(f'agents/{MODEL_NAME}/models/{server.index}')
        # CALCULAMOS EL NUMERO DE MODELOS (NO CONTAMOS LOS SUBDIRECTORIOS) DENTRO DEL DIRECTORIO
        num_models = len([archivo for archivo in agent_models_list if os.path.isfile(os.path.join(f'agents/{MODEL_NAME}/models/{server.index}', archivo))])
    
        # HASTA QUE NO HAYA EL DOBLE DE MODELOS DENTRO DEL DIRECTORIO QUE EL NUMERO DE AGENTES (PORQUE HABRA DOS FICHEROS POR AGENTE, MAIN Y TARGET), ESPERAMOS
        while not num_models == 2*num_agents:
            print('Esperando a que esten todos los modelos en el directorio :')
            print(f'agents/{MODEL_NAME}/models/{server.index}')
            print(agent_models_list)
            agent_models_list = os.listdir(f'agents/{MODEL_NAME}/models/{server.index}')
            num_models = len([archivo for archivo in agent_models_list if os.path.isfile(os.path.join(f'agents/{MODEL_NAME}/models/{server.index}', archivo))])
            time.sleep(30)
            
        # CUANDO YA ESTAN TODOS LOS MODELOS GUARDADOS EN EL DIRECTORIO SE HACE LA MEDIA DE SUS PESOS
        new_main_model_weights, new_target_model_weights = mean_weights(agent_models_list, server.index)

        # SE SETEAN LOS NUEVOS PESOS PARA LAS DQN DEL SERVER PRINCIPAL
        print("Actualización de pesos del server")
        server.model.set_weights(new_main_model_weights)
        server.target_model.set_weights(new_target_model_weights)
        
        # SE AUMENTA EL INDICE DEL SERVER
        server.index +=1
        
        print("Se crea el nuevo directorio para el nuevo index", server.index)
        
        # Y SE GUARDAN LOS MODELOS CON LOS NUEVOS PESOS EN EL DIRECTORIO CON EL NUEVO INDEX
        if not os.path.isdir('server/{MODEL_NAME}/{server.index}'):
                os.makedirs('server/{MODEL_NAME}/{server.index}')  
        
        print("Se guardan los nuevos modelos en el directorio con nuevo indice")
        server.model.save(f'server/{MODEL_NAME}/{server.index}/main.hdf5')
        server.target_model.save(f'server/{MODEL_NAME}/{server.index}/target.hdf5')
    
   
    
    
    
        
        


