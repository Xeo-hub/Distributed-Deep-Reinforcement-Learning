import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread
from tqdm import tqdm
tf.config.run_functions_eagerly(True)
from distributed_dql import *

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

if __name__ == '__main__':
    agent_index = 0
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

    # Create agent and environment
    
    # TENEMOS EL SERVER_INDEX QUE EMPEZARA SIENDO 0, Y AUMENTARA CUANDO EL AGENTE SE CARGUE LOS NUEVOS PESOS DEL SERVER EN SUS MODELOS LOCALES
    server_index = 0
    
    # SE GENERA EL AGENTE Y EL ENVIRONMENT (AHORA SETEADO AL PUERTO 2000)
    agent = DQNAgent(agent_index, gpus[0])
    env = CarEnv(agent_index, 2000 + 100*agent_index) #env = CarEnv(agent_index, 2000+agent_index)
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
        
    while not agent.training_initialized:
        time.sleep(0.01)
        
    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    
    # SI NO EXISTE EL DIRECTORIO DE MODELOS DE LOS AGENTES PARA ESTE SERVER_INDEX LO CREAMOS (EL SERVER ESPERARA HASTA QUE ESTE CREADO PARA REVISAR SI ESTAN TODOS LOS MODELOS DENTRO)
    if not os.path.isdir(f'agents/{MODEL_NAME}/models/{server_index}'):
        os.makedirs(f'agents/{MODEL_NAME}/models/{server_index}')   
        
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # CON LA FRECUENCIA QUE HAYAMOS ESPECIFICADO EN SERVER_UPDATE_FREC, GUARDAMOS LOS MODELOS LOCALES DEL AGENTE Y ESPERAMOS A QUE ESTEN LOS NUEVOS DEL SERVER PARA UTILIZARLOS
        if episode % SERVER_UPDATE_FREC == 0:
        
            #GUARDAMOS LOS MODELOS DEL AGENTE (TARGET Y MAIN). IMPORTANTE, SOLO SE HACE UNA VEZ CADA SERVER_UPDATE_FREC EPISODIOS
            agent.model.save(f'agents/{MODEL_NAME}/models/{server_index}/main_{agent.identifier}.hdf5')
            agent.target_model.save(f'agents/{MODEL_NAME}/models/{server_index}/target_{agent.identifier}.hdf5')
            
            # SI EL EPISODIO NO ES EL 0, AUMENTAMOS EL INDICE (PORQUE EN EL 0 NO HAY QUE HACER NINGUNA ACTUALIZACION DE PESOS)
            if episode !=0:
                print("Aumenta server index en: ", agent.identifier)
                server_index += 1
            
            # GUARDA EN EL INDEX N+1 EL MODELO QUE HA ENTRENADO EN EL INDEX N
            
            # ESTO SE VUELVE A HACER AQUI POR SI SE HUBIERA AUMENTADO EL SERVER_INDEX, PARA CREAR EL NUEVO DIRECTORIO PARA LOS NUEVOS MODELOS DE LOS AGENTES
            if not os.path.isdir(f'agents/{MODEL_NAME}/models/{server_index}'):
                os.makedirs(f'agents/{MODEL_NAME}/models/{server_index}')          
            
            # GUARDAMOS LOS MODELOS DEL AGENTE (TARGET Y MAIN). IMPORTANTE, SOLO SE HACE UNA VEZ CADA SERVER_UPDATE_FREC EPISODIOS
            #agent.model.save(f'agents/{MODEL_NAME}/models/{server_index}/main_{agent.identifier}.hdf5')
            #agent.target_model.save(f'agents/{MODEL_NAME}/models/{server_index}/target_{agent.identifier}.hdf5')
            
            # ESPERAMOS A QUE LOS NUEVOS MODELOS DEL SERVER ESTEN LISTOS
            while not (os.path.exists(os.path.join(f'server/{MODEL_NAME}/{server_index}', 'main.hdf5')) and os.path.exists(os.path.join(f'server/{MODEL_NAME}/{server_index}', 'target.hdf5'))):
                print("Esperando a los nuevos modelos del server", agent.identifier)
                time.sleep(1)
        
            # NOTE: IMPORTANTE, ESTO PODRÍA HACERSE CON UN GET WEIGHTS Y SET WEIGHTS DEL SERVER AL AGENTE PROBABLEMENTE TAMBIEN
            
            # OBTENEMOS LOS MODELOS PARA ESTE INDEX
            agent.model = tf.keras.models.load_model(f'server/{MODEL_NAME}/{server_index}/main.hdf5')
            agent.target_model = tf.keras.models.load_model(f'server/{MODEL_NAME}/{server_index}/target.hdf5')
            print("Nuevos modelos cargados en: ", agent.identifier)
        
        # A PARTIR DE AQUI TODO SE HACE PARA CADA EPISODIO, LO ANTERIOR SOLO SE HACE SEGUN LA FRECUENCIA DE ACTUALIZACION
          
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

        # ESTE TRACKING NO SE HACE CADA SERVER_UPDATE_FREC, SINO QUE SE HACE MAS FRECUENTEMENTE PARA LLEVAR UN TRACKING A MENOR NIVEL DE LO QUE VA OBTENIENDO CADA AGENTE
        # Append episode reward to a list and log stats (every given number of episodes)
        print("Reward: ", episode_reward, agent.identifier)
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
            acc_loss_data = pd.DataFrame(data = agent.history)
                
            if not os.path.isdir(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}'):
                os.makedirs(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}')  
            df.to_csv(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}/output.csv')
            acc_loss_data.to_csv(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}/acc_loss_tracking.csv')
                    
            # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    df = pd.DataFrame(data = monitor_data)
    df.to_csv(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}/output.csv')
    
    acc_loss_data = pd.DataFrame(data = agent.history)
    acc_loss_data.to_csv(f'agents/{MODEL_NAME}/models/{server_index}/agent_{agent.identifier}/acc_loss_tracking.csv')
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
        
    trainer_thread.join()



