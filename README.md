# Distributed-Deep-Reinforcement-Learning
This project implements a multi-agent architecture for autonomous vehicles using Deep Reinforcement Learning in CARLA Simulator. Through a centralized server that coordinates this continuous learning architecture, it is possible to extract and process the knowledge acquired by each agent during its learning process. This strategy facilitates transfer learning, allowing vehicles not only to learn from their own experiences but also from the experiences of other agents in diverse environments.
The correct order for running the architecture is:
    1. Determine the number of agents over which the architecture should be scaled.
    2. Run as many instances of Carla as agents selected (with ./CarlaUE4.sh and defining a different connection port for each agent).
    3. Run the server script: server.py.
    4. Run the intermediary script between the simulator and the architecture: distributed_dql.py.
    5. Run as many agents as selected (executing agent.py with different parameters for each agent depending on the listening port, name, etc.).
