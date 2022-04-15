# import gym

# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from collections import deque


# class Reinforce:

#     def __init__(self, n_actions=2, n_nodes=[32, 16], learning_rate=0.05, gamma=1):
#         """
#         :param n_actions: Number of actions available to agent
#         :param n_nodes: Number of nodes in hidden layers
#         :param learning_rate: Learning rate of neural network
#         :param gamma: discount factor, used to calculate the targets during the update
#         """
#         self.gamma = gamma
#         self.learning_rate = learning_rate
#         self.n_actions = n_actions
#         self.n_nodes = n_nodes
#         self.model = self.make_model()
        

#     def update(self, observations, grad):
#         """Update model"""
        
#         states = []
#         targets = []
        
#         R = 0

#         for s, a, r, _, done in observations[::-1]:
#             states.append(s)
            
#             R = r + self.gamma * R
#             prob = self.model.predict(s)
#             grad += R * -np.log(prob[0][a])
#             g = np.zeros(prob.shape, np.float32)
#             g[0][a] = 1
#             g = g - prob
#             g *= grad
#             g += 1e-4 * prob
#             targets.append(g)
            
#         states = np.concatenate(states)
#         targets = np.concatenate(targets)
        
#         self.model.fit(states, targets, epochs=1, verbose=0)

#     def make_model(self):
#         """Construct the Q network with input, hidden and output layers"""
        
#         # Make input layer
#         inputs = layers.Input(shape=(4,))
#         layers_list = [inputs]
        
#         # Make hidden layers
#         layers_list.append(layers.Dense(self.n_nodes[0], activation='relu')(layers_list[-1]))
#         layers_list.append(layers.Dense(self.n_nodes[1], activation='relu')(layers_list[-1]))
        
#         # Make output layer
#         output = layers.Dense(self.n_actions, activation="softmax")(layers_list[-1])
        
#         # Combine input and output layers to construct the network
#         model = keras.Model(inputs=inputs, outputs=output)
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        
#         return model


# def reinforce(n_episodes=250, learning_rate=0.05, gamma=1, 
#                 n_nodes=[32, 16], render=False):
#     ''' runs a single repetition of Reinforce
#     Return: rewards, a vector with the observed rewards at each timestep '''

#     agent = Reinforce(gamma=gamma, learning_rate=learning_rate)
    
#     print(agent.model)
    
#     reward_per_episode = []
    
#     # Make the CartPole environment
#     env = gym.make("CartPole-v1")

#     grad = 0
    
#     for i in range(n_episodes):
#         rewards = []
#         state = env.reset()
#         state = np.reshape(state, [1, 4])
#         episode = []
#         done = False
#         while not done:
#             if render:
#                 env.render()
                
#             # Select action for given state and strategy
#             action = np.random.choice([0,1], p=agent.model.predict(state)[0])
            
#             # Simulate environment
#             next_state, reward, done, info = env.step(action)
#             next_state = np.reshape(next_state, [1, 4])
#             episode.append((state, action, reward, next_state, done))
#             rewards.append(reward)
#             state = next_state
        
#         reward_per_episode.append(np.sum(rewards))
        
#         agent.update(episode, grad)
        
#         print("episode: ", i, " score: ", reward_per_episode[-1])
            
#     env.close()
#     return reward_per_episode


# def test():
#     """Test function which prints the obtained rewards for the parameters below"""
    
#     n_episodes = 500
#     gamma = 1
#     learning_rate = 0.1

#     # Hidden layers
#     n_nodes = [32, 16]

#     # Plotting parameters
#     render = False
#     rewards = reinforce(n_episodes=n_episodes,
#                           learning_rate=learning_rate,
#                           gamma=gamma,
#                           n_nodes=n_nodes,
#                           render=render)
#     print("Obtained rewards: {}".format(rewards))


# if __name__ == '__main__':
#     test()

import gym

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import tensorflow_probability as tfp


class Reinforce:

    def __init__(self, n_actions=2, n_nodes=[32, 16], learning_rate=0.05, gamma=1):
        """
        :param n_actions: Number of actions available to agent
        :param n_nodes: Number of nodes in hidden layers
        :param learning_rate: Learning rate of neural network
        :param gamma: discount factor, used to calculate the targets during the update
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.model = self.make_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
    

    def update(self, observations):
        """Update model"""
        
        states = []
        targets = []
        
        grad = 0
        R = 0

        for s, a, r, _, done in observations[::-1]:
            states.append(s)          
            R = r + self.gamma * R

            with tf.GradientTape() as tape:
                #dit doet iets:
                prob =  self.model(s)
                d = tfp.distributions.Categorical(prob) #for some reason 
                #gewoon dit gebruiken krijg ik niet aan de praat, maar wat ik nu doe lijkt dezelfde waarden te geven
                # prob = self.model.predict(s) 
                #grad += R * -np.log(prob[0][a])
                #grad = tf.convert_to_tensor(grad) #dit geeft hetzelfde type, maar ook een error?
                logprob_a = d.log_prob(a)
                grad = -(logprob_a*R)
                if tf.math.count_nonzero(grad) == 0: #if the gradient is zero we do not update 
                    continue
                
                grads = tape.gradient(grad, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                


    def make_model(self):
        """Construct the Q network with input, hidden and output layers"""
        
        # Make input layer
        inputs = layers.Input(shape=(4,))
        layers_list = [inputs]
        
        # Make hidden layers
        layers_list.append(layers.Dense(self.n_nodes[0], activation='relu')(layers_list[-1]))
        layers_list.append(layers.Dense(self.n_nodes[1], activation='relu')(layers_list[-1]))
        
        # Make output layer
        output = layers.Dense(self.n_actions, activation="softmax")(layers_list[-1])
        
        # Combine input and output layers to construct the network
        model = keras.Model(inputs=inputs, outputs=output)
        #model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        
        return model


def reinforce(n_episodes=250, learning_rate=0.05, gamma=1, 
                n_nodes=[32, 16], render=False):
    ''' runs a single repetition of Reinforce
    Return: rewards, a vector with the observed rewards at each timestep '''

    agent = Reinforce(gamma=gamma, learning_rate=learning_rate)
    
    print(agent.model)
    
    reward_per_episode = []
    
    # Make the CartPole environment
    env = gym.make("CartPole-v1")

    
    for i in range(n_episodes):
        rewards = []
        state = env.reset()
        state = np.reshape(state, [1, 4])
        episode = []
        done = False
        while not done:
            if render:
                env.render()
                
            # Select action for given state and strategy
            action = np.random.choice([0,1], p=agent.model.predict(state)[0])
            
            # Simulate environment
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            episode.append((state, action, reward, next_state, done))
            rewards.append(reward)
            state = next_state
        
        reward_per_episode.append(np.sum(rewards))
        
        agent.update(episode)
        
        print("episode: ", i, " score: ", reward_per_episode[-1])
            
    env.close()
    return reward_per_episode


def test():
    """Test function which prints the obtained rewards for the parameters below"""
    
    n_episodes = 500
    gamma = 1
    learning_rate = 0.1

    # Hidden layers
    n_nodes = [32, 16]

    # Plotting parameters
    render = False
    rewards = reinforce(n_episodes=n_episodes,
                          learning_rate=learning_rate,
                          gamma=gamma,
                          n_nodes=n_nodes,
                          render=render)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
