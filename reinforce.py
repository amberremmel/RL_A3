import gym

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


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
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        


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

        return model


def reinforce(n_episodes=1500, learning_rate=0.001, gamma=1,
               n_nodes=[32, 16], baseline_subtraction = False, bootstrapping = False, bootstrapping_depth = 1,
               render=False, print_episodes=False):
    ''' runs a single repetition of Reinforce
    Return: rewards, a vector with the observed rewards at each timestep '''

    agent = Reinforce(gamma=gamma, learning_rate=learning_rate)

    reward_per_episode = []
    
    # Make the CartPole environment
    env = gym.make("CartPole-v0")

    grad = 0
    
    for i in range(n_episodes):
        rewards = []
        state = env.reset()
        state = np.reshape(state, [1, 4])
        episode = []
        done = False
        with tf.GradientTape() as tape:
            while not done:
                if render:
                    env.render()

                # Select action for given state and strategy
                probabillities = agent.model(state)
                action = np.random.choice([0,1], p=probabillities.numpy()[0])

                # Simulate environment
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                episode.append((action, reward, probabillities))
                rewards.append(reward)
                state = next_state


            reward_per_episode.append(np.sum(rewards))

            losses = []
            R = 0
            for action, reward, probabillities in episode[::-1]:
                R = gamma * R + reward
                loss = R * -tf.math.log(probabillities[0][action])
                losses.insert(0, loss)
            total_loss = sum(losses)
            gradient = tape.gradient(total_loss, agent.model.trainable_weights)
            agent.optimizer.apply_gradients(zip(gradient, agent.model.trainable_weights))

        if print_episodes:
            print("episode: ", i, " score: ", reward_per_episode[-1])
            
    env.close()
    return reward_per_episode


def test():
    """Test function which prints the obtained rewards for the parameters below"""
    
    n_episodes = 1000
    gamma = 0.8
    learning_rate = 0.01

    # Hidden layers
    n_nodes = [32, 16]

    # Plotting parameters
    render = False
    rewards = reinforce(n_episodes=n_episodes,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         n_nodes=n_nodes,
                         render=render,
                        print_episodes=True)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    now = time.time()
    test()
    print('Running one iteration takes {} minutes'.format((time.time()-now)/60))
