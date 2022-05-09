import gym

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque


class Actor_critic:

    def __init__(self,n_nodes=[32, 16], learning_rate=0.05, gamma=1, environment="CartPole-v0"):
        """
        :param n_actions: Number of actions available to agent
        :param n_nodes: Number of nodes in hidden layers
        :param learning_rate: Learning rate of neural network
        :param gamma: discount factor, used to calculate the targets during the update
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_inputs = 0
        self.n_actions = 0
        self.n_nodes = n_nodes
        self.model = self.make_model(environment)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        


    def make_model(self, environment):
        """Construct the Q network with input, hidden and output layers"""

        self.n_inputs, self.n_actions, output_activation = get_layer_sizes(environment)

        # Make input layer
        inputs = layers.Input(shape=(self.n_inputs,))
        layers_list = [inputs]
        
        # Make hidden layers
        layers_list.append(layers.Dense(self.n_nodes[0], activation='relu')(layers_list[-1]))
        layers_list.append(layers.Dense(self.n_nodes[1], activation='relu')(layers_list[-1]))
        
        # Make output layer
        actor = layers.Dense(self.n_actions, activation=output_activation)(layers_list[-1])
        critic = layers.Dense(1, activation="linear")(layers_list[-1])
        
        # Combine input and output layers to construct the network
        model = keras.Model(inputs=inputs, outputs=[actor, critic])

        return model


def get_layer_sizes(environment):
    '''For a given environment, return the size of the state vector, and the number of necessary outputs, along with the
    required activation function'''
    sizes = {'CartPole-v0': (4, 2, 'softmax'),
             'Acrobot-v1': (6, 3, 'softmax'),
             'MountainCar-v0': (2, 3, 'softmax'),
             "LunarLander-v2": (8, 4, 'softmax')}
    return sizes[environment]



def actor_critic(n_episodes=250, learning_rate=0.05, gamma=1,
               n_nodes=[32, 16], baseline_subtraction=True, bootstrapping=True, bootstrapping_depth=1,
               render=False, print_episodes=False, environment='CartPole-v0'):
    ''' runs a single repetition of Reinforce
    Return: rewards, a vector with the observed rewards at each timestep '''
    agent = Actor_critic(gamma=gamma, learning_rate=learning_rate, n_nodes=n_nodes, environment=environment)
    reward_per_episode = []
    # Make the CartPole environment
    env = gym.make(environment)
    grad = 0
    for i in range(n_episodes):
        # for each episode, initialise storage for the relevant information, and initialise enviroment
        rewards = []
        actions = []
        probabilities = []
        predictions = []
        state = env.reset()
        state = np.reshape(state, [1, agent.n_inputs])
        done = False
        #initialise GradientTape, this helps determine the gradient of the network when it needs to be updated
        with tf.GradientTape() as tape:
            '''
            GradientTape enables the automated calculation of the gradient over the whole episode.
            it requires that all operations on the data occur within the same context.
            in order to achieve this, both the running of the episode, and the calculation of the loss are done within 
            the same function
            '''
            while not done:
                # run the enviroment until the end of the episode, and store the relevant details about the state and actions taken
                if render:
                    env.render()

                # Select action for given state and strategy
                probs, prediction = agent.model(state)
                action = np.random.choice(a=agent.n_actions, p=probs.numpy()[0])

                # Simulate environment
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, agent.n_inputs])
                actions.append(action)
                rewards.append(reward)
                probabilities.append(probs)
                predictions.append(prediction)
                state = next_state

            # after the episode finishes, calculate total reward obtained
            reward_per_episode.append(np.sum(rewards))

            #calculate loss for the policy head and the actor head.
            actor_losses = []
            critic_losses = []
            R = 0

            for j in range(len(actions)-1, -1, -1):
                action, reward, probs, prediction = actions[j], rewards[j], probabilities[j], predictions[j]

                # Calculate the reward function for this timestep, depending on if bootstrapping is enabled, and the bootstrapping depth
                if bootstrapping:
                    # get the bootstrapping depth, shorten this depth if the episode ends sooner than the set depth
                    k = min(j + bootstrapping_depth, len(actions) - 1) - j
                    if k == bootstrapping_depth:
                        # if the episode is longer than the bootstrapping depth, then use estimated value at time t+n (j+k)
                        R = np.sum([rewards[j+z]*(agent.gamma ** z) for z in range(k)])+predictions[j + k]*(agent.gamma ** k)
                    else:
                        # if the bootstrapping depth is longer than the rest of the episode, then use actual rewards.
                        R = np.sum([rewards[j+z]*(agent.gamma ** z) for z in range(k)])
                else:
                    # use standard discounted rewards if bootstrapping is disabled
                    R = gamma * R + reward

                #Calculate the actor loss this timestep
                if baseline_subtraction:
                    # perform baseline subtraction if enabled.
                    loss = (R - prediction) * -tf.math.log(probs[0][action])
                else:
                    loss = R * -tf.math.log(probs[0][action])
                actor_losses.append(loss)

                # calculate critic loss this step, squared error between predicted and obtained reward for each timestep
                critic_loss = tf.pow(prediction - R, 2)
                critic_losses.append(critic_loss)

            # calculate total loss, update gradient, and perform the update step
            total_loss = sum(actor_losses) + sum((critic_losses))
            gradient = tape.gradient(total_loss, agent.model.trainable_weights)
            agent.optimizer.apply_gradients(zip(gradient, agent.model.trainable_weights))
            #end of episode, GradientTape is used, new tape is initalised at the top of the loop

        if print_episodes:
            print("episode: ", i, " score: ", reward_per_episode[-1])
            
    env.close()
    return reward_per_episode


def test():
    """Test function which prints the obtained rewards for the parameters below"""
    
    n_episodes = 1000
    gamma = 0.8
    learning_rate = 0.005
    bootstrapping_depth = 3

    # Hidden layers
    n_nodes = [32, 16]

    # Plotting parameters
    render = True
    rewards = actor_critic(n_episodes=n_episodes,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         bootstrapping_depth=bootstrapping_depth,
                         n_nodes=n_nodes,
                         render=render,
                         print_episodes=True,
                         environment='LunarLander-v2')
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
