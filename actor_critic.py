import gym

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque


class Actor_critic:

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
        actor = layers.Dense(self.n_actions, activation="softmax")(layers_list[-1])
        critic = layers.Dense(1, activation="linear")(layers_list[-1])
        
        # Combine input and output layers to construct the network
        model = keras.Model(inputs=inputs, outputs=[actor, critic])

        return model


def actor_critic(n_episodes=250, learning_rate=0.05, gamma=1,
               n_nodes=[32, 16], baseline_subtraction=True, bootstrapping=True, bootstrapping_depth=1, render=False, print_episodes=False):
    ''' runs a single repetition of Reinforce
    Return: rewards, a vector with the observed rewards at each timestep '''

    agent = Actor_critic(gamma=gamma, learning_rate=learning_rate)

    reward_per_episode = []
    
    # Make the CartPole environment
    env = gym.make("CartPole-v0")

    grad = 0

    for i in range(n_episodes):
        #initialise the lists for collecting data, and initialise enviroment
        rewards = []
        actions = []
        probabilities = []
        predictions = []
        state = env.reset()
        state = np.reshape(state, [1, 4])
        done = False
        #initialise GradientTape, this helps determine the gradient of the network when it needs to be updated
        with tf.GradientTape() as tape:
            while not done:  # run the enviroment until the end of the episode, and store the relevant details about the state and actions taken
                if render:
                    env.render()

                # Select action for given state and strategy
                probs, prediction = agent.model(state)
                #print(prediction, probabilities)
                action = np.random.choice([0,1], p=probs.numpy()[0])


                # Simulate environment
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                #episode.append((action, reward, probs, prediction[0, 0]))
                actions.append(action)
                rewards.append(reward)
                probabilities.append(probs)
                predictions.append(prediction)
                state = next_state

            # after the episode finishes, then calculate total reward obtained
            reward_per_episode.append(np.sum(rewards))

            #calculate loss for the policy head and the actor head.
            actor_losses = []
            critic_losses = []
            R = 0
            for j in range(len(actions)-1, -1, -1): # calculate the policy loss and value loss for each timestep
                action, reward, probs, prediction = actions[j], rewards[j], probabilities[j], predictions[j]
                if bootstrapping:   # Use bootstrapping in the reward function if it is enabled
                    k = min(j + bootstrapping_depth, len(actions) - 1) - j # get the bootstrapping depth, minding the end of the episode
                    if k == bootstrapping_depth: # if the episode is longer than the bootstrapping depth, then use estimated value
                        R = np.sum([rewards[j+z]*(agent.gamma ** z) for z in range(k)])+predictions[j + k]*(agent.gamma ** k)
                    else: # if the bootstrapping depth is longer than the rest of the episode, then use actual rewards.
                        R = np.sum([rewards[j+z]*(agent.gamma ** z) for z in range(k)])
                else:
                    R = gamma * R + reward # use standard discounted rewards if bootstrapping is disabled

                if baseline_subtraction:
                    loss = (R - prediction) * -tf.math.log(probs[0][action]) #perform baseline subtraction if enabled
                else:
                    loss = R * -tf.math.log(probs[0][action])

                actor_losses.append(loss)
                critic_loss = tf.pow(prediction - R, 2) # calculate critic loss, simple squared error for each timestep, comparing the prediction and the obtained discounted rewards
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
    
    n_episodes = 500
    gamma = 0.8
    learning_rate = 0.015

    # Hidden layers
    n_nodes = [32, 16]

    # Plotting parameters
    render = False
    rewards = actor_critic(n_episodes=n_episodes,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         n_nodes=n_nodes,
                         render=render,
                         print_episodes=True)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
