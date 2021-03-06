import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow import keras
from keras.layers import Dense, Activation, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np


class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=5, layer1_size=16, layer2_size=16, layer3_size=16, input_dims=9, fname='model.h5'):
        self.GAMMA = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.fc3_dims = layer3_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]
        self.model_file = fname
    
    #this is the function that will define the policy agent and and will make predictions
    def build_policy_network(self):
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        dense3 = Dense(self.fc3_dims, activation='relu')(dense2)
        probs = Dense(self.n_actions, activation='softmax')(dense3)

        def custom_loss(y_true, y_predict):
            out = K.clip(y_predict, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * advantages)

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)
        predict = Model(input=[input], output=[probs])
        return policy, predict
    
    #we will choose an actiom for each observation within the batch by making a random choice weighted by the probabilities predicted by our model
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    #this is a helper function that stores the history of the model
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    # this is the main driver function that we will call to train the agent
    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1
        
        # calculating the gain
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *= self.GAMMA
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std
        
        #calculating the cost
        cost = self.policy.train_on_batch([state_memory, self.G], actions)
        
        #resetting memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    
    #helper function to save and load the model
    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)
