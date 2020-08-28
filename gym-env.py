import random
import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta

class ontime_dataset_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(ontime_dataset_env, self).__init__()
        self.df = df
        # getting the maximum of each variable because variables have to be bound between 0 and 1
        self.max_x2 = max(df['x2'])
        self.max_x3 = max(df['x3'])
        self.max_x4 = max(df['x4'])
        self.max_x5 = max(df['x5'])
        self.max_x6 = max(df['x6'])
        self.max_x7 = max(df['x7'])
        self.max_x8 = max(df['x8'])
        self.max_x9 = max(df['x9'])
        self.prev_time_step = datetime.strptime(
            '2019-03-30 23:45:01', '%Y-%m-%d %H:%M:%S')
        n_actions = 5
        #defining action space, we are going to discritize our actions into 5 bins
        self.action_space = spaces.Discrete(n_actions)
        
        #defining observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float16)

    def _next_observation(self):
        # construct the state (observation) for every step, the state will include all the airplains in a window of 15 minutes and we will pad the resulting array with zeros to reach a uniform shape
        obs = np.array([
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x1'].values,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x2'].values / self.max_x2,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x3'].values / self.max_x3,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x4'].values / self.max_x4,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x5'].values / self.max_x5,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x6'].values / self.max_x6,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x7'].values / self.max_x7,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x8'].values / self.max_x8,
            self.df.loc[self.current_step: self.current_step
                        + timedelta(minutes=15), 'x9'].values / self.max_x9
        ])
        #fetch next row until the window is done then get next window
        if self.prev_time_step != self.current_step:
            self.row_idx = 0
            self.input_length = obs.shape[1]
        obs = obs[:, self.row_idx]
        self.prev_time_step = self.current_step
        return obs
    
    def _take_action(self, action):
        # choose a random action
        current_dep_latency = random.randint(0, 4)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        obs = self._next_observation()        
        #if we get to the end of the dataset while taking a step, we return to the begining

        
        #calculating the reward, it will be one for each correct lable and -1* abs(true_label - predicted_label)/4 for each incorrect label
        reward = np.absolute(obs[0] - action)*-1/4
        if reward == 0:
            reward = 1
        self.row_idx += 1
        
        # we will end each episode when we have trained on each row within a 15 minutes window, also if we randomly chose a starting ppoint in the last 15 minutes of the dataset we will be reset to the begining
        done = False if self.row_idx < self.input_length else True
        if done:
            self.current_step = self.current_step = self.df.iloc[random.randint(0, len(self.df.loc[:, 'x2'].values) - 6)].name 
            if self.current_step > datetime.strptime('2019-04-30 23:45:01', '%Y-%m-%d %H:%M:%S'):
                self.current_step = datetime.strptime(
                    '2019-04-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        return obs, reward, done, {}

    def reset(self):
        # Set the current step to a random point within the data frame
        self.current_step = self.df.iloc[random.randint(
            0, len(self.df.loc[:, 'x2'].values) - 6)].name
        return self._next_observation()


    def render(self, mode='human', close=False):
        pass
