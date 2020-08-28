# Implementation of a Policy Gradient Model using Keras and a custom Gym environment
Data used in this project can be downloaded from https://www.transtats.bts.gov/Fields.asp. The preprocessing of the data to create the custom gym environment is adapted from this paper https://ieeexplore.ieee.org/document/7411275. 

#### We will use the data from April 2019 in this demonstration


```python
import pandas as pd
from datetime import datetime, date, time, timedelta
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv').drop('Unnamed: 24', axis=1)
df = df[(df.CANCELLED == 0) & (df.DIVERTED == 0)].drop(
    ['CANCELLED', 'DIVERTED'], axis=1)
```

# Data Processing
X1 is going to be Taxi Out time, our target


```python
df['x1'] = df['TAXI_OUT']
```

## Windowing
To be able to achieve the windowing described in the paper we will need to convert DEP_TIME and ARR_TIME into a python datetime object. We will also use the date from FL_DATE


```python
def time_converter(in_):
    in_date = in_[0]
    in_time = in_[1]
    in_time = str(in_time)
    if '.' in in_time:
        in_time = in_time[:-2]
    while len(str(in_time)) < 4:
        in_time = '0' + in_time
    if in_time[:2] == '24':
        in_time = '00' + in_time[2:]
    time_ = time(int(in_time[:2]), int(in_time[2:4]), 0)
    in_date = in_date.split('-')
    date_ = date(int(in_date[0]), int(in_date[1]), int(in_date[2]))
    return datetime.combine(date_, time_)


time_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF',
             'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME', ]

for i, col in enumerate(time_cols):
    df[col] = df[['FL_DATE', col]].apply(time_converter, axis=1)
    print(f"{round(100*(i+1)/len(time_cols),2)}% done.")
```

    16.67% done.
    33.33% done.
    50.0% done.
    66.67% done.
    83.33% done.
    100.0% done.



```python
def get_bounds(in_time):
    time_d = timedelta(minutes=20)
    return (in_time - time_d),  (in_time + time_d)


def window_func(TIME, variable, df):
    lower_bound, upper_bound = get_bounds(TIME)
    df_ = df[(df[variable] >= lower_bound) & (df[variable] <=
                                              upper_bound)].copy()
    count_airplanes = len(df_) - 1
    return count_airplanes


def pct_tracker(to_be_done):
    done = 0
    count = 0
    while True:
        yield
        done += 1
        count += 1
        if done > to_be_done:
            done -= to_be_done
        if (count >= to_be_done * 0.02):
            count = 0
            pct = round((float(done) / float(to_be_done)) * 100, 2)
            print(f"{pct}% done.")


def track_progress(func=window_func, progress=pct_tracker(len(df))):
    def call_func(*args, **kwargs):
        progress.send(None)
        return func(*args, **kwargs)
    return call_func


def airplane_counter(func, variable, airport_type, list, add_cols=[]):
    listy = []
    sub_df = df.sort_values(by=[airport_type, variable])[
        [airport_type, variable] + add_cols].copy()
    for i, airport in enumerate(list):
        x = sub_df[sub_df[airport_type] == airport].copy()
        x['target'] = x[variable].apply(
            track_progress(func), args=([variable, x]))
        listy.append(x)
    series = pd.concat(listy).sort_values(by=[airport_type, variable]).target
    return series


df = df.sort_values(by=['ORIGIN', 'DEP_TIME']).copy()
origins = df.ORIGIN.unique().tolist()

x2 = airplane_counter(window_func, 'DEP_TIME', 'ORIGIN', origins)
```

    2.0% done.
    4.0% done.
    6.0% done.
    8.0% done.
    10.0% done.
    12.0% done.
    14.0% done.
    16.0% done.
    18.0% done.
    20.0% done.
    22.0% done.
    24.0% done.
    26.0% done.
    28.0% done.
    30.0% done.
    32.0% done.
    34.0% done.
    36.0% done.
    38.0% done.
    40.0% done.
    42.0% done.
    44.0% done.
    46.0% done.
    48.0% done.
    50.0% done.
    52.0% done.
    54.0% done.
    56.0% done.
    58.0% done.
    60.0% done.
    62.0% done.
    64.0% done.
    66.0% done.
    68.0% done.
    70.0% done.
    72.0% done.
    74.0% done.
    76.0% done.
    78.0% done.
    80.0% done.
    82.0% done.
    84.0% done.
    86.0% done.
    88.0% done.
    90.0% done.
    92.0% done.
    94.0% done.
    96.0% done.
    98.0% done.


Similarly, we will calculate the number of arrival aircrafts sharing the runway with the aircraft being considered in the +-20 minutes window (x3)


```python
df['x2'] = x2

df = df.sort_values(by=['DEST', 'ARR_TIME']).copy()
destinations = df.DEST.unique().tolist()

sub_df = df.sort_values(by=['DEST', 'ARR_TIME'])[
    ['DEST', 'ARR_TIME', 'ARR_DELAY']].copy()
dict_ = {dest: sub_df[sub_df['DEST'] == dest].copy() for dest in destinations}


def window_func_arr(in_):
    ORIGIN, TIME = in_[0], in_[1]
    lower_bound, upper_bound = get_bounds(TIME)
    df = dict_[ORIGIN]
    df_ = df[(df['ARR_TIME'] >= lower_bound) & (df['ARR_TIME'] <=
                                                upper_bound)].copy()
    count_airplanes = len(df_)
    return count_airplanes


df['x3'] = df[['ORIGIN', 'DEP_TIME']].apply(
    track_progress(window_func_arr), axis=1)
```

    0.01% done.
    2.01% done.
    4.01% done.
    6.01% done.
    8.01% done.
    10.01% done.
    12.01% done.
    14.01% done.
    16.01% done.
    18.01% done.
    20.01% done.
    22.01% done.
    24.01% done.
    26.01% done.
    28.01% done.
    30.01% done.
    32.01% done.
    34.01% done.
    36.01% done.
    38.01% done.
    40.01% done.
    42.01% done.
    44.01% done.
    46.01% done.
    48.01% done.
    50.01% done.
    52.01% done.
    54.01% done.
    56.01% done.
    58.01% done.
    60.01% done.
    62.01% done.
    64.01% done.
    66.01% done.
    68.01% done.
    70.01% done.
    72.01% done.
    74.01% done.
    76.01% done.
    78.01% done.
    80.01% done.
    82.01% done.
    84.01% done.
    86.01% done.
    88.01% done.
    90.01% done.
    92.01% done.
    94.01% done.
    96.01% done.
    98.01% done.


-we will calculate the average taxi out time for each airoprt (x4)

-We will also define a function to check if the flight is or is not during peak time (x5)

-we will take (x6) to be air time and we will take (x7) to be taxi in


```python
def is_peak(dep_time):
    dep_time = dep_time.time()
    if (dep_time >= time(4, 0) and (dep_time <= time(16, 0))):
        return 1
    else:
        return 2


dict_averages={origin: df[df.ORIGIN == origin]
                 ['TAXI_OUT'].mean() for origin in origins}


def avg_taxi_out(origin, dict=dict_averages):
    return dict[origin]


df['x4']=df.ORIGIN.apply(avg_taxi_out)

df['x5']=df.DEP_TIME.apply(is_peak)

df['x6']=df.AIR_TIME

df['x7']=df.TAXI_IN

df.to_csv('snapshot.csv', index=False)
```

-we will define x* as the number of late aircrafts sharing the runway with the aircraft being considered in the time window


```python
def late_to_leave(TIME, variable, df):
    lower_bound, upper_bound=get_bounds(TIME)
    df_=df[(df[variable] >= lower_bound) & (df[variable] <=
                                              upper_bound) & ((df['DEP_DELAY'] < -15) | (df['DEP_DELAY'] > 15))].copy()
    count_airplanes=len(df_)
    return count_airplanes

x8=airplane_counter(late_to_leave, 'DEP_TIME', 'ORIGIN',
                    origins, add_cols=['DEP_DELAY'])

df['x8']=x8
```

    0.01% done.
    2.01% done.
    4.01% done.
    6.01% done.
    8.01% done.
    10.01% done.
    12.01% done.
    14.01% done.
    16.01% done.
    18.01% done.
    20.01% done.
    22.01% done.
    24.01% done.
    26.01% done.
    28.01% done.
    30.01% done.
    32.01% done.
    34.01% done.
    36.01% done.
    38.01% done.
    40.01% done.
    42.01% done.
    44.01% done.
    46.01% done.
    48.01% done.
    50.01% done.
    52.01% done.
    54.01% done.
    56.01% done.
    58.01% done.
    60.01% done.
    62.01% done.
    64.01% done.
    66.01% done.
    68.01% done.
    70.01% done.
    72.01% done.
    74.01% done.
    76.01% done.
    78.01% done.
    80.01% done.
    82.01% done.
    84.01% done.
    86.01% done.
    88.01% done.
    90.01% done.
    92.01% done.
    94.01% done.
    96.02% done.
    98.02% done.


x9 is the number of late arriving aircrafts on the runway in the time window


```python
def late_to_arrive(in_):
    ORIGIN, TIME=in_[0], in_[1]
    lower_bound, upper_bound=get_bounds(TIME)
    df=dict_[ORIGIN]
    df_=df[(df['ARR_TIME'] >= lower_bound) & (df['ARR_TIME'] <=
                                                upper_bound) & ((df['ARR_DELAY'] < -15) | (df['ARR_DELAY'] > 15))].copy()
    count_airplanes=len(df_)
    return count_airplanes

df['x9']=df[['ORIGIN', 'DEP_TIME']].apply(
    track_progress(late_to_arrive), axis=1)

df[['ORIGIN', 'DEP_TIME'] +
    ['x' + str(i) for i in range(1, 10)]].to_csv('variables.csv', index=False)

work_df=df[['ORIGIN', 'DEP_TIME'] +
    ['x' + str(i) for i in range(1, 10)]]
```

    0.02% done.
    2.02% done.
    4.02% done.
    6.02% done.
    8.02% done.
    10.02% done.
    12.02% done.
    14.02% done.
    16.02% done.
    18.02% done.
    20.02% done.
    22.02% done.
    24.02% done.
    26.02% done.
    28.02% done.
    30.02% done.
    32.02% done.
    34.02% done.
    36.02% done.
    38.02% done.
    40.02% done.
    42.02% done.
    44.02% done.
    46.02% done.
    48.02% done.
    50.02% done.
    52.02% done.
    54.02% done.
    56.02% done.
    58.02% done.
    60.02% done.
    62.02% done.
    64.02% done.
    66.02% done.
    68.02% done.
    70.02% done.
    72.02% done.
    74.02% done.
    76.02% done.
    78.02% done.
    80.02% done.
    82.02% done.
    84.02% done.
    86.02% done.
    88.02% done.
    90.02% done.
    92.02% done.
    94.02% done.
    96.02% done.
    98.02% done.


Finally we will discretize the target value (taxi out time). our target x1 is going to be descritized into 5 bins


```python
work_df['DEP_TIME']=pd.to_datetime(work_df['DEP_TIME'])
work_df=pd.read_csv('./env/data/variables.csv')
print('non discretized environment variables')
work_df.iloc[:10,:]
```

    non discretized environment variables





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ORIGIN</th>
      <th>DEP_TIME</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SFB</td>
      <td>2019-04-01 06:58:00</td>
      <td>15.0</td>
      <td>7</td>
      <td>0</td>
      <td>13.801970</td>
      <td>1</td>
      <td>127.0</td>
      <td>4.0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLT</td>
      <td>2019-04-01 09:46:00</td>
      <td>27.0</td>
      <td>33</td>
      <td>28</td>
      <td>21.445305</td>
      <td>1</td>
      <td>68.0</td>
      <td>5.0</td>
      <td>9</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DTW</td>
      <td>2019-04-01 10:09:00</td>
      <td>18.0</td>
      <td>39</td>
      <td>16</td>
      <td>17.244655</td>
      <td>1</td>
      <td>62.0</td>
      <td>4.0</td>
      <td>13</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ATL</td>
      <td>2019-04-01 10:39:00</td>
      <td>11.0</td>
      <td>58</td>
      <td>42</td>
      <td>15.761485</td>
      <td>1</td>
      <td>95.0</td>
      <td>3.0</td>
      <td>18</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PHL</td>
      <td>2019-04-01 12:47:00</td>
      <td>14.0</td>
      <td>22</td>
      <td>9</td>
      <td>21.666012</td>
      <td>1</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SFB</td>
      <td>2019-04-01 12:13:00</td>
      <td>13.0</td>
      <td>1</td>
      <td>3</td>
      <td>13.801970</td>
      <td>1</td>
      <td>120.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ORD</td>
      <td>2019-04-01 13:56:00</td>
      <td>33.0</td>
      <td>65</td>
      <td>44</td>
      <td>21.773309</td>
      <td>1</td>
      <td>84.0</td>
      <td>5.0</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTW</td>
      <td>2019-04-01 15:40:00</td>
      <td>21.0</td>
      <td>42</td>
      <td>22</td>
      <td>17.244655</td>
      <td>1</td>
      <td>61.0</td>
      <td>3.0</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ATL</td>
      <td>2019-04-01 15:49:00</td>
      <td>15.0</td>
      <td>37</td>
      <td>53</td>
      <td>15.761485</td>
      <td>1</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>10</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CLT</td>
      <td>2019-04-01 16:24:00</td>
      <td>19.0</td>
      <td>52</td>
      <td>19</td>
      <td>21.445305</td>
      <td>2</td>
      <td>74.0</td>
      <td>6.0</td>
      <td>7</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# Making the environment
We will make a custom gym environment from this data following the code examples in their github.

We will use comments inside the code below to document.


```python
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
```

# Making the Model
Our model is based on stable-baselines PGM model, it's also built on the code available in their github.


```python
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
```

    Using TensorFlow backend.


# Testing the environment


```python
import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv('./env/data/variables.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal')
df['x1'] = discretizer.fit_transform(
df['x1'].to_numpy().reshape(-1, 1))

env = ontime_dataset_env(df)
```


```python
env.reset()
```




    array([1.        , 0.03448276, 0.02380952, 0.58825559, 0.5       ,
           0.05890805, 0.0221519 , 0.        , 0.01724138])




```python
env.step(4)
```




    (array([4.        , 0.14942529, 0.70238095, 0.61140748, 0.5       ,
            0.16954023, 0.00949367, 0.1       , 0.15517241]),
     1,
     False,
     {})




```python
env.step(4)
```




    (array([0.        , 0.11494253, 0.27380952, 0.68524502, 1.        ,
            0.27011494, 0.01265823, 0.03333333, 0.13793103]),
     -1.0,
     False,
     {})



Now that we are satisfied that our custom environment is working, we will start training the model.

# Training the model
Unfortunately no matter what hyper parameters were used, this model was not able to be trained on this dataset effectively.


```python
agent = Agent(ALPHA=0.001, GAMMA=0.99, n_actions=5,
                  layer1_size=32, layer2_size=16,
                  layer3_size=8, fname='model.h5')
env = ontime_dataset_env(df)
score_history = []

n_episodes = 2500

for i in range(n_episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observation_
        score += reward
    score_history.append(score)
    agent.learn()
    print('episode', i, 'score %.1f' % score, 'average_score %.1f' %
         np.mean(score_history[-100:]))

agent.save_model()
```

    /home/mostafakm/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:43: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=[<tf.Tenso...)`
    /home/mostafakm/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:45: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=[<tf.Tenso...)`


    episode 0 score -45.2 average_score -45.2
    episode 1 score -80.0 average_score -62.6
    episode 2 score -71.5 average_score -65.6
    episode 3 score -32.5 average_score -57.3
    episode 4 score -51.8 average_score -56.2
    episode 5 score -74.5 average_score -59.2
    episode 6 score -65.8 average_score -60.2
    episode 7 score -41.0 average_score -57.8
    episode 8 score -85.8 average_score -60.9
    episode 9 score -86.5 average_score -63.5
    episode 10 score -123.0 average_score -68.9
    episode 11 score -65.5 average_score -68.6
    episode 12 score -71.5 average_score -68.8
    episode 13 score -62.5 average_score -68.4
    episode 14 score -88.2 average_score -69.7
    episode 15 score -41.0 average_score -67.9
    episode 16 score -90.5 average_score -69.2
    episode 17 score -46.5 average_score -68.0
    episode 18 score -65.2 average_score -67.8
    episode 19 score -63.5 average_score -67.6
    episode 20 score -60.0 average_score -67.2
    episode 21 score -35.8 average_score -65.8
    episode 22 score -59.0 average_score -65.5
    episode 23 score -80.2 average_score -66.1
    episode 24 score -92.5 average_score -67.2
    episode 25 score -60.8 average_score -66.9
    episode 26 score -66.2 average_score -66.9
    episode 27 score -62.0 average_score -66.7
    episode 28 score -49.0 average_score -66.1
    episode 29 score -62.0 average_score -66.0
    episode 30 score -65.8 average_score -66.0
    episode 31 score -73.0 average_score -66.2
    episode 32 score -80.0 average_score -66.6
    episode 33 score -57.2 average_score -66.3
    episode 34 score -50.5 average_score -65.9
    episode 35 score -59.8 average_score -65.7
    episode 36 score -79.2 average_score -66.1
    episode 37 score -4.5 average_score -64.5
    episode 38 score -61.2 average_score -64.4
    episode 39 score -95.2 average_score -65.2
    episode 40 score -62.0 average_score -65.1
    episode 41 score -45.0 average_score -64.6
    episode 42 score -61.0 average_score -64.5
    episode 43 score -51.8 average_score -64.2
    episode 44 score -34.5 average_score -63.6
    episode 45 score -69.8 average_score -63.7
    episode 46 score -18.8 average_score -62.7
    episode 47 score -80.5 average_score -63.1
    episode 48 score -40.2 average_score -62.6
    episode 49 score -53.2 average_score -62.5
    episode 50 score -70.8 average_score -62.6
    episode 51 score 1.0 average_score -61.4
    episode 52 score -16.5 average_score -60.5
    episode 53 score -79.0 average_score -60.9
    episode 54 score -83.0 average_score -61.3
    episode 55 score -59.5 average_score -61.3
    episode 56 score -81.8 average_score -61.6
    episode 57 score -81.8 average_score -62.0
    episode 58 score -92.8 average_score -62.5
    episode 59 score -54.8 average_score -62.4
    episode 60 score -41.5 average_score -62.0
    episode 61 score -54.2 average_score -61.9
    episode 62 score -54.5 average_score -61.8
    episode 63 score -38.2 average_score -61.4
    episode 64 score -59.8 average_score -61.4
    episode 65 score -56.8 average_score -61.3
    episode 66 score -2.5 average_score -60.4
    episode 67 score -68.0 average_score -60.5
    episode 68 score -83.5 average_score -60.9
    episode 69 score -51.5 average_score -60.7
    episode 70 score -66.8 average_score -60.8
    episode 71 score -96.0 average_score -61.3
    episode 72 score -54.5 average_score -61.2
    episode 73 score -54.5 average_score -61.1
    episode 74 score -42.5 average_score -60.9
    episode 75 score -67.2 average_score -61.0
    episode 76 score -83.5 average_score -61.3
    episode 77 score -81.8 average_score -61.5
    episode 78 score -55.5 average_score -61.4
    episode 79 score -56.5 average_score -61.4
    episode 80 score -71.5 average_score -61.5
    episode 81 score -74.0 average_score -61.7
    episode 82 score -57.8 average_score -61.6
    episode 83 score -54.5 average_score -61.5
    episode 84 score -49.2 average_score -61.4
    episode 85 score -62.2 average_score -61.4
    episode 86 score -56.8 average_score -61.3
    episode 87 score -44.0 average_score -61.1
    episode 88 score -79.0 average_score -61.3
    episode 89 score -54.5 average_score -61.3
    episode 90 score -49.5 average_score -61.1
    episode 91 score -49.0 average_score -61.0
    episode 92 score -59.0 average_score -61.0
    episode 93 score -36.5 average_score -60.7
    episode 94 score -93.5 average_score -61.1
    episode 95 score -38.0 average_score -60.8
    episode 96 score -63.5 average_score -60.9
    episode 97 score -30.8 average_score -60.6
    episode 98 score -74.8 average_score -60.7
    episode 99 score -65.0 average_score -60.7
    episode 100 score -59.8 average_score -60.9
    episode 101 score -79.5 average_score -60.9
    episode 102 score -73.0 average_score -60.9
    episode 103 score -66.0 average_score -61.2
    episode 104 score -35.5 average_score -61.1
    episode 105 score -61.2 average_score -60.9
    episode 106 score -54.2 average_score -60.8
    episode 107 score -94.2 average_score -61.4
    episode 108 score -115.5 average_score -61.6
    episode 109 score -87.8 average_score -61.7
    episode 110 score -142.5 average_score -61.9
    episode 111 score -111.8 average_score -62.3
    episode 112 score -79.5 average_score -62.4
    episode 113 score -43.2 average_score -62.2
    episode 114 score -85.0 average_score -62.2
    episode 115 score -57.5 average_score -62.3
    episode 116 score -48.8 average_score -61.9
    episode 117 score -95.0 average_score -62.4
    episode 118 score -49.0 average_score -62.2
    episode 119 score -60.0 average_score -62.2
    episode 120 score -101.8 average_score -62.6
    episode 121 score -54.8 average_score -62.8
    episode 122 score -91.8 average_score -63.1
    episode 123 score -43.8 average_score -62.8
    episode 124 score -56.5 average_score -62.4
    episode 125 score -62.0 average_score -62.4
    episode 126 score -74.2 average_score -62.5
    episode 127 score -73.5 average_score -62.6
    episode 128 score -63.5 average_score -62.8
    episode 129 score -54.2 average_score -62.7
    episode 130 score -62.8 average_score -62.7
    episode 131 score -74.8 average_score -62.7
    episode 132 score -57.8 average_score -62.5
    episode 133 score -71.8 average_score -62.6
    episode 134 score -74.0 average_score -62.8
    episode 135 score -61.8 average_score -62.9
    episode 136 score -58.2 average_score -62.6
    episode 137 score -46.5 average_score -63.1
    episode 138 score -66.0 average_score -63.1
    episode 139 score -67.5 average_score -62.8
    episode 140 score -47.0 average_score -62.7
    episode 141 score -72.2 average_score -63.0
    episode 142 score -84.0 average_score -63.2
    episode 143 score -0.8 average_score -62.7
    episode 144 score -59.0 average_score -62.9
    episode 145 score -62.8 average_score -62.9
    episode 146 score -33.2 average_score -63.0
    episode 147 score -2.0 average_score -62.2
    episode 148 score -74.0 average_score -62.6
    episode 149 score -54.5 average_score -62.6
    episode 150 score -55.2 average_score -62.4
    episode 151 score -50.2 average_score -62.9
    episode 152 score -59.0 average_score -63.3
    episode 153 score -76.5 average_score -63.3
    episode 154 score -53.5 average_score -63.0
    episode 155 score -53.5 average_score -63.0
    episode 156 score -68.8 average_score -62.8
    episode 157 score -65.2 average_score -62.7
    episode 158 score -74.2 average_score -62.5
    episode 159 score -72.2 average_score -62.7
    episode 160 score -73.5 average_score -63.0
    episode 161 score -46.8 average_score -62.9
    episode 162 score -78.0 average_score -63.1
    episode 163 score -51.0 average_score -63.3
    episode 164 score -60.5 average_score -63.3
    episode 165 score -60.5 average_score -63.3
    episode 166 score -57.5 average_score -63.9
    episode 167 score -46.5 average_score -63.6
    episode 168 score -6.0 average_score -62.9
    episode 169 score -34.0 average_score -62.7
    episode 170 score -103.0 average_score -63.1
    episode 171 score -65.2 average_score -62.8
    episode 172 score -40.2 average_score -62.6
    episode 173 score -76.0 average_score -62.8
    episode 174 score -64.0 average_score -63.0
    episode 175 score -80.5 average_score -63.2
    episode 176 score -45.8 average_score -62.8
    episode 177 score -50.0 average_score -62.5
    episode 178 score -41.5 average_score -62.3
    episode 179 score -3.0 average_score -61.8
    episode 180 score -71.2 average_score -61.8
    episode 181 score -70.0 average_score -61.8
    episode 182 score -61.8 average_score -61.8
    episode 183 score -55.5 average_score -61.8
    episode 184 score -51.2 average_score -61.8
    episode 185 score -15.2 average_score -61.4
    episode 186 score -48.8 average_score -61.3
    episode 187 score -59.8 average_score -61.4
    episode 188 score -68.8 average_score -61.3
    episode 189 score -56.8 average_score -61.4
    episode 190 score -55.5 average_score -61.4
    episode 191 score -89.2 average_score -61.8
    episode 192 score -58.2 average_score -61.8
    episode 193 score -55.8 average_score -62.0
    episode 194 score -66.5 average_score -61.7
    episode 195 score -65.5 average_score -62.0
    episode 196 score -59.0 average_score -62.0
    episode 197 score -47.2 average_score -62.1
    episode 198 score -69.2 average_score -62.1
    episode 199 score -63.0 average_score -62.1
    episode 200 score -39.8 average_score -61.9
    episode 201 score -57.8 average_score -61.6
    episode 202 score -28.2 average_score -61.2
    episode 203 score -69.0 average_score -61.2
    episode 204 score -46.2 average_score -61.3
    episode 205 score -60.0 average_score -61.3
    episode 206 score -31.2 average_score -61.1
    episode 207 score -68.2 average_score -60.8
    episode 208 score -68.5 average_score -60.4
    episode 209 score -54.5 average_score -60.0
    episode 210 score -41.2 average_score -59.0
    episode 211 score -19.5 average_score -58.1
    episode 212 score -65.2 average_score -57.9
    episode 213 score -61.2 average_score -58.1
    episode 214 score -62.8 average_score -57.9
    episode 215 score -69.5 average_score -58.0
    episode 216 score -80.0 average_score -58.3
    episode 217 score -48.8 average_score -57.9
    episode 218 score -71.2 average_score -58.1
    episode 219 score -52.8 average_score -58.0
    episode 220 score -76.0 average_score -57.8
    episode 221 score -47.0 average_score -57.7
    episode 222 score -59.5 average_score -57.4
    episode 223 score -78.8 average_score -57.7
    episode 224 score -29.0 average_score -57.4
    episode 225 score -63.0 average_score -57.5
    episode 226 score -8.8 average_score -56.8
    episode 227 score -83.2 average_score -56.9
    episode 228 score -60.5 average_score -56.9
    episode 229 score -60.0 average_score -56.9
    episode 230 score -80.2 average_score -57.1
    episode 231 score -56.8 average_score -56.9
    episode 232 score -64.2 average_score -57.0
    episode 233 score -28.8 average_score -56.6
    episode 234 score -79.8 average_score -56.6
    episode 235 score -65.0 average_score -56.6
    episode 236 score -66.8 average_score -56.7
    episode 237 score -51.5 average_score -56.8
    episode 238 score -81.5 average_score -56.9
    episode 239 score -67.2 average_score -56.9
    episode 240 score -52.2 average_score -57.0
    episode 241 score -71.2 average_score -57.0
    episode 242 score -48.5 average_score -56.6
    episode 243 score -58.2 average_score -57.2
    episode 244 score -6.0 average_score -56.7
    episode 245 score -45.2 average_score -56.5
    episode 246 score -38.5 average_score -56.5
    episode 247 score -52.2 average_score -57.0
    episode 248 score -50.0 average_score -56.8
    episode 249 score -54.5 average_score -56.8
    episode 250 score -72.0 average_score -57.0
    episode 251 score -49.5 average_score -57.0
    episode 252 score -48.0 average_score -56.9
    episode 253 score -35.0 average_score -56.4
    episode 254 score -36.2 average_score -56.3
    episode 255 score -74.0 average_score -56.5
    episode 256 score -31.5 average_score -56.1
    episode 257 score -35.0 average_score -55.8
    episode 258 score -61.5 average_score -55.7
    episode 259 score -11.5 average_score -55.1
    episode 260 score -110.2 average_score -55.4
    episode 261 score -106.2 average_score -56.0
    episode 262 score 2.5 average_score -55.2
    episode 263 score -59.2 average_score -55.3
    episode 264 score -65.5 average_score -55.4
    episode 265 score -56.0 average_score -55.3
    episode 266 score -55.0 average_score -55.3
    episode 267 score -47.2 average_score -55.3
    episode 268 score -31.5 average_score -55.5
    episode 269 score -73.8 average_score -55.9
    episode 270 score -87.5 average_score -55.8
    episode 271 score -77.5 average_score -55.9
    episode 272 score -20.0 average_score -55.7
    episode 273 score -39.0 average_score -55.3
    episode 274 score -64.8 average_score -55.3
    episode 275 score -36.2 average_score -54.9
    episode 276 score -61.0 average_score -55.1
    episode 277 score -63.5 average_score -55.2
    episode 278 score -37.5 average_score -55.1
    episode 279 score -45.5 average_score -55.6
    episode 280 score -42.5 average_score -55.3
    episode 281 score -64.5 average_score -55.2
    episode 282 score -60.2 average_score -55.2
    episode 283 score -64.5 average_score -55.3
    episode 284 score -83.2 average_score -55.6
    episode 285 score -43.0 average_score -55.9
    episode 286 score -59.2 average_score -56.0
    episode 287 score -33.2 average_score -55.7
    episode 288 score -40.5 average_score -55.5
    episode 289 score -70.2 average_score -55.6
    episode 290 score -65.8 average_score -55.7
    episode 291 score -57.5 average_score -55.4
    episode 292 score -60.8 average_score -55.4
    episode 293 score -77.2 average_score -55.6
    episode 294 score -70.8 average_score -55.7
    episode 295 score -88.8 average_score -55.9
    episode 296 score -109.8 average_score -56.4
    episode 297 score -50.8 average_score -56.4
    episode 298 score -41.5 average_score -56.2
    episode 299 score -55.5 average_score -56.1
    episode 300 score -52.2 average_score -56.2
    episode 301 score -66.5 average_score -56.3
    episode 302 score -65.0 average_score -56.7
    episode 303 score -68.8 average_score -56.7
    episode 304 score -52.8 average_score -56.7
    episode 305 score -24.2 average_score -56.4
    episode 306 score -15.2 average_score -56.2
    episode 307 score -41.2 average_score -55.9
    episode 308 score -73.0 average_score -56.0
    episode 309 score -35.5 average_score -55.8
    episode 310 score -65.5 average_score -56.0
    episode 311 score -56.0 average_score -56.4
    episode 312 score -44.5 average_score -56.2
    episode 313 score -59.8 average_score -56.2
    episode 314 score -82.8 average_score -56.4
    episode 315 score -62.5 average_score -56.3
    episode 316 score -52.0 average_score -56.0
    episode 317 score -46.2 average_score -56.0
    episode 318 score -83.5 average_score -56.1
    episode 319 score -50.8 average_score -56.1
    episode 320 score -51.0 average_score -55.9
    episode 321 score -17.8 average_score -55.6
    episode 322 score -43.0 average_score -55.4
    episode 323 score -79.0 average_score -55.4
    episode 324 score -59.2 average_score -55.7
    episode 325 score -12.0 average_score -55.2
    episode 326 score -48.5 average_score -55.6
    episode 327 score -67.0 average_score -55.4
    episode 328 score -13.8 average_score -55.0
    episode 329 score -8.2 average_score -54.4
    episode 330 score -53.2 average_score -54.2
    episode 331 score -43.8 average_score -54.0
    episode 332 score -53.0 average_score -53.9
    episode 333 score -54.0 average_score -54.2
    episode 334 score -60.8 average_score -54.0
    episode 335 score -54.5 average_score -53.9
    episode 336 score -64.2 average_score -53.9
    episode 337 score -49.0 average_score -53.8
    episode 338 score -59.8 average_score -53.6
    episode 339 score -68.8 average_score -53.6
    episode 340 score -41.5 average_score -53.5
    episode 341 score -39.8 average_score -53.2
    episode 342 score -68.0 average_score -53.4
    episode 343 score -57.0 average_score -53.4
    episode 344 score 1.2 average_score -53.3
    episode 345 score -55.8 average_score -53.4
    episode 346 score -90.2 average_score -53.9
    episode 347 score -52.0 average_score -53.9
    episode 348 score -69.5 average_score -54.1
    episode 349 score -48.2 average_score -54.1
    episode 350 score -79.5 average_score -54.2
    episode 351 score -68.8 average_score -54.3
    episode 352 score -67.2 average_score -54.5
    episode 353 score -38.8 average_score -54.6
    episode 354 score -60.0 average_score -54.8
    episode 355 score -57.2 average_score -54.6
    episode 356 score -53.8 average_score -54.9
    episode 357 score -27.0 average_score -54.8
    episode 358 score -53.0 average_score -54.7
    episode 359 score -59.5 average_score -55.2
    episode 360 score -73.0 average_score -54.8
    episode 361 score -59.0 average_score -54.3
    episode 362 score -57.8 average_score -54.9
    episode 363 score -45.5 average_score -54.8
    episode 364 score -35.5 average_score -54.5
    episode 365 score -46.0 average_score -54.4
    episode 366 score -78.8 average_score -54.6
    episode 367 score -90.8 average_score -55.1
    episode 368 score -48.5 average_score -55.2
    episode 369 score -63.0 average_score -55.1
    episode 370 score -39.8 average_score -54.7
    episode 371 score -42.0 average_score -54.3
    episode 372 score -29.0 average_score -54.4
    episode 373 score -26.8 average_score -54.3
    episode 374 score -58.2 average_score -54.2
    episode 375 score -82.8 average_score -54.7
    episode 376 score -71.2 average_score -54.8
    episode 377 score -62.0 average_score -54.8
    episode 378 score -51.5 average_score -54.9
    episode 379 score -52.8 average_score -55.0
    episode 380 score -71.2 average_score -55.3
    episode 381 score -69.8 average_score -55.3
    episode 382 score -68.2 average_score -55.4
    episode 383 score -58.8 average_score -55.3
    episode 384 score -63.0 average_score -55.1
    episode 385 score -44.0 average_score -55.1
    episode 386 score -5.2 average_score -54.6
    episode 387 score -27.5 average_score -54.5
    episode 388 score -59.8 average_score -54.7
    episode 389 score -43.0 average_score -54.5
    episode 390 score -59.0 average_score -54.4
    episode 391 score -67.5 average_score -54.5
    episode 392 score -68.5 average_score -54.6
    episode 393 score -48.5 average_score -54.3
    episode 394 score -48.0 average_score -54.1
    episode 395 score -66.2 average_score -53.8
    episode 396 score -61.2 average_score -53.4
    episode 397 score -55.5 average_score -53.4
    episode 398 score -40.5 average_score -53.4
    episode 399 score -64.5 average_score -53.5
    episode 400 score -56.5 average_score -53.5
    episode 401 score -29.8 average_score -53.2
    episode 402 score -50.5 average_score -53.0
    episode 403 score -12.2 average_score -52.4
    episode 404 score -47.2 average_score -52.4
    episode 405 score -47.8 average_score -52.6
    episode 406 score -82.8 average_score -53.3
    episode 407 score -13.8 average_score -53.0
    episode 408 score -36.8 average_score -52.7
    episode 409 score -48.0 average_score -52.8
    episode 410 score -41.2 average_score -52.5
    episode 411 score -34.2 average_score -52.3
    episode 412 score -67.2 average_score -52.6
    episode 413 score -50.8 average_score -52.5
    episode 414 score -54.2 average_score -52.2
    episode 415 score -57.0 average_score -52.1
    episode 416 score -70.5 average_score -52.3
    episode 417 score -52.2 average_score -52.4
    episode 418 score -77.8 average_score -52.3
    episode 419 score -39.0 average_score -52.2
    episode 420 score -41.5 average_score -52.1
    episode 421 score -27.8 average_score -52.2
    episode 422 score -38.8 average_score -52.2
    episode 423 score -48.5 average_score -51.9
    episode 424 score -17.2 average_score -51.4
    episode 425 score -64.0 average_score -52.0
    episode 426 score -58.0 average_score -52.0
    episode 427 score -44.8 average_score -51.8
    episode 428 score -3.8 average_score -51.7
    episode 429 score -49.0 average_score -52.1
    episode 430 score -40.0 average_score -52.0
    episode 431 score -36.0 average_score -51.9
    episode 432 score -21.5 average_score -51.6
    episode 433 score -50.0 average_score -51.6
    episode 434 score -47.0 average_score -51.4
    episode 435 score -52.5 average_score -51.4
    episode 436 score -67.8 average_score -51.4
    episode 437 score -39.2 average_score -51.3
    episode 438 score -68.5 average_score -51.4
    episode 439 score -59.8 average_score -51.3
    episode 440 score -31.5 average_score -51.2
    episode 441 score -56.5 average_score -51.4
    episode 442 score -66.5 average_score -51.4
    episode 443 score -40.2 average_score -51.2
    episode 444 score -67.8 average_score -51.9
    episode 445 score -55.2 average_score -51.9
    episode 446 score -73.8 average_score -51.7
    episode 447 score -41.0 average_score -51.6
    episode 448 score -63.5 average_score -51.6
    episode 449 score -47.5 average_score -51.6
    episode 450 score -22.5 average_score -51.0
    episode 451 score -53.8 average_score -50.9
    episode 452 score -67.8 average_score -50.9
    episode 453 score -63.0 average_score -51.1
    episode 454 score -62.2 average_score -51.1
    episode 455 score -67.5 average_score -51.2
    episode 456 score -48.8 average_score -51.2
    episode 457 score -62.0 average_score -51.5
    episode 458 score -57.5 average_score -51.6
    episode 459 score -15.8 average_score -51.1
    episode 460 score -26.8 average_score -50.7
    episode 461 score -37.0 average_score -50.4
    episode 462 score -17.5 average_score -50.0
    episode 463 score -52.8 average_score -50.1
    episode 464 score -58.8 average_score -50.4
    episode 465 score -49.0 average_score -50.4
    episode 466 score -41.0 average_score -50.0
    episode 467 score -42.8 average_score -49.5
    episode 468 score -86.2 average_score -49.9
    episode 469 score -71.0 average_score -50.0
    episode 470 score -44.2 average_score -50.0
    episode 471 score -52.0 average_score -50.1
    episode 472 score -54.2 average_score -50.4
    episode 473 score -65.5 average_score -50.8
    episode 474 score -60.5 average_score -50.8
    episode 475 score -47.2 average_score -50.4
    episode 476 score -57.0 average_score -50.3
    episode 477 score -54.5 average_score -50.2
    episode 478 score -23.0 average_score -49.9
    episode 479 score -49.2 average_score -49.9
    episode 480 score -28.8 average_score -49.5
    episode 481 score -45.8 average_score -49.2
    episode 482 score -29.2 average_score -48.8
    episode 483 score -49.0 average_score -48.7
    episode 484 score -35.8 average_score -48.5
    episode 485 score -41.5 average_score -48.4
    episode 486 score -61.8 average_score -49.0
    episode 487 score -54.8 average_score -49.3
    episode 488 score -36.0 average_score -49.0
    episode 489 score -51.5 average_score -49.1
    episode 490 score -63.5 average_score -49.2
    episode 491 score -70.0 average_score -49.2
    episode 492 score -46.2 average_score -49.0
    episode 493 score -66.5 average_score -49.2
    episode 494 score -53.2 average_score -49.2
    episode 495 score -15.5 average_score -48.7
    episode 496 score -35.2 average_score -48.4
    episode 497 score -78.5 average_score -48.7
    episode 498 score -32.5 average_score -48.6
    episode 499 score -43.5 average_score -48.4
    episode 500 score -36.0 average_score -48.2
    episode 501 score -56.5 average_score -48.4
    episode 502 score -42.8 average_score -48.4
    episode 503 score -56.8 average_score -48.8
    episode 504 score -28.0 average_score -48.6
    episode 505 score -44.8 average_score -48.6
    episode 506 score -49.2 average_score -48.3
    episode 507 score -52.2 average_score -48.6
    episode 508 score -62.5 average_score -48.9
    episode 509 score -53.0 average_score -48.9
    episode 510 score -36.0 average_score -48.9
    episode 511 score -47.8 average_score -49.0
    episode 512 score -55.0 average_score -48.9
    episode 513 score -57.5 average_score -49.0
    episode 514 score -48.8 average_score -48.9
    episode 515 score -69.0 average_score -49.0
    episode 516 score -1.2 average_score -48.3
    episode 517 score -60.8 average_score -48.4
    episode 518 score -76.5 average_score -48.4
    episode 519 score -27.8 average_score -48.3
    episode 520 score -42.5 average_score -48.3
    episode 521 score -69.2 average_score -48.7
    episode 522 score -52.8 average_score -48.9
    episode 523 score -56.5 average_score -49.0
    episode 524 score -43.0 average_score -49.2
    episode 525 score -67.8 average_score -49.2
    episode 526 score -63.0 average_score -49.3
    episode 527 score -55.2 average_score -49.4
    episode 528 score -64.2 average_score -50.0
    episode 529 score -23.0 average_score -49.7
    episode 530 score -9.0 average_score -49.4
    episode 531 score -75.8 average_score -49.8
    episode 532 score -35.8 average_score -50.0
    episode 533 score -37.0 average_score -49.8
    episode 534 score -38.5 average_score -49.8
    episode 535 score -55.5 average_score -49.8
    episode 536 score -44.5 average_score -49.6
    episode 537 score -26.0 average_score -49.4
    episode 538 score -43.0 average_score -49.2
    episode 539 score -9.5 average_score -48.7
    episode 540 score -19.8 average_score -48.6
    episode 541 score -65.0 average_score -48.6
    episode 542 score -56.0 average_score -48.5
    episode 543 score -47.0 average_score -48.6
    episode 544 score -40.0 average_score -48.3
    episode 545 score -51.0 average_score -48.3
    episode 546 score -21.8 average_score -47.8
    episode 547 score -31.0 average_score -47.7
    episode 548 score -38.2 average_score -47.4
    episode 549 score -5.5 average_score -47.0
    episode 550 score -40.0 average_score -47.2
    episode 551 score -40.0 average_score -47.0
    episode 552 score -59.2 average_score -46.9
    episode 553 score -45.8 average_score -46.8
    episode 554 score -45.5 average_score -46.6
    episode 555 score -27.5 average_score -46.2
    episode 556 score -35.8 average_score -46.1
    episode 557 score -25.2 average_score -45.7
    episode 558 score -33.0 average_score -45.5
    episode 559 score -37.8 average_score -45.7
    episode 560 score -43.0 average_score -45.8
    episode 561 score -56.2 average_score -46.0
    episode 562 score -69.2 average_score -46.5
    episode 563 score -43.0 average_score -46.5
    episode 564 score -71.0 average_score -46.6
    episode 565 score -43.8 average_score -46.5
    episode 566 score -50.5 average_score -46.6
    episode 567 score -42.2 average_score -46.6
    episode 568 score -28.8 average_score -46.0
    episode 569 score -72.5 average_score -46.1
    episode 570 score -38.0 average_score -46.0
    episode 571 score -58.2 average_score -46.1
    episode 572 score -21.2 average_score -45.7
    episode 573 score -45.0 average_score -45.5
    episode 574 score -33.0 average_score -45.2
    episode 575 score -40.2 average_score -45.2
    episode 576 score -48.5 average_score -45.1
    episode 577 score -34.0 average_score -44.9
    episode 578 score -56.5 average_score -45.2
    episode 579 score -28.0 average_score -45.0
    episode 580 score -41.2 average_score -45.1
    episode 581 score -75.0 average_score -45.4
    episode 582 score -35.5 average_score -45.5
    episode 583 score -39.2 average_score -45.4
    episode 584 score -35.0 average_score -45.4
    episode 585 score -43.5 average_score -45.4
    episode 586 score -66.5 average_score -45.4
    episode 587 score -65.2 average_score -45.6
    episode 588 score -34.8 average_score -45.5
    episode 589 score -57.5 average_score -45.6
    episode 590 score -48.5 average_score -45.5
    episode 591 score -34.0 average_score -45.1
    episode 592 score -49.0 average_score -45.1
    episode 593 score -34.5 average_score -44.8
    episode 594 score -45.0 average_score -44.7
    episode 595 score -35.5 average_score -44.9
    episode 596 score -69.0 average_score -45.3
    episode 597 score -42.2 average_score -44.9
    episode 598 score -28.8 average_score -44.9
    episode 599 score -40.2 average_score -44.8
    episode 600 score -75.5 average_score -45.2
    episode 601 score -49.5 average_score -45.1
    episode 602 score -30.8 average_score -45.0
    episode 603 score -21.8 average_score -44.7
    episode 604 score -47.2 average_score -44.9
    episode 605 score -10.5 average_score -44.5
    episode 606 score -55.0 average_score -44.6
    episode 607 score -48.2 average_score -44.5
    episode 608 score -42.5 average_score -44.3
    episode 609 score -45.0 average_score -44.3
    episode 610 score -68.8 average_score -44.6
    episode 611 score -40.8 average_score -44.5
    episode 612 score -24.2 average_score -44.2
    episode 613 score -34.8 average_score -44.0
    episode 614 score -60.5 average_score -44.1
    episode 615 score -59.0 average_score -44.0
    episode 616 score -60.2 average_score -44.6
    episode 617 score -75.5 average_score -44.7
    episode 618 score -29.8 average_score -44.3
    episode 619 score -33.5 average_score -44.3
    episode 620 score -40.2 average_score -44.3
    episode 621 score -50.2 average_score -44.1
    episode 622 score -49.8 average_score -44.1
    episode 623 score -54.2 average_score -44.1
    episode 624 score -31.0 average_score -43.9
    episode 625 score -59.2 average_score -43.9
    episode 626 score -49.0 average_score -43.7
    episode 627 score -32.2 average_score -43.5
    episode 628 score -69.0 average_score -43.5
    episode 629 score -28.8 average_score -43.6
    episode 630 score -57.5 average_score -44.1
    episode 631 score -51.8 average_score -43.8
    episode 632 score -67.8 average_score -44.2
    episode 633 score -2.2 average_score -43.8
    episode 634 score -36.8 average_score -43.8
    episode 635 score -50.2 average_score -43.7
    episode 636 score -46.0 average_score -43.8
    episode 637 score -40.5 average_score -43.9
    episode 638 score -52.2 average_score -44.0
    episode 639 score -9.5 average_score -44.0
    episode 640 score -54.0 average_score -44.3
    episode 641 score -55.2 average_score -44.2
    episode 642 score -44.0 average_score -44.1
    episode 643 score -28.0 average_score -43.9
    episode 644 score -40.8 average_score -43.9
    episode 645 score -20.0 average_score -43.6
    episode 646 score -34.8 average_score -43.8
    episode 647 score -89.0 average_score -44.3
    episode 648 score -71.8 average_score -44.7
    episode 649 score -32.5 average_score -44.9
    episode 650 score -52.8 average_score -45.1
    episode 651 score -77.0 average_score -45.4
    episode 652 score -36.2 average_score -45.2
    episode 653 score -57.0 average_score -45.3
    episode 654 score -49.0 average_score -45.4
    episode 655 score -24.8 average_score -45.3
    episode 656 score -45.2 average_score -45.4
    episode 657 score -47.5 average_score -45.6
    episode 658 score -66.2 average_score -46.0
    episode 659 score -23.0 average_score -45.8
    episode 660 score -27.2 average_score -45.7
    episode 661 score -57.2 average_score -45.7
    episode 662 score -51.8 average_score -45.5
    episode 663 score -56.8 average_score -45.6
    episode 664 score -33.8 average_score -45.3
    episode 665 score -49.0 average_score -45.3
    episode 666 score -9.8 average_score -44.9
    episode 667 score -30.0 average_score -44.8
    episode 668 score -37.2 average_score -44.9
    episode 669 score -43.2 average_score -44.6
    episode 670 score -52.2 average_score -44.7
    episode 671 score -39.0 average_score -44.5
    episode 672 score -54.2 average_score -44.9
    episode 673 score -58.2 average_score -45.0
    episode 674 score -48.0 average_score -45.2
    episode 675 score -39.5 average_score -45.1
    episode 676 score -54.5 average_score -45.2
    episode 677 score -45.0 average_score -45.3
    episode 678 score -63.2 average_score -45.4
    episode 679 score -44.5 average_score -45.5
    episode 680 score -43.0 average_score -45.6
    episode 681 score -54.2 average_score -45.4
    episode 682 score -33.8 average_score -45.3
    episode 683 score -36.2 average_score -45.3
    episode 684 score -13.0 average_score -45.1
    episode 685 score -70.0 average_score -45.4
    episode 686 score -31.2 average_score -45.0
    episode 687 score -54.0 average_score -44.9
    episode 688 score -44.2 average_score -45.0
    episode 689 score -49.2 average_score -44.9
    episode 690 score -59.2 average_score -45.0
    episode 691 score -78.2 average_score -45.5
    episode 692 score -46.2 average_score -45.4
    episode 693 score -56.2 average_score -45.6
    episode 694 score -48.5 average_score -45.7
    episode 695 score -39.0 average_score -45.7
    episode 696 score -39.2 average_score -45.4
    episode 697 score -65.2 average_score -45.6
    episode 698 score -42.8 average_score -45.8
    episode 699 score -53.0 average_score -45.9
    episode 700 score -68.5 average_score -45.8
    episode 701 score -46.2 average_score -45.8
    episode 702 score -49.5 average_score -46.0
    episode 703 score -1.2 average_score -45.8
    episode 704 score -36.8 average_score -45.7
    episode 705 score -31.2 average_score -45.9
    episode 706 score -71.2 average_score -46.1
    episode 707 score -45.0 average_score -46.0
    episode 708 score -22.5 average_score -45.8
    episode 709 score -39.2 average_score -45.8
    episode 710 score -53.8 average_score -45.6
    episode 711 score -21.0 average_score -45.4
    episode 712 score -20.0 average_score -45.4
    episode 713 score -49.2 average_score -45.5
    episode 714 score -33.8 average_score -45.3
    episode 715 score -48.5 average_score -45.1
    episode 716 score -22.0 average_score -44.8
    episode 717 score -7.5 average_score -44.1
    episode 718 score -39.8 average_score -44.2
    episode 719 score -9.0 average_score -43.9
    episode 720 score -28.8 average_score -43.8
    episode 721 score -78.2 average_score -44.1
    episode 722 score -10.8 average_score -43.7
    episode 723 score -51.8 average_score -43.7
    episode 724 score -68.0 average_score -44.1
    episode 725 score -52.0 average_score -44.0
    episode 726 score -56.5 average_score -44.1
    episode 727 score -30.2 average_score -44.0
    episode 728 score -65.0 average_score -44.0
    episode 729 score -29.2 average_score -44.0
    episode 730 score -29.5 average_score -43.7
    episode 731 score -43.0 average_score -43.6
    episode 732 score -40.0 average_score -43.4
    episode 733 score -48.2 average_score -43.8
    episode 734 score -45.5 average_score -43.9
    episode 735 score -28.5 average_score -43.7
    episode 736 score -32.8 average_score -43.6
    episode 737 score -49.2 average_score -43.6
    episode 738 score -39.0 average_score -43.5
    episode 739 score -17.2 average_score -43.6
    episode 740 score -47.5 average_score -43.5
    episode 741 score -5.8 average_score -43.0
    episode 742 score -18.5 average_score -42.8
    episode 743 score -56.8 average_score -43.1
    episode 744 score -54.0 average_score -43.2
    episode 745 score -21.5 average_score -43.2
    episode 746 score -58.5 average_score -43.5
    episode 747 score -50.8 average_score -43.1
    episode 748 score -21.5 average_score -42.6
    episode 749 score -70.0 average_score -42.9
    episode 750 score -61.2 average_score -43.0
    episode 751 score -14.0 average_score -42.4
    episode 752 score -32.0 average_score -42.4
    episode 753 score -35.8 average_score -42.1
    episode 754 score -49.8 average_score -42.1
    episode 755 score -67.2 average_score -42.6
    episode 756 score -40.5 average_score -42.5
    episode 757 score -49.5 average_score -42.5
    episode 758 score -40.5 average_score -42.3
    episode 759 score -41.8 average_score -42.5
    episode 760 score -22.5 average_score -42.4
    episode 761 score -27.2 average_score -42.1
    episode 762 score -20.2 average_score -41.8
    episode 763 score -14.5 average_score -41.4
    episode 764 score -65.8 average_score -41.7
    episode 765 score -50.5 average_score -41.7
    episode 766 score -48.5 average_score -42.1
    episode 767 score -43.5 average_score -42.2
    episode 768 score -71.0 average_score -42.6
    episode 769 score -40.8 average_score -42.6
    episode 770 score -35.5 average_score -42.4
    episode 771 score -72.5 average_score -42.7
    episode 772 score -48.8 average_score -42.7
    episode 773 score -18.5 average_score -42.3
    episode 774 score -46.0 average_score -42.3
    episode 775 score 5.5 average_score -41.8
    episode 776 score -20.0 average_score -41.5
    episode 777 score -26.8 average_score -41.3
    episode 778 score -58.0 average_score -41.2
    episode 779 score -20.2 average_score -41.0
    episode 780 score -49.2 average_score -41.0
    episode 781 score -53.8 average_score -41.0
    episode 782 score -59.0 average_score -41.3
    episode 783 score -34.8 average_score -41.3
    episode 784 score -47.0 average_score -41.6
    episode 785 score -64.5 average_score -41.6
    episode 786 score -30.8 average_score -41.6
    episode 787 score -24.8 average_score -41.3
    episode 788 score -25.2 average_score -41.1
    episode 789 score -18.2 average_score -40.8
    episode 790 score -37.2 average_score -40.5
    episode 791 score -13.8 average_score -39.9
    episode 792 score -39.5 average_score -39.8
    episode 793 score -36.2 average_score -39.6
    episode 794 score -37.2 average_score -39.5
    episode 795 score -57.0 average_score -39.7
    episode 796 score -19.8 average_score -39.5
    episode 797 score -22.8 average_score -39.1
    episode 798 score -46.2 average_score -39.1
    episode 799 score -3.8 average_score -38.6
    episode 800 score -34.8 average_score -38.3
    episode 801 score -42.5 average_score -38.2
    episode 802 score -2.8 average_score -37.8
    episode 803 score -39.2 average_score -38.2
    episode 804 score -18.0 average_score -38.0
    episode 805 score -42.2 average_score -38.1
    episode 806 score -62.8 average_score -38.0
    episode 807 score -50.8 average_score -38.1
    episode 808 score -42.8 average_score -38.3
    episode 809 score -39.2 average_score -38.3
    episode 810 score -43.2 average_score -38.2
    episode 811 score -23.8 average_score -38.2
    episode 812 score -34.8 average_score -38.3
    episode 813 score -29.0 average_score -38.1
    episode 814 score -54.5 average_score -38.3
    episode 815 score -58.2 average_score -38.4
    episode 816 score -24.0 average_score -38.5
    episode 817 score -28.0 average_score -38.7
    episode 818 score -41.0 average_score -38.7
    episode 819 score -32.2 average_score -38.9
    episode 820 score -39.8 average_score -39.0
    episode 821 score -41.5 average_score -38.6
    episode 822 score -13.5 average_score -38.7
    episode 823 score -47.2 average_score -38.6
    episode 824 score -43.0 average_score -38.4
    episode 825 score -29.2 average_score -38.1
    episode 826 score -35.2 average_score -37.9
    episode 827 score -44.2 average_score -38.1
    episode 828 score -47.0 average_score -37.9
    episode 829 score -13.0 average_score -37.7
    episode 830 score -40.0 average_score -37.8
    episode 831 score -55.0 average_score -38.0
    episode 832 score -18.5 average_score -37.7
    episode 833 score -22.8 average_score -37.5
    episode 834 score -28.0 average_score -37.3
    episode 835 score -33.8 average_score -37.4
    episode 836 score -50.8 average_score -37.5
    episode 837 score -23.2 average_score -37.3
    episode 838 score 0.5 average_score -36.9
    episode 839 score -43.0 average_score -37.1
    episode 840 score -25.8 average_score -36.9
    episode 841 score -38.5 average_score -37.3
    episode 842 score -49.0 average_score -37.6
    episode 843 score -13.2 average_score -37.1
    episode 844 score -34.2 average_score -36.9
    episode 845 score -49.8 average_score -37.2
    episode 846 score -43.8 average_score -37.1
    episode 847 score -46.0 average_score -37.0
    episode 848 score -47.8 average_score -37.3
    episode 849 score -42.0 average_score -37.0
    episode 850 score -42.5 average_score -36.8
    episode 851 score -31.2 average_score -37.0
    episode 852 score -37.8 average_score -37.0
    episode 853 score -42.8 average_score -37.1
    episode 854 score -23.0 average_score -36.8
    episode 855 score -7.5 average_score -36.2
    episode 856 score -31.2 average_score -36.2
    episode 857 score -27.8 average_score -35.9
    episode 858 score -12.5 average_score -35.7
    episode 859 score -31.8 average_score -35.6
    episode 860 score -53.5 average_score -35.9
    episode 861 score -56.0 average_score -36.2
    episode 862 score -54.2 average_score -36.5
    episode 863 score -42.2 average_score -36.8
    episode 864 score -29.0 average_score -36.4
    episode 865 score -63.5 average_score -36.5
    episode 866 score -37.8 average_score -36.4
    episode 867 score -44.2 average_score -36.4
    episode 868 score -34.8 average_score -36.1
    episode 869 score -19.0 average_score -35.9
    episode 870 score -21.8 average_score -35.7
    episode 871 score -30.8 average_score -35.3
    episode 872 score -49.0 average_score -35.3
    episode 873 score -22.2 average_score -35.3
    episode 874 score -1.2 average_score -34.9
    episode 875 score -47.5 average_score -35.4
    episode 876 score -52.8 average_score -35.8
    episode 877 score -21.8 average_score -35.7
    episode 878 score -45.0 average_score -35.6
    episode 879 score -66.0 average_score -36.0
    episode 880 score 2.0 average_score -35.5
    episode 881 score 12.2 average_score -34.9
    episode 882 score -36.5 average_score -34.6
    episode 883 score -10.2 average_score -34.4
    episode 884 score -41.8 average_score -34.3
    episode 885 score -21.2 average_score -33.9
    episode 886 score -22.8 average_score -33.8
    episode 887 score -21.8 average_score -33.8
    episode 888 score -41.8 average_score -34.0
    episode 889 score -57.2 average_score -34.3
    episode 890 score -47.8 average_score -34.5
    episode 891 score 10.0 average_score -34.2
    episode 892 score -31.8 average_score -34.1
    episode 893 score -33.5 average_score -34.1
    episode 894 score -3.5 average_score -33.8
    episode 895 score -41.8 average_score -33.6
    episode 896 score -33.5 average_score -33.8
    episode 897 score -18.5 average_score -33.7
    episode 898 score -32.8 average_score -33.6
    episode 899 score -24.8 average_score -33.8
    episode 900 score -26.5 average_score -33.7
    episode 901 score -18.2 average_score -33.5
    episode 902 score -55.2 average_score -34.0
    episode 903 score -41.0 average_score -34.0
    episode 904 score -29.5 average_score -34.1
    episode 905 score -24.8 average_score -33.9
    episode 906 score -38.2 average_score -33.7
    episode 907 score -31.0 average_score -33.5
    episode 908 score -11.0 average_score -33.2
    episode 909 score -54.5 average_score -33.3
    episode 910 score -49.0 average_score -33.4
    episode 911 score 4.5 average_score -33.1
    episode 912 score -24.0 average_score -33.0
    episode 913 score -51.5 average_score -33.2
    episode 914 score -59.0 average_score -33.3
    episode 915 score -55.5 average_score -33.2
    episode 916 score -35.0 average_score -33.4
    episode 917 score -32.5 average_score -33.4
    episode 918 score -46.8 average_score -33.5
    episode 919 score -0.8 average_score -33.1
    episode 920 score -44.5 average_score -33.2
    episode 921 score -23.8 average_score -33.0
    episode 922 score 1.0 average_score -32.9
    episode 923 score -15.5 average_score -32.6
    episode 924 score -40.5 average_score -32.5
    episode 925 score -40.2 average_score -32.6
    episode 926 score -32.8 average_score -32.6
    episode 927 score -44.2 average_score -32.6
    episode 928 score -37.5 average_score -32.5
    episode 929 score -39.5 average_score -32.8
    episode 930 score -54.2 average_score -32.9
    episode 931 score -6.8 average_score -32.4
    episode 932 score -14.8 average_score -32.4
    episode 933 score -52.5 average_score -32.7
    episode 934 score -31.8 average_score -32.7
    episode 935 score -15.2 average_score -32.6
    episode 936 score -53.8 average_score -32.6
    episode 937 score -20.2 average_score -32.6
    episode 938 score -40.2 average_score -33.0
    episode 939 score -40.2 average_score -32.9
    episode 940 score -23.8 average_score -32.9
    episode 941 score -30.8 average_score -32.8
    episode 942 score -34.8 average_score -32.7
    episode 943 score -25.0 average_score -32.8
    episode 944 score -35.8 average_score -32.8
    episode 945 score -36.5 average_score -32.7
    episode 946 score -20.5 average_score -32.5
    episode 947 score -43.8 average_score -32.4
    episode 948 score -36.2 average_score -32.3
    episode 949 score -32.5 average_score -32.2
    episode 950 score 6.2 average_score -31.7
    episode 951 score -41.0 average_score -31.8
    episode 952 score -46.2 average_score -31.9
    episode 953 score -26.0 average_score -31.8
    episode 954 score -30.8 average_score -31.8
    episode 955 score -44.8 average_score -32.2
    episode 956 score -59.5 average_score -32.5
    episode 957 score -66.8 average_score -32.9
    episode 958 score -17.2 average_score -32.9
    episode 959 score -25.8 average_score -32.9
    episode 960 score -26.0 average_score -32.6
    episode 961 score -34.5 average_score -32.4
    episode 962 score -17.8 average_score -32.0
    episode 963 score -40.2 average_score -32.0
    episode 964 score -57.0 average_score -32.3
    episode 965 score -5.2 average_score -31.7
    episode 966 score -18.2 average_score -31.5
    episode 967 score -12.2 average_score -31.2
    episode 968 score -18.5 average_score -31.0
    episode 969 score -2.2 average_score -30.8
    episode 970 score -16.0 average_score -30.8
    episode 971 score -36.8 average_score -30.9
    episode 972 score -11.8 average_score -30.5
    episode 973 score -48.5 average_score -30.7
    episode 974 score -12.2 average_score -30.9
    episode 975 score -36.5 average_score -30.7
    episode 976 score -38.5 average_score -30.6
    episode 977 score -40.5 average_score -30.8
    episode 978 score -52.2 average_score -30.9
    episode 979 score -39.8 average_score -30.6
    episode 980 score -43.0 average_score -31.0
    episode 981 score -26.5 average_score -31.4
    episode 982 score -24.8 average_score -31.3
    episode 983 score -37.0 average_score -31.6
    episode 984 score -36.0 average_score -31.5
    episode 985 score -23.0 average_score -31.5
    episode 986 score -81.5 average_score -32.1
    episode 987 score -32.2 average_score -32.2
    episode 988 score -27.0 average_score -32.1
    episode 989 score -12.5 average_score -31.6
    episode 990 score -45.2 average_score -31.6
    episode 991 score -22.0 average_score -31.9
    episode 992 score -35.0 average_score -32.0
    episode 993 score -19.8 average_score -31.8
    episode 994 score -45.0 average_score -32.2
    episode 995 score -53.8 average_score -32.4
    episode 996 score -46.8 average_score -32.5
    episode 997 score -53.2 average_score -32.8
    episode 998 score -34.5 average_score -32.9
    episode 999 score -28.0 average_score -32.9
    episode 1000 score -36.8 average_score -33.0
    episode 1001 score -36.2 average_score -33.2
    episode 1002 score -27.2 average_score -32.9
    episode 1003 score -21.0 average_score -32.7
    episode 1004 score -31.5 average_score -32.7
    episode 1005 score -20.5 average_score -32.7
    episode 1006 score -37.2 average_score -32.7
    episode 1007 score -33.8 average_score -32.7
    episode 1008 score 4.0 average_score -32.5
    episode 1009 score -48.8 average_score -32.5
    episode 1010 score -37.5 average_score -32.4
    episode 1011 score -45.8 average_score -32.9
    episode 1012 score -51.2 average_score -33.1
    episode 1013 score -41.8 average_score -33.0
    episode 1014 score -1.5 average_score -32.5
    episode 1015 score 4.8 average_score -31.9
    episode 1016 score -65.0 average_score -32.2
    episode 1017 score -34.5 average_score -32.2
    episode 1018 score -6.8 average_score -31.8
    episode 1019 score -63.8 average_score -32.4
    episode 1020 score -32.5 average_score -32.3
    episode 1021 score -25.0 average_score -32.3
    episode 1022 score -16.8 average_score -32.5
    episode 1023 score -32.0 average_score -32.7
    episode 1024 score -16.8 average_score -32.4
    episode 1025 score -36.2 average_score -32.4
    episode 1026 score -31.8 average_score -32.4
    episode 1027 score -40.0 average_score -32.3
    episode 1028 score -6.0 average_score -32.0
    episode 1029 score -21.0 average_score -31.8
    episode 1030 score -45.2 average_score -31.7
    episode 1031 score -30.8 average_score -32.0
    episode 1032 score -18.2 average_score -32.0
    episode 1033 score -25.0 average_score -31.7
    episode 1034 score -30.8 average_score -31.7
    episode 1035 score -29.2 average_score -31.9
    episode 1036 score -36.2 average_score -31.7
    episode 1037 score -17.8 average_score -31.7
    episode 1038 score -58.0 average_score -31.8
    episode 1039 score -14.0 average_score -31.6
    episode 1040 score -30.2 average_score -31.6
    episode 1041 score -60.8 average_score -31.9
    episode 1042 score -53.0 average_score -32.1
    episode 1043 score -17.0 average_score -32.0
    episode 1044 score -30.8 average_score -32.0
    episode 1045 score -6.0 average_score -31.7
    episode 1046 score -20.8 average_score -31.7
    episode 1047 score -23.0 average_score -31.5
    episode 1048 score -37.2 average_score -31.5
    episode 1049 score -36.2 average_score -31.5
    episode 1050 score -23.0 average_score -31.8
    episode 1051 score -51.5 average_score -31.9
    episode 1052 score -22.8 average_score -31.7
    episode 1053 score -42.5 average_score -31.9
    episode 1054 score -15.5 average_score -31.7
    episode 1055 score -40.0 average_score -31.7
    episode 1056 score -7.0 average_score -31.1
    episode 1057 score -29.0 average_score -30.8
    episode 1058 score -62.2 average_score -31.2
    episode 1059 score -4.0 average_score -31.0
    episode 1060 score -29.8 average_score -31.0
    episode 1061 score -15.5 average_score -30.8
    episode 1062 score -36.2 average_score -31.0
    episode 1063 score -21.2 average_score -30.8
    episode 1064 score -45.8 average_score -30.7
    episode 1065 score -44.2 average_score -31.1
    episode 1066 score -30.5 average_score -31.2
    episode 1067 score -15.5 average_score -31.3
    episode 1068 score -24.5 average_score -31.3
    episode 1069 score 31.0 average_score -31.0
    episode 1070 score -26.5 average_score -31.1
    episode 1071 score -31.0 average_score -31.0
    episode 1072 score -46.2 average_score -31.4
    episode 1073 score -21.0 average_score -31.1
    episode 1074 score -41.8 average_score -31.4
    episode 1075 score -30.5 average_score -31.3
    episode 1076 score -56.0 average_score -31.5
    episode 1077 score -43.8 average_score -31.6
    episode 1078 score -50.0 average_score -31.5
    episode 1079 score -21.0 average_score -31.3
    episode 1080 score -23.5 average_score -31.1
    episode 1081 score -43.2 average_score -31.3
    episode 1082 score -36.8 average_score -31.4
    episode 1083 score -4.0 average_score -31.1
    episode 1084 score -20.5 average_score -31.0
    episode 1085 score -32.5 average_score -31.0
    episode 1086 score -27.5 average_score -30.5
    episode 1087 score -41.2 average_score -30.6
    episode 1088 score -58.8 average_score -30.9
    episode 1089 score -54.2 average_score -31.3
    episode 1090 score -17.5 average_score -31.1
    episode 1091 score -49.8 average_score -31.3
    episode 1092 score -35.5 average_score -31.3
    episode 1093 score -30.2 average_score -31.4
    episode 1094 score -17.0 average_score -31.2
    episode 1095 score -28.2 average_score -30.9
    episode 1096 score -8.2 average_score -30.5
    episode 1097 score -31.0 average_score -30.3
    episode 1098 score -37.5 average_score -30.3
    episode 1099 score -15.0 average_score -30.2
    episode 1100 score -40.2 average_score -30.2
    episode 1101 score -29.2 average_score -30.2
    episode 1102 score -43.8 average_score -30.3
    episode 1103 score -30.2 average_score -30.4
    episode 1104 score -48.5 average_score -30.6
    episode 1105 score -65.5 average_score -31.0
    episode 1106 score -34.8 average_score -31.0
    episode 1107 score -82.2 average_score -31.5
    episode 1108 score -44.2 average_score -32.0
    episode 1109 score -33.8 average_score -31.8
    episode 1110 score -30.2 average_score -31.8
    episode 1111 score -19.2 average_score -31.5
    episode 1112 score -47.5 average_score -31.5
    episode 1113 score -31.5 average_score -31.4
    episode 1114 score -69.2 average_score -32.0
    episode 1115 score -18.5 average_score -32.3
    episode 1116 score -13.0 average_score -31.7
    episode 1117 score -31.2 average_score -31.7
    episode 1118 score -48.8 average_score -32.1
    episode 1119 score -13.8 average_score -31.6
    episode 1120 score -32.2 average_score -31.6
    episode 1121 score -26.2 average_score -31.6
    episode 1122 score -34.8 average_score -31.8
    episode 1123 score -34.0 average_score -31.8
    episode 1124 score -13.5 average_score -31.8
    episode 1125 score -13.5 average_score -31.6
    episode 1126 score -32.8 average_score -31.6
    episode 1127 score -37.5 average_score -31.6
    episode 1128 score -30.0 average_score -31.8
    episode 1129 score -31.5 average_score -31.9
    episode 1130 score -3.2 average_score -31.5
    episode 1131 score -40.5 average_score -31.6
    episode 1132 score -35.5 average_score -31.8
    episode 1133 score -27.8 average_score -31.8
    episode 1134 score -43.0 average_score -31.9
    episode 1135 score -34.2 average_score -32.0
    episode 1136 score -25.8 average_score -31.9
    episode 1137 score -38.8 average_score -32.1
    episode 1138 score -41.2 average_score -31.9
    episode 1139 score -39.5 average_score -32.2
    episode 1140 score -32.5 average_score -32.2
    episode 1141 score -44.0 average_score -32.0
    episode 1142 score 2.5 average_score -31.5
    episode 1143 score -26.0 average_score -31.5
    episode 1144 score -11.2 average_score -31.4
    episode 1145 score -7.8 average_score -31.4
    episode 1146 score -55.2 average_score -31.7
    episode 1147 score -41.8 average_score -31.9
    episode 1148 score -26.0 average_score -31.8
    episode 1149 score -74.0 average_score -32.2
    episode 1150 score -9.0 average_score -32.0
    episode 1151 score -26.2 average_score -31.8
    episode 1152 score -31.0 average_score -31.9
    episode 1153 score -39.5 average_score -31.8
    episode 1154 score -28.5 average_score -32.0
    episode 1155 score -38.2 average_score -31.9
    episode 1156 score -42.0 average_score -32.3
    episode 1157 score -34.0 average_score -32.3
    episode 1158 score -20.5 average_score -31.9
    episode 1159 score -3.2 average_score -31.9
    episode 1160 score -28.8 average_score -31.9
    episode 1161 score -47.8 average_score -32.2
    episode 1162 score -54.5 average_score -32.4
    episode 1163 score -29.0 average_score -32.5
    episode 1164 score -24.8 average_score -32.3
    episode 1165 score -18.0 average_score -32.0
    episode 1166 score -14.2 average_score -31.9
    episode 1167 score -26.5 average_score -32.0
    episode 1168 score -59.0 average_score -32.3
    episode 1169 score -46.2 average_score -33.1
    episode 1170 score -8.5 average_score -32.9
    episode 1171 score -20.8 average_score -32.8
    episode 1172 score -28.2 average_score -32.6
    episode 1173 score -16.2 average_score -32.6
    episode 1174 score -35.2 average_score -32.5
    episode 1175 score -14.5 average_score -32.3
    episode 1176 score -18.2 average_score -32.0
    episode 1177 score -38.8 average_score -31.9
    episode 1178 score -37.8 average_score -31.8
    episode 1179 score -27.0 average_score -31.9
    episode 1180 score -37.8 average_score -32.0
    episode 1181 score -23.8 average_score -31.8
    episode 1182 score -44.5 average_score -31.9
    episode 1183 score -19.8 average_score -32.0
    episode 1184 score -46.8 average_score -32.3
    episode 1185 score -40.5 average_score -32.4
    episode 1186 score -2.2 average_score -32.1
    episode 1187 score -31.0 average_score -32.0
    episode 1188 score -23.0 average_score -31.7
    episode 1189 score -56.0 average_score -31.7
    episode 1190 score -33.2 average_score -31.8
    episode 1191 score -26.2 average_score -31.6
    episode 1192 score -44.5 average_score -31.7
    episode 1193 score -10.5 average_score -31.5
    episode 1194 score -30.2 average_score -31.6
    episode 1195 score -6.8 average_score -31.4
    episode 1196 score -39.8 average_score -31.7
    episode 1197 score -10.2 average_score -31.5
    episode 1198 score -21.8 average_score -31.4
    episode 1199 score -9.2 average_score -31.3
    episode 1200 score -20.5 average_score -31.1
    episode 1201 score -28.0 average_score -31.1
    episode 1202 score -32.8 average_score -31.0
    episode 1203 score -27.8 average_score -31.0
    episode 1204 score 1.2 average_score -30.5
    episode 1205 score 2.2 average_score -29.8
    episode 1206 score -13.5 average_score -29.6
    episode 1207 score -8.8 average_score -28.8
    episode 1208 score -68.8 average_score -29.1
    episode 1209 score -30.8 average_score -29.1
    episode 1210 score -25.5 average_score -29.0
    episode 1211 score -28.8 average_score -29.1
    episode 1212 score -23.5 average_score -28.9
    episode 1213 score -41.2 average_score -29.0
    episode 1214 score -13.8 average_score -28.4
    episode 1215 score -17.5 average_score -28.4
    episode 1216 score -50.5 average_score -28.8
    episode 1217 score -25.0 average_score -28.7
    episode 1218 score -38.8 average_score -28.6
    episode 1219 score -27.5 average_score -28.7
    episode 1220 score -4.2 average_score -28.5
    episode 1221 score -16.2 average_score -28.4
    episode 1222 score -41.2 average_score -28.4
    episode 1223 score -40.8 average_score -28.5
    episode 1224 score -9.5 average_score -28.5
    episode 1225 score -4.0 average_score -28.4
    episode 1226 score -36.2 average_score -28.4
    episode 1227 score -26.5 average_score -28.3
    episode 1228 score -34.0 average_score -28.3
    episode 1229 score -40.2 average_score -28.4
    episode 1230 score -28.2 average_score -28.7
    episode 1231 score -44.0 average_score -28.7
    episode 1232 score 1.2 average_score -28.3
    episode 1233 score -36.2 average_score -28.4
    episode 1234 score -1.2 average_score -28.0
    episode 1235 score -17.8 average_score -27.8
    episode 1236 score -12.8 average_score -27.7
    episode 1237 score -17.8 average_score -27.5
    episode 1238 score -22.0 average_score -27.3
    episode 1239 score -35.8 average_score -27.3
    episode 1240 score -25.8 average_score -27.2
    episode 1241 score -41.8 average_score -27.2
    episode 1242 score -44.0 average_score -27.6
    episode 1243 score -31.0 average_score -27.7
    episode 1244 score -42.8 average_score -28.0
    episode 1245 score 9.2 average_score -27.8
    episode 1246 score -36.8 average_score -27.7
    episode 1247 score -41.5 average_score -27.6
    episode 1248 score -20.8 average_score -27.6
    episode 1249 score -27.0 average_score -27.1
    episode 1250 score -27.0 average_score -27.3
    episode 1251 score -24.8 average_score -27.3
    episode 1252 score -23.8 average_score -27.2
    episode 1253 score -27.5 average_score -27.1
    episode 1254 score -15.8 average_score -27.0
    episode 1255 score -27.0 average_score -26.9
    episode 1256 score -28.5 average_score -26.7
    episode 1257 score -15.5 average_score -26.5
    episode 1258 score -15.0 average_score -26.5
    episode 1259 score -13.0 average_score -26.6
    episode 1260 score -21.8 average_score -26.5
    episode 1261 score -45.2 average_score -26.5
    episode 1262 score -23.5 average_score -26.2
    episode 1263 score -52.2 average_score -26.4
    episode 1264 score -48.8 average_score -26.6
    episode 1265 score -54.8 average_score -27.0
    episode 1266 score -39.2 average_score -27.3
    episode 1267 score -24.8 average_score -27.2
    episode 1268 score -13.2 average_score -26.8
    episode 1269 score -32.5 average_score -26.7
    episode 1270 score -37.5 average_score -26.9
    episode 1271 score -35.8 average_score -27.1
    episode 1272 score -10.2 average_score -26.9
    episode 1273 score -49.8 average_score -27.2
    episode 1274 score -31.5 average_score -27.2
    episode 1275 score -35.0 average_score -27.4
    episode 1276 score -11.5 average_score -27.4
    episode 1277 score -13.5 average_score -27.1
    episode 1278 score -30.0 average_score -27.0
    episode 1279 score -26.0 average_score -27.0
    episode 1280 score -15.5 average_score -26.8
    episode 1281 score -20.8 average_score -26.8
    episode 1282 score -39.2 average_score -26.7
    episode 1283 score -36.0 average_score -26.9
    episode 1284 score -62.2 average_score -27.0
    episode 1285 score -51.0 average_score -27.1
    episode 1286 score -70.2 average_score -27.8
    episode 1287 score -17.0 average_score -27.7
    episode 1288 score -15.8 average_score -27.6
    episode 1289 score -41.5 average_score -27.4
    episode 1290 score -26.2 average_score -27.4
    episode 1291 score -34.5 average_score -27.5
    episode 1292 score -42.2 average_score -27.4
    episode 1293 score -46.5 average_score -27.8
    episode 1294 score -16.8 average_score -27.7
    episode 1295 score -4.5 average_score -27.6
    episode 1296 score -15.8 average_score -27.4
    episode 1297 score -43.2 average_score -27.7
    episode 1298 score -92.0 average_score -28.4
    episode 1299 score -36.5 average_score -28.7
    episode 1300 score -42.2 average_score -28.9
    episode 1301 score -19.8 average_score -28.8
    episode 1302 score -44.5 average_score -29.0
    episode 1303 score -21.2 average_score -28.9
    episode 1304 score -25.8 average_score -29.2
    episode 1305 score -33.5 average_score -29.5
    episode 1306 score -40.2 average_score -29.8
    episode 1307 score -23.0 average_score -29.9
    episode 1308 score -56.0 average_score -29.8
    episode 1309 score -27.8 average_score -29.8
    episode 1310 score -22.2 average_score -29.7
    episode 1311 score -4.2 average_score -29.5
    episode 1312 score -29.5 average_score -29.6
    episode 1313 score -25.8 average_score -29.4
    episode 1314 score -28.0 average_score -29.5
    episode 1315 score -9.8 average_score -29.5
    episode 1316 score -31.2 average_score -29.3
    episode 1317 score -67.0 average_score -29.7
    episode 1318 score -14.8 average_score -29.5
    episode 1319 score -17.8 average_score -29.4
    episode 1320 score -9.0 average_score -29.4
    episode 1321 score -23.2 average_score -29.5
    episode 1322 score -23.5 average_score -29.3
    episode 1323 score -21.2 average_score -29.1
    episode 1324 score -28.8 average_score -29.3
    episode 1325 score -1.8 average_score -29.3
    episode 1326 score -31.5 average_score -29.2
    episode 1327 score -36.2 average_score -29.3
    episode 1328 score -31.5 average_score -29.3
    episode 1329 score -17.2 average_score -29.1
    episode 1330 score -22.8 average_score -29.0
    episode 1331 score -25.5 average_score -28.8
    episode 1332 score -40.0 average_score -29.2
    episode 1333 score -27.8 average_score -29.2
    episode 1334 score -62.0 average_score -29.8
    episode 1335 score -11.5 average_score -29.7
    episode 1336 score -40.5 average_score -30.0
    episode 1337 score -54.5 average_score -30.3
    episode 1338 score -13.5 average_score -30.3
    episode 1339 score -11.0 average_score -30.0
    episode 1340 score -9.0 average_score -29.8
    episode 1341 score -35.8 average_score -29.8
    episode 1342 score -58.0 average_score -29.9
    episode 1343 score -41.5 average_score -30.0
    episode 1344 score -41.5 average_score -30.0
    episode 1345 score -51.5 average_score -30.6
    episode 1346 score -24.8 average_score -30.5
    episode 1347 score -34.2 average_score -30.4
    episode 1348 score -21.0 average_score -30.4
    episode 1349 score -46.2 average_score -30.6
    episode 1350 score -27.5 average_score -30.6
    episode 1351 score -29.2 average_score -30.7
    episode 1352 score -31.2 average_score -30.8
    episode 1353 score -3.0 average_score -30.5
    episode 1354 score -9.5 average_score -30.4
    episode 1355 score -42.0 average_score -30.6
    episode 1356 score -75.5 average_score -31.1
    episode 1357 score -48.5 average_score -31.4
    episode 1358 score -30.5 average_score -31.6
    episode 1359 score -43.8 average_score -31.9
    episode 1360 score -33.8 average_score -32.0
    episode 1361 score 1.8 average_score -31.5
    episode 1362 score -9.8 average_score -31.4
    episode 1363 score 1.8 average_score -30.8
    episode 1364 score -24.8 average_score -30.6
    episode 1365 score -34.8 average_score -30.4
    episode 1366 score -27.5 average_score -30.3
    episode 1367 score -70.5 average_score -30.7
    episode 1368 score -30.5 average_score -30.9
    episode 1369 score -45.5 average_score -31.0
    episode 1370 score -5.5 average_score -30.7
    episode 1371 score -27.8 average_score -30.6
    episode 1372 score -28.2 average_score -30.8
    episode 1373 score -42.5 average_score -30.7
    episode 1374 score -37.0 average_score -30.8
    episode 1375 score -32.2 average_score -30.8
    episode 1376 score -26.0 average_score -30.9
    episode 1377 score -11.5 average_score -30.9
    episode 1378 score -13.2 average_score -30.7
    episode 1379 score -36.0 average_score -30.8
    episode 1380 score -41.2 average_score -31.1
    episode 1381 score -44.8 average_score -31.3
    episode 1382 score -30.5 average_score -31.2
    episode 1383 score -37.5 average_score -31.2
    episode 1384 score -35.8 average_score -31.0
    episode 1385 score -22.8 average_score -30.7
    episode 1386 score -52.8 average_score -30.5
    episode 1387 score -10.2 average_score -30.5
    episode 1388 score -30.8 average_score -30.6
    episode 1389 score -10.8 average_score -30.3
    episode 1390 score -43.5 average_score -30.5
    episode 1391 score -57.5 average_score -30.7
    episode 1392 score -53.8 average_score -30.8
    episode 1393 score -36.8 average_score -30.7
    episode 1394 score -19.2 average_score -30.7
    episode 1395 score -28.8 average_score -31.0
    episode 1396 score -44.0 average_score -31.3
    episode 1397 score -43.2 average_score -31.3
    episode 1398 score -22.2 average_score -30.6
    episode 1399 score -26.2 average_score -30.5
    episode 1400 score -7.5 average_score -30.1
    episode 1401 score -21.0 average_score -30.1
    episode 1402 score -24.2 average_score -29.9
    episode 1403 score 10.0 average_score -29.6
    episode 1404 score -28.8 average_score -29.7
    episode 1405 score -29.5 average_score -29.6
    episode 1406 score -23.8 average_score -29.4
    episode 1407 score -22.5 average_score -29.4
    episode 1408 score -39.8 average_score -29.3
    episode 1409 score -11.0 average_score -29.1
    episode 1410 score -10.2 average_score -29.0
    episode 1411 score -39.5 average_score -29.3
    episode 1412 score -36.5 average_score -29.4
    episode 1413 score -0.2 average_score -29.2
    episode 1414 score -36.2 average_score -29.2
    episode 1415 score -33.2 average_score -29.5
    episode 1416 score -37.2 average_score -29.5
    episode 1417 score -48.8 average_score -29.4
    episode 1418 score -55.5 average_score -29.8
    episode 1419 score -41.0 average_score -30.0
    episode 1420 score -29.8 average_score -30.2
    episode 1421 score -44.0 average_score -30.4
    episode 1422 score -27.0 average_score -30.4
    episode 1423 score -45.2 average_score -30.7
    episode 1424 score -27.8 average_score -30.7
    episode 1425 score -16.2 average_score -30.8
    episode 1426 score 15.8 average_score -30.3
    episode 1427 score -41.8 average_score -30.4
    episode 1428 score -33.2 average_score -30.4
    episode 1429 score -35.0 average_score -30.6
    episode 1430 score -40.2 average_score -30.8
    episode 1431 score -23.0 average_score -30.7
    episode 1432 score -48.8 average_score -30.8
    episode 1433 score -19.5 average_score -30.8
    episode 1434 score -60.8 average_score -30.7
    episode 1435 score -56.8 average_score -31.2
    episode 1436 score -43.8 average_score -31.2
    episode 1437 score -14.5 average_score -30.8
    episode 1438 score -33.5 average_score -31.0
    episode 1439 score -52.0 average_score -31.4
    episode 1440 score -8.0 average_score -31.4
    episode 1441 score -28.5 average_score -31.4
    episode 1442 score -36.2 average_score -31.1
    episode 1443 score -33.8 average_score -31.1
    episode 1444 score -47.8 average_score -31.1
    episode 1445 score -20.2 average_score -30.8
    episode 1446 score -11.8 average_score -30.7
    episode 1447 score -45.5 average_score -30.8
    episode 1448 score -25.0 average_score -30.8
    episode 1449 score -41.0 average_score -30.8
    episode 1450 score -3.0 average_score -30.5
    episode 1451 score -10.2 average_score -30.3
    episode 1452 score -26.5 average_score -30.3
    episode 1453 score 6.0 average_score -30.2
    episode 1454 score -51.8 average_score -30.6
    episode 1455 score -48.0 average_score -30.7
    episode 1456 score -26.2 average_score -30.2
    episode 1457 score -40.8 average_score -30.1
    episode 1458 score -31.0 average_score -30.1
    episode 1459 score -67.5 average_score -30.4
    episode 1460 score -38.0 average_score -30.4
    episode 1461 score -43.8 average_score -30.9
    episode 1462 score 22.2 average_score -30.5
    episode 1463 score -24.0 average_score -30.8
    episode 1464 score -16.8 average_score -30.7
    episode 1465 score -10.0 average_score -30.5
    episode 1466 score -77.0 average_score -31.0
    episode 1467 score -34.5 average_score -30.6
    episode 1468 score -16.5 average_score -30.5
    episode 1469 score -28.0 average_score -30.3
    episode 1470 score -33.2 average_score -30.6
    episode 1471 score -48.8 average_score -30.8
    episode 1472 score -26.8 average_score -30.8
    episode 1473 score 4.5 average_score -30.3
    episode 1474 score -41.2 average_score -30.3
    episode 1475 score -16.2 average_score -30.2
    episode 1476 score -28.8 average_score -30.2
    episode 1477 score -12.0 average_score -30.2
    episode 1478 score -69.2 average_score -30.8
    episode 1479 score -35.5 average_score -30.8
    episode 1480 score -14.2 average_score -30.5
    episode 1481 score -27.5 average_score -30.3
    episode 1482 score -25.2 average_score -30.3
    episode 1483 score -5.8 average_score -29.9
    episode 1484 score -42.0 average_score -30.0
    episode 1485 score -3.5 average_score -29.8
    episode 1486 score -14.5 average_score -29.4
    episode 1487 score -19.2 average_score -29.5
    episode 1488 score -37.5 average_score -29.6
    episode 1489 score -25.5 average_score -29.7
    episode 1490 score -22.0 average_score -29.5
    episode 1491 score -19.2 average_score -29.1
    episode 1492 score -33.8 average_score -28.9
    episode 1493 score -53.0 average_score -29.1
    episode 1494 score -47.2 average_score -29.4
    episode 1495 score -38.5 average_score -29.5
    episode 1496 score -39.0 average_score -29.4
    episode 1497 score -12.8 average_score -29.1
    episode 1498 score -23.5 average_score -29.1
    episode 1499 score -28.8 average_score -29.2
    episode 1500 score -19.2 average_score -29.3
    episode 1501 score -36.0 average_score -29.4
    episode 1502 score -34.8 average_score -29.5
    episode 1503 score -51.8 average_score -30.2
    episode 1504 score -25.5 average_score -30.1
    episode 1505 score 0.2 average_score -29.8
    episode 1506 score -41.5 average_score -30.0
    episode 1507 score -44.5 average_score -30.2
    episode 1508 score -24.2 average_score -30.1
    episode 1509 score -45.0 average_score -30.4
    episode 1510 score -62.8 average_score -30.9
    episode 1511 score 9.0 average_score -30.4
    episode 1512 score -60.5 average_score -30.7
    episode 1513 score -50.8 average_score -31.2
    episode 1514 score -26.5 average_score -31.1
    episode 1515 score -32.8 average_score -31.1
    episode 1516 score -14.5 average_score -30.9
    episode 1517 score 1.2 average_score -30.4
    episode 1518 score -14.2 average_score -29.9
    episode 1519 score -22.8 average_score -29.8
    episode 1520 score -12.0 average_score -29.6
    episode 1521 score -59.2 average_score -29.7
    episode 1522 score -21.8 average_score -29.7
    episode 1523 score -33.0 average_score -29.6
    episode 1524 score -21.5 average_score -29.5
    episode 1525 score -23.2 average_score -29.6
    episode 1526 score -38.2 average_score -30.1
    episode 1527 score -58.2 average_score -30.3
    episode 1528 score -37.8 average_score -30.3
    episode 1529 score -28.8 average_score -30.3
    episode 1530 score -50.2 average_score -30.4
    episode 1531 score -47.2 average_score -30.6
    episode 1532 score -50.2 average_score -30.6
    episode 1533 score -29.0 average_score -30.7
    episode 1534 score -54.0 average_score -30.6
    episode 1535 score -43.8 average_score -30.5
    episode 1536 score -26.0 average_score -30.3
    episode 1537 score -16.8 average_score -30.4
    episode 1538 score 13.5 average_score -29.9
    episode 1539 score -29.2 average_score -29.7
    episode 1540 score -29.8 average_score -29.9
    episode 1541 score -38.2 average_score -30.0
    episode 1542 score -25.5 average_score -29.9
    episode 1543 score -40.8 average_score -29.9
    episode 1544 score -4.8 average_score -29.5
    episode 1545 score -39.5 average_score -29.7
    episode 1546 score -37.5 average_score -30.0
    episode 1547 score -47.8 average_score -30.0
    episode 1548 score -43.8 average_score -30.2
    episode 1549 score -24.2 average_score -30.0
    episode 1550 score -20.0 average_score -30.2
    episode 1551 score -40.8 average_score -30.5
    episode 1552 score -41.2 average_score -30.6
    episode 1553 score -40.0 average_score -31.1
    episode 1554 score -65.5 average_score -31.2
    episode 1555 score -23.5 average_score -31.0
    episode 1556 score -26.0 average_score -31.0
    episode 1557 score -5.0 average_score -30.6
    episode 1558 score -28.2 average_score -30.6
    episode 1559 score -16.8 average_score -30.1
    episode 1560 score -48.8 average_score -30.2
    episode 1561 score -27.8 average_score -30.0
    episode 1562 score -30.2 average_score -30.6
    episode 1563 score -34.0 average_score -30.7
    episode 1564 score -29.0 average_score -30.8
    episode 1565 score 4.5 average_score -30.6
    episode 1566 score -49.0 average_score -30.4
    episode 1567 score -61.2 average_score -30.6
    episode 1568 score 1.5 average_score -30.4
    episode 1569 score -3.8 average_score -30.2
    episode 1570 score -33.8 average_score -30.2
    episode 1571 score -9.8 average_score -29.8
    episode 1572 score -6.5 average_score -29.6
    episode 1573 score -34.5 average_score -30.0
    episode 1574 score -28.0 average_score -29.9
    episode 1575 score -55.0 average_score -30.3
    episode 1576 score -2.2 average_score -30.0
    episode 1577 score -28.0 average_score -30.2
    episode 1578 score -38.5 average_score -29.8
    episode 1579 score -48.8 average_score -30.0
    episode 1580 score -17.5 average_score -30.0
    episode 1581 score -38.5 average_score -30.1
    episode 1582 score -19.5 average_score -30.1
    episode 1583 score -32.8 average_score -30.3
    episode 1584 score -40.2 average_score -30.3
    episode 1585 score -32.5 average_score -30.6
    episode 1586 score -36.5 average_score -30.8
    episode 1587 score -22.5 average_score -30.9
    episode 1588 score -31.5 average_score -30.8
    episode 1589 score 12.2 average_score -30.4
    episode 1590 score -25.0 average_score -30.4
    episode 1591 score -18.8 average_score -30.4
    episode 1592 score -36.8 average_score -30.5
    episode 1593 score 10.8 average_score -29.8
    episode 1594 score -33.5 average_score -29.7
    episode 1595 score -13.5 average_score -29.4
    episode 1596 score 2.8 average_score -29.0
    episode 1597 score -22.0 average_score -29.1
    episode 1598 score -17.5 average_score -29.1
    episode 1599 score -21.5 average_score -29.0
    episode 1600 score -3.0 average_score -28.8
    episode 1601 score -29.2 average_score -28.8
    episode 1602 score -32.8 average_score -28.7
    episode 1603 score -35.2 average_score -28.6
    episode 1604 score -46.0 average_score -28.8
    episode 1605 score -7.5 average_score -28.9
    episode 1606 score -41.0 average_score -28.9
    episode 1607 score -21.2 average_score -28.6
    episode 1608 score -31.5 average_score -28.7
    episode 1609 score -11.0 average_score -28.4
    episode 1610 score -23.5 average_score -28.0
    episode 1611 score -15.5 average_score -28.2
    episode 1612 score -44.0 average_score -28.0
    episode 1613 score -37.5 average_score -27.9
    episode 1614 score -26.5 average_score -27.9
    episode 1615 score -14.8 average_score -27.7
    episode 1616 score -33.0 average_score -27.9
    episode 1617 score -30.8 average_score -28.2
    episode 1618 score -36.8 average_score -28.5
    episode 1619 score -39.0 average_score -28.6
    episode 1620 score -27.5 average_score -28.8
    episode 1621 score -21.0 average_score -28.4
    episode 1622 score -24.8 average_score -28.4
    episode 1623 score -38.5 average_score -28.5
    episode 1624 score -11.0 average_score -28.4
    episode 1625 score -18.0 average_score -28.3
    episode 1626 score -19.2 average_score -28.1
    episode 1627 score -39.8 average_score -27.9
    episode 1628 score -17.8 average_score -27.7
    episode 1629 score -33.0 average_score -27.8
    episode 1630 score -34.8 average_score -27.6
    episode 1631 score -28.2 average_score -27.4
    episode 1632 score -51.5 average_score -27.5
    episode 1633 score -27.2 average_score -27.4
    episode 1634 score -35.8 average_score -27.3
    episode 1635 score -28.8 average_score -27.1
    episode 1636 score -36.0 average_score -27.2
    episode 1637 score 6.5 average_score -27.0
    episode 1638 score -11.8 average_score -27.2
    episode 1639 score -25.5 average_score -27.2
    episode 1640 score -21.8 average_score -27.1
    episode 1641 score 6.5 average_score -26.7
    episode 1642 score -16.5 average_score -26.6
    episode 1643 score -25.5 average_score -26.4
    episode 1644 score -21.2 average_score -26.6
    episode 1645 score -52.8 average_score -26.7
    episode 1646 score -43.2 average_score -26.8
    episode 1647 score 4.2 average_score -26.3
    episode 1648 score -49.0 average_score -26.3
    episode 1649 score -26.0 average_score -26.3
    episode 1650 score -37.5 average_score -26.5
    episode 1651 score -21.0 average_score -26.3
    episode 1652 score -22.5 average_score -26.1
    episode 1653 score -23.5 average_score -25.9
    episode 1654 score -37.5 average_score -25.7
    episode 1655 score -52.2 average_score -26.0
    episode 1656 score -28.0 average_score -26.0
    episode 1657 score -48.5 average_score -26.4
    episode 1658 score -41.0 average_score -26.5
    episode 1659 score -27.2 average_score -26.6
    episode 1660 score -36.2 average_score -26.5
    episode 1661 score -18.5 average_score -26.4
    episode 1662 score -31.0 average_score -26.4
    episode 1663 score -29.8 average_score -26.4
    episode 1664 score -32.0 average_score -26.4
    episode 1665 score -28.2 average_score -26.8
    episode 1666 score -52.8 average_score -26.8
    episode 1667 score -30.2 average_score -26.5
    episode 1668 score -53.0 average_score -27.0
    episode 1669 score -56.8 average_score -27.6
    episode 1670 score -38.8 average_score -27.6
    episode 1671 score -22.8 average_score -27.7
    episode 1672 score -60.8 average_score -28.3
    episode 1673 score -29.2 average_score -28.2
    episode 1674 score -51.2 average_score -28.5
    episode 1675 score -33.2 average_score -28.2
    episode 1676 score -23.0 average_score -28.4
    episode 1677 score 6.2 average_score -28.1
    episode 1678 score -20.5 average_score -27.9
    episode 1679 score -5.8 average_score -27.5
    episode 1680 score -10.0 average_score -27.4
    episode 1681 score -43.0 average_score -27.5
    episode 1682 score -56.0 average_score -27.8
    episode 1683 score -13.0 average_score -27.6
    episode 1684 score -33.5 average_score -27.6
    episode 1685 score -24.2 average_score -27.5
    episode 1686 score -34.8 average_score -27.5
    episode 1687 score -45.0 average_score -27.7
    episode 1688 score -78.2 average_score -28.2
    episode 1689 score -38.5 average_score -28.7
    episode 1690 score -62.5 average_score -29.0
    episode 1691 score -15.8 average_score -29.0
    episode 1692 score -13.5 average_score -28.8
    episode 1693 score -43.8 average_score -29.3
    episode 1694 score -4.8 average_score -29.0
    episode 1695 score -17.5 average_score -29.1
    episode 1696 score -36.0 average_score -29.5
    episode 1697 score -3.0 average_score -29.3
    episode 1698 score -6.0 average_score -29.2
    episode 1699 score -65.5 average_score -29.6
    episode 1700 score -43.2 average_score -30.0
    episode 1701 score -28.2 average_score -30.0
    episode 1702 score -25.5 average_score -29.9
    episode 1703 score -21.0 average_score -29.8
    episode 1704 score -40.2 average_score -29.7
    episode 1705 score -23.0 average_score -29.9
    episode 1706 score -34.2 average_score -29.8
    episode 1707 score -49.5 average_score -30.1
    episode 1708 score -13.2 average_score -29.9
    episode 1709 score -20.8 average_score -30.0
    episode 1710 score -44.5 average_score -30.2
    episode 1711 score -30.5 average_score -30.4
    episode 1712 score -42.0 average_score -30.3
    episode 1713 score -43.2 average_score -30.4
    episode 1714 score -3.8 average_score -30.2
    episode 1715 score -67.8 average_score -30.7
    episode 1716 score -38.0 average_score -30.8
    episode 1717 score -40.2 average_score -30.8
    episode 1718 score -61.8 average_score -31.1
    episode 1719 score -16.8 average_score -30.9
    episode 1720 score -40.8 average_score -31.0
    episode 1721 score -20.8 average_score -31.0
    episode 1722 score -1.5 average_score -30.8
    episode 1723 score -12.2 average_score -30.5
    episode 1724 score -23.0 average_score -30.6
    episode 1725 score -59.8 average_score -31.0
    episode 1726 score -74.5 average_score -31.6
    episode 1727 score -13.5 average_score -31.3
    episode 1728 score -51.5 average_score -31.7
    episode 1729 score -0.2 average_score -31.3
    episode 1730 score -16.2 average_score -31.2
    episode 1731 score -35.8 average_score -31.2
    episode 1732 score -53.5 average_score -31.3
    episode 1733 score -14.0 average_score -31.1
    episode 1734 score -3.8 average_score -30.8
    episode 1735 score -20.8 average_score -30.7
    episode 1736 score -43.5 average_score -30.8
    episode 1737 score -47.2 average_score -31.3
    episode 1738 score -31.2 average_score -31.5
    episode 1739 score -28.2 average_score -31.6
    episode 1740 score -26.5 average_score -31.6
    episode 1741 score -24.0 average_score -31.9
    episode 1742 score -36.5 average_score -32.1
    episode 1743 score -36.2 average_score -32.2
    episode 1744 score -50.5 average_score -32.5
    episode 1745 score -28.8 average_score -32.3
    episode 1746 score -22.5 average_score -32.1
    episode 1747 score -22.5 average_score -32.3
    episode 1748 score -52.5 average_score -32.4
    episode 1749 score -33.0 average_score -32.4
    episode 1750 score -24.0 average_score -32.3
    episode 1751 score -27.8 average_score -32.4
    episode 1752 score -35.0 average_score -32.5
    episode 1753 score -30.2 average_score -32.6
    episode 1754 score -60.2 average_score -32.8
    episode 1755 score -34.8 average_score -32.6
    episode 1756 score -44.2 average_score -32.8
    episode 1757 score -54.5 average_score -32.8
    episode 1758 score -34.5 average_score -32.8
    episode 1759 score -45.2 average_score -33.0
    episode 1760 score -29.5 average_score -32.9
    episode 1761 score -19.8 average_score -32.9
    episode 1762 score -28.0 average_score -32.9
    episode 1763 score -24.8 average_score -32.8
    episode 1764 score -18.8 average_score -32.7
    episode 1765 score -66.0 average_score -33.1
    episode 1766 score -15.5 average_score -32.7
    episode 1767 score -47.5 average_score -32.9
    episode 1768 score -55.2 average_score -32.9
    episode 1769 score -20.0 average_score -32.5
    episode 1770 score -36.0 average_score -32.5
    episode 1771 score -17.5 average_score -32.4
    episode 1772 score -44.5 average_score -32.3
    episode 1773 score -27.8 average_score -32.3
    episode 1774 score -5.0 average_score -31.8
    episode 1775 score -27.2 average_score -31.7
    episode 1776 score -28.0 average_score -31.8
    episode 1777 score -40.5 average_score -32.3
    episode 1778 score -12.8 average_score -32.2
    episode 1779 score -25.0 average_score -32.4
    episode 1780 score -5.5 average_score -32.3
    episode 1781 score -24.2 average_score -32.1
    episode 1782 score -9.2 average_score -31.7
    episode 1783 score -27.0 average_score -31.8
    episode 1784 score -40.2 average_score -31.9
    episode 1785 score -31.0 average_score -31.9
    episode 1786 score -37.8 average_score -32.0
    episode 1787 score -13.5 average_score -31.7
    episode 1788 score -33.0 average_score -31.2
    episode 1789 score -33.8 average_score -31.2
    episode 1790 score -29.2 average_score -30.8
    episode 1791 score -60.5 average_score -31.3
    episode 1792 score -22.8 average_score -31.4
    episode 1793 score -23.5 average_score -31.2
    episode 1794 score -21.0 average_score -31.3
    episode 1795 score -47.2 average_score -31.6
    episode 1796 score -18.2 average_score -31.4
    episode 1797 score -22.5 average_score -31.6
    episode 1798 score -32.8 average_score -31.9
    episode 1799 score -26.5 average_score -31.5
    episode 1800 score -35.2 average_score -31.4
    episode 1801 score -51.5 average_score -31.7
    episode 1802 score -25.2 average_score -31.7
    episode 1803 score -59.8 average_score -32.1
    episode 1804 score -26.0 average_score -31.9
    episode 1805 score -37.5 average_score -32.1
    episode 1806 score -21.0 average_score -31.9
    episode 1807 score 19.5 average_score -31.2
    episode 1808 score -34.5 average_score -31.4
    episode 1809 score -32.0 average_score -31.6
    episode 1810 score -38.8 average_score -31.5
    episode 1811 score -46.2 average_score -31.7
    episode 1812 score -21.0 average_score -31.4
    episode 1813 score -23.5 average_score -31.3
    episode 1814 score -13.5 average_score -31.4
    episode 1815 score -23.5 average_score -30.9
    episode 1816 score -12.5 average_score -30.7
    episode 1817 score -57.0 average_score -30.8
    episode 1818 score -41.0 average_score -30.6
    episode 1819 score -28.2 average_score -30.7
    episode 1820 score -18.5 average_score -30.5
    episode 1821 score -22.5 average_score -30.5
    episode 1822 score -25.8 average_score -30.8
    episode 1823 score -27.8 average_score -30.9
    episode 1824 score -32.0 average_score -31.0
    episode 1825 score -30.2 average_score -30.7
    episode 1826 score -34.8 average_score -30.3
    episode 1827 score -33.2 average_score -30.5
    episode 1828 score -15.8 average_score -30.2
    episode 1829 score -31.8 average_score -30.5
    episode 1830 score -31.5 average_score -30.6
    episode 1831 score 19.8 average_score -30.1
    episode 1832 score -58.2 average_score -30.1
    episode 1833 score -15.2 average_score -30.1
    episode 1834 score -34.5 average_score -30.4
    episode 1835 score -45.0 average_score -30.7
    episode 1836 score -33.5 average_score -30.6
    episode 1837 score -21.2 average_score -30.3
    episode 1838 score -53.0 average_score -30.5
    episode 1839 score -18.5 average_score -30.4
    episode 1840 score -0.2 average_score -30.2
    episode 1841 score -27.0 average_score -30.2
    episode 1842 score -30.0 average_score -30.1
    episode 1843 score 11.0 average_score -29.7
    episode 1844 score -32.5 average_score -29.5
    episode 1845 score -19.0 average_score -29.4
    episode 1846 score -60.2 average_score -29.8
    episode 1847 score -25.2 average_score -29.8
    episode 1848 score -29.5 average_score -29.6
    episode 1849 score -48.2 average_score -29.7
    episode 1850 score -12.5 average_score -29.6
    episode 1851 score -19.8 average_score -29.5
    episode 1852 score -58.5 average_score -29.8
    episode 1853 score -32.0 average_score -29.8
    episode 1854 score -17.5 average_score -29.4
    episode 1855 score -2.8 average_score -29.0
    episode 1856 score -24.2 average_score -28.8
    episode 1857 score -11.5 average_score -28.4
    episode 1858 score -35.2 average_score -28.4
    episode 1859 score -19.5 average_score -28.1
    episode 1860 score -24.8 average_score -28.1
    episode 1861 score -14.8 average_score -28.1
    episode 1862 score -28.2 average_score -28.1
    episode 1863 score -7.0 average_score -27.9
    episode 1864 score -11.2 average_score -27.8
    episode 1865 score -47.2 average_score -27.6
    episode 1866 score -53.5 average_score -28.0
    episode 1867 score -12.8 average_score -27.6
    episode 1868 score -71.0 average_score -27.8
    episode 1869 score -45.8 average_score -28.1
    episode 1870 score -23.0 average_score -27.9
    episode 1871 score -70.2 average_score -28.5
    episode 1872 score -37.5 average_score -28.4
    episode 1873 score 1.5 average_score -28.1
    episode 1874 score -37.2 average_score -28.4
    episode 1875 score -50.2 average_score -28.6
    episode 1876 score -45.8 average_score -28.8
    episode 1877 score -42.8 average_score -28.9
    episode 1878 score -25.8 average_score -29.0
    episode 1879 score -19.0 average_score -28.9
    episode 1880 score -38.5 average_score -29.2
    episode 1881 score -10.2 average_score -29.1
    episode 1882 score -12.8 average_score -29.1
    episode 1883 score -36.2 average_score -29.2
    episode 1884 score -9.5 average_score -28.9
    episode 1885 score -25.2 average_score -28.9
    episode 1886 score -59.5 average_score -29.1
    episode 1887 score -29.2 average_score -29.2
    episode 1888 score -4.5 average_score -29.0
    episode 1889 score 0.2 average_score -28.6
    episode 1890 score -4.8 average_score -28.4
    episode 1891 score -56.0 average_score -28.3
    episode 1892 score -37.5 average_score -28.5
    episode 1893 score -45.5 average_score -28.7
    episode 1894 score -13.0 average_score -28.6
    episode 1895 score -16.2 average_score -28.3
    episode 1896 score -28.0 average_score -28.4
    episode 1897 score -57.0 average_score -28.8
    episode 1898 score -30.0 average_score -28.7
    episode 1899 score -28.8 average_score -28.7
    episode 1900 score -28.8 average_score -28.7
    episode 1901 score -43.0 average_score -28.6
    episode 1902 score -54.0 average_score -28.9
    episode 1903 score -20.2 average_score -28.5
    episode 1904 score -40.2 average_score -28.6
    episode 1905 score -8.8 average_score -28.3
    episode 1906 score -11.5 average_score -28.2
    episode 1907 score -42.8 average_score -28.9
    episode 1908 score -20.8 average_score -28.7
    episode 1909 score -16.0 average_score -28.6
    episode 1910 score -16.5 average_score -28.4
    episode 1911 score -48.2 average_score -28.4
    episode 1912 score -18.5 average_score -28.3
    episode 1913 score -11.0 average_score -28.2
    episode 1914 score -41.8 average_score -28.5
    episode 1915 score 2.2 average_score -28.2
    episode 1916 score -56.2 average_score -28.7
    episode 1917 score -39.0 average_score -28.5
    episode 1918 score -53.5 average_score -28.6
    episode 1919 score -22.2 average_score -28.6
    episode 1920 score -37.5 average_score -28.8
    episode 1921 score -62.8 average_score -29.2
    episode 1922 score -23.0 average_score -29.1
    episode 1923 score -38.2 average_score -29.2
    episode 1924 score -30.2 average_score -29.2
    episode 1925 score -6.0 average_score -29.0
    episode 1926 score -4.8 average_score -28.7
    episode 1927 score -60.0 average_score -28.9
    episode 1928 score -40.0 average_score -29.2
    episode 1929 score -41.8 average_score -29.3
    episode 1930 score -0.8 average_score -29.0
    episode 1931 score -38.8 average_score -29.6
    episode 1932 score -21.2 average_score -29.2
    episode 1933 score -49.0 average_score -29.5
    episode 1934 score -29.0 average_score -29.5
    episode 1935 score -36.8 average_score -29.4
    episode 1936 score -23.8 average_score -29.3
    episode 1937 score -31.2 average_score -29.4
    episode 1938 score -38.8 average_score -29.3
    episode 1939 score -15.5 average_score -29.2
    episode 1940 score -27.2 average_score -29.5
    episode 1941 score -57.5 average_score -29.8
    episode 1942 score -26.2 average_score -29.8
    episode 1943 score -40.5 average_score -30.3
    episode 1944 score -37.5 average_score -30.3
    episode 1945 score -46.5 average_score -30.6
    episode 1946 score -31.5 average_score -30.3
    episode 1947 score -38.0 average_score -30.4
    episode 1948 score -76.0 average_score -30.9
    episode 1949 score -39.5 average_score -30.8
    episode 1950 score -32.8 average_score -31.0
    episode 1951 score -17.2 average_score -31.0
    episode 1952 score -30.0 average_score -30.7
    episode 1953 score -40.5 average_score -30.8
    episode 1954 score -45.2 average_score -31.1
    episode 1955 score -15.0 average_score -31.2
    episode 1956 score -44.2 average_score -31.4
    episode 1957 score -39.2 average_score -31.7
    episode 1958 score -34.2 average_score -31.7
    episode 1959 score -32.2 average_score -31.8
    episode 1960 score -33.5 average_score -31.9
    episode 1961 score -7.8 average_score -31.8
    episode 1962 score -53.0 average_score -32.1
    episode 1963 score -34.8 average_score -32.3
    episode 1964 score -23.0 average_score -32.5
    episode 1965 score -18.8 average_score -32.2
    episode 1966 score -20.5 average_score -31.8
    episode 1967 score -22.0 average_score -31.9
    episode 1968 score -27.2 average_score -31.5
    episode 1969 score 4.8 average_score -31.0
    episode 1970 score -24.8 average_score -31.0
    episode 1971 score -17.0 average_score -30.5
    episode 1972 score -8.5 average_score -30.2
    episode 1973 score -15.5 average_score -30.4
    episode 1974 score 16.2 average_score -29.8
    episode 1975 score -5.8 average_score -29.4
    episode 1976 score -47.2 average_score -29.4
    episode 1977 score -29.8 average_score -29.3
    episode 1978 score -18.5 average_score -29.2
    episode 1979 score -38.5 average_score -29.4
    episode 1980 score -45.8 average_score -29.5
    episode 1981 score -18.8 average_score -29.5
    episode 1982 score -58.8 average_score -30.0
    episode 1983 score -11.8 average_score -29.8
    episode 1984 score -34.0 average_score -30.0
    episode 1985 score -52.0 average_score -30.3
    episode 1986 score -21.8 average_score -29.9
    episode 1987 score -60.5 average_score -30.2
    episode 1988 score -30.5 average_score -30.5
    episode 1989 score -27.5 average_score -30.7
    episode 1990 score -8.2 average_score -30.8
    episode 1991 score 1.5 average_score -30.2
    episode 1992 score -16.2 average_score -30.0
    episode 1993 score 7.2 average_score -29.5
    episode 1994 score -31.8 average_score -29.6
    episode 1995 score -20.5 average_score -29.7
    episode 1996 score 2.8 average_score -29.4
    episode 1997 score -30.2 average_score -29.1
    episode 1998 score 5.8 average_score -28.8
    episode 1999 score -30.2 average_score -28.8
    episode 2000 score -35.8 average_score -28.8
    episode 2001 score 0.8 average_score -28.4
    episode 2002 score -43.0 average_score -28.3
    episode 2003 score -48.5 average_score -28.6
    episode 2004 score -42.0 average_score -28.6
    episode 2005 score -26.0 average_score -28.8
    episode 2006 score -51.8 average_score -29.2
    episode 2007 score -50.2 average_score -29.2
    episode 2008 score -7.0 average_score -29.1
    episode 2009 score -51.8 average_score -29.5
    episode 2010 score -16.2 average_score -29.5
    episode 2011 score -48.5 average_score -29.5
    episode 2012 score -23.0 average_score -29.5
    episode 2013 score -22.2 average_score -29.6
    episode 2014 score -30.5 average_score -29.5
    episode 2015 score -38.5 average_score -29.9
    episode 2016 score -33.8 average_score -29.7
    episode 2017 score -43.5 average_score -29.7
    episode 2018 score -58.5 average_score -29.8
    episode 2019 score -12.0 average_score -29.7
    episode 2020 score -42.5 average_score -29.7
    episode 2021 score -31.2 average_score -29.4
    episode 2022 score -27.5 average_score -29.5
    episode 2023 score -33.2 average_score -29.4
    episode 2024 score -36.0 average_score -29.5
    episode 2025 score -29.0 average_score -29.7
    episode 2026 score -20.8 average_score -29.9
    episode 2027 score -23.5 average_score -29.5
    episode 2028 score -2.0 average_score -29.1
    episode 2029 score 4.2 average_score -28.7
    episode 2030 score -31.8 average_score -29.0
    episode 2031 score 11.8 average_score -28.5
    episode 2032 score -56.8 average_score -28.8
    episode 2033 score -29.8 average_score -28.6
    episode 2034 score -44.5 average_score -28.8
    episode 2035 score -41.8 average_score -28.8
    episode 2036 score -12.5 average_score -28.7
    episode 2037 score -28.5 average_score -28.7
    episode 2038 score -40.8 average_score -28.7
    episode 2039 score -16.0 average_score -28.7
    episode 2040 score -16.8 average_score -28.6
    episode 2041 score -27.2 average_score -28.3
    episode 2042 score -23.8 average_score -28.3
    episode 2043 score -13.2 average_score -28.0
    episode 2044 score -21.0 average_score -27.8
    episode 2045 score -35.2 average_score -27.7
    episode 2046 score -12.8 average_score -27.5
    episode 2047 score -49.8 average_score -27.7
    episode 2048 score -30.2 average_score -27.2
    episode 2049 score -35.5 average_score -27.2
    episode 2050 score -21.5 average_score -27.1
    episode 2051 score -33.2 average_score -27.2
    episode 2052 score -35.8 average_score -27.3
    episode 2053 score -35.5 average_score -27.2
    episode 2054 score -51.5 average_score -27.3
    episode 2055 score -17.8 average_score -27.3
    episode 2056 score -28.0 average_score -27.1
    episode 2057 score -27.0 average_score -27.0
    episode 2058 score -20.0 average_score -26.9
    episode 2059 score 0.8 average_score -26.6
    episode 2060 score -11.2 average_score -26.3
    episode 2061 score -29.2 average_score -26.5
    episode 2062 score -29.0 average_score -26.3
    episode 2063 score -34.0 average_score -26.3
    episode 2064 score -36.2 average_score -26.4
    episode 2065 score -45.8 average_score -26.7
    episode 2066 score -64.2 average_score -27.1
    episode 2067 score -39.2 average_score -27.3
    episode 2068 score -37.5 average_score -27.4
    episode 2069 score -25.2 average_score -27.7
    episode 2070 score -57.2 average_score -28.0
    episode 2071 score -13.5 average_score -28.0
    episode 2072 score -36.2 average_score -28.3
    episode 2073 score -49.0 average_score -28.6
    episode 2074 score -28.2 average_score -29.1
    episode 2075 score -13.2 average_score -29.1
    episode 2076 score -24.2 average_score -28.9
    episode 2077 score -3.2 average_score -28.6
    episode 2078 score -51.2 average_score -29.0
    episode 2079 score -13.0 average_score -28.7
    episode 2080 score 5.0 average_score -28.2
    episode 2081 score -46.2 average_score -28.5
    episode 2082 score -21.8 average_score -28.1
    episode 2083 score -12.2 average_score -28.1
    episode 2084 score -20.0 average_score -28.0
    episode 2085 score -28.0 average_score -27.7
    episode 2086 score -7.0 average_score -27.6
    episode 2087 score -29.2 average_score -27.3
    episode 2088 score -26.0 average_score -27.2
    episode 2089 score -25.0 average_score -27.2
    episode 2090 score -29.8 average_score -27.4
    episode 2091 score -31.2 average_score -27.7
    episode 2092 score -12.0 average_score -27.7
    episode 2093 score -42.0 average_score -28.2
    episode 2094 score -38.5 average_score -28.3
    episode 2095 score -14.8 average_score -28.2
    episode 2096 score -28.2 average_score -28.5
    episode 2097 score -29.5 average_score -28.5
    episode 2098 score 5.0 average_score -28.5
    episode 2099 score -21.5 average_score -28.4
    episode 2100 score -29.5 average_score -28.4
    episode 2101 score -32.8 average_score -28.7
    episode 2102 score -30.0 average_score -28.6
    episode 2103 score -33.8 average_score -28.4
    episode 2104 score -24.2 average_score -28.2
    episode 2105 score -52.2 average_score -28.5
    episode 2106 score -8.8 average_score -28.1
    episode 2107 score -21.5 average_score -27.8
    episode 2108 score -43.5 average_score -28.2
    episode 2109 score -27.8 average_score -27.9
    episode 2110 score -35.2 average_score -28.1
    episode 2111 score -31.0 average_score -27.9
    episode 2112 score -21.8 average_score -27.9
    episode 2113 score -17.2 average_score -27.9
    episode 2114 score -45.0 average_score -28.0
    episode 2115 score -27.2 average_score -27.9
    episode 2116 score -24.2 average_score -27.8
    episode 2117 score -38.0 average_score -27.8
    episode 2118 score -48.0 average_score -27.6
    episode 2119 score -33.0 average_score -27.9
    episode 2120 score -38.5 average_score -27.8
    episode 2121 score -46.8 average_score -28.0
    episode 2122 score -46.8 average_score -28.2
    episode 2123 score -33.8 average_score -28.2
    episode 2124 score -8.0 average_score -27.9
    episode 2125 score -18.5 average_score -27.8
    episode 2126 score -31.8 average_score -27.9
    episode 2127 score -45.2 average_score -28.1
    episode 2128 score -32.8 average_score -28.4
    episode 2129 score -60.0 average_score -29.1
    episode 2130 score -37.8 average_score -29.1
    episode 2131 score -32.5 average_score -29.6
    episode 2132 score -21.2 average_score -29.2
    episode 2133 score -9.5 average_score -29.0
    episode 2134 score -42.2 average_score -29.0
    episode 2135 score -39.0 average_score -29.0
    episode 2136 score -6.0 average_score -28.9
    episode 2137 score -10.5 average_score -28.7
    episode 2138 score -39.2 average_score -28.7
    episode 2139 score -14.0 average_score -28.7
    episode 2140 score -33.5 average_score -28.8
    episode 2141 score -16.0 average_score -28.7
    episode 2142 score -15.2 average_score -28.6
    episode 2143 score -44.2 average_score -29.0
    episode 2144 score -67.2 average_score -29.4
    episode 2145 score -18.2 average_score -29.2
    episode 2146 score -33.5 average_score -29.5
    episode 2147 score -48.5 average_score -29.4
    episode 2148 score -40.2 average_score -29.5
    episode 2149 score -7.5 average_score -29.3
    episode 2150 score -56.8 average_score -29.6
    episode 2151 score -68.8 average_score -30.0
    episode 2152 score -26.2 average_score -29.9
    episode 2153 score -31.5 average_score -29.8
    episode 2154 score -31.0 average_score -29.6
    episode 2155 score -57.8 average_score -30.0
    episode 2156 score -77.0 average_score -30.5
    episode 2157 score -34.5 average_score -30.6
    episode 2158 score -55.2 average_score -30.9
    episode 2159 score -2.5 average_score -31.0
    episode 2160 score -36.0 average_score -31.2
    episode 2161 score -21.0 average_score -31.1
    episode 2162 score -14.5 average_score -31.0
    episode 2163 score 1.2 average_score -30.6
    episode 2164 score -31.5 average_score -30.6
    episode 2165 score -39.2 average_score -30.5
    episode 2166 score -36.8 average_score -30.3
    episode 2167 score -13.8 average_score -30.0
    episode 2168 score -35.2 average_score -30.0
    episode 2169 score -13.0 average_score -29.9
    episode 2170 score -47.2 average_score -29.8
    episode 2171 score -37.8 average_score -30.0
    episode 2172 score -34.0 average_score -30.0
    episode 2173 score -31.0 average_score -29.8
    episode 2174 score -13.2 average_score -29.7
    episode 2175 score -50.5 average_score -30.0
    episode 2176 score -23.0 average_score -30.0
    episode 2177 score -5.5 average_score -30.0
    episode 2178 score -8.0 average_score -29.6
    episode 2179 score -40.0 average_score -29.9
    episode 2180 score -47.0 average_score -30.4
    episode 2181 score -43.5 average_score -30.4
    episode 2182 score -55.8 average_score -30.7
    episode 2183 score -30.8 average_score -30.9
    episode 2184 score -21.8 average_score -30.9
    episode 2185 score -38.8 average_score -31.0
    episode 2186 score -26.5 average_score -31.2
    episode 2187 score -25.2 average_score -31.2
    episode 2188 score -50.5 average_score -31.4
    episode 2189 score -12.5 average_score -31.3
    episode 2190 score -37.8 average_score -31.4
    episode 2191 score -20.5 average_score -31.3
    episode 2192 score -23.0 average_score -31.4
    episode 2193 score -41.5 average_score -31.4
    episode 2194 score -52.5 average_score -31.5
    episode 2195 score -25.8 average_score -31.6
    episode 2196 score -41.2 average_score -31.7
    episode 2197 score -22.0 average_score -31.7
    episode 2198 score -30.2 average_score -32.0
    episode 2199 score -37.8 average_score -32.2
    episode 2200 score -31.8 average_score -32.2
    episode 2201 score -46.5 average_score -32.3
    episode 2202 score -34.2 average_score -32.4
    episode 2203 score -35.5 average_score -32.4
    episode 2204 score -20.8 average_score -32.4
    episode 2205 score -49.5 average_score -32.3
    episode 2206 score -46.5 average_score -32.7
    episode 2207 score -44.5 average_score -33.0
    episode 2208 score -25.5 average_score -32.8
    episode 2209 score -34.0 average_score -32.8
    episode 2210 score -9.2 average_score -32.6
    episode 2211 score -56.8 average_score -32.8
    episode 2212 score -70.8 average_score -33.3
    episode 2213 score -42.0 average_score -33.6
    episode 2214 score -16.0 average_score -33.3
    episode 2215 score -30.2 average_score -33.3
    episode 2216 score -13.5 average_score -33.2
    episode 2217 score 10.8 average_score -32.7
    episode 2218 score -44.2 average_score -32.7
    episode 2219 score -3.0 average_score -32.4
    episode 2220 score -32.5 average_score -32.3
    episode 2221 score -57.2 average_score -32.4
    episode 2222 score -25.0 average_score -32.2
    episode 2223 score -36.0 average_score -32.2
    episode 2224 score -53.5 average_score -32.7
    episode 2225 score -21.5 average_score -32.7
    episode 2226 score -27.2 average_score -32.7
    episode 2227 score -14.8 average_score -32.4
    episode 2228 score -62.8 average_score -32.7
    episode 2229 score -72.2 average_score -32.8
    episode 2230 score 1.2 average_score -32.4
    episode 2231 score -18.5 average_score -32.3
    episode 2232 score -55.0 average_score -32.6
    episode 2233 score -18.8 average_score -32.7
    episode 2234 score -8.0 average_score -32.3
    episode 2235 score -20.2 average_score -32.2
    episode 2236 score -35.2 average_score -32.4
    episode 2237 score -46.8 average_score -32.8
    episode 2238 score -14.8 average_score -32.6
    episode 2239 score -35.8 average_score -32.8
    episode 2240 score -1.2 average_score -32.5
    episode 2241 score -46.2 average_score -32.8
    episode 2242 score -32.2 average_score -32.9
    episode 2243 score -58.2 average_score -33.1
    episode 2244 score -38.0 average_score -32.8
    episode 2245 score -25.5 average_score -32.9
    episode 2246 score -11.8 average_score -32.6
    episode 2247 score -42.0 average_score -32.6
    episode 2248 score -21.8 average_score -32.4
    episode 2249 score -21.0 average_score -32.5
    episode 2250 score -66.5 average_score -32.6
    episode 2251 score -35.5 average_score -32.3
    episode 2252 score -50.5 average_score -32.5
    episode 2253 score -37.2 average_score -32.6
    episode 2254 score -19.2 average_score -32.5
    episode 2255 score -82.2 average_score -32.7
    episode 2256 score -14.2 average_score -32.1
    episode 2257 score -33.5 average_score -32.1
    episode 2258 score -7.5 average_score -31.6
    episode 2259 score -37.0 average_score -31.9
    episode 2260 score -30.2 average_score -31.9
    episode 2261 score -11.0 average_score -31.8
    episode 2262 score -22.2 average_score -31.9
    episode 2263 score -34.0 average_score -32.2
    episode 2264 score -28.8 average_score -32.2
    episode 2265 score -13.0 average_score -31.9
    episode 2266 score -12.8 average_score -31.7
    episode 2267 score -26.0 average_score -31.8
    episode 2268 score -29.0 average_score -31.7
    episode 2269 score -20.8 average_score -31.8
    episode 2270 score -27.5 average_score -31.6
    episode 2271 score -30.5 average_score -31.6
    episode 2272 score -21.5 average_score -31.4
    episode 2273 score -4.5 average_score -31.2
    episode 2274 score -30.8 average_score -31.3
    episode 2275 score -38.8 average_score -31.2
    episode 2276 score -32.0 average_score -31.3
    episode 2277 score -52.8 average_score -31.8
    episode 2278 score -24.2 average_score -31.9
    episode 2279 score -31.0 average_score -31.9
    episode 2280 score -7.5 average_score -31.5
    episode 2281 score -23.0 average_score -31.3
    episode 2282 score -51.5 average_score -31.2
    episode 2283 score -22.8 average_score -31.1
    episode 2284 score -35.8 average_score -31.3
    episode 2285 score -11.8 average_score -31.0
    episode 2286 score -41.0 average_score -31.1
    episode 2287 score -50.5 average_score -31.4
    episode 2288 score -21.8 average_score -31.1
    episode 2289 score -36.2 average_score -31.4
    episode 2290 score -45.0 average_score -31.4
    episode 2291 score -47.5 average_score -31.7
    episode 2292 score -43.8 average_score -31.9
    episode 2293 score -6.0 average_score -31.5
    episode 2294 score -0.5 average_score -31.0
    episode 2295 score -40.2 average_score -31.2
    episode 2296 score -36.8 average_score -31.1
    episode 2297 score -47.5 average_score -31.4
    episode 2298 score -29.8 average_score -31.4
    episode 2299 score -22.0 average_score -31.2
    episode 2300 score -35.8 average_score -31.3
    episode 2301 score -41.2 average_score -31.2
    episode 2302 score -28.8 average_score -31.1
    episode 2303 score -7.2 average_score -30.9
    episode 2304 score -18.2 average_score -30.8
    episode 2305 score -31.5 average_score -30.7
    episode 2306 score -19.2 average_score -30.4
    episode 2307 score -47.0 average_score -30.4
    episode 2308 score -15.5 average_score -30.3
    episode 2309 score -18.8 average_score -30.2
    episode 2310 score -29.0 average_score -30.4
    episode 2311 score -28.5 average_score -30.1
    episode 2312 score -38.0 average_score -29.8
    episode 2313 score -34.0 average_score -29.7
    episode 2314 score -40.8 average_score -29.9
    episode 2315 score -37.0 average_score -30.0
    episode 2316 score -38.2 average_score -30.2
    episode 2317 score -40.0 average_score -30.7
    episode 2318 score -38.2 average_score -30.7
    episode 2319 score -40.2 average_score -31.1
    episode 2320 score -44.8 average_score -31.2
    episode 2321 score -31.0 average_score -30.9
    episode 2322 score -56.0 average_score -31.2
    episode 2323 score -4.0 average_score -30.9
    episode 2324 score -28.0 average_score -30.6
    episode 2325 score -12.2 average_score -30.6
    episode 2326 score -14.0 average_score -30.4
    episode 2327 score -43.0 average_score -30.7
    episode 2328 score -12.2 average_score -30.2
    episode 2329 score -49.0 average_score -30.0
    episode 2330 score -33.0 average_score -30.3
    episode 2331 score -12.0 average_score -30.2
    episode 2332 score -17.0 average_score -29.9
    episode 2333 score -18.5 average_score -29.9
    episode 2334 score -10.5 average_score -29.9
    episode 2335 score -60.5 average_score -30.3
    episode 2336 score -42.2 average_score -30.4
    episode 2337 score -48.0 average_score -30.4
    episode 2338 score -30.2 average_score -30.5
    episode 2339 score -52.8 average_score -30.7
    episode 2340 score -48.5 average_score -31.2
    episode 2341 score -18.0 average_score -30.9
    episode 2342 score -38.5 average_score -30.9
    episode 2343 score -23.0 average_score -30.6
    episode 2344 score -34.0 average_score -30.6
    episode 2345 score -29.5 average_score -30.6
    episode 2346 score -25.5 average_score -30.7
    episode 2347 score -35.8 average_score -30.7
    episode 2348 score -18.8 average_score -30.6
    episode 2349 score -48.0 average_score -30.9
    episode 2350 score 0.2 average_score -30.2
    episode 2351 score -29.5 average_score -30.2
    episode 2352 score -6.8 average_score -29.7
    episode 2353 score -45.0 average_score -29.8
    episode 2354 score -24.2 average_score -29.9
    episode 2355 score -71.2 average_score -29.8
    episode 2356 score -36.0 average_score -30.0
    episode 2357 score -6.5 average_score -29.7
    episode 2358 score -57.0 average_score -30.2
    episode 2359 score -26.8 average_score -30.1
    episode 2360 score -27.0 average_score -30.1
    episode 2361 score -24.0 average_score -30.2
    episode 2362 score -28.2 average_score -30.3
    episode 2363 score -20.8 average_score -30.1
    episode 2364 score -10.5 average_score -29.9
    episode 2365 score -38.8 average_score -30.2
    episode 2366 score 13.2 average_score -29.9
    episode 2367 score -42.0 average_score -30.1
    episode 2368 score -49.0 average_score -30.3
    episode 2369 score -58.2 average_score -30.7
    episode 2370 score -28.8 average_score -30.7
    episode 2371 score -30.8 average_score -30.7
    episode 2372 score -18.0 average_score -30.7
    episode 2373 score -60.5 average_score -31.2
    episode 2374 score 3.2 average_score -30.9
    episode 2375 score -12.2 average_score -30.6
    episode 2376 score -32.5 average_score -30.6
    episode 2377 score -5.8 average_score -30.1
    episode 2378 score -37.2 average_score -30.3
    episode 2379 score -14.2 average_score -30.1
    episode 2380 score -6.2 average_score -30.1
    episode 2381 score -34.2 average_score -30.2
    episode 2382 score -31.5 average_score -30.0
    episode 2383 score -25.8 average_score -30.0
    episode 2384 score -48.8 average_score -30.2
    episode 2385 score -32.2 average_score -30.4
    episode 2386 score -46.8 average_score -30.4
    episode 2387 score -24.5 average_score -30.2
    episode 2388 score -44.8 average_score -30.4
    episode 2389 score -27.0 average_score -30.3
    episode 2390 score -48.8 average_score -30.4
    episode 2391 score -16.8 average_score -30.0
    episode 2392 score -33.2 average_score -29.9
    episode 2393 score -29.2 average_score -30.2
    episode 2394 score -14.8 average_score -30.3
    episode 2395 score -32.8 average_score -30.2
    episode 2396 score -19.5 average_score -30.1
    episode 2397 score -42.0 average_score -30.0
    episode 2398 score -41.5 average_score -30.1
    episode 2399 score -18.8 average_score -30.1
    episode 2400 score -18.2 average_score -29.9
    episode 2401 score -33.5 average_score -29.8
    episode 2402 score -52.0 average_score -30.1
    episode 2403 score -10.8 average_score -30.1
    episode 2404 score -28.8 average_score -30.2
    episode 2405 score -34.5 average_score -30.2
    episode 2406 score -0.8 average_score -30.1
    episode 2407 score -26.5 average_score -29.9
    episode 2408 score -43.8 average_score -30.1
    episode 2409 score -66.8 average_score -30.6
    episode 2410 score -19.5 average_score -30.5
    episode 2411 score -35.8 average_score -30.6
    episode 2412 score -4.8 average_score -30.3
    episode 2413 score -31.0 average_score -30.2
    episode 2414 score -12.2 average_score -29.9
    episode 2415 score -18.2 average_score -29.8
    episode 2416 score -38.5 average_score -29.8
    episode 2417 score -54.8 average_score -29.9
    episode 2418 score -16.5 average_score -29.7
    episode 2419 score -54.2 average_score -29.8
    episode 2420 score -27.8 average_score -29.7
    episode 2421 score -42.5 average_score -29.8
    episode 2422 score -38.2 average_score -29.6
    episode 2423 score -27.8 average_score -29.8
    episode 2424 score -54.5 average_score -30.1
    episode 2425 score -52.2 average_score -30.5
    episode 2426 score -4.0 average_score -30.4
    episode 2427 score -50.0 average_score -30.5
    episode 2428 score -19.5 average_score -30.5
    episode 2429 score -54.8 average_score -30.6
    episode 2430 score -12.2 average_score -30.4
    episode 2431 score -49.5 average_score -30.8
    episode 2432 score -26.8 average_score -30.9
    episode 2433 score -31.2 average_score -31.0
    episode 2434 score -11.2 average_score -31.0
    episode 2435 score -30.5 average_score -30.7
    episode 2436 score -22.0 average_score -30.5
    episode 2437 score -31.2 average_score -30.3
    episode 2438 score -36.5 average_score -30.4
    episode 2439 score -26.8 average_score -30.1
    episode 2440 score -17.2 average_score -29.8
    episode 2441 score -25.8 average_score -29.9
    episode 2442 score -26.8 average_score -29.8
    episode 2443 score -24.0 average_score -29.8
    episode 2444 score -37.2 average_score -29.8
    episode 2445 score 0.0 average_score -29.5
    episode 2446 score -37.8 average_score -29.7
    episode 2447 score 14.2 average_score -29.2
    episode 2448 score -81.8 average_score -29.8
    episode 2449 score -27.2 average_score -29.6
    episode 2450 score -44.8 average_score -30.0
    episode 2451 score -57.8 average_score -30.3
    episode 2452 score -38.5 average_score -30.6
    episode 2453 score -21.2 average_score -30.4
    episode 2454 score -34.5 average_score -30.5
    episode 2455 score -50.0 average_score -30.3
    episode 2456 score -33.8 average_score -30.3
    episode 2457 score -32.2 average_score -30.5
    episode 2458 score -30.2 average_score -30.2
    episode 2459 score -22.0 average_score -30.2
    episode 2460 score -24.5 average_score -30.2
    episode 2461 score 3.0 average_score -29.9
    episode 2462 score -40.0 average_score -30.0
    episode 2463 score -16.8 average_score -30.0
    episode 2464 score -25.2 average_score -30.1
    episode 2465 score -42.8 average_score -30.2
    episode 2466 score -36.0 average_score -30.7
    episode 2467 score -42.0 average_score -30.7
    episode 2468 score -18.2 average_score -30.4
    episode 2469 score -29.2 average_score -30.1
    episode 2470 score -19.5 average_score -30.0
    episode 2471 score -43.8 average_score -30.1
    episode 2472 score -55.5 average_score -30.5
    episode 2473 score -4.8 average_score -29.9
    episode 2474 score -19.5 average_score -30.1
    episode 2475 score -66.5 average_score -30.7
    episode 2476 score -13.8 average_score -30.5
    episode 2477 score -37.2 average_score -30.8
    episode 2478 score -47.0 average_score -30.9
    episode 2479 score -35.2 average_score -31.1
    episode 2480 score -23.0 average_score -31.3
    episode 2481 score -8.5 average_score -31.0
    episode 2482 score -19.2 average_score -30.9
    episode 2483 score -31.5 average_score -31.0
    episode 2484 score -48.8 average_score -31.0
    episode 2485 score -16.0 average_score -30.8
    episode 2486 score -30.2 average_score -30.6
    episode 2487 score -23.5 average_score -30.6
    episode 2488 score -27.5 average_score -30.5
    episode 2489 score 15.2 average_score -30.0
    episode 2490 score -25.2 average_score -29.8
    episode 2491 score -8.0 average_score -29.7
    episode 2492 score -16.0 average_score -29.5
    episode 2493 score -58.2 average_score -29.8
    episode 2494 score -16.5 average_score -29.8
    episode 2495 score -2.5 average_score -29.5
    episode 2496 score -25.0 average_score -29.6
    episode 2497 score -19.2 average_score -29.4
    episode 2498 score -18.2 average_score -29.1
    episode 2499 score -30.0 average_score -29.3


# Trying the same agent on a gym environment
Below is the same model defined above, just edited to take the new input shapes from Gym's LunarLander-v2 environment. the model was able to be trained on this environment and was able to get positive rewards.


```python
from gym_model.Agent import GymAgent
from gym_model.gym_env import run_model
```

    Using TensorFlow backend.



```python
run_model()
```

    /home/mostafakm/Documents/School/Practical Data Science/Assignment 7/gym_model/Agent.py:41: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=[<tf.Tenso...)`
      policy = Model(input=[input, advantages], output=[probs])
    /home/mostafakm/Documents/School/Practical Data Science/Assignment 7/gym_model/Agent.py:43: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=[<tf.Tenso..., outputs=[<tf.Tenso...)`
      predict = Model(input=[input], output=[probs])


    episode 0 score -321.2 average_score -321.2
    episode 1 score -111.1 average_score -216.1
    episode 2 score -157.6 average_score -196.6
    episode 3 score -222.1 average_score -203.0
    episode 4 score -142.3 average_score -190.8
    episode 5 score -291.4 average_score -207.6
    episode 6 score -25.4 average_score -181.6
    episode 7 score -28.4 average_score -162.4
    episode 8 score -261.0 average_score -173.4
    episode 9 score -103.2 average_score -166.4
    episode 10 score -10.5 average_score -152.2
    episode 11 score -153.5 average_score -152.3
    episode 12 score -287.2 average_score -162.7
    episode 13 score -168.1 average_score -163.1
    episode 14 score -300.6 average_score -172.2
    episode 15 score -225.9 average_score -175.6
    episode 16 score -133.4 average_score -173.1
    episode 17 score -141.5 average_score -171.4
    episode 18 score -143.3 average_score -169.9
    episode 19 score -109.8 average_score -166.9
    episode 20 score -149.8 average_score -166.1
    episode 21 score -106.6 average_score -163.4
    episode 22 score -72.3 average_score -159.4
    episode 23 score -170.1 average_score -159.8
    episode 24 score -161.9 average_score -159.9
    episode 25 score -295.5 average_score -165.1
    episode 26 score -199.0 average_score -166.4
    episode 27 score -181.8 average_score -166.9
    episode 28 score -178.7 average_score -167.3
    episode 29 score -46.8 average_score -163.3
    episode 30 score 11.9 average_score -157.7
    episode 31 score -134.3 average_score -156.9
    episode 32 score -90.4 average_score -154.9
    episode 33 score -312.2 average_score -159.6
    episode 34 score -297.6 average_score -163.5
    episode 35 score 38.0 average_score -157.9
    episode 36 score -371.6 average_score -163.7
    episode 37 score -127.7 average_score -162.7
    episode 38 score -74.7 average_score -160.5
    episode 39 score -170.0 average_score -160.7
    episode 40 score -290.9 average_score -163.9
    episode 41 score -69.9 average_score -161.6
    episode 42 score -153.5 average_score -161.5
    episode 43 score -142.8 average_score -161.0
    episode 44 score -107.7 average_score -159.8
    episode 45 score -131.0 average_score -159.2
    episode 46 score -73.7 average_score -157.4
    episode 47 score -80.4 average_score -155.8
    episode 48 score -222.6 average_score -157.2
    episode 49 score -233.2 average_score -158.7
    episode 50 score -244.0 average_score -160.4
    episode 51 score -337.2 average_score -163.8
    episode 52 score -74.1 average_score -162.1
    episode 53 score -156.8 average_score -162.0
    episode 54 score -91.8 average_score -160.7
    episode 55 score -76.0 average_score -159.2
    episode 56 score -85.7 average_score -157.9
    episode 57 score -201.6 average_score -158.6
    episode 58 score -61.3 average_score -157.0
    episode 59 score -174.3 average_score -157.3
    episode 60 score -85.3 average_score -156.1
    episode 61 score -160.6 average_score -156.2
    episode 62 score -152.4 average_score -156.1
    episode 63 score -38.4 average_score -154.3
    episode 64 score -119.7 average_score -153.7
    episode 65 score -92.5 average_score -152.8
    episode 66 score -83.9 average_score -151.8
    episode 67 score -2.7 average_score -149.6
    episode 68 score -391.1 average_score -153.1
    episode 69 score -172.2 average_score -153.4
    episode 70 score -218.5 average_score -154.3
    episode 71 score -161.7 average_score -154.4
    episode 72 score -94.3 average_score -153.6
    episode 73 score -167.3 average_score -153.7
    episode 74 score -114.7 average_score -153.2
    episode 75 score -366.0 average_score -156.0
    episode 76 score -96.4 average_score -155.3
    episode 77 score -150.1 average_score -155.2
    episode 78 score -262.2 average_score -156.5
    episode 79 score -67.9 average_score -155.4
    episode 80 score -167.4 average_score -155.6
    episode 81 score -57.5 average_score -154.4
    episode 82 score -69.0 average_score -153.4
    episode 83 score -75.3 average_score -152.4
    episode 84 score -120.8 average_score -152.1
    episode 85 score -104.5 average_score -151.5
    episode 86 score -137.8 average_score -151.3
    episode 87 score -155.0 average_score -151.4
    episode 88 score -202.1 average_score -152.0
    episode 89 score -256.4 average_score -153.1
    episode 90 score -102.3 average_score -152.6
    episode 91 score -71.0 average_score -151.7
    episode 92 score -150.1 average_score -151.7
    episode 93 score -311.7 average_score -153.4
    episode 94 score -225.9 average_score -154.1
    episode 95 score -123.4 average_score -153.8
    episode 96 score -102.0 average_score -153.3
    episode 97 score -142.9 average_score -153.2
    episode 98 score -42.9 average_score -152.0
    episode 99 score -113.6 average_score -151.7
    episode 100 score -126.1 average_score -149.7
    episode 101 score -132.5 average_score -149.9
    episode 102 score -203.0 average_score -150.4
    episode 103 score -125.6 average_score -149.4
    episode 104 score -186.2 average_score -149.9
    episode 105 score -134.3 average_score -148.3
    episode 106 score -43.3 average_score -148.5
    episode 107 score -286.3 average_score -151.0
    episode 108 score -87.8 average_score -149.3
    episode 109 score -74.4 average_score -149.0
    episode 110 score -104.8 average_score -150.0
    episode 111 score -118.9 average_score -149.6
    episode 112 score -106.3 average_score -147.8
    episode 113 score -217.9 average_score -148.3
    episode 114 score -119.6 average_score -146.5
    episode 115 score -249.3 average_score -146.7
    episode 116 score -117.6 average_score -146.6
    episode 117 score -112.2 average_score -146.3
    episode 118 score -111.5 average_score -146.0
    episode 119 score -85.9 average_score -145.7
    episode 120 score -140.0 average_score -145.6
    episode 121 score -73.6 average_score -145.3
    episode 122 score -292.5 average_score -147.5
    episode 123 score -170.7 average_score -147.5
    episode 124 score -24.1 average_score -146.1
    episode 125 score 39.5 average_score -142.8
    episode 126 score -87.8 average_score -141.7
    episode 127 score -41.5 average_score -140.3
    episode 128 score -87.3 average_score -139.3
    episode 129 score -49.0 average_score -139.4
    episode 130 score -58.1 average_score -140.1
    episode 131 score -221.7 average_score -140.9
    episode 132 score -210.5 average_score -142.1
    episode 133 score -147.7 average_score -140.5
    episode 134 score -62.3 average_score -138.1
    episode 135 score -274.9 average_score -141.3
    episode 136 score -92.0 average_score -138.5
    episode 137 score -114.1 average_score -138.3
    episode 138 score -243.5 average_score -140.0
    episode 139 score -127.2 average_score -139.6
    episode 140 score -56.1 average_score -137.3
    episode 141 score -141.1 average_score -138.0
    episode 142 score -112.4 average_score -137.6
    episode 143 score -176.2 average_score -137.9
    episode 144 score -75.5 average_score -137.6
    episode 145 score -122.7 average_score -137.5
    episode 146 score -316.5 average_score -139.9
    episode 147 score -82.4 average_score -139.9
    episode 148 score -83.8 average_score -138.5
    episode 149 score -60.6 average_score -136.8
    episode 150 score -125.3 average_score -135.6
    episode 151 score -121.9 average_score -133.5
    episode 152 score -64.4 average_score -133.4
    episode 153 score -87.8 average_score -132.7
    episode 154 score 18.9 average_score -131.6
    episode 155 score -117.3 average_score -132.0
    episode 156 score -116.0 average_score -132.3
    episode 157 score -161.2 average_score -131.9
    episode 158 score -385.4 average_score -135.1
    episode 159 score -44.6 average_score -133.8
    episode 160 score -206.1 average_score -135.1
    episode 161 score -145.0 average_score -134.9
    episode 162 score -34.6 average_score -133.7
    episode 163 score -78.2 average_score -134.1
    episode 164 score -186.9 average_score -134.8
    episode 165 score -74.5 average_score -134.6
    episode 166 score -97.0 average_score -134.7
    episode 167 score -81.5 average_score -135.5
    episode 168 score -127.4 average_score -132.9
    episode 169 score -177.9 average_score -132.9
    episode 170 score -110.6 average_score -131.9
    episode 171 score -78.8 average_score -131.0
    episode 172 score -104.5 average_score -131.1
    episode 173 score -364.0 average_score -133.1
    episode 174 score -139.2 average_score -133.3
    episode 175 score -117.6 average_score -130.9
    episode 176 score -484.6 average_score -134.7
    episode 177 score -80.4 average_score -134.0
    episode 178 score -84.2 average_score -132.3
    episode 179 score -63.7 average_score -132.2
    episode 180 score -189.7 average_score -132.5
    episode 181 score -216.6 average_score -134.0
    episode 182 score -145.7 average_score -134.8
    episode 183 score -172.7 average_score -135.8
    episode 184 score -221.7 average_score -136.8
    episode 185 score -75.1 average_score -136.5
    episode 186 score -61.5 average_score -135.7
    episode 187 score -17.8 average_score -134.4
    episode 188 score -181.3 average_score -134.2
    episode 189 score -97.3 average_score -132.6
    episode 190 score -293.7 average_score -134.5
    episode 191 score -65.9 average_score -134.4
    episode 192 score -183.1 average_score -134.8
    episode 193 score -78.7 average_score -132.4
    episode 194 score -346.4 average_score -133.6
    episode 195 score -143.5 average_score -133.8
    episode 196 score -60.9 average_score -133.4
    episode 197 score -110.3 average_score -133.1
    episode 198 score -115.0 average_score -133.8
    episode 199 score -326.4 average_score -135.9
    episode 200 score -268.0 average_score -137.4
    episode 201 score -73.4 average_score -136.8
    episode 202 score -175.9 average_score -136.5
    episode 203 score -129.7 average_score -136.5
    episode 204 score -414.1 average_score -138.8
    episode 205 score -54.1 average_score -138.0
    episode 206 score -178.8 average_score -139.4
    episode 207 score -31.8 average_score -136.8
    episode 208 score -128.9 average_score -137.2
    episode 209 score -349.9 average_score -140.0
    episode 210 score -104.0 average_score -140.0
    episode 211 score -90.9 average_score -139.7
    episode 212 score -175.8 average_score -140.4
    episode 213 score -89.4 average_score -139.1
    episode 214 score -120.8 average_score -139.1
    episode 215 score -54.2 average_score -137.2
    episode 216 score -18.9 average_score -136.2
    episode 217 score -41.4 average_score -135.5
    episode 218 score -124.2 average_score -135.6
    episode 219 score -54.9 average_score -135.3
    episode 220 score -63.8 average_score -134.5
    episode 221 score -138.7 average_score -135.2
    episode 222 score -85.9 average_score -133.1
    episode 223 score -165.6 average_score -133.1
    episode 224 score -115.1 average_score -134.0
    episode 225 score -104.0 average_score -135.4
    episode 226 score -72.9 average_score -135.3
    episode 227 score -57.9 average_score -135.4
    episode 228 score -320.6 average_score -137.8
    episode 229 score -296.0 average_score -140.2
    episode 230 score -161.0 average_score -141.3
    episode 231 score -19.3 average_score -139.2
    episode 232 score -212.1 average_score -139.3
    episode 233 score -72.1 average_score -138.5
    episode 234 score -74.5 average_score -138.6
    episode 235 score -51.6 average_score -136.4
    episode 236 score -258.2 average_score -138.1
    episode 237 score -264.3 average_score -139.6
    episode 238 score -100.2 average_score -138.1
    episode 239 score -78.2 average_score -137.6
    episode 240 score -143.5 average_score -138.5
    episode 241 score -52.2 average_score -137.6
    episode 242 score -105.8 average_score -137.6
    episode 243 score -83.1 average_score -136.6
    episode 244 score -9.9 average_score -136.0
    episode 245 score -78.1 average_score -135.5
    episode 246 score -108.2 average_score -133.4
    episode 247 score -44.0 average_score -133.1
    episode 248 score -146.9 average_score -133.7
    episode 249 score -103.0 average_score -134.1
    episode 250 score -86.5 average_score -133.7
    episode 251 score -71.7 average_score -133.2
    episode 252 score -142.3 average_score -134.0
    episode 253 score -164.5 average_score -134.8
    episode 254 score -173.3 average_score -136.7
    episode 255 score -88.1 average_score -136.4
    episode 256 score -145.5 average_score -136.7
    episode 257 score -32.6 average_score -135.4
    episode 258 score -308.6 average_score -134.6
    episode 259 score -148.9 average_score -135.7
    episode 260 score -73.2 average_score -134.4
    episode 261 score -266.9 average_score -135.6
    episode 262 score -261.3 average_score -137.8
    episode 263 score -230.9 average_score -139.4
    episode 264 score -95.0 average_score -138.4
    episode 265 score -158.0 average_score -139.3
    episode 266 score -330.4 average_score -141.6
    episode 267 score -103.5 average_score -141.8
    episode 268 score -91.8 average_score -141.5
    episode 269 score -299.5 average_score -142.7
    episode 270 score -68.9 average_score -142.3
    episode 271 score -103.4 average_score -142.5
    episode 272 score -119.2 average_score -142.7
    episode 273 score -95.2 average_score -140.0
    episode 274 score -48.0 average_score -139.1
    episode 275 score -325.9 average_score -141.2
    episode 276 score -43.5 average_score -136.7
    episode 277 score -355.9 average_score -139.5
    episode 278 score -96.4 average_score -139.6
    episode 279 score -75.9 average_score -139.7
    episode 280 score -204.4 average_score -139.9
    episode 281 score -163.9 average_score -139.4
    episode 282 score -77.1 average_score -138.7
    episode 283 score -76.5 average_score -137.7
    episode 284 score -381.1 average_score -139.3
    episode 285 score 14.9 average_score -138.4
    episode 286 score -80.4 average_score -138.6
    episode 287 score -215.8 average_score -140.6
    episode 288 score -4.5 average_score -138.8
    episode 289 score -105.9 average_score -138.9
    episode 290 score -12.7 average_score -136.1
    episode 291 score -120.7 average_score -136.6
    episode 292 score -221.4 average_score -137.0
    episode 293 score -106.3 average_score -137.3
    episode 294 score -253.0 average_score -136.4
    episode 295 score -41.0 average_score -135.3
    episode 296 score -95.9 average_score -135.7
    episode 297 score -60.8 average_score -135.2
    episode 298 score -54.4 average_score -134.6
    episode 299 score -391.2 average_score -135.2
    episode 300 score -117.2 average_score -133.7
    episode 301 score -170.5 average_score -134.7
    episode 302 score -44.5 average_score -133.4
    episode 303 score -33.0 average_score -132.4
    episode 304 score -29.9 average_score -128.6
    episode 305 score -140.7 average_score -129.4
    episode 306 score -142.4 average_score -129.1
    episode 307 score -103.8 average_score -129.8
    episode 308 score -207.3 average_score -130.6
    episode 309 score -53.7 average_score -127.6
    episode 310 score -20.9 average_score -126.8
    episode 311 score -112.7 average_score -127.0
    episode 312 score -82.8 average_score -126.1
    episode 313 score -248.5 average_score -127.7
    episode 314 score -64.7 average_score -127.1
    episode 315 score -101.5 average_score -127.6
    episode 316 score -39.4 average_score -127.8
    episode 317 score -73.8 average_score -128.1
    episode 318 score -108.9 average_score -128.0
    episode 319 score -149.3 average_score -128.9
    episode 320 score -254.0 average_score -130.8
    episode 321 score -69.9 average_score -130.1
    episode 322 score -140.3 average_score -130.7
    episode 323 score -28.1 average_score -129.3
    episode 324 score -43.3 average_score -128.6
    episode 325 score -281.3 average_score -130.3
    episode 326 score -56.5 average_score -130.2
    episode 327 score -62.5 average_score -130.2
    episode 328 score -3.4 average_score -127.0
    episode 329 score -378.6 average_score -127.9
    episode 330 score -94.8 average_score -127.2
    episode 331 score -225.5 average_score -129.3
    episode 332 score -74.0 average_score -127.9
    episode 333 score -58.0 average_score -127.7
    episode 334 score -54.2 average_score -127.5
    episode 335 score -226.2 average_score -129.3
    episode 336 score -480.8 average_score -131.5
    episode 337 score -133.1 average_score -130.2
    episode 338 score -225.1 average_score -131.5
    episode 339 score -50.4 average_score -131.2
    episode 340 score -338.8 average_score -133.1
    episode 341 score -86.3 average_score -133.5
    episode 342 score -78.1 average_score -133.2
    episode 343 score -251.7 average_score -134.9
    episode 344 score -331.4 average_score -138.1
    episode 345 score -81.8 average_score -138.1
    episode 346 score -99.3 average_score -138.0
    episode 347 score -122.3 average_score -138.8
    episode 348 score -107.8 average_score -138.4
    episode 349 score -48.8 average_score -137.9
    episode 350 score -139.3 average_score -138.4
    episode 351 score -260.2 average_score -140.3
    episode 352 score -56.1 average_score -139.4
    episode 353 score -178.1 average_score -139.6
    episode 354 score -27.2 average_score -138.1
    episode 355 score -424.6 average_score -141.5
    episode 356 score -130.6 average_score -141.3
    episode 357 score -215.0 average_score -143.2
    episode 358 score -63.5 average_score -140.7
    episode 359 score -85.1 average_score -140.1
    episode 360 score -284.3 average_score -142.2
    episode 361 score -79.8 average_score -140.3
    episode 362 score -82.1 average_score -138.5
    episode 363 score -221.7 average_score -138.4
    episode 364 score -93.6 average_score -138.4
    episode 365 score -79.0 average_score -137.6
    episode 366 score -51.2 average_score -134.8
    episode 367 score -71.6 average_score -134.5
    episode 368 score -74.6 average_score -134.3
    episode 369 score -167.6 average_score -133.0
    episode 370 score -194.5 average_score -134.3
    episode 371 score 29.9 average_score -132.9
    episode 372 score -117.2 average_score -132.9
    episode 373 score -267.8 average_score -134.6
    episode 374 score -52.6 average_score -134.7
    episode 375 score -91.7 average_score -132.4
    episode 376 score -260.8 average_score -134.5
    episode 377 score -170.7 average_score -132.7
    episode 378 score -179.8 average_score -133.5
    episode 379 score 33.3 average_score -132.4
    episode 380 score -203.3 average_score -132.4
    episode 381 score -126.4 average_score -132.0
    episode 382 score -74.4 average_score -132.0
    episode 383 score -81.4 average_score -132.0
    episode 384 score -221.2 average_score -130.5
    episode 385 score -83.9 average_score -131.4
    episode 386 score -267.4 average_score -133.3
    episode 387 score -91.3 average_score -132.1
    episode 388 score -59.2 average_score -132.6
    episode 389 score -87.6 average_score -132.4
    episode 390 score -73.1 average_score -133.0
    episode 391 score -53.5 average_score -132.4
    episode 392 score -158.6 average_score -131.7
    episode 393 score -67.7 average_score -131.3
    episode 394 score -80.5 average_score -129.6
    episode 395 score -126.9 average_score -130.5
    episode 396 score -105.5 average_score -130.6
    episode 397 score -364.6 average_score -133.6
    episode 398 score -43.6 average_score -133.5
    episode 399 score -374.1 average_score -133.3
    episode 400 score -23.2 average_score -132.4
    episode 401 score -98.8 average_score -131.7
    episode 402 score -114.8 average_score -132.4
    episode 403 score -167.8 average_score -133.7
    episode 404 score -379.6 average_score -137.2
    episode 405 score -27.6 average_score -136.1
    episode 406 score -105.0 average_score -135.7
    episode 407 score -75.2 average_score -135.4
    episode 408 score -129.1 average_score -134.7
    episode 409 score -85.5 average_score -135.0
    episode 410 score -296.3 average_score -137.7
    episode 411 score -64.5 average_score -137.2
    episode 412 score -108.5 average_score -137.5
    episode 413 score -154.8 average_score -136.6
    episode 414 score -95.6 average_score -136.9
    episode 415 score -164.4 average_score -137.5
    episode 416 score -32.5 average_score -137.4
    episode 417 score -16.1 average_score -136.9
    episode 418 score -122.5 average_score -137.0
    episode 419 score -58.4 average_score -136.1
    episode 420 score -50.8 average_score -134.0
    episode 421 score -253.5 average_score -135.9
    episode 422 score -86.1 average_score -135.3
    episode 423 score -70.0 average_score -135.8
    episode 424 score -217.8 average_score -137.5
    episode 425 score -269.2 average_score -137.4
    episode 426 score -126.1 average_score -138.1
    episode 427 score -124.4 average_score -138.7
    episode 428 score -130.3 average_score -140.0
    episode 429 score -28.5 average_score -136.5
    episode 430 score -50.8 average_score -136.0
    episode 431 score -159.7 average_score -135.4
    episode 432 score -63.8 average_score -135.3
    episode 433 score -110.6 average_score -135.8
    episode 434 score -123.0 average_score -136.5
    episode 435 score -33.6 average_score -134.6
    episode 436 score -64.8 average_score -130.4
    episode 437 score -31.2 average_score -129.4
    episode 438 score -376.7 average_score -130.9
    episode 439 score -39.1 average_score -130.8
    episode 440 score -125.5 average_score -128.6
    episode 441 score -92.7 average_score -128.7
    episode 442 score -150.7 average_score -129.4
    episode 443 score -71.9 average_score -127.6
    episode 444 score -225.5 average_score -126.6
    episode 445 score -67.9 average_score -126.4
    episode 446 score -16.8 average_score -125.6
    episode 447 score -88.8 average_score -125.3
    episode 448 score -41.8 average_score -124.6
    episode 449 score -53.0 average_score -124.7
    episode 450 score -3.9 average_score -123.3
    episode 451 score -15.6 average_score -120.9
    episode 452 score -198.4 average_score -122.3
    episode 453 score -5.7 average_score -120.6
    episode 454 score -105.6 average_score -121.3
    episode 455 score -109.2 average_score -118.2
    episode 456 score -90.5 average_score -117.8
    episode 457 score -191.4 average_score -117.6
    episode 458 score -131.2 average_score -118.2
    episode 459 score -82.0 average_score -118.2
    episode 460 score -239.0 average_score -117.7
    episode 461 score 2.1 average_score -116.9
    episode 462 score -93.5 average_score -117.0
    episode 463 score -49.2 average_score -115.3
    episode 464 score -100.9 average_score -115.4
    episode 465 score -50.8 average_score -115.1
    episode 466 score -28.6 average_score -114.9
    episode 467 score -165.7 average_score -115.8
    episode 468 score -386.7 average_score -118.9
    episode 469 score -81.9 average_score -118.1
    episode 470 score -33.5 average_score -116.5
    episode 471 score -127.5 average_score -118.1
    episode 472 score -117.6 average_score -118.1
    episode 473 score -237.4 average_score -117.8
    episode 474 score -295.1 average_score -120.2
    episode 475 score -94.5 average_score -120.2
    episode 476 score -183.6 average_score -119.4
    episode 477 score -40.0 average_score -118.1
    episode 478 score -203.5 average_score -118.4
    episode 479 score -129.2 average_score -120.0
    episode 480 score -47.8 average_score -118.4
    episode 481 score -304.8 average_score -120.2
    episode 482 score -151.0 average_score -121.0
    episode 483 score -65.5 average_score -120.8
    episode 484 score -57.4 average_score -119.2
    episode 485 score -453.8 average_score -122.9
    episode 486 score -134.0 average_score -121.6
    episode 487 score -103.3 average_score -121.7
    episode 488 score -230.3 average_score -123.4
    episode 489 score -256.8 average_score -125.1
    episode 490 score -22.2 average_score -124.6
    episode 491 score -351.0 average_score -127.5
    episode 492 score -63.5 average_score -126.6
    episode 493 score -86.8 average_score -126.8
    episode 494 score -285.2 average_score -128.8
    episode 495 score -164.0 average_score -129.2
    episode 496 score -213.2 average_score -130.3
    episode 497 score -57.6 average_score -127.2
    episode 498 score -106.6 average_score -127.8
    episode 499 score -45.8 average_score -124.6
    episode 500 score -97.1 average_score -125.3
    episode 501 score -77.2 average_score -125.1
    episode 502 score -40.4 average_score -124.3
    episode 503 score 23.6 average_score -122.4
    episode 504 score -200.1 average_score -120.6
    episode 505 score -17.0 average_score -120.5
    episode 506 score -158.3 average_score -121.1
    episode 507 score 30.7 average_score -120.0
    episode 508 score 67.6 average_score -118.0
    episode 509 score -15.6 average_score -117.3
    episode 510 score -81.1 average_score -115.2
    episode 511 score -83.1 average_score -115.4
    episode 512 score -317.2 average_score -117.4
    episode 513 score -45.7 average_score -116.4
    episode 514 score -194.0 average_score -117.3
    episode 515 score -26.5 average_score -116.0
    episode 516 score -75.7 average_score -116.4
    episode 517 score -6.0 average_score -116.3
    episode 518 score -65.7 average_score -115.7
    episode 519 score -213.4 average_score -117.3
    episode 520 score -269.1 average_score -119.5
    episode 521 score -51.1 average_score -117.4
    episode 522 score -66.3 average_score -117.2
    episode 523 score -60.8 average_score -117.1
    episode 524 score 15.6 average_score -114.8
    episode 525 score -24.3 average_score -112.4
    episode 526 score -76.0 average_score -111.9
    episode 527 score -78.8 average_score -111.4
    episode 528 score 2.7 average_score -110.1
    episode 529 score -212.8 average_score -111.9
    episode 530 score -66.9 average_score -112.1
    episode 531 score -64.6 average_score -111.1
    episode 532 score -197.3 average_score -112.5
    episode 533 score -37.4 average_score -111.7
    episode 534 score -334.0 average_score -113.8
    episode 535 score -46.8 average_score -114.0
    episode 536 score -74.0 average_score -114.1
    episode 537 score -84.8 average_score -114.6
    episode 538 score -264.4 average_score -113.5
    episode 539 score -107.3 average_score -114.2
    episode 540 score -44.9 average_score -113.3
    episode 541 score -36.9 average_score -112.8
    episode 542 score -47.7 average_score -111.8
    episode 543 score -281.7 average_score -113.9
    episode 544 score -78.6 average_score -112.4
    episode 545 score -311.9 average_score -114.8
    episode 546 score 3.4 average_score -114.6
    episode 547 score -283.0 average_score -116.6
    episode 548 score -22.4 average_score -116.4
    episode 549 score -81.7 average_score -116.7
    episode 550 score -28.3 average_score -116.9
    episode 551 score -130.4 average_score -118.1
    episode 552 score -131.8 average_score -117.4
    episode 553 score -126.9 average_score -118.6
    episode 554 score 25.7 average_score -117.3
    episode 555 score -53.1 average_score -116.7
    episode 556 score 1.6 average_score -115.8
    episode 557 score -119.5 average_score -115.1
    episode 558 score -92.3 average_score -114.7
    episode 559 score -12.1 average_score -114.0
    episode 560 score -96.5 average_score -112.6
    episode 561 score -102.9 average_score -113.6
    episode 562 score -86.8 average_score -113.6
    episode 563 score -35.1 average_score -113.4
    episode 564 score -230.1 average_score -114.7
    episode 565 score -71.2 average_score -114.9
    episode 566 score -5.8 average_score -114.7
    episode 567 score -64.2 average_score -113.7
    episode 568 score -53.0 average_score -110.3
    episode 569 score -77.4 average_score -110.3
    episode 570 score -28.7 average_score -110.2
    episode 571 score -77.9 average_score -109.7
    episode 572 score -47.1 average_score -109.0
    episode 573 score -45.1 average_score -107.1
    episode 574 score -233.0 average_score -106.5
    episode 575 score -87.8 average_score -106.4
    episode 576 score 9.7 average_score -104.5
    episode 577 score -18.6 average_score -104.3
    episode 578 score -53.5 average_score -102.8
    episode 579 score -46.3 average_score -102.0
    episode 580 score -100.8 average_score -102.5
    episode 581 score -63.9 average_score -100.1
    episode 582 score -3.1 average_score -98.6
    episode 583 score -59.8 average_score -98.5
    episode 584 score -94.1 average_score -98.9
    episode 585 score -56.3 average_score -94.9
    episode 586 score -86.4 average_score -94.5
    episode 587 score -56.7 average_score -94.0
    episode 588 score -29.5 average_score -92.0
    episode 589 score -163.2 average_score -91.0
    episode 590 score -4.0 average_score -90.9
    episode 591 score -240.8 average_score -89.8
    episode 592 score -143.6 average_score -90.6
    episode 593 score -60.4 average_score -90.3
    episode 594 score -174.0 average_score -89.2
    episode 595 score -77.3 average_score -88.3
    episode 596 score -132.5 average_score -87.5
    episode 597 score -72.9 average_score -87.7
    episode 598 score -75.9 average_score -87.4
    episode 599 score -140.2 average_score -88.3
    episode 600 score 23.3 average_score -87.1
    episode 601 score -34.8 average_score -86.7
    episode 602 score -52.2 average_score -86.8
    episode 603 score -74.4 average_score -87.8
    episode 604 score -24.8 average_score -86.0
    episode 605 score -41.7 average_score -86.3
    episode 606 score -97.9 average_score -85.7
    episode 607 score 9.1 average_score -85.9
    episode 608 score -175.8 average_score -88.3
    episode 609 score -139.6 average_score -89.5
    episode 610 score -114.3 average_score -89.9
    episode 611 score -66.2 average_score -89.7
    episode 612 score -10.8 average_score -86.6
    episode 613 score -42.6 average_score -86.6
    episode 614 score -15.6 average_score -84.8
    episode 615 score -62.6 average_score -85.2
    episode 616 score -19.4 average_score -84.6
    episode 617 score -49.0 average_score -85.1
    episode 618 score -20.0 average_score -84.6
    episode 619 score -53.8 average_score -83.0
    episode 620 score -262.2 average_score -82.9
    episode 621 score -176.0 average_score -84.2
    episode 622 score -29.2 average_score -83.8
    episode 623 score -38.8 average_score -83.6
    episode 624 score -219.9 average_score -86.0
    episode 625 score -86.8 average_score -86.6
    episode 626 score -146.3 average_score -87.3
    episode 627 score -44.3 average_score -86.9
    episode 628 score 35.1 average_score -86.6
    episode 629 score -306.8 average_score -87.6
    episode 630 score -32.3 average_score -87.2
    episode 631 score 64.4 average_score -85.9
    episode 632 score -74.7 average_score -84.7
    episode 633 score -61.0 average_score -84.9
    episode 634 score -124.3 average_score -82.8
    episode 635 score -22.9 average_score -82.6
    episode 636 score -9.1 average_score -81.9
    episode 637 score 30.2 average_score -80.8
    episode 638 score -31.3 average_score -78.5
    episode 639 score -43.2 average_score -77.8
    episode 640 score -144.8 average_score -78.8
    episode 641 score -8.8 average_score -78.5
    episode 642 score -9.0 average_score -78.1
    episode 643 score -48.4 average_score -75.8
    episode 644 score -36.6 average_score -75.4
    episode 645 score -14.1 average_score -72.4
    episode 646 score -93.5 average_score -73.4
    episode 647 score -13.5 average_score -70.7
    episode 648 score -52.4 average_score -71.0
    episode 649 score -5.3 average_score -70.2
    episode 650 score -183.5 average_score -71.8
    episode 651 score -178.8 average_score -72.3
    episode 652 score 1.4 average_score -70.9
    episode 653 score -139.5 average_score -71.1
    episode 654 score -135.0 average_score -72.7
    episode 655 score -60.8 average_score -72.7
    episode 656 score -64.9 average_score -73.4
    episode 657 score -95.2 average_score -73.2
    episode 658 score -38.8 average_score -72.6
    episode 659 score 42.0 average_score -72.1
    episode 660 score -57.3 average_score -71.7
    episode 661 score -40.3 average_score -71.1
    episode 662 score -54.3 average_score -70.7
    episode 663 score -76.9 average_score -71.2
    episode 664 score -75.6 average_score -69.6
    episode 665 score -46.5 average_score -69.4
    episode 666 score -172.1 average_score -71.0
    episode 667 score -89.9 average_score -71.3
    episode 668 score -17.4 average_score -70.9
    episode 669 score 35.2 average_score -69.8
    episode 670 score -17.6 average_score -69.7
    episode 671 score -16.8 average_score -69.1
    episode 672 score -117.2 average_score -69.8
    episode 673 score -62.4 average_score -70.0
    episode 674 score -58.1 average_score -68.2
    episode 675 score -36.5 average_score -67.7
    episode 676 score -55.6 average_score -68.3
    episode 677 score -141.4 average_score -69.6
    episode 678 score -17.9 average_score -69.2
    episode 679 score -27.1 average_score -69.0
    episode 680 score -8.9 average_score -68.1
    episode 681 score -25.1 average_score -67.7
    episode 682 score -83.4 average_score -68.5
    episode 683 score 21.6 average_score -67.7
    episode 684 score -51.0 average_score -67.3
    episode 685 score -17.4 average_score -66.9
    episode 686 score -75.8 average_score -66.8
    episode 687 score -138.2 average_score -67.6
    episode 688 score -32.6 average_score -67.6
    episode 689 score -283.5 average_score -68.8
    episode 690 score -92.8 average_score -69.7
    episode 691 score -204.7 average_score -69.4
    episode 692 score -144.1 average_score -69.4
    episode 693 score -76.4 average_score -69.5
    episode 694 score -44.4 average_score -68.2
    episode 695 score -57.7 average_score -68.0
    episode 696 score -115.8 average_score -67.9
    episode 697 score -47.8 average_score -67.6
    episode 698 score 5.7 average_score -66.8
    episode 699 score -113.0 average_score -66.5
    episode 700 score -55.2 average_score -67.3
    episode 701 score -18.6 average_score -67.1
    episode 702 score -94.1 average_score -67.6
    episode 703 score -209.2 average_score -68.9
    episode 704 score 11.8 average_score -68.6
    episode 705 score 71.9 average_score -67.4
    episode 706 score -66.2 average_score -67.1
    episode 707 score -23.0 average_score -67.4
    episode 708 score 37.5 average_score -65.3
    episode 709 score 21.9 average_score -63.7
    episode 710 score 3.6 average_score -62.5
    episode 711 score -19.3 average_score -62.0
    episode 712 score -42.4 average_score -62.3
    episode 713 score -59.3 average_score -62.5
    episode 714 score -6.6 average_score -62.4
    episode 715 score -33.1 average_score -62.1
    episode 716 score -4.2 average_score -62.0
    episode 717 score -42.4 average_score -61.9
    episode 718 score -7.2 average_score -61.8
    episode 719 score 24.9 average_score -61.0
    episode 720 score -38.7 average_score -58.8
    episode 721 score -58.4 average_score -57.6
    episode 722 score 7.9 average_score -57.2
    episode 723 score 1.1 average_score -56.8
    episode 724 score -262.4 average_score -57.2
    episode 725 score 56.9 average_score -55.8
    episode 726 score -68.9 average_score -55.0
    episode 727 score -77.0 average_score -55.3
    episode 728 score -161.0 average_score -57.3
    episode 729 score -57.3 average_score -54.8
    episode 730 score -16.2 average_score -54.7
    episode 731 score -166.9 average_score -57.0
    episode 732 score -127.3 average_score -57.5
    episode 733 score -241.4 average_score -59.3
    episode 734 score 5.6 average_score -58.0
    episode 735 score -193.4 average_score -59.7
    episode 736 score -90.0 average_score -60.5
    episode 737 score 19.4 average_score -60.6
    episode 738 score -75.8 average_score -61.1
    episode 739 score -63.8 average_score -61.3
    episode 740 score 6.6 average_score -59.8
    episode 741 score -59.1 average_score -60.3
    episode 742 score -371.8 average_score -63.9
    episode 743 score -68.9 average_score -64.1
    episode 744 score -20.6 average_score -63.9
    episode 745 score -61.8 average_score -64.4
    episode 746 score -2.2 average_score -63.5
    episode 747 score 46.8 average_score -62.9
    episode 748 score -61.6 average_score -63.0
    episode 749 score -68.4 average_score -63.6
    episode 750 score -75.5 average_score -62.5
    episode 751 score -8.3 average_score -60.8
    episode 752 score -165.7 average_score -62.5
    episode 753 score -52.7 average_score -61.6
    episode 754 score -14.8 average_score -60.4
    episode 755 score -24.9 average_score -60.1
    episode 756 score -18.1 average_score -59.6
    episode 757 score -27.8 average_score -58.9
    episode 758 score -5.1 average_score -58.6
    episode 759 score -74.4 average_score -59.8
    episode 760 score -8.7 average_score -59.3
    episode 761 score 17.3 average_score -58.7
    episode 762 score -148.0 average_score -59.6
    episode 763 score -89.4 average_score -59.8
    episode 764 score -139.2 average_score -60.4
    episode 765 score -102.9 average_score -61.0
    episode 766 score 28.6 average_score -58.9
    episode 767 score -16.0 average_score -58.2
    episode 768 score -42.1 average_score -58.5
    episode 769 score -4.1 average_score -58.9
    episode 770 score -30.1 average_score -59.0
    episode 771 score -14.9 average_score -59.0
    episode 772 score -35.9 average_score -58.1
    episode 773 score -227.7 average_score -59.8
    episode 774 score 22.9 average_score -59.0
    episode 775 score -40.3 average_score -59.0
    episode 776 score -141.5 average_score -59.9
    episode 777 score -128.2 average_score -59.8
    episode 778 score -11.5 average_score -59.7
    episode 779 score -24.0 average_score -59.7
    episode 780 score -35.9 average_score -59.9
    episode 781 score 16.1 average_score -59.5
    episode 782 score -282.1 average_score -61.5
    episode 783 score -78.8 average_score -62.5
    episode 784 score 23.9 average_score -61.8
    episode 785 score 6.1 average_score -61.5
    episode 786 score 3.2 average_score -60.7
    episode 787 score -263.0 average_score -62.0
    episode 788 score -40.9 average_score -62.1
    episode 789 score -81.0 average_score -60.0
    episode 790 score 13.4 average_score -59.0
    episode 791 score -64.2 average_score -57.6
    episode 792 score -84.8 average_score -57.0
    episode 793 score -37.3 average_score -56.6
    episode 794 score -3.0 average_score -56.2
    episode 795 score -2.1 average_score -55.6
    episode 796 score -52.8 average_score -55.0
    episode 797 score -62.7 average_score -55.1
    episode 798 score -91.1 average_score -56.1
    episode 799 score -8.3 average_score -55.1
    episode 800 score -161.4 average_score -56.1
    episode 801 score -72.8 average_score -56.7
    episode 802 score 20.9 average_score -55.5
    episode 803 score -28.1 average_score -53.7
    episode 804 score -152.5 average_score -55.3
    episode 805 score -174.5 average_score -57.8
    episode 806 score 17.9 average_score -57.0
    episode 807 score -42.1 average_score -57.2
    episode 808 score -43.9 average_score -58.0
    episode 809 score -230.2 average_score -60.5
    episode 810 score -28.0 average_score -60.8
    episode 811 score -10.3 average_score -60.7
    episode 812 score -190.5 average_score -62.2
    episode 813 score -63.3 average_score -62.2
    episode 814 score -48.0 average_score -62.7
    episode 815 score -1.4 average_score -62.3
    episode 816 score -69.1 average_score -63.0
    episode 817 score -87.9 average_score -63.4
    episode 818 score -19.0 average_score -63.6
    episode 819 score 20.5 average_score -63.6
    episode 820 score 41.6 average_score -62.8
    episode 821 score -224.3 average_score -64.5
    episode 822 score -377.9 average_score -68.3
    episode 823 score -169.2 average_score -70.0
    episode 824 score -76.4 average_score -68.2
    episode 825 score -132.1 average_score -70.1
    episode 826 score -338.0 average_score -72.7
    episode 827 score -68.9 average_score -72.7
    episode 828 score -49.1 average_score -71.5
    episode 829 score 28.3 average_score -70.7
    episode 830 score -199.5 average_score -72.5
    episode 831 score -278.5 average_score -73.6
    episode 832 score 20.8 average_score -72.2
    episode 833 score -28.9 average_score -70.0
    episode 834 score 51.9 average_score -69.6
    episode 835 score -26.4 average_score -67.9
    episode 836 score -26.5 average_score -67.3
    episode 837 score -245.6 average_score -69.9
    episode 838 score -241.4 average_score -71.6
    episode 839 score 71.9 average_score -70.2
    episode 840 score -194.4 average_score -72.2
    episode 841 score -147.5 average_score -73.1
    episode 842 score 20.1 average_score -69.2
    episode 843 score -14.9 average_score -68.6
    episode 844 score -114.4 average_score -69.6
    episode 845 score 24.8 average_score -68.7
    episode 846 score -24.9 average_score -68.9
    episode 847 score -85.9 average_score -70.3
    episode 848 score -38.2 average_score -70.0
    episode 849 score -35.6 average_score -69.7
    episode 850 score -160.5 average_score -70.6
    episode 851 score -14.4 average_score -70.6
    episode 852 score 23.4 average_score -68.7
    episode 853 score 51.1 average_score -67.7
    episode 854 score -277.4 average_score -70.3
    episode 855 score -36.4 average_score -70.4
    episode 856 score -174.2 average_score -72.0
    episode 857 score -63.5 average_score -72.4
    episode 858 score -121.8 average_score -73.5
    episode 859 score -198.8 average_score -74.8
    episode 860 score 18.7 average_score -74.5
    episode 861 score -157.6 average_score -76.2
    episode 862 score -99.7 average_score -75.8
    episode 863 score -54.7 average_score -75.4
    episode 864 score 26.8 average_score -73.7
    episode 865 score 48.7 average_score -72.2
    episode 866 score -78.0 average_score -73.3
    episode 867 score -44.5 average_score -73.6
    episode 868 score -210.2 average_score -75.3
    episode 869 score -63.2 average_score -75.9
    episode 870 score 46.5 average_score -75.1
    episode 871 score -290.4 average_score -77.8
    episode 872 score 6.8 average_score -77.4
    episode 873 score 57.7 average_score -74.6
    episode 874 score 11.7 average_score -74.7
    episode 875 score 9.5 average_score -74.2
    episode 876 score -75.4 average_score -73.5
    episode 877 score -99.0 average_score -73.2
    episode 878 score -20.5 average_score -73.3
    episode 879 score -163.9 average_score -74.7
    episode 880 score -5.4 average_score -74.4
    episode 881 score 32.4 average_score -74.2
    episode 882 score 17.4 average_score -71.3
    episode 883 score 57.1 average_score -69.9
    episode 884 score 66.9 average_score -69.5
    episode 885 score -9.5 average_score -69.6
    episode 886 score -47.7 average_score -70.1
    episode 887 score -45.6 average_score -68.0
    episode 888 score -39.0 average_score -67.9
    episode 889 score -27.4 average_score -67.4
    episode 890 score 3.6 average_score -67.5
    episode 891 score 19.4 average_score -66.7
    episode 892 score -219.5 average_score -68.0
    episode 893 score 61.3 average_score -67.0
    episode 894 score 48.4 average_score -66.5
    episode 895 score -10.2 average_score -66.6
    episode 896 score -52.1 average_score -66.6
    episode 897 score -312.9 average_score -69.1
    episode 898 score 17.1 average_score -68.0
    episode 899 score -48.7 average_score -68.4
    episode 900 score 19.1 average_score -66.6
    episode 901 score 18.2 average_score -65.7
    episode 902 score -10.1 average_score -66.0
    episode 903 score 18.8 average_score -65.5
    episode 904 score 34.3 average_score -63.7
    episode 905 score -13.0 average_score -62.0
    episode 906 score 1.3 average_score -62.2
    episode 907 score -4.0 average_score -61.8
    episode 908 score -193.8 average_score -63.3
    episode 909 score -76.9 average_score -61.8
    episode 910 score -34.1 average_score -61.9
    episode 911 score -57.3 average_score -62.3
    episode 912 score -15.4 average_score -60.6
    episode 913 score -29.7 average_score -60.2
    episode 914 score 22.9 average_score -59.5
    episode 915 score 12.3 average_score -59.4
    episode 916 score 13.6 average_score -58.6
    episode 917 score -236.9 average_score -60.1
    episode 918 score -179.2 average_score -61.7
    episode 919 score 24.5 average_score -61.6
    episode 920 score 28.4 average_score -61.8
    episode 921 score -35.7 average_score -59.9
    episode 922 score 30.6 average_score -55.8
    episode 923 score -24.2 average_score -54.3
    episode 924 score -24.9 average_score -53.8
    episode 925 score 0.4 average_score -52.5
    episode 926 score 17.7 average_score -48.9
    episode 927 score 44.7 average_score -47.8
    episode 928 score -33.4 average_score -47.6
    episode 929 score 12.3 average_score -47.8
    episode 930 score 47.3 average_score -45.3
    episode 931 score -7.0 average_score -42.6
    episode 932 score -33.2 average_score -43.2
    episode 933 score -18.2 average_score -43.0
    episode 934 score -102.3 average_score -44.6
    episode 935 score -97.0 average_score -45.3
    episode 936 score -39.3 average_score -45.4
    episode 937 score -38.1 average_score -43.4
    episode 938 score -13.1 average_score -41.1
    episode 939 score -9.4 average_score -41.9
    episode 940 score 16.3 average_score -39.8
    episode 941 score 8.1 average_score -38.2
    episode 942 score -38.5 average_score -38.8
    episode 943 score 25.1 average_score -38.4
    episode 944 score 26.5 average_score -37.0
    episode 945 score 30.5 average_score -36.9
    episode 946 score 31.4 average_score -36.4
    episode 947 score 3.8 average_score -35.5
    episode 948 score 2.4 average_score -35.1
    episode 949 score 1.6 average_score -34.7
    episode 950 score -4.2 average_score -33.1
    episode 951 score -8.9 average_score -33.1
    episode 952 score 21.3 average_score -33.1
    episode 953 score -87.4 average_score -34.5
    episode 954 score -177.6 average_score -33.5
    episode 955 score -65.5 average_score -33.8
    episode 956 score 17.9 average_score -31.9
    episode 957 score -92.7 average_score -32.2
    episode 958 score 37.0 average_score -30.6
    episode 959 score 36.1 average_score -28.2
    episode 960 score -30.7 average_score -28.7
    episode 961 score -18.0 average_score -27.3
    episode 962 score 33.6 average_score -26.0
    episode 963 score 55.7 average_score -24.9
    episode 964 score -15.2 average_score -25.3
    episode 965 score -17.3 average_score -26.0
    episode 966 score -4.8 average_score -25.2
    episode 967 score -27.4 average_score -25.1
    episode 968 score -13.6 average_score -23.1
    episode 969 score 14.8 average_score -22.3
    episode 970 score 3.3 average_score -22.7
    episode 971 score 8.4 average_score -19.8
    episode 972 score 45.3 average_score -19.4
    episode 973 score -90.5 average_score -20.9
    episode 974 score -54.4 average_score -21.5
    episode 975 score -102.7 average_score -22.6
    episode 976 score 98.7 average_score -20.9
    episode 977 score -134.4 average_score -21.2
    episode 978 score 26.9 average_score -20.8
    episode 979 score 16.9 average_score -19.0
    episode 980 score 12.1 average_score -18.8
    episode 981 score 55.1 average_score -18.6
    episode 982 score -53.3 average_score -19.3
    episode 983 score -15.6 average_score -20.0
    episode 984 score 3.9 average_score -20.6
    episode 985 score -228.3 average_score -22.8
    episode 986 score 10.4 average_score -22.2
    episode 987 score -70.2 average_score -22.5
    episode 988 score 61.2 average_score -21.5
    episode 989 score -60.7 average_score -21.8
    episode 990 score -13.9 average_score -22.0
    episode 991 score -13.1 average_score -22.3
    episode 992 score -2.6 average_score -20.1
    episode 993 score -32.8 average_score -21.1
    episode 994 score 37.5 average_score -21.2
    episode 995 score 29.1 average_score -20.8
    episode 996 score 12.0 average_score -20.2
    episode 997 score 15.4 average_score -16.9
    episode 998 score 74.1 average_score -16.3
    episode 999 score 49.8 average_score -15.3
    episode 1000 score 39.5 average_score -15.1
    episode 1001 score 55.4 average_score -14.7
    episode 1002 score -8.3 average_score -14.7
    episode 1003 score -24.5 average_score -15.2
    episode 1004 score -0.9 average_score -15.5
    episode 1005 score -56.6 average_score -15.9
    episode 1006 score 6.1 average_score -15.9
    episode 1007 score -42.9 average_score -16.3
    episode 1008 score -25.7 average_score -14.6
    episode 1009 score -50.8 average_score -14.3
    episode 1010 score -12.6 average_score -14.1
    episode 1011 score -27.7 average_score -13.8
    episode 1012 score -36.6 average_score -14.1
    episode 1013 score 9.1 average_score -13.7
    episode 1014 score 92.4 average_score -13.0
    episode 1015 score 10.6 average_score -13.0
    episode 1016 score 22.2 average_score -12.9
    episode 1017 score 2.0 average_score -10.5
    episode 1018 score -115.0 average_score -9.9
    episode 1019 score -10.8 average_score -10.2
    episode 1020 score -42.3 average_score -10.9
    episode 1021 score 31.4 average_score -10.3
    episode 1022 score 25.6 average_score -10.3
    episode 1023 score 48.1 average_score -9.6
    episode 1024 score -14.0 average_score -9.5
    episode 1025 score -23.1 average_score -9.7
    episode 1026 score -107.2 average_score -11.0
    episode 1027 score 10.4 average_score -11.3
    episode 1028 score 76.1 average_score -10.2
    episode 1029 score -37.9 average_score -10.7
    episode 1030 score -1.0 average_score -11.2
    episode 1031 score -9.1 average_score -11.2
    episode 1032 score -17.7 average_score -11.1
    episode 1033 score -1.4 average_score -10.9
    episode 1034 score 8.5 average_score -9.8
    episode 1035 score -7.0 average_score -8.9
    episode 1036 score 61.0 average_score -7.9
    episode 1037 score -87.8 average_score -8.4
    episode 1038 score 5.6 average_score -8.2
    episode 1039 score 15.3 average_score -7.9
    episode 1040 score 49.0 average_score -7.6
    episode 1041 score 23.3 average_score -7.5
    episode 1042 score 25.5 average_score -6.8
    episode 1043 score 78.3 average_score -6.3
    episode 1044 score 70.1 average_score -5.9
    episode 1045 score -32.5 average_score -6.5
    episode 1046 score -14.3 average_score -6.9
    episode 1047 score 26.4 average_score -6.7
    episode 1048 score -63.1 average_score -7.4
    episode 1049 score -18.5 average_score -7.6
    episode 1050 score -18.5 average_score -7.7
    episode 1051 score -78.8 average_score -8.4
    episode 1052 score -11.9 average_score -8.7
    episode 1053 score 79.8 average_score -7.1
    episode 1054 score -90.1 average_score -6.2
    episode 1055 score 11.8 average_score -5.4
    episode 1056 score -19.6 average_score -5.8
    episode 1057 score 14.8 average_score -4.7
    episode 1058 score -126.4 average_score -6.4
    episode 1059 score 26.0 average_score -6.5
    episode 1060 score 79.7 average_score -5.4
    episode 1061 score -74.3 average_score -5.9
    episode 1062 score -56.5 average_score -6.8
    episode 1063 score 49.7 average_score -6.9
    episode 1064 score -23.4 average_score -7.0
    episode 1065 score -4.8 average_score -6.8
    episode 1066 score -48.9 average_score -7.3
    episode 1067 score 66.1 average_score -6.3
    episode 1068 score -202.1 average_score -8.2
    episode 1069 score -53.7 average_score -8.9
    episode 1070 score -37.3 average_score -9.3
    episode 1071 score -48.2 average_score -9.9
    episode 1072 score -2.5 average_score -10.4
    episode 1073 score 93.8 average_score -8.5
    episode 1074 score -0.4 average_score -8.0
    episode 1075 score 27.7 average_score -6.7
    episode 1076 score 41.4 average_score -7.2
    episode 1077 score 8.6 average_score -5.8
    episode 1078 score 0.9 average_score -6.1
    episode 1079 score 59.0 average_score -5.7
    episode 1080 score 34.2 average_score -5.4
    episode 1081 score 70.4 average_score -5.3
    episode 1082 score -155.9 average_score -6.3
    episode 1083 score -106.2 average_score -7.2
    episode 1084 score -19.0 average_score -7.4
    episode 1085 score 37.4 average_score -4.8
    episode 1086 score -177.3 average_score -6.7
    episode 1087 score -58.5 average_score -6.5
    episode 1088 score 82.9 average_score -6.3
    episode 1089 score -194.5 average_score -7.7
    episode 1090 score -13.1 average_score -7.7
    episode 1091 score 23.9 average_score -7.3
    episode 1092 score 23.1 average_score -7.0
    episode 1093 score -154.7 average_score -8.3
    episode 1094 score -1.4 average_score -8.6
    episode 1095 score 69.2 average_score -8.2
    episode 1096 score 57.3 average_score -7.8
    episode 1097 score -178.2 average_score -9.7
    episode 1098 score -123.4 average_score -11.7
    episode 1099 score -25.6 average_score -12.5
    episode 1100 score -49.9 average_score -13.3
    episode 1101 score 61.8 average_score -13.3
    episode 1102 score 55.1 average_score -12.6
    episode 1103 score -248.1 average_score -14.9
    episode 1104 score 5.3 average_score -14.8
    episode 1105 score -23.7 average_score -14.5
    episode 1106 score 57.9 average_score -14.0
    episode 1107 score 71.6 average_score -12.8
    episode 1108 score -176.3 average_score -14.3
    episode 1109 score -14.1 average_score -14.0
    episode 1110 score -173.6 average_score -15.6
    episode 1111 score -75.5 average_score -16.1
    episode 1112 score 21.7 average_score -15.5
    episode 1113 score 66.5 average_score -14.9
    episode 1114 score -181.3 average_score -17.6
    episode 1115 score 79.9 average_score -16.9
    episode 1116 score -7.8 average_score -17.2
    episode 1117 score -21.8 average_score -17.5
    episode 1118 score -9.7 average_score -16.4
    episode 1119 score -71.1 average_score -17.0
    episode 1120 score -10.0 average_score -16.7
    episode 1121 score -16.6 average_score -17.2
    episode 1122 score 7.4 average_score -17.4
    episode 1123 score -122.6 average_score -19.1
    episode 1124 score 54.5 average_score -18.4
    episode 1125 score 63.6 average_score -17.5
    episode 1126 score 59.5 average_score -15.9
    episode 1127 score -33.2 average_score -16.3
    episode 1128 score 28.5 average_score -16.8
    episode 1129 score 85.1 average_score -15.5
    episode 1130 score -12.5 average_score -15.7
    episode 1131 score -14.8 average_score -15.7
    episode 1132 score 26.5 average_score -15.3
    episode 1133 score 47.3 average_score -14.8
    episode 1134 score 0.8 average_score -14.9
    episode 1135 score -1.2 average_score -14.8
    episode 1136 score 26.5 average_score -15.1
    episode 1137 score 22.7 average_score -14.0
    episode 1138 score 57.4 average_score -13.5
    episode 1139 score 66.2 average_score -13.0
    episode 1140 score 116.7 average_score -12.3
    episode 1141 score 98.4 average_score -11.6
    episode 1142 score -13.0 average_score -12.0
    episode 1143 score 110.6 average_score -11.6
    episode 1144 score 110.8 average_score -11.2
    episode 1145 score 36.5 average_score -10.6
    episode 1146 score -1.7 average_score -10.4
    episode 1147 score 120.2 average_score -9.5
    episode 1148 score -52.8 average_score -9.4
    episode 1149 score 19.2 average_score -9.0
    episode 1150 score 94.8 average_score -7.9
    episode 1151 score 42.6 average_score -6.7
    episode 1152 score -7.6 average_score -6.6
    episode 1153 score 17.3 average_score -7.2
    episode 1154 score 89.0 average_score -5.4
    episode 1155 score 74.9 average_score -4.8
    episode 1156 score 57.3 average_score -4.0
    episode 1157 score -17.1 average_score -4.4
    episode 1158 score 56.9 average_score -2.5
    episode 1159 score 99.2 average_score -1.8
    episode 1160 score 66.7 average_score -1.9
    episode 1161 score -33.6 average_score -1.5
    episode 1162 score -141.2 average_score -2.4
    episode 1163 score 120.1 average_score -1.7
    episode 1164 score 79.3 average_score -0.6
    episode 1165 score 5.2 average_score -0.5
    episode 1166 score 3.0 average_score -0.0
    episode 1167 score 120.7 average_score 0.5
    episode 1168 score -2.5 average_score 2.5
    episode 1169 score -30.0 average_score 2.8
    episode 1170 score 84.4 average_score 4.0
    episode 1171 score 110.4 average_score 5.6
    episode 1172 score -180.3 average_score 3.8
    episode 1173 score 82.9 average_score 3.7
    episode 1174 score -11.6 average_score 3.6
    episode 1175 score 1.5 average_score 3.3
    episode 1176 score 46.0 average_score 3.3
    episode 1177 score 48.8 average_score 3.7
    episode 1178 score 35.8 average_score 4.1
    episode 1179 score 3.3 average_score 3.5
    episode 1180 score -24.8 average_score 2.9
    episode 1181 score -42.2 average_score 1.8
    episode 1182 score 28.6 average_score 3.7
    episode 1183 score 12.9 average_score 4.9
    episode 1184 score 4.7 average_score 5.1
    episode 1185 score -21.0 average_score 4.5
    episode 1186 score 42.7 average_score 6.7
    episode 1187 score 137.0 average_score 8.7
    episode 1188 score -14.9 average_score 7.7
    episode 1189 score -12.8 average_score 9.5
    episode 1190 score -17.3 average_score 9.5
    episode 1191 score 2.9 average_score 9.3
    episode 1192 score -1.3 average_score 9.0
    episode 1193 score 46.6 average_score 11.0
    episode 1194 score -30.3 average_score 10.7
    episode 1195 score 10.8 average_score 10.2
    episode 1196 score -18.4 average_score 9.4
    episode 1197 score 15.6 average_score 11.3
    episode 1198 score 32.5 average_score 12.9
    episode 1199 score -66.5 average_score 12.5
    episode 1200 score -35.6 average_score 12.6
    episode 1201 score -8.1 average_score 11.9
    episode 1202 score 83.5 average_score 12.2
    episode 1203 score -28.8 average_score 14.4
    episode 1204 score -89.7 average_score 13.5
    episode 1205 score 8.0 average_score 13.8
    episode 1206 score -39.0 average_score 12.8
    episode 1207 score 138.9 average_score 13.5
    episode 1208 score 5.4 average_score 15.3
    episode 1209 score -64.6 average_score 14.8
    episode 1210 score 39.3 average_score 16.9
    episode 1211 score 0.5 average_score 17.7
    episode 1212 score 24.0 average_score 17.7
    episode 1213 score -58.1 average_score 16.5
    episode 1214 score -13.5 average_score 18.1
    episode 1215 score -42.3 average_score 16.9
    episode 1216 score -77.2 average_score 16.2
    episode 1217 score -60.8 average_score 15.8
    episode 1218 score 59.3 average_score 16.5
    episode 1219 score 38.7 average_score 17.6
    episode 1220 score -158.2 average_score 16.1
    episode 1221 score -107.5 average_score 15.2
    episode 1222 score 45.8 average_score 15.6
    episode 1223 score 50.6 average_score 17.3
    episode 1224 score 120.6 average_score 18.0
    episode 1225 score -37.0 average_score 17.0
    episode 1226 score -25.1 average_score 16.1
    episode 1227 score 6.3 average_score 16.5
    episode 1228 score 42.4 average_score 16.7
    episode 1229 score -45.8 average_score 15.4
    episode 1230 score 130.7 average_score 16.8
    episode 1231 score 25.0 average_score 17.2
    episode 1232 score -17.0 average_score 16.8
    episode 1233 score 69.4 average_score 17.0
    episode 1234 score -88.9 average_score 16.1
    episode 1235 score -0.7 average_score 16.1
    episode 1236 score 22.3 average_score 16.1
    episode 1237 score 1.5 average_score 15.8
    episode 1238 score -19.1 average_score 15.1
    episode 1239 score -4.8 average_score 14.4
    episode 1240 score 3.5 average_score 13.2
    episode 1241 score 117.1 average_score 13.4
    episode 1242 score 45.6 average_score 14.0
    episode 1243 score -75.4 average_score 12.1
    episode 1244 score -1.0 average_score 11.0
    episode 1245 score -244.4 average_score 8.2
    episode 1246 score 102.5 average_score 9.3
    episode 1247 score 24.5 average_score 8.3
    episode 1248 score 141.7 average_score 10.2
    episode 1249 score -46.8 average_score 9.6
    episode 1250 score 131.1 average_score 10.0
    episode 1251 score 31.9 average_score 9.8
    episode 1252 score 66.6 average_score 10.6
    episode 1253 score -114.3 average_score 9.3
    episode 1254 score 125.8 average_score 9.6
    episode 1255 score 74.0 average_score 9.6
    episode 1256 score 99.4 average_score 10.0
    episode 1257 score -46.1 average_score 9.8
    episode 1258 score 74.1 average_score 9.9
    episode 1259 score 28.5 average_score 9.2
    episode 1260 score -15.5 average_score 8.4
    episode 1261 score 32.8 average_score 9.1
    episode 1262 score 41.8 average_score 10.9
    episode 1263 score -20.5 average_score 9.5
    episode 1264 score 126.3 average_score 10.0
    episode 1265 score -52.0 average_score 9.4
    episode 1266 score -41.4 average_score 8.9
    episode 1267 score 77.1 average_score 8.5
    episode 1268 score 118.5 average_score 9.7
    episode 1269 score 132.7 average_score 11.3
    episode 1270 score -6.0 average_score 10.4
    episode 1271 score 87.0 average_score 10.2
    episode 1272 score -55.3 average_score 11.5
    episode 1273 score -0.5 average_score 10.6
    episode 1274 score 18.9 average_score 10.9
    episode 1275 score -21.9 average_score 10.7
    episode 1276 score -17.4 average_score 10.1
    episode 1277 score -70.6 average_score 8.9
    episode 1278 score 60.4 average_score 9.1
    episode 1279 score -132.0 average_score 7.8
    episode 1280 score -99.1 average_score 7.0
    episode 1281 score -106.0 average_score 6.4
    episode 1282 score -19.9 average_score 5.9
    episode 1283 score 96.9 average_score 6.7
    episode 1284 score -14.3 average_score 6.5
    episode 1285 score 39.5 average_score 7.1
    episode 1286 score 122.8 average_score 7.9
    episode 1287 score 56.6 average_score 7.1
    episode 1288 score -79.7 average_score 6.5
    episode 1289 score -16.2 average_score 6.5
    episode 1290 score -58.7 average_score 6.0
    episode 1291 score -58.6 average_score 5.4
    episode 1292 score -105.4 average_score 4.4
    episode 1293 score 127.8 average_score 5.2
    episode 1294 score 100.4 average_score 6.5
    episode 1295 score 100.9 average_score 7.4
    episode 1296 score 100.9 average_score 8.6
    episode 1297 score -34.4 average_score 8.1
    episode 1298 score -142.9 average_score 6.4
    episode 1299 score 35.5 average_score 7.4
    episode 1300 score 38.4 average_score 8.1
    episode 1301 score -13.0 average_score 8.1
    episode 1302 score 25.6 average_score 7.5
    episode 1303 score 7.3 average_score 7.8
    episode 1304 score 7.0 average_score 8.8
    episode 1305 score -9.0 average_score 8.6
    episode 1306 score -32.2 average_score 8.7
    episode 1307 score -0.2 average_score 7.3
    episode 1308 score 12.2 average_score 7.4
    episode 1309 score 35.8 average_score 8.4
    episode 1310 score 82.8 average_score 8.8
    episode 1311 score 32.2 average_score 9.1
    episode 1312 score -0.9 average_score 8.9
    episode 1313 score -7.7 average_score 9.4
    episode 1314 score 26.9 average_score 9.8
    episode 1315 score -18.3 average_score 10.0
    episode 1316 score -14.3 average_score 10.7
    episode 1317 score 56.7 average_score 11.8
    episode 1318 score -17.0 average_score 11.1
    episode 1319 score 27.4 average_score 11.0
    episode 1320 score -3.6 average_score 12.5
    episode 1321 score 115.0 average_score 14.7
    episode 1322 score 16.2 average_score 14.4
    episode 1323 score -2.3 average_score 13.9
    episode 1324 score 30.9 average_score 13.0
    episode 1325 score -9.8 average_score 13.3
    episode 1326 score 8.0 average_score 13.6
    episode 1327 score 22.4 average_score 13.8
    episode 1328 score 33.5 average_score 13.7
    episode 1329 score 77.1 average_score 14.9
    episode 1330 score 24.2 average_score 13.9
    episode 1331 score 160.5 average_score 15.2
    episode 1332 score 14.5 average_score 15.5
    episode 1333 score 5.6 average_score 14.9
    episode 1334 score 76.9 average_score 16.5
    episode 1335 score 33.3 average_score 16.9
    episode 1336 score 14.5 average_score 16.8
    episode 1337 score 29.0 average_score 17.1
    episode 1338 score 43.8 average_score 17.7
    episode 1339 score 141.0 average_score 19.2
    episode 1340 score -16.4 average_score 19.0
    episode 1341 score 44.8 average_score 18.2
    episode 1342 score -4.2 average_score 17.8
    episode 1343 score -5.2 average_score 18.5
    episode 1344 score 13.7 average_score 18.6
    episode 1345 score 211.0 average_score 23.2
    episode 1346 score -24.2 average_score 21.9
    episode 1347 score 36.2 average_score 22.0
    episode 1348 score 153.5 average_score 22.1
    episode 1349 score 23.5 average_score 22.8
    episode 1350 score -42.3 average_score 21.1
    episode 1351 score 18.3 average_score 21.0
    episode 1352 score 117.2 average_score 21.5
    episode 1353 score 15.8 average_score 22.8
    episode 1354 score 11.5 average_score 21.6
    episode 1355 score 48.3 average_score 21.4
    episode 1356 score 1.4 average_score 20.4
    episode 1357 score 53.1 average_score 21.4
    episode 1358 score -4.6 average_score 20.6
    episode 1359 score 64.7 average_score 20.9
    episode 1360 score 15.0 average_score 21.3
    episode 1361 score 45.9 average_score 21.4
    episode 1362 score 64.1 average_score 21.6
    episode 1363 score 39.1 average_score 22.2
    episode 1364 score 141.9 average_score 22.4
    episode 1365 score -18.0 average_score 22.7
    episode 1366 score 21.9 average_score 23.3
    episode 1367 score 2.4 average_score 22.6
    episode 1368 score 18.1 average_score 21.6
    episode 1369 score 27.9 average_score 20.5
    episode 1370 score -10.6 average_score 20.5
    episode 1371 score -12.3 average_score 19.5
    episode 1372 score -2.9 average_score 20.0
    episode 1373 score 26.0 average_score 20.3
    episode 1374 score -44.1 average_score 19.7
    episode 1375 score 6.2 average_score 19.9
    episode 1376 score -86.9 average_score 19.2
    episode 1377 score 97.2 average_score 20.9
    episode 1378 score 2.7 average_score 20.3
    episode 1379 score -7.5 average_score 21.6
    episode 1380 score -95.2 average_score 21.6
    episode 1381 score -9.0 average_score 22.6
    episode 1382 score 37.0 average_score 23.2
    episode 1383 score 5.5 average_score 22.3
    episode 1384 score 35.4 average_score 22.7
    episode 1385 score 20.9 average_score 22.6
    episode 1386 score -2.2 average_score 21.3
    episode 1387 score -113.7 average_score 19.6
    episode 1388 score 27.1 average_score 20.7
    episode 1389 score -0.7 average_score 20.8
    episode 1390 score 27.2 average_score 21.7
    episode 1391 score 19.1 average_score 22.5
    episode 1392 score 38.3 average_score 23.9
    episode 1393 score 176.3 average_score 24.4
    episode 1394 score 54.3 average_score 23.9
    episode 1395 score 5.2 average_score 23.0
    episode 1396 score 23.3 average_score 22.2
    episode 1397 score 20.3 average_score 22.7
    episode 1398 score -61.1 average_score 23.6
    episode 1399 score -13.7 average_score 23.1
    episode 1400 score 23.4 average_score 22.9
    episode 1401 score -33.4 average_score 22.7
    episode 1402 score 21.5 average_score 22.7
    episode 1403 score 2.4 average_score 22.6
    episode 1404 score -12.9 average_score 22.4
    episode 1405 score 3.7 average_score 22.6
    episode 1406 score 163.7 average_score 24.5
    episode 1407 score 4.3 average_score 24.6
    episode 1408 score 31.7 average_score 24.8
    episode 1409 score -7.5 average_score 24.3
    episode 1410 score 136.6 average_score 24.9
    episode 1411 score -17.3 average_score 24.4
    episode 1412 score 22.2 average_score 24.6
    episode 1413 score 40.6 average_score 25.1
    episode 1414 score 146.1 average_score 26.3
    episode 1415 score 11.3 average_score 26.6
    episode 1416 score 136.1 average_score 28.1
    episode 1417 score -13.8 average_score 27.4
    episode 1418 score 13.1 average_score 27.7
    episode 1419 score 32.2 average_score 27.7
    episode 1420 score 105.4 average_score 28.8
    episode 1421 score -53.6 average_score 27.1
    episode 1422 score 24.3 average_score 27.2
    episode 1423 score 161.4 average_score 28.8
    episode 1424 score 34.1 average_score 28.9
    episode 1425 score 0.4 average_score 29.0
    episode 1426 score 6.9 average_score 29.0
    episode 1427 score 20.2 average_score 28.9
    episode 1428 score 14.5 average_score 28.7
    episode 1429 score 1.0 average_score 28.0
    episode 1430 score -37.3 average_score 27.4
    episode 1431 score -49.3 average_score 25.3
    episode 1432 score -22.3 average_score 24.9
    episode 1433 score 147.5 average_score 26.3
    episode 1434 score 29.1 average_score 25.8
    episode 1435 score 9.2 average_score 25.6
    episode 1436 score 152.9 average_score 27.0
    episode 1437 score -21.6 average_score 26.5
    episode 1438 score 72.1 average_score 26.8
    episode 1439 score -206.5 average_score 23.3
    episode 1440 score 0.7 average_score 23.5
    episode 1441 score -32.7 average_score 22.7
    episode 1442 score 35.5 average_score 23.1
    episode 1443 score 35.0 average_score 23.5
    episode 1444 score 174.2 average_score 25.1
    episode 1445 score 15.3 average_score 23.1
    episode 1446 score -12.0 average_score 23.3
    episode 1447 score 263.3 average_score 25.5
    episode 1448 score 55.4 average_score 24.5
    episode 1449 score -27.2 average_score 24.0
    episode 1450 score -31.3 average_score 24.2
    episode 1451 score 39.6 average_score 24.4
    episode 1452 score 37.9 average_score 23.6
    episode 1453 score -20.2 average_score 23.2
    episode 1454 score 101.4 average_score 24.1
    episode 1455 score -8.1 average_score 23.5
    episode 1456 score 142.3 average_score 25.0
    episode 1457 score 78.3 average_score 25.2
    episode 1458 score 47.4 average_score 25.7
    episode 1459 score 104.3 average_score 26.1
    episode 1460 score -58.1 average_score 25.4
    episode 1461 score 180.4 average_score 26.7
    episode 1462 score -20.1 average_score 25.9
    episode 1463 score -18.5 average_score 25.3
    episode 1464 score -12.7 average_score 23.8
    episode 1465 score 153.1 average_score 25.5
    episode 1466 score 8.1 average_score 25.3
    episode 1467 score 29.5 average_score 25.6
    episode 1468 score 13.8 average_score 25.6
    episode 1469 score -48.1 average_score 24.8
    episode 1470 score -27.4 average_score 24.6
    episode 1471 score -27.5 average_score 24.5
    episode 1472 score -91.8 average_score 23.6
    episode 1473 score 26.4 average_score 23.6
    episode 1474 score 78.3 average_score 24.8
    episode 1475 score 115.1 average_score 25.9
    episode 1476 score -27.0 average_score 26.5
    episode 1477 score 103.3 average_score 26.6
    episode 1478 score -46.6 average_score 26.1
    episode 1479 score -111.9 average_score 25.0
    episode 1480 score 153.6 average_score 27.5
    episode 1481 score 22.0 average_score 27.8
    episode 1482 score 87.5 average_score 28.3
    episode 1483 score 35.6 average_score 28.6
    episode 1484 score -8.3 average_score 28.2
    episode 1485 score 30.7 average_score 28.3
    episode 1486 score -1.8 average_score 28.3
    episode 1487 score 19.1 average_score 29.6
    episode 1488 score 147.7 average_score 30.8
    episode 1489 score 3.5 average_score 30.9
    episode 1490 score -10.7 average_score 30.5
    episode 1491 score 100.6 average_score 31.3
    episode 1492 score 135.1 average_score 32.3
    episode 1493 score 44.4 average_score 31.0
    episode 1494 score 105.6 average_score 31.5
    episode 1495 score 23.1 average_score 31.7
    episode 1496 score 141.2 average_score 32.8
    episode 1497 score 151.4 average_score 34.2
    episode 1498 score 46.3 average_score 35.2
    episode 1499 score -16.8 average_score 35.2
    episode 1500 score 63.9 average_score 35.6
    episode 1501 score 24.3 average_score 36.2
    episode 1502 score -67.3 average_score 35.3
    episode 1503 score 31.4 average_score 35.6
    episode 1504 score -12.9 average_score 35.6
    episode 1505 score 2.3 average_score 35.6
    episode 1506 score 2.8 average_score 34.0
    episode 1507 score 9.3 average_score 34.0
    episode 1508 score -9.4 average_score 33.6
    episode 1509 score -5.5 average_score 33.6
    episode 1510 score 264.3 average_score 34.9
    episode 1511 score 2.6 average_score 35.1
    episode 1512 score -87.9 average_score 34.0
    episode 1513 score 17.1 average_score 33.8
    episode 1514 score 158.0 average_score 33.9
    episode 1515 score 17.9 average_score 33.9
    episode 1516 score 266.1 average_score 35.2
    episode 1517 score -6.4 average_score 35.3
    episode 1518 score 26.1 average_score 35.4
    episode 1519 score 253.3 average_score 37.7
    episode 1520 score 23.2 average_score 36.8
    episode 1521 score 33.2 average_score 37.7
    episode 1522 score 35.6 average_score 37.8
    episode 1523 score 167.5 average_score 37.9
    episode 1524 score 0.9 average_score 37.5
    episode 1525 score 250.5 average_score 40.0
    episode 1526 score 227.3 average_score 42.2
    episode 1527 score 6.5 average_score 42.1
    episode 1528 score 38.8 average_score 42.4
    episode 1529 score -10.4 average_score 42.2
    episode 1530 score 6.3 average_score 42.7
    episode 1531 score 150.9 average_score 44.7
    episode 1532 score -6.2 average_score 44.8
    episode 1533 score 18.8 average_score 43.6
    episode 1534 score 203.2 average_score 45.3
    episode 1535 score 243.1 average_score 47.6
    episode 1536 score 278.9 average_score 48.9
    episode 1537 score 263.0 average_score 51.7
    episode 1538 score 259.6 average_score 53.6
    episode 1539 score 60.3 average_score 56.3
    episode 1540 score -31.3 average_score 56.0
    episode 1541 score 29.6 average_score 56.6
    episode 1542 score 40.7 average_score 56.6
    episode 1543 score -17.8 average_score 56.1
    episode 1544 score -179.5 average_score 52.6
    episode 1545 score 41.0 average_score 52.8
    episode 1546 score -167.4 average_score 51.3
    episode 1547 score -92.9 average_score 47.7
    episode 1548 score 220.5 average_score 49.4
    episode 1549 score -36.2 average_score 49.3
    episode 1550 score 17.5 average_score 49.8
    episode 1551 score 252.2 average_score 51.9
    episode 1552 score 230.3 average_score 53.8
    episode 1553 score 297.9 average_score 57.0
    episode 1554 score 14.9 average_score 56.1
    episode 1555 score 255.2 average_score 58.8
    episode 1556 score 292.9 average_score 60.3
    episode 1557 score 260.9 average_score 62.1
    episode 1558 score -4.9 average_score 61.6
    episode 1559 score -185.0 average_score 58.7
    episode 1560 score 203.2 average_score 61.3
    episode 1561 score -47.0 average_score 59.0
    episode 1562 score 273.4 average_score 62.0
    episode 1563 score 37.2 average_score 62.5
    episode 1564 score 243.3 average_score 65.1
    episode 1565 score -11.9 average_score 63.4
    episode 1566 score 273.6 average_score 66.1
    episode 1567 score -7.3 average_score 65.7
    episode 1568 score 38.5 average_score 66.0
    episode 1569 score 75.2 average_score 67.2
    episode 1570 score 303.3 average_score 70.5
    episode 1571 score -23.5 average_score 70.5
    episode 1572 score 247.1 average_score 73.9
    episode 1573 score -3.1 average_score 73.6
    episode 1574 score -41.7 average_score 72.4
    episode 1575 score 20.2 average_score 71.5
    episode 1576 score -6.2 average_score 71.7
    episode 1577 score 268.8 average_score 73.3
    episode 1578 score 282.4 average_score 76.6
    episode 1579 score -51.3 average_score 77.2
    episode 1580 score -42.1 average_score 75.3
    episode 1581 score 263.7 average_score 77.7
    episode 1582 score -15.5 average_score 76.7
    episode 1583 score -51.8 average_score 75.8
    episode 1584 score 13.4 average_score 76.0
    episode 1585 score 178.0 average_score 77.5
    episode 1586 score 202.7 average_score 79.5
    episode 1587 score 14.1 average_score 79.5
    episode 1588 score -78.5 average_score 77.2
    episode 1589 score -72.7 average_score 76.5
    episode 1590 score -43.1 average_score 76.1
    episode 1591 score 195.1 average_score 77.1
    episode 1592 score -133.6 average_score 74.4
    episode 1593 score 17.4 average_score 74.1
    episode 1594 score 13.8 average_score 73.2
    episode 1595 score 23.0 average_score 73.2
    episode 1596 score 191.7 average_score 73.7
    episode 1597 score -11.5 average_score 72.1
    episode 1598 score 28.8 average_score 71.9
    episode 1599 score 252.4 average_score 74.6
    episode 1600 score 207.8 average_score 76.0
    episode 1601 score -21.4 average_score 75.6
    episode 1602 score 27.2 average_score 76.5
    episode 1603 score 186.6 average_score 78.1
    episode 1604 score 37.2 average_score 78.6
    episode 1605 score 19.1 average_score 78.7
    episode 1606 score -8.6 average_score 78.6
    episode 1607 score -53.1 average_score 78.0
    episode 1608 score 271.7 average_score 80.8
    episode 1609 score -137.2 average_score 79.5
    episode 1610 score 14.8 average_score 77.0
    episode 1611 score 237.9 average_score 79.4
    episode 1612 score 36.0 average_score 80.6
    episode 1613 score 4.5 average_score 80.5
    episode 1614 score 214.8 average_score 81.0
    episode 1615 score -18.4 average_score 80.7
    episode 1616 score 31.7 average_score 78.3
    episode 1617 score -21.2 average_score 78.2
    episode 1618 score 40.4 average_score 78.3
    episode 1619 score -55.8 average_score 75.2
    episode 1620 score -15.3 average_score 74.8
    episode 1621 score -9.6 average_score 74.4
    episode 1622 score -60.6 average_score 73.5
    episode 1623 score 0.8 average_score 71.8
    episode 1624 score 255.4 average_score 74.3
    episode 1625 score 29.3 average_score 72.1
    episode 1626 score 199.2 average_score 71.8
    episode 1627 score -7.4 average_score 71.7
    episode 1628 score 260.6 average_score 73.9
    episode 1629 score -134.3 average_score 72.7
    episode 1630 score 14.8 average_score 72.8
    episode 1631 score 235.1 average_score 73.6
    episode 1632 score 29.8 average_score 74.0
    episode 1633 score -54.5 average_score 73.2
    episode 1634 score 27.1 average_score 71.5
    episode 1635 score 27.4 average_score 69.3
    episode 1636 score 267.1 average_score 69.2
    episode 1637 score 222.4 average_score 68.8
    episode 1638 score 21.6 average_score 66.4
    episode 1639 score 31.2 average_score 66.1
    episode 1640 score 221.5 average_score 68.7
    episode 1641 score 239.7 average_score 70.8
    episode 1642 score 281.7 average_score 73.2
    episode 1643 score 292.2 average_score 76.3
    episode 1644 score 254.3 average_score 80.6
    episode 1645 score 273.7 average_score 82.9
    episode 1646 score 218.8 average_score 86.8
    episode 1647 score -34.7 average_score 87.4
    episode 1648 score -1.3 average_score 85.2
    episode 1649 score 2.9 average_score 85.5
    episode 1650 score 20.7 average_score 85.6
    episode 1651 score 52.8 average_score 83.6
    episode 1652 score -111.9 average_score 80.2
    episode 1653 score 207.7 average_score 79.3
    episode 1654 score 30.7 average_score 79.4
    episode 1655 score 274.4 average_score 79.6
    episode 1656 score 18.4 average_score 76.9
    episode 1657 score 231.5 average_score 76.6
    episode 1658 score -51.2 average_score 76.1
    episode 1659 score 32.5 average_score 78.3
    episode 1660 score 244.8 average_score 78.7
    episode 1661 score 240.9 average_score 81.6
    episode 1662 score -66.9 average_score 78.2
    episode 1663 score 289.6 average_score 80.7
    episode 1664 score 242.3 average_score 80.7
    episode 1665 score 284.5 average_score 83.6
    episode 1666 score 262.0 average_score 83.5
    episode 1667 score 57.4 average_score 84.2
    episode 1668 score 273.9 average_score 86.5
    episode 1669 score -45.5 average_score 85.3
    episode 1670 score 270.1 average_score 85.0
    episode 1671 score 251.7 average_score 87.7
    episode 1672 score -105.5 average_score 84.2
    episode 1673 score 4.0 average_score 84.3
    episode 1674 score -13.5 average_score 84.6
    episode 1675 score 277.8 average_score 87.1
    episode 1676 score -63.9 average_score 86.6
    episode 1677 score 205.7 average_score 85.9
    episode 1678 score 266.1 average_score 85.8
    episode 1679 score 244.4 average_score 88.7
    episode 1680 score 272.9 average_score 91.9
    episode 1681 score -17.6 average_score 89.1
    episode 1682 score 245.6 average_score 91.7
    episode 1683 score 2.0 average_score 92.2
    episode 1684 score -60.0 average_score 91.5
    episode 1685 score 10.7 average_score 89.8
    episode 1686 score 249.4 average_score 90.3
    episode 1687 score 15.0 average_score 90.3
    episode 1688 score 30.0 average_score 91.4
    episode 1689 score 22.5 average_score 92.3
    episode 1690 score 215.3 average_score 94.9
    episode 1691 score 241.4 average_score 95.4
    episode 1692 score -61.7 average_score 96.1
    episode 1693 score 254.3 average_score 98.5
    episode 1694 score 243.2 average_score 100.8
    episode 1695 score 47.2 average_score 101.0
    episode 1696 score 45.9 average_score 99.5
    episode 1697 score -51.0 average_score 99.1
    episode 1698 score 42.0 average_score 99.3
    episode 1699 score 5.4 average_score 96.8
    episode 1700 score 73.4 average_score 95.5
    episode 1701 score 23.7 average_score 95.9
    episode 1702 score 290.0 average_score 98.5
    episode 1703 score 18.6 average_score 96.9
    episode 1704 score 279.1 average_score 99.3
    episode 1705 score 18.8 average_score 99.3
    episode 1706 score -14.9 average_score 99.2
    episode 1707 score 275.4 average_score 102.5
    episode 1708 score -15.8 average_score 99.6
    episode 1709 score -21.3 average_score 100.8
    episode 1710 score 9.3 average_score 100.7
    episode 1711 score 263.6 average_score 101.0
    episode 1712 score 287.5 average_score 103.5
    episode 1713 score 32.2 average_score 103.8
    episode 1714 score -7.1 average_score 101.6
    episode 1715 score -15.2 average_score 101.6
    episode 1716 score 3.2 average_score 101.3
    episode 1717 score -12.6 average_score 101.4
    episode 1718 score 264.1 average_score 103.6
    episode 1719 score 267.5 average_score 106.9
    episode 1720 score -62.8 average_score 106.4
    episode 1721 score -3.6 average_score 106.5
    episode 1722 score 10.9 average_score 107.2
    episode 1723 score 269.0 average_score 109.9
    episode 1724 score -50.2 average_score 106.8
    episode 1725 score 270.1 average_score 109.2
    episode 1726 score -161.6 average_score 105.6
    episode 1727 score -3.5 average_score 105.6
    episode 1728 score -47.6 average_score 102.5
    episode 1729 score -19.7 average_score 103.7
    episode 1730 score 32.9 average_score 103.9
    episode 1731 score 55.2 average_score 102.1
    episode 1732 score 8.8 average_score 101.9
    episode 1733 score 248.4 average_score 104.9
    episode 1734 score -48.4 average_score 104.1
    episode 1735 score 132.4 average_score 105.2
    episode 1736 score 238.9 average_score 104.9
    episode 1737 score 290.0 average_score 105.6
    episode 1738 score -141.5 average_score 104.0
    episode 1739 score -40.1 average_score 103.2
    episode 1740 score -29.4 average_score 100.7
    episode 1741 score -88.7 average_score 97.4
    episode 1742 score 16.9 average_score 94.8
    episode 1743 score -8.4 average_score 91.8
    episode 1744 score -51.8 average_score 88.7
    episode 1745 score 262.0 average_score 88.6
    episode 1746 score 210.0 average_score 88.5
    episode 1747 score 7.1 average_score 88.9
    episode 1748 score -183.3 average_score 87.1
    episode 1749 score 217.5 average_score 89.3
    episode 1750 score -47.0 average_score 88.6
    episode 1751 score -4.7 average_score 88.0
    episode 1752 score -22.5 average_score 88.9
    episode 1753 score 244.4 average_score 89.3
    episode 1754 score 173.5 average_score 90.7
    episode 1755 score -42.1 average_score 87.5
    episode 1756 score 267.9 average_score 90.0
    episode 1757 score 250.1 average_score 90.2
    episode 1758 score -7.3 average_score 90.7
    episode 1759 score 5.4 average_score 90.4
    episode 1760 score 12.2 average_score 88.1
    episode 1761 score -68.2 average_score 85.0
    episode 1762 score -9.2 average_score 85.6
    episode 1763 score 13.2 average_score 82.8
    episode 1764 score 215.6 average_score 82.5
    episode 1765 score 259.1 average_score 82.3
    episode 1766 score -33.9 average_score 79.3
    episode 1767 score 259.4 average_score 81.3
    episode 1768 score 179.8 average_score 80.4
    episode 1769 score -6.5 average_score 80.8
    episode 1770 score 171.8 average_score 79.8
    episode 1771 score -102.2 average_score 76.3
    episode 1772 score 249.4 average_score 79.8
    episode 1773 score -24.1 average_score 79.5
    episode 1774 score -10.5 average_score 79.6
    episode 1775 score 210.8 average_score 78.9
    episode 1776 score 208.9 average_score 81.6
    episode 1777 score 7.6 average_score 79.6
    episode 1778 score -57.3 average_score 76.4
    episode 1779 score 286.7 average_score 76.8
    episode 1780 score 23.5 average_score 74.3
    episode 1781 score -18.1 average_score 74.3
    episode 1782 score -5.8 average_score 71.8
    episode 1783 score -42.2 average_score 71.4
    episode 1784 score 284.3 average_score 74.8
    episode 1785 score 278.4 average_score 77.5
    episode 1786 score 252.5 average_score 77.5
    episode 1787 score 41.7 average_score 77.8
    episode 1788 score 261.4 average_score 80.1
    episode 1789 score 24.9 average_score 80.1
    episode 1790 score -24.6 average_score 77.7
    episode 1791 score -13.2 average_score 75.2
    episode 1792 score 184.1 average_score 77.6
    episode 1793 score -21.3 average_score 74.9
    episode 1794 score 241.7 average_score 74.9
    episode 1795 score 222.0 average_score 76.6
    episode 1796 score 54.8 average_score 76.7
    episode 1797 score -34.2 average_score 76.9
    episode 1798 score -80.7 average_score 75.6
    episode 1799 score -92.1 average_score 74.7
    episode 1800 score -47.5 average_score 73.5
    episode 1801 score 210.6 average_score 75.3
    episode 1802 score -42.3 average_score 72.0
    episode 1803 score -9.1 average_score 71.7
    episode 1804 score -24.6 average_score 68.7
    episode 1805 score 226.1 average_score 70.8
    episode 1806 score 288.7 average_score 73.8
    episode 1807 score 196.9 average_score 73.0
    episode 1808 score -9.5 average_score 73.1
    episode 1809 score 8.9 average_score 73.4
    episode 1810 score 4.5 average_score 73.3
    episode 1811 score 24.2 average_score 70.9
    episode 1812 score -1.7 average_score 68.0
    episode 1813 score 233.2 average_score 70.1
    episode 1814 score 14.2 average_score 70.3
    episode 1815 score 237.6 average_score 72.8
    episode 1816 score -0.4 average_score 72.8
    episode 1817 score -89.5 average_score 72.0
    episode 1818 score -34.5 average_score 69.0
    episode 1819 score -10.7 average_score 66.2
    episode 1820 score 266.7 average_score 69.5
    episode 1821 score -55.2 average_score 69.0
    episode 1822 score 256.4 average_score 71.5
    episode 1823 score -0.2 average_score 68.8
    episode 1824 score -6.2 average_score 69.2
    episode 1825 score 251.8 average_score 69.0
    episode 1826 score 262.5 average_score 73.3
    episode 1827 score 300.4 average_score 76.3
    episode 1828 score 15.0 average_score 76.9
    episode 1829 score 207.9 average_score 79.2
    episode 1830 score 255.5 average_score 81.4
    episode 1831 score -152.6 average_score 79.3
    episode 1832 score -12.2 average_score 79.1
    episode 1833 score 255.2 average_score 79.2
    episode 1834 score 267.1 average_score 82.4
    episode 1835 score 248.2 average_score 83.5
    episode 1836 score 8.2 average_score 81.2
    episode 1837 score 201.5 average_score 80.3
    episode 1838 score 237.1 average_score 84.1
    episode 1839 score 216.1 average_score 86.7
    episode 1840 score 279.8 average_score 89.8
    episode 1841 score 172.3 average_score 92.4
    episode 1842 score 48.4 average_score 92.7
    episode 1843 score 233.5 average_score 95.1
    episode 1844 score -58.7 average_score 95.0
    episode 1845 score 253.2 average_score 95.0
    episode 1846 score 260.7 average_score 95.5
    episode 1847 score 145.9 average_score 96.8
    episode 1848 score 246.7 average_score 101.1
    episode 1849 score 301.4 average_score 102.0
    episode 1850 score 301.4 average_score 105.5
    episode 1851 score 244.7 average_score 108.0
    episode 1852 score -17.7 average_score 108.0
    episode 1853 score 53.4 average_score 106.1
    episode 1854 score -103.1 average_score 103.3
    episode 1855 score 274.5 average_score 106.5
    episode 1856 score 263.6 average_score 106.5
    episode 1857 score -7.1 average_score 103.9
    episode 1858 score 21.0 average_score 104.2
    episode 1859 score 260.6 average_score 106.7
    episode 1860 score -42.1 average_score 106.2
    episode 1861 score 304.4 average_score 109.9
    episode 1862 score 303.7 average_score 113.0
    episode 1863 score 28.0 average_score 113.2
    episode 1864 score 30.0 average_score 111.3
    episode 1865 score 261.4 average_score 111.4
    episode 1866 score 202.2 average_score 113.7
    episode 1867 score -50.8 average_score 110.6
    episode 1868 score -19.1 average_score 108.6
    episode 1869 score 211.5 average_score 110.8
    episode 1870 score -29.3 average_score 108.8
    episode 1871 score 268.2 average_score 112.5
    episode 1872 score 235.0 average_score 112.4
    episode 1873 score 277.0 average_score 115.4
    episode 1874 score 230.8 average_score 117.8
    episode 1875 score 244.5 average_score 118.1
    episode 1876 score 254.0 average_score 118.6
    episode 1877 score 261.8 average_score 121.1
    episode 1878 score 29.9 average_score 122.0
    episode 1879 score 261.1 average_score 121.7
    episode 1880 score 309.1 average_score 124.6
    episode 1881 score 45.7 average_score 125.2
    episode 1882 score -52.0 average_score 124.8
    episode 1883 score 18.7 average_score 125.4
    episode 1884 score 264.9 average_score 125.2
    episode 1885 score 44.1 average_score 122.8
    episode 1886 score 215.4 average_score 122.5
    episode 1887 score 180.4 average_score 123.8
    episode 1888 score -12.7 average_score 121.1
    episode 1889 score -71.0 average_score 120.1
    episode 1890 score 274.4 average_score 123.1
    episode 1891 score 252.8 average_score 125.8
    episode 1892 score 279.5 average_score 126.7
    episode 1893 score 233.6 average_score 129.3
    episode 1894 score 273.2 average_score 129.6
    episode 1895 score 12.4 average_score 127.5
    episode 1896 score 4.5 average_score 127.0
    episode 1897 score 166.8 average_score 129.0
    episode 1898 score 27.4 average_score 130.1
    episode 1899 score 237.8 average_score 133.4
    episode 1900 score -71.7 average_score 133.2
    episode 1901 score 24.1 average_score 131.3
    episode 1902 score 244.7 average_score 134.2
    episode 1903 score 247.9 average_score 136.7
    episode 1904 score 4.6 average_score 137.0
    episode 1905 score 260.6 average_score 137.4
    episode 1906 score 14.3 average_score 134.6
    episode 1907 score 266.2 average_score 135.3
    episode 1908 score 287.2 average_score 138.3
    episode 1909 score -13.1 average_score 138.1
    episode 1910 score 186.3 average_score 139.9
    episode 1911 score -2.3 average_score 139.6
    episode 1912 score 12.0 average_score 139.8
    episode 1913 score 19.1 average_score 137.6
    episode 1914 score 256.4 average_score 140.0
    episode 1915 score 20.7 average_score 137.9
    episode 1916 score 168.9 average_score 139.6
    episode 1917 score 230.0 average_score 142.8
    episode 1918 score -1.1 average_score 143.1
    episode 1919 score 237.0 average_score 145.6
    episode 1920 score 52.8 average_score 143.4
    episode 1921 score 222.5 average_score 146.2
    episode 1922 score 245.4 average_score 146.1
    episode 1923 score 12.8 average_score 146.2
    episode 1924 score 13.1 average_score 146.4
    episode 1925 score 22.0 average_score 144.1
    episode 1926 score 207.3 average_score 143.6
    episode 1927 score 295.6 average_score 143.5
    episode 1928 score -2.8 average_score 143.3
    episode 1929 score 55.4 average_score 141.8
    episode 1930 score 60.1 average_score 139.9
    episode 1931 score -8.9 average_score 141.3
    episode 1932 score 272.5 average_score 144.1
    episode 1933 score -29.0 average_score 141.3
    episode 1934 score 58.5 average_score 139.2
    episode 1935 score 14.8 average_score 136.9
    episode 1936 score -30.9 average_score 136.5
    episode 1937 score 255.7 average_score 137.0
    episode 1938 score 0.1 average_score 134.7
    episode 1939 score 226.0 average_score 134.8
    episode 1940 score 256.7 average_score 134.5
    episode 1941 score 20.4 average_score 133.0
    episode 1942 score 276.0 average_score 135.3
    episode 1943 score 236.4 average_score 135.3
    episode 1944 score 152.3 average_score 137.4
    episode 1945 score 58.4 average_score 135.5
    episode 1946 score 254.7 average_score 135.4
    episode 1947 score 16.2 average_score 134.1
    episode 1948 score 26.4 average_score 131.9
    episode 1949 score 11.9 average_score 129.0
    episode 1950 score -9.3 average_score 125.9
    episode 1951 score 58.4 average_score 124.1
    episode 1952 score 49.5 average_score 124.7
    episode 1953 score 274.8 average_score 126.9
    episode 1954 score 24.4 average_score 128.2
    episode 1955 score 16.7 average_score 125.6
    episode 1956 score 294.1 average_score 125.9
    episode 1957 score 277.1 average_score 128.8
    episode 1958 score 42.0 average_score 129.0
    episode 1959 score 291.2 average_score 129.3
    episode 1960 score 269.6 average_score 132.4
    episode 1961 score 257.7 average_score 132.0
    episode 1962 score 60.6 average_score 129.5
    episode 1963 score 161.1 average_score 130.9
    episode 1964 score 22.5 average_score 130.8
    episode 1965 score 22.0 average_score 128.4
    episode 1966 score 47.4 average_score 126.8
    episode 1967 score -38.8 average_score 127.0
    episode 1968 score 31.7 average_score 127.5
    episode 1969 score 274.8 average_score 128.1
    episode 1970 score 265.5 average_score 131.0
    episode 1971 score 34.1 average_score 128.7
    episode 1972 score -2.4 average_score 126.3
    episode 1973 score 220.9 average_score 125.8
    episode 1974 score 254.2 average_score 126.0
    episode 1975 score 272.5 average_score 126.3
    episode 1976 score 228.3 average_score 126.0
    episode 1977 score 286.5 average_score 126.3
    episode 1978 score 261.0 average_score 128.6
    episode 1979 score 240.5 average_score 128.4
    episode 1980 score 7.8 average_score 125.4
    episode 1981 score -7.9 average_score 124.8
    episode 1982 score 266.5 average_score 128.0
    episode 1983 score 231.7 average_score 130.1
    episode 1984 score 23.1 average_score 127.7
    episode 1985 score -127.0 average_score 126.0
    episode 1986 score 257.0 average_score 126.4
    episode 1987 score 248.7 average_score 127.1
    episode 1988 score 248.5 average_score 129.7
    episode 1989 score 34.1 average_score 130.8
    episode 1990 score -4.7 average_score 128.0
    episode 1991 score 256.1 average_score 128.0
    episode 1992 score 261.8 average_score 127.8
    episode 1993 score 203.9 average_score 127.5
    episode 1994 score 264.3 average_score 127.5
    episode 1995 score 45.5 average_score 127.8
    episode 1996 score 26.2 average_score 128.0
    episode 1997 score 40.6 average_score 126.7
    episode 1998 score 233.5 average_score 128.8
    episode 1999 score 237.6 average_score 128.8


# Conclusion
After assignment 6, this is my second attempt to apply the approach outlined the paper "Reinforcement Learning for Taxi-out Time Prediction: An improved Q-learning Approach, to predict taxi-out time on the on-time dataset.

In assignment 6, i used deep q-learning and in this assignment I tried PGM. both attempts did not yield significant results. I'm lead to believe that this dataset might not be the best candidate for the RL approach.
