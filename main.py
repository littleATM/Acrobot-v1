import gym
import random
import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

ENV = 'Acrobot-v1'
env = gym.make(ENV)
action = env.action_space.n

def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=len(env.reset()), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action, activation='linear'))
    model.compile(loss="mse",
                  optimizer=Adam(lr=0.0001, clipvalue=1))
    return model

def get_best_action(ss:[]):
    max_v=-10000
    index = -1
    for i in ss:
        if i[0]>max_v:
            index = ss.index(i)
            max_v = i[0]
    return index

def main():

    model = build_model()
    s = env.reset()
    episodes = 0
    step = 0
    ss = [None]*action
    while True:
        for i in range(action):
            env.set_state(s)
            s_, r, done, info = env.step(i)
            child_steps = 0
            while True:
                if done or child_steps>100:
                    if child_steps<100:
                        print(child_steps)
                    ss[i] = s_
                    break
                else:
                    act_values = model.predict(np.reshape(s_, [1, len(s_)]))
                    a = np.argmax(act_values[0])
                    s_, r, done, info = env.step(a)
                    child_steps = child_steps +1
        target_f = model.predict(np.reshape(s, [1, len(s)]))
        env.set_state(s)
        s, r, done, info = env.step(get_best_action(ss))
        step = step + 1
        env.render()
        if done:
            episodes = episodes + 1
            print("step:", step, "episodes", episodes)
            step = 0
            s = env.reset()
            continue
        for i in range(action):
            target_f[0][i] = -1
        target_f[0][get_best_action(ss)] = 1
        model.fit(np.reshape(s, [1, len(s)]), target_f, epochs=1, verbose=0)


if __name__ == "__main__":
    main()
