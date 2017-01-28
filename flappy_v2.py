##version avec replay de l'algo

import random
from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, sgd, adam
from keras.layers.recurrent import LSTM
from keras.models import load_model


# charge le jeu, version avec et sans affichage graphique

game = FlappyBird()
ps = PLE(game,fps=30, frame_skip=2, force_fps=False, display_screen=True,num_steps=1)
ps.init()
pws = PLE(game,fps=30, frame_skip=2, force_fps=False, display_screen=False,num_steps=1)
pws.init()


#creation du modele

#model=load_model("model_flappyv2.dqf")
model = Sequential()
#
model.add(Dense(400, init='lecun_uniform', input_shape=(17,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(500, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('linear'))
adam = adam(lr=1e-5)
model.compile(loss='mse', optimizer=adam)

#parametres

epochs =20000 #combien de fois on joue
gamma = .99 #discount
epsilon = 1 #parametre probabilite
lda =.99999 #parametre de decroissance de la probabilite

alpha=.1 #learning rate
experience_replay=False
batch_size=1000
buffer_size=5000
counter_replay=0
not_full=True
replay=np.zeros((buffer_size, 17+1+1+17))


#quelques fonctions
def form_state(dic):
    state=np.zeros(8)
    for key, value in dic.iteritems():
        if key=='player_y':
            state[0]=value
        if key=='player_vel':
            state[1]=value
        if key=='next_pipe_dist_to_player':
            state[2]=value
        if key=='next_pipe_top_y':
            state[3]=value
        if key=='next_pipe_bottom_y':
            state[4]=value
        if key=='next_next_pipe_dist_to_player':
            state[5]=value
        if key=='next_next_pipe_top_y':
            state[6]=value
        if key=='next_next_pipe_bottom_y':
            state[7]=value
    return state


def heuristique(state):
    r=np.random.rand()
    if state[9]> state[13] and r>.05:
        return 119
    if state[9]< state[12]:
        return None
    if  state[9]> state[12] and r>1-((state[12]-state[9])/(state[12]-state[13]))*1 and state[11]>0:
        return 119
    if state[11]<10 and r>.9:
        return 119

    if state[9]< state[12] and r>.4:
        return 119
    return None

def play(p, model):
    p.reset_game()
    state = np.zeros(17)
    reward = p.act(None)
    new_state = update_state(p,0,state)
    while not p.game_over():

        state=new_state
        q = model.predict(state.reshape((1, -1))).reshape(2)
        act = None

        #boucle de prise de decision
        prob = np.random.rand()
        if prob>epsilon:
            action = int(np.argmax(q))
            if action==1:
                act=119
        else:
            act=heuristique(state)
            action =int(action==119)

        # action
        reward = jeu_get_reward(p, act)

        #combien de blocks on a passe
        if reward==5:
            count_blocks[j]=count_blocks[j]+1

        #formation du nouveau etat
        new_state =   update_state(p,action,state)


def jeu_get_reward(p,action):
    r =p.act(action)
    if r==1:
        return 5
    if r==0:
        return 5/10
    else:
        return -5

def update_state(p, action, state):
    # return etat dans la forme etat_t-1 action etat_t
    raw_state = game.getGameState()
    new_state = np.zeros(17)
    new_state[9:] = form_state(raw_state)
    new_state[8] = action
    new_state[:8] = state[9:]
    return new_state



def replay_update(state,action,reward, new_state,verb):
    global not_full, counter_replay
    if not_full:

        replay[counter_replay,0:17]=state
        replay[counter_replay,17]=action
        replay[counter_replay,18]=reward
        replay[counter_replay,19:]=new_state
        counter_replay = counter_replay + 1
        not_full=(counter_replay<(buffer_size-1))
        return None
    else:
        if counter_replay == buffer_size-1:
            counter_replay=0
        else:
            counter_replay= counter_replay + 1
        replay[counter_replay,0:17]=state
        replay[counter_replay,17]=action
        replay[counter_replay,18]=reward
        replay[counter_replay,19:]=new_state
        minibatch = np.array( random.sample(replay, batch_size))
        old_states = minibatch[:,:17]
        new_states =  minibatch[:,19:]
        actions = minibatch[:,17]
        rewards = minibatch[:,18]
        Qs = model.predict(old_states)
        newQs = model.predict(new_states)
        X_train = old_states
        y_train = np.zeros((batch_size, 2))
        for k in range(len(rewards)):
            maxQ = np.max(newQs[k,:])
            y_train[k] = Qs[k,:]
            if reward != -5: #non-terminal state
                update = (reward + (gamma * maxQ))
            else: #terminal state
                update = reward
            y_train[k,int(actions[k])] = update

        model.fit(X_train, y_train, batch_size=batch_size, verbose=verb)
        return None



## start boucle apprentissage
q = np.zeros((2))
count_blocks=np.zeros(epochs)
state = np.zeros((17))
action = 0

p=pws
for j in range(epochs):
    if (j%50)==1:
        print(epsilon   )
        p=ps
        verbose=2
    else:
        p=pws
        verbose=0
    p.reset_game()

    reward = p.act(None)
    new_state = update_state(p,0,state)

    while not p.game_over():

        state=new_state
        q = model.predict(state.reshape((1, -1))).reshape(2)
        act = None

        #boucle de prise de decision
        prob = np.random.rand()
        if prob>epsilon:
            action = int(np.argmax(q))
            if action==1:
                act=119
        else:
            act=heuristique(state)
            action =int(action==119)

        # action
        reward = jeu_get_reward(p, act)

        #combien de blocks on a passe
        if reward==5:
            count_blocks[j]=count_blocks[j]+1

        #formation du nouveau etat
        new_state =   update_state(p,action,state)

        replay_update(state, action, reward, new_state,verbose)

    #affichage des resultats
    if j%50==0:
        print('epsilon =  %f', epsilon)
        print('jumped blocks = ')
        print(np.mean(count_blocks[j-50:j]))

    #mise a jour de la temperature
    epsilon = lda*epsilon
    if j%1000 == 0:
        model.save("model_flappyv2.dqf")
print('maintenant tout seul')
play(ps, model)
