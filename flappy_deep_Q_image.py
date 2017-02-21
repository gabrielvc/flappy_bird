#Deep Q net for flappy bird

import random
from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy import misc, ndimage

from collections import deque
from keras import initializations
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam



# Load two types of game, one with display of the game being played and other used for training

game = FlappyBird()
ps = PLE(game,fps=30, frame_skip=1, force_fps=False, display_screen=True,num_steps=1)
ps.init()
pws = PLE(game,fps=30, frame_skip=1, force_fps=True, display_screen=False,num_steps=1)
pws.init()


#creation du modele


# model=load_model("model_flappy_image.dqf")
model=Sequential()
model.add(Convolution2D(32, 6, 6, subsample=(4,4),init='lecun_uniform', border_mode='same',input_shape=(4,60,100)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2,2),init='lecun_uniform', border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1),init='lecun_uniform', border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(2,init='lecun_uniform'))

adam = Adam(lr=1e-6)
model.compile(loss='mse',optimizer=adam)


#buffer memory and batch size

batch_size=32
buffer_size=5000


def get_background(image, state):
    #take the bird out of the picture, in order to only get the background
    image_without_bottom = image[50:,:405]
    image_only_background = image_without_bottom
    image_only_background[:50,int(state[0]):int(state[0])+25] = image_without_bottom[50:100,int(state[0]):int(state[0])+25]
    return image_only_background

def treat_image(image, background):
    #take an image and background and gives the black and with image of what is not the background (bird and blocks)
    image_without_background=(image[50:,:405]-background)
    ind_blanc = image_without_background>0
    image_treated = image_without_background
    image_treated[ind_blanc] = 1
    image_treated = ndimage.morphology.binary_closing(image_treated).astype(np.int)
    image_treated = misc.imresize(image_treated,(60,100))
    return image_treated


state_game_memory = []
fcounter_get_state = 0
def game_get_state(game):
    #this functions corrects the getGameState function of ple, that considered that one has passed the block as one passes the middle of it
    global state_game_memory, fcounter_get_state
    brut_state = form_state(game.getGameState())
    if (brut_state[2]<8) and (fcounter_get_state==0):
        fcounter_get_state=1
        state_game_memory=brut_state
    if (fcounter_get_state>0) and fcounter_get_state<19:
        state_game=state_game_memory + np.array([0,0,8,0,0,8,0,0])*(19-fcounter_get_state+1)
        state_game[0:1]=brut_state[0:1]
        fcounter_get_state=(fcounter_get_state+1)%19
        return state_game
    if (fcounter_get_state==0) and not(brut_state[2]<8):
        return brut_state + np.array([0,0,8,0,0,8,0,0])*19

reward_fcounter=0
def jeu_get_reward(p,action,game):
    #modifies the reward function in ple, and shifts it in time as the previous function
    global reward_fcounter
    r =p.act(action)
    raw_state = game_get_state(game)
    if reward_fcounter>0 and reward_fcounter<12:

        reward_fcounter=reward_fcounter+1
        return (r==-5)*(-10) + (r==0)*2
    if reward_fcounter==12:

        reward_fcounter=0
        return 10*(r==0)+(r==-5)*(-10)

    if r==1:

        reward_fcounter=1
        return 2
    if r==0:
        if raw_state[0]>raw_state[3] and raw_state[0]<raw_state[4]-23:
            return 2
        else:
            return -1
    else:

        return -10

def form_state(dic):
    #Take a dictionnary and returns a vector
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



count_frames=0
def replay_update(state,action,reward, new_state,gamma):
    #this function deals with buffer and training
    global count_frames
    count_frames = count_frames+1
    replay.append((state,action,reward, new_state))
    n_events= len(replay)
    if n_events>buffer_size:
        replay.popleft()
    if n_events>batch_size and count_frames%5==1:
        minibatch = random.sample(replay, batch_size)
        states_train = np.zeros((batch_size,4,60,100))
        new_states_train = np.zeros((batch_size,4,60,100))
        actions_train = np.zeros((batch_size,1))
        rewards_train = np.zeros((batch_size,1))
        for k in range(batch_size):
            states_train[k:k+1,:,:,:]=minibatch[k][0]
            new_states_train[k:k+1,:,:,:]=minibatch[k][3]
            actions_train[k,0]=minibatch[k][1]
            rewards_train[k,0]=minibatch[k][2]

        Qnew = model.predict(new_states_train)
        update = model.predict(states_train)
        for k in range(batch_size):
            if not rewards_train[k,0]==-10:
                update[k,int(actions_train[k,0])]= rewards_train[k,0]+gamma*np.max(Qnew[k,:])
            else:
                update[k,int(actions_train[k,0])]=-10



        history = model.fit(states_train, update, nb_epoch=1,verbose=0)

        return history
    return None


## Main program block


epochs =200000 
gamma = .9 #discount

p=ps

loss=[]
replay = deque()
q = np.zeros((2))
count_blocks=np.zeros(epochs)
action = 0
epsilon =0.6
for j in range(epochs):

    if (j%100)==1:
        p=ps
    else:
        p=pws


    p.reset_game()
    reward = jeu_get_reward(p,None,game)


    #get_background
    raw_state =game_get_state(game)
    img = p.getScreenGrayscale()
    img = np.array(img)
    background = get_background(img,raw_state)

    img = treat_image(img, background)

    new_state = np.stack((img,img,img,img), axis=0)
    new_state = new_state.reshape(1, new_state.shape[0], new_state.shape[1], new_state.shape[2])

    while not p.game_over():

        state=new_state
        act=None
        prob = np.random.rand()
        
        #decides if the action is going to be at random or not and chooses the action
        if prob<epsilon:
            action = int(prob<epsilon/2.0)
        else:
            action = np.argmax(model_2.predict(state).reshape(2))
        act = (action==1)*119

        # action
        reward = jeu_get_reward(p, act, game)

        #count how many blocks we've passed in this game
        if reward==10:
            count_blocks[j]=count_blocks[j]+1

        #update new state
        raw_state = game_get_state(game)
        img = p.getScreenGrayscale()
        img = np.array(img)
        img = treat_image(img, background)
        img = img.reshape((1,1,60,100))
        new_state = np.append(img, state[:, :3, :, :], axis=1)
        
        #update buffer and train
        history = replay_update(state, action, reward, new_state, gamma)
        
        if history is not None:
            loss+=history.history["loss"]



    #Print of different informations
    print(j)
    if j%100==0:
        print('Epsilon = ', epsilon)
        print('jumped blocks = ')
        print(np.mean(count_blocks[j-50:j]))
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.plot(loss)

        ax2 = fig.add_subplot(212)
        ax2.plot([np.mean(count_blocks[u-100:u]) for u in range(100,j)] )

        plt.show(block=False)


    #update epsilon
    epsilon = epsilon - .6/float(epochs/2.0)
    
    #saves model
    if j%200 == 1:
        model.save("model_flappy_image.dqf")

