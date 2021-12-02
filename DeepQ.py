import sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # fraction of memory. this helps significantly with speed.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
'''
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
TICTACTOE GAME CLASS

this class inspired heavily by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
'''

class tictactoe:
    # players are identified as 1 and -1.
    # board is just a 3x3 grid of zeros at first.
    # when a player moves, update that spot from
    # a zero to a 1 or -1.
    def __init__(self, player1, player2, numrows=3, numcols=3, wincount=3, gravitymove=False, randinit=False):
        self.numrows = numrows
        self.numcols = numcols
        self.wincount = wincount
        self.board = np.zeros((self.numrows, self.numcols))
        self.player1 = player1
        self.player2 = player2
        self.gravitymove=gravitymove
        self.randinit=randinit
        self.isEnd = False
        self.boardID = None
        # init p1 plays first
        self.playerSymbol = 1
        
    # unique identifier for current board state
    def getBoardID(self):
        '''
        Get unique identifier for current board state.
        Literally just use current board.
        '''
        self.boardID = self.board.copy()
        return self.boardID
    
    def availablePositions(self):
        '''
        Get all available positions.
        For the Deep problems, which are all gravity-based, just all columns that aren't full.
        '''
        positions = []
        for j in range(0, self.numcols):
            if self.board[0, j] == 0:
                positions.append(j)
        #print(positions)
        return positions
    
    def updateMove(self, position):
        '''
        Drop token into column.
        Find lowest spot in column that's empty and has a non-empty cell below it.
        '''
        rownum = self.numrows-1
        # if there exists one above bottom row, get row where current spot is clear and spot below is full
        for row in range(0, self.numrows-1):
            if self.board[(row+1, position)] != 0 and self.board[(row, position)] == 0:
                rownum = row
                
        self.board[(rownum, position)] = self.playerSymbol
        # switch to player -1 or 1
        self.playerSymbol = -1*self.playerSymbol
        
    def checkWinner(self):
        '''
        Check winner.
        Check rows first, see if there's any groups of columns in a certain row that sum
        to the win condition.
        Check columns next, see if there's any groups of rows in a certain column that sum
        to the win condition.
        Finally, check diagonals. See if there's any squares of wincondition x wincondition
        where the diagonals sum to the win condition.
        '''
        for i in range(0, self.numrows):
            #print('row')
            for j in range(0, self.numcols-self.wincount+1):
                #print("col")
                if sum(self.board[i,j:j+self.wincount]) == self.wincount:
                    self.isEnd=True
                    return 1
                if sum(self.board[i,j:j+self.wincount]) == -1*self.wincount:
                    self.isEnd=True
                    return -1
        # cols
        for i in range(0, self.numcols):
            for j in range(0, self.numrows-self.wincount+1):
                if sum(self.board[j:j+self.wincount,i]) == self.wincount:
                    #print("win")
                    self.isEnd=True
                    return 1
                if sum(self.board[j:j+self.wincount,i]) == -1*self.wincount:
                    self.isEnd=True
                    return -1
            '''
            if sum(self.board[:,i]) == self.wincount:
                self.isEnd=True
                return 1
            if sum(self.board[:,i]) == -1*self.wincount:
                self.isEnd=True
                return -1
            '''
        # diags
        # top left to bot right
        for i in range(0, self.numrows-self.wincount+1):
            for j in range(0, self.numcols-self.wincount+1):
                diag1 = sum([self.board[i+k,j+k] for k in range(0, self.wincount)])
                if abs(diag1) == self.wincount:
                    self.isEnd = True
                    return int(diag1//self.wincount)
        # top right to bottom left
        for i in range(0, self.numrows-self.wincount+1):
            for j in range(self.wincount-1, self.numcols):
                diag2 = sum([self.board[i+k,j-k] for k in range(0, self.wincount)])
                if abs(diag2) == self.wincount:
                    self.isEnd = True
                    return int(diag2//self.wincount)
        '''
        diag1 = sum([self.board[i,i] for i in range(0, self.wincount)])
        diag2 = sum([self.board[i,self.numcols-i-1] for i in range(0, self.wincount)])
        diag = max(abs(diag1), abs(diag2))
        if diag == self.wincount:
            self.isEnd = True
            if diag1 == self.wincount or diag2 == self.wincount:
                return 1
            else:
                return -1
        '''
        # tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        return None
    
    def giveReward(self):
        '''
        Gives rewards to players.
        Checks winner and calls their feed reward function.
        Rewards for win, loss, and tie for each player are specified here.
        '''
        winner = self.checkWinner()
        if winner == 1:
            self.player1.feedReward(1)
            self.player2.feedReward(0)
        elif winner == -1:
            self.player1.feedReward(0)
            self.player2.feedReward(1)
        else:
            # keep in mind that a tie is worse for player 1 who has more moves to win with.
            # but we need to give a little bit of a reward for tying, otherwise he way
            # overprioritizes winning.
            self.player1.feedReward(0.2)
            self.player2.feedReward(0.6)
            
    def reset(self):
        self.board = np.zeros((self.numrows, self.numcols))
        self.boardID = None
        self.isEnd = False
        self.playerSymbol = 1
        
    '''
    PLAYING
    '''
    
    '''
    play against self
    '''
    def play(self, trainingrounds=100):
        '''
        Main training function. 
        Stores wins and updates in a list.
        '''
        wins = []
        p1updates = []
        p2updates = []
        '''
        Here, store the mean value of how many random moves we'll make
        to initialize the game.
        '''
        meanrand = max((self.player1.explore_rate/4)*self.numrows*self.numcols-1, 0)
        print(meanrand)
        for i in range(0, trainingrounds):
            print(f"{i} of {trainingrounds}")
            rand=False
            randmoves1 = 0
            randmoves2 = 0
            '''
            If we're randomly initializing, generate number of moves to play
            randomly. Store.
            '''
            if self.randinit:
                randmoves1 = int(round(np.random.normal(meanrand, meanrand/2)))
                randmoves2 = int(round(np.random.normal(meanrand, meanrand/2)))
            '''
            While game is not over, make moves (first randomly, then by 
            calling the chooseAction function). Check winner after each move.
            If a winner is found, reset board and give rewards to players.
            '''
            while not self.isEnd:
                preboard = self.getBoardID()
                positions = self.availablePositions()
                if randmoves1 > 0:
                    rand=True
                    randmoves1 = randmoves1-1
                player1move = self.player1.chooseAction(positions, 
                                                        self.board.reshape((self.numrows, self.numcols, 1)), 
                                                        self.playerSymbol, rand=rand)
                rand=False
                self.updateMove(player1move)
                boardID = self.getBoardID()
                self.player1.addState(preboard, player1move)
                
                win = self.checkWinner()
                # if player 1 wins
                if win != None:
                    self.giveReward()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset()
                    break
                # if player 1 did not win
                else:
                    preboard = self.getBoardID()
                    positions = self.availablePositions()
                    if randmoves2 > 0:
                        rand=True
                        randmoves2 -= 1
                    player2move = self.player2.chooseAction(positions, 
                                                            self.board.reshape((self.numrows, self.numcols, 1)), 
                                                            self.playerSymbol, rand=rand)
                    rand=False
                    self.updateMove(player2move)
                    boardID = self.getBoardID()
                    self.player2.addState(preboard, player2move)

                    win = self.checkWinner()
                    # if player 1 wins
                    if win != None:
                        self.giveReward()
                        self.player1.reset()
                        self.player2.reset()
                        self.reset()
                        break
            '''
            After we've found a winner, append who won to wins list
            and append updates players make to their Q table to pupdates.
            '''
            wins.append(win)
            p1updates.append(self.player1.mostrecentupdate)
            p2updates.append(self.player2.mostrecentupdate)
            #self.giveReward()
            #self.player1.reset()
            #self.player2.reset()
            #self.reset()
        '''
        Return a bunch of other stuff to makedynamic to make graphs.
        '''
        avgupdate1 = self.player1.avgupdate
        avgupdate2 = self.player2.avgupdate
        p1stop = self.player1.stoplearning
        p2stop = self.player2.stoplearning
        p1badavg = self.player1.badavg
        p2badavg = self.player2.badavg
        #self.player1.movestowinavg = sum(self.player1.movestowin)/len(self.player1.movestowin)
        movestowin = self.player1.movestowin
        return wins, p1updates, p2updates, avgupdate1, avgupdate2, p1stop, p2stop, p1badavg, p2badavg, movestowin
    '''
    Play against human player
    '''
    def playhuman(self, prints=True):
        '''
        Play against human player. Fetch actions from both players.
        Can also be used to play two bots against each other.
        
        Pretty much same logic as inside play() function.
        While game not ended, lets each player make moves.
        '''
        self.showBoard()
        while not self.isEnd:
            positions = self.availablePositions()
            print(positions)
            player1move = self.player1.chooseAction(positions, self.board, self.playerSymbol, playhuman=True)
            print(player1move)
            self.updateMove(player1move)
            self.showBoard()
            win = self.checkWinner()
            if win != None:
                self.reset()
                break
            else:
                positions = self.availablePositions()
                print(positions)
                player2move = self.player2.chooseAction(positions, self.board, self.playerSymbol, playhuman=True)
                self.updateMove(player2move)
                self.showBoard()
                win = self.checkWinner()
                if win != None:
                    self.reset()
                    break
        if prints:
            if win == 1:
                print(self.player1.name, "wins!")
            elif win == -1:
                print(self.player2.name, "wins!")
            else:
                print("tie!")
        return win
                    
    def showBoard(self):
        '''
        Shows visual representation of board.
        '''
        # player1: x  player2: o
        colcounts = "   "
        for j in range(0, self.numcols):
            colcounts += f"| {j} "
        colcounts += "|"
        print(colcounts)
        for i in range(0, self.numrows):
            divider = '-----'
            for j in range(0, self.numcols):
                divider += '----'
            print(divider)
            out = f'{i}  | '
            for j in range(0, self.numcols):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print(divider+'\n')
        
''' 
PLAYER CLASS

Also heavily inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
in addition to inspiration from Stefan Voigt https://www.voigtstefan.me/post/connectx/

In DeepQ, we use a neural network instead of a states_values dictionary. 

'''
class player:
    def __init__(self, name, explore_rate=0.3, learning_rate = 0.2, board_size = [3, 3, 1], winc=3, layers=3,
                numfilters=4, filtersize=3, squareloss=False, avgupdate=100, stoplearning=False, badavg=0):
        # board size must be a tuple
        self.name = name
        self.statesactions = []
        self.learningrate = learning_rate
        self.explore_rate = explore_rate
        self.explore_min = 0.01
        self.decay_gamma = 0.98
        self.state_size = board_size
        self.action_size = board_size[1]
        self.b = board_size[0]*board_size[1]
        self.winc = winc
        self.layers = layers
        self.numfilters=numfilters
        self.filtersize=filtersize
        self.squareloss=squareloss
        # store the size for our layers here. basically do boardsize * (actionsize/boardsize) ^ (layer_at/layers_total)
        # this doesn't work, but no point getting rid of it.
        self.layersizes = []
        for i in range(1, layers):
            self.layersizes.append(int(round(self.b * (self.action_size/self.b)**(i/layers))))
        self.states_values = self._build_network()
        self.mostrecentupdate = 0
        self.avgupdate = avgupdate
        self.stoplearning = stoplearning
        self.badavg = badavg
        self.movestowin = []
        
    def _build_network(self):
        '''
        Loosely inspired by https://www.voigtstefan.me/post/connectx/.
        
        Network:
            convolutions with number specified by user.
            dense layer with neurons equal to board size.
            dense layer with neurons equal to board size.
            output layer ReLU.
        '''
        input_shape = self.state_size
        #input_shape.append(1)
        #print(input_shape)
        
        model = Sequential()
        # first, convolutions of 3x3 to capture 3 in a row. will also capture all groups of 2x2.
        model.add(Conv2D(self.numfilters, (self.filtersize, self.filtersize), padding="same", 
                         input_shape = input_shape, activation="relu"))
        # next, convolutions of 2x2. no max pooling, want to capture everything. this is to capture 
        # more complex moves. not necessary for tic tac toe.
        #model.add(Conv2D(2, (2, 2), padding="same", input_shape = input_shape, activation="relu"))
        # next, 2 FC layers, both with same size as board.
        model.add(Dense(self.b, activation="elu"))
        model.add(Dense(self.b, activation="elu"))
        # action size is now just number of columns. so it's 3, for example, for tic tac toe.
        model.add(Dense(self.action_size, activation="sigmoid"))
        # we just set the LR to 1 since we already have a learning rate built into the loss
        # we compute for each game as part of the Q learning algorithm.
        model.compile(loss="mse", optimizer=Adam())
        return model
        
    def getBoardID(self, board):
        '''
        Gets board ID.
        Turns board into string.
        '''
        boardID = str(board.reshape(len(board)*len(board[0])))
        return boardID
        
    def chooseAction(self, positions, current_board, symbol, rand=False, playhuman=False):
        '''
        Chooses action.
        Choose action randomly. If we draw a uniform number
        lower than our explore rate, keep it. otherwise, predict
        best action from our neural network. 
        '''
        # choose whether to explore or not
        action = positions[np.random.choice(len(positions))]
        #print("a")
        if not rand:
            max_value = -10000
            if np.random.uniform(0, 1) >= self.explore_rate:
                #print("b")
                # predict values of actions from current board -- argmax across available positions
                #if playhuman:
                #    print(current_board.reshape((1, 
                #                                                                  self.state_size[0], 
                #                                                                  self.state_size[1], 
                #                                                                  self.state_size[2])))
                action_values = self.states_values.predict(current_board.reshape((1, 
                                                                                  self.state_size[0], 
                                                                                  self.state_size[1], 
                                                                                  self.state_size[2])))
                #if playhuman:
                    #print(action_values)
                #print(action_values)
                #action = int(np.mean(np.argmax(action_values, axis=3)))
                #print(action_values)
                for p in positions:
                    # can now just assign action to index in output layer.
                    if action_values[0,0,0,p] > max_value:
                        max_value = action_values[0,0,0,p]
                        action = p
        return action
    
    def addState(self, state, action):
        '''
        appends a state to statesactions list.
        '''
        self.statesactions.append((state, action))
    
    def feedReward(self, reward):
        '''
        Feeds rewards to neural network. 
        Iterates backwards through states, storing previous state's updated
        value as its new reward. Uses this reward and the current action values of the
        state to establish the new action values for the state.
        Gives these updated action values back to neural network and has it fit on them.
        
        Basically exactly what we do for tabular Q-learning, just feeding to a neural network instead.
        '''
        if self.stoplearning:
            self.learningrate = 0.000001
        addupdate = True
        
        '''
        If we've won in only winc moves, we don't get a reward. Prevents getting stuck in a
        condition where both players play chicken. Only if board is reasonably large.
        '''
        print(f"moves to win: {len(self.statesactions)}")
        
        #Code for penalizing finishing too early.
        #Algo ends up just stalling until it can skip the penalty.
        # still useful to have to spur better training.
        #if len(self.statesactions) <= 6:
            #print(self.statesactions[-1])
        '''
        if 2*self.winc < self.b/2:
            #print("less")
            if len(self.statesactions) <= int(np.sqrt(self.b)):
                reward = -1
        '''
        
        
        '''
        Instead, make reward scale up with how far into game we are.
        At end of game, get full reward of 1.
        '''
        if reward == 1:
            reward = 0.05 + (0.95/(self.b/2))*len(self.statesactions)
            
        '''
        Keep track of moves to win
        '''
        self.movestowin.append(len(self.statesactions))
            
        print(f"reward: {reward}")
        for stated, action in reversed(self.statesactions):
            
            '''
            Find best action value for state by predicting on state and then argmaxing.
            '''
            state = stated.reshape((1, self.state_size[0], self.state_size[1], self.state_size[2]))
            statesvalues = self.states_values.predict(state)
            statevaluehere = np.float64(statesvalues[0][0][0][action])
            #statevaluehere = np.float64(np.amax(statesvalues))
            
            '''
            Compute sixe of update.
            '''
            update = np.float64(self.learningrate) * (np.float64(self.decay_gamma * reward) - statevaluehere)
            '''
            Cap updates so they don't balloon into oblivion. This has been a problem before!
            '''
            if self.squareloss:
                if update > np.float64(3.0):
                    update = np.float64(3.0)
                if update < np.float64(-3.0):
                    update = np.float64(-3.0)
            else:
                if update > np.float64(9.0):
                    update = np.float64(9.0)
                if update < np.float64(-9.0):
                    update = np.float64(-9.0)
            if self.squareloss:
                # even though we're squaring loss, should keep sign -- direction of update is important!
                sign = update/abs(update)
                update = update ** 2
                update = update * sign
            
            
            '''
            Had a humongous bug here. My updates were off because the state value was 
            literally a float 32 and everything else was a float 64 and adding it 
            was producing some real funky stuff. Literally threw my entire
            training off.
            '''
            statevaluehere = np.float64(statevaluehere)
            target = np.float64(statevaluehere) + update
            reward = target
            
            if addupdate:
                if not self.stoplearning:
                    self.mostrecentupdate = update 
                else:
                    self.mostrecentupdate = np.float64(0.000001)
                self.avgupdate = 0.99 * self.avgupdate + 0.01 * abs(self.mostrecentupdate)
                addupdate = False
                print(f"update: {self.mostrecentupdate}")
                #print(self.avgupdate)
                '''
                in case we get any nans. 
                '''
                if self.mostrecentupdate == float('nan'):
                    self.mostrecentupdate = np.float64(1)
                    self.avgupdate = np.float64(1)
                '''
                Stop learning if updates are too big.
                I decided to get rid of this because
                it actually didnt' help.
                '''
                #if abs(self.mostrecentupdate) > self.avgupdate * 1.4:
                    #print("stop learn")
                    #self.stoplearning = True
                    #self.badavg = self.avgupdate+np.float64(0.00001)
                '''
                If we haven't overfit in a while, can restart. This will restart
                us at a lower exp and learn rate where hopefully we do a bit better.
                Again, this didn't really help, so I'm not using it.
                '''
                #if self.stoplearning:
                    #if self.avgupdate < np.float64(0.67)*self.badavg:
                        #self.stoplearning = False
                        #print("learning again.")
                        #self.learningrate = np.float64(0.2)
                        #self.mostrecentupdate = np.float64(0.01)
                        #self.avgupdate = np.float64(4)
                
            target_f = statesvalues
            # target f where we have max state is what we argmaxed for.
            # update that max value that we found to be our new target -- 
            # location of best action
            '''
            Update predicted action values with updated action values
            that we've just computed based on value from next state 
            (reward of best action) and
            current value from this state.
            '''
            indexhere = np.unravel_index(np.argmax(target_f), 
                                     target_f.shape)
            for k in range(0, target_f.shape[0]):
                for i in range(0, target_f.shape[2]):
                    for j in range(0, target_f.shape[1]):
                #target_f[np.unravel_index(np.argmax(target_f), 
                                     #target_f.shape)] = target
                        target_f[k, j, i, action] = target
            #print(target_f)
            self.states_values.fit(state, target_f, epochs=1, verbose=0)
            
            
            
            
    def reset(self):
        self.states = []
        self.statesactions = []
        
    def savePolicy(self):
        '''
        saves policy.
        '''
        #fw = open('policy_' + str(self.name), 'wb')
        self.states_values.save(f'models/{self.name}')
        #fw.close()

    def loadPolicy(self, file):
        '''
        loads policy.
        '''
        #fr = open(file, 'rb')
        self.states_values = keras.models.load_model(f'models/{self.name}')
        #fr.close()

        
class HumanPlayer:
    '''
    Human player class. Allows for inputs from human of their actions.
    Also has some dummy functions for functions that are called for
    all players.
    '''
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, board, playerSymbol, playhuman=False):
        while True:
            col = int(input("Input your action col:"))
            action = col
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass

    

def makedynamic(numrounds=100, numexplores=11, expstart=0.4, movingalpha=False,
                p1dynamic=None, p2dynamic=None, 
                rows=3, cols=3, winc=3, gmove=False, 
                randinit=False, stratnames='dynamic', aggover=100,
                numfilters=4, filtersize=3, selftrain=False, squareloss=False):
    '''
    This is where the magic happens! But this function is an absolute mess.
    This is the function to call to train our network.
    '''
    boardsize = [rows, cols, 1]
    x= []
    ydynamic = []
    p1updynamicavg = []
    p2updynamicavg = []
    expdiff = 0
    alphadiff = 0
    if numexplores > 1:
        expdiff = expstart/(numexplores-1)
    print(expdiff)
    winsdynamic = []
    p1updynamic = []
    p2updynamic = []
    moveslist = []
    movesdynamicavg = []
    
    '''
    If our alpha is moving, set its movement rate.
    If squareloss, don't let alpha get too low.
    '''
    if not movingalpha:
        alpha=0.2
    else:
        alphadiff = (expstart/2)/(numexplores-1)
        alpha=expstart
        
    
    '''
    Create players if we weren't given pre-existing players to train by our function.
    '''
    if p1dynamic == None:
        p1dynamic = player(stratnames+"1", explore_rate=expstart, learning_rate=alpha, board_size = boardsize,
                          numfilters=numfilters, filtersize=filtersize, squareloss=squareloss)
    if p2dynamic == None:
        p2dynamic = player(stratnames+"2", explore_rate=expstart, learning_rate=alpha, board_size = boardsize,
                          numfilters=numfilters, filtersize=filtersize, squareloss=squareloss)

    '''
    For all exploration epochs, play tictactoe games at the current epsilon and alpha.
    Save policy when done, and get returned a bunch of stuff for making graphs.
    Then reload policy into new player object to pass back to next epoch.
    '''
    for i in tqdm(range(0, numexplores)):
        exprate = round(expstart-expdiff*i, 2)
        # reset learning rate for next stage
        if movingalpha:
            alpha=round(expstart-alphadiff*i, 2)
            if alpha==0:
                alpha = 0.01
        if selftrain:
            st = tictactoe(p1dynamic, p1dynamic, numrows=rows, numcols=cols, wincount=winc, gravitymove=gmove, randinit=randinit)
        else:
            st = tictactoe(p1dynamic, p2dynamic, numrows=rows, numcols=cols, wincount=winc, gravitymove=gmove, randinit=randinit)
        #print(f"training for explore rate {exprate} and alpha {alpha}")
        wins, p1updates, p2updates, avgupdate1, avgupdate2, p1stop, p2stop, p1badavg, p2badavg, movestowin = st.play(numrounds)
        winsdynamic.extend(wins)
        p1updynamic.extend(p1updates)
        p2updynamic.extend(p2updates)
        moveslist.extend(movestowin)

        p1dynamic.savePolicy()
        p2dynamic.savePolicy()
        
        p1dynamic = player(stratnames+"1", explore_rate=exprate, learning_rate=alpha, board_size = boardsize,
                          numfilters=numfilters, filtersize=filtersize, squareloss=squareloss, avgupdate=avgupdate1, 
                           stoplearning=p1stop, badavg=p1badavg)
        p2dynamic = player(stratnames+"2", explore_rate=exprate, learning_rate=alpha, board_size = boardsize,
                          numfilters=numfilters, filtersize=filtersize, squareloss=squareloss, avgupdate=avgupdate2, 
                           stoplearning=p2stop, badavg=p2badavg)
        p1dynamic.loadPolicy("policy_"+stratnames+"1")
        p2dynamic.loadPolicy("policy_"+stratnames+"2")
        
    '''
    Aggregate over training data that was returned to make nicer looking graphs. 
    '''
    for i in range(0, numrounds*numexplores//aggover):
        x.append(i)
        ydynamic.append(np.mean(np.abs(winsdynamic[(i*aggover):(i+1)*aggover])))
        p1updynamicavg.append(np.mean(np.abs(p1updynamic[(i*aggover):(i+1)*aggover])))
        p2updynamicavg.append(np.mean(np.abs(p2updynamic[(i*aggover):(i+1)*aggover])))
        movesdynamicavg.append(np.mean(np.abs(moveslist[(i*aggover):(i+1)*aggover])))
    
    '''
    Make all of our plots.
    '''
    plt.scatter(x, ydynamic, s=4)
    plt.ylim(0, 1)
    plt.xlabel(f"Training round / {aggover}")
    plt.ylabel("Rate of games with clear winner")
    plt.title(f"Rate of wins vs. training rounds with {numexplores} explore rates from {0.4} down to 0")
    #for i in range(0, numexplores):
        #print(i*numrounds//aggover)
        #plt.vlines(i*numrounds//aggover, 0, 1)
    plt.show()
    
    plt.plot(x, p1updynamicavg, label="Player 1")
    plt.plot(x, p2updynamicavg, label="Player 2")
    plt.xlabel(f"Training round / {aggover}")
    plt.ylabel("Update size")
    plt.show()
    
    plt.plot(x, movesdynamicavg, label="Player 1")
    plt.xlabel(f"Training round / {aggover}")
    plt.ylabel("Average moves to win")
    plt.show()
    
    return p1dynamic, p2dynamic