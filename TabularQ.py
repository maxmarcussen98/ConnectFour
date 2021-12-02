import numpy as np
import tensorflow as tf
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
        self.boardID = str(self.board.reshape(self.numrows*self.numcols))
        return self.boardID
    
    def availablePositions(self):
        positions = []
        for i in range(0, self.numrows):
            for j in range(0, self.numcols):
                if self.board[i, j] == 0:
                    if self.gravitymove:
                        #print(i, j)
                        if i+1 == self.numrows:
                            positions.append((i, j))
                        elif (i+1 < self.numrows and self.board[i+1, j] != 0):
                            #print("use")
                            #print(i, j)
                            positions.append((i, j))
                    else:
                        positions.append((i, j))
        #print(positions)
        return positions
    
    def updateMove(self, position):
        self.board[position] = self.playerSymbol
        # switch to player -1 or 1
        self.playerSymbol = -1*self.playerSymbol
        
    def checkWinner(self):
        #print("checking winner")
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
        winner = self.checkWinner()
        if winner == 1:
            self.player1.feedReward(1)
            self.player2.feedReward(-0.1)
        if winner == -1:
            self.player1.feedReward(0)
            self.player2.feedReward(1)
        else:
            # keep in mind that a tie is worse for player 1 who has more moves to win with
            self.player1.feedReward(0.1)
            self.player2.feedReward(0.9)
            
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
        wins = []
        meanrand = (self.player1.explore_rate**2)*self.numrows*self.numcols
        for i in tqdm(range(0, trainingrounds)):
            rand=False
            randmoves1 = 0
            randmoves2 = 0
            if self.randinit:
                randmoves1 = int(round(np.random.normal(meanrand, meanrand/2)))
                randmoves2 = int(round(np.random.normal(meanrand, meanrand/2)))
            while not self.isEnd:
                positions = self.availablePositions()
                if randmoves1 > 0:
                    rand=True
                    randmoves1 = randmoves1-1
                player1move = self.player1.chooseAction(positions, self.board, self.playerSymbol, rand=rand)
                rand=False
                self.updateMove(player1move)
                boardID = self.getBoardID()
                self.player1.addState(boardID)
                
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
                    positions = self.availablePositions()
                    if randmoves2 > 0:
                        rand=True
                        randmoves2 -= 1
                    player2move = self.player2.chooseAction(positions, self.board, self.playerSymbol, rand=rand)
                    rand=False
                    self.updateMove(player2move)
                    boardID = self.getBoardID()
                    self.player2.addState(boardID)

                    win = self.checkWinner()
                    # if player 1 wins
                    if win != None:
                        self.giveReward()
                        self.player1.reset()
                        self.player2.reset()
                        self.reset()
                        break
            wins.append(win)
        return wins
    '''
    Play against human player
    '''
    def playhuman(self, prints=True):
        self.showBoard()
        while not self.isEnd:
            positions = self.availablePositions()
            player1move = self.player1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateMove(player1move)
            self.showBoard()
            win = self.checkWinner()
            if win != None:
                self.reset()
                break
            else:
                positions = self.availablePositions()
                player2move = self.player2.chooseAction(positions, self.board, self.playerSymbol)
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

'''
class player:
    def __init__(self, name, explore_rate=0.3, learning_rate = 0.2):
        self.name = name
        self.states = []
        self.learningrate = learning_rate
        self.explore_rate = explore_rate
        self.decay_gamma = 0.9
        self.states_values = {}
        
    def getBoardID(self, board):
        boardID = str(board.reshape(len(board)*len(board[0])))
        return boardID
        
    def chooseAction(self, positions, current_board, symbol, rand=False):
        # choose whether to explore or not
        action = positions[np.random.choice(len(positions))]
        if not rand:
            if np.random.uniform(0, 1) >= self.explore_rate:
                value_max = -10000
                for position in positions:
                    nextBoard = current_board.copy()
                    nextBoard[position] = symbol
                    nextBoardHash = self.getBoardID(nextBoard)
                    value = 0
                    if self.states_values.get(nextBoardHash) != None:
                        value = self.states_values.get(nextBoardHash)
                    if value >= value_max:
                        value_max = value
                        action = position
        return action
    
    def addState(self, state):
        self.states.append(state)
    
    def feedReward(self, reward):
        for state in reversed(self.states):
            if self.states_values.get(state) == None:
                self.states_values[state] = 0
            self.states_values[state] += self.learningrate * (self.decay_gamma * reward - self.states_values[state])
            reward = self.states_values[state]
            
    def reset(self):
        self.states = []
        
    def savePolicy(self):
        fw = open('policies/policy_' + str(self.name), 'wb')
        pickle.dump(self.states_values, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open('policies/'+file, 'rb')
        self.states_values = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, board, playerSymbol):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
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

'''
DYNAMIC EXPLORE RATE
'''
def makedynamic(numrounds=15000, numexplores=11, expstart=0.4, movingalpha=False,
                p1dynamic=None, p2dynamic=None, 
                rows=3, cols=3, winc=3, gmove=False, 
                randinit=False, stratnames='dynamic'):
    # moving alpha - have alpha (learning rate) be equal to explore rate -- so learn faster
    # in exploration stage, learn slower in convergence stage
    x= []
    ydynamic = []
    expdiff = 0
    if numexplores > 1:
        expdiff = expstart/(numexplores-1)
    print(expdiff)
    winsdynamic = []
    if not movingalpha:
        alpha=0.2
    else:
        alpha=expstart
    if p1dynamic == None:
        p1dynamic = player(stratnames+"1", explore_rate=expstart, learning_rate=alpha)
    if p2dynamic == None:
        p2dynamic = player(stratnames+"2", explore_rate=expstart, learning_rate=alpha)

    for i in range(0, numexplores):
        exprate = round(expstart-expdiff*i, 2)
        # reset learning rate for next stage
        if movingalpha:
            alpha=exprate
            if alpha==0:
                alpha = 0.01
        
        st = tictactoe(p1dynamic, p2dynamic, numrows=rows, numcols=cols, wincount=winc, gravitymove=gmove, randinit=randinit)
        print(f"training for explore rate {exprate} and alpha {alpha}")
        winsdynamic.extend(st.play(numrounds))

        p1dynamic.savePolicy()
        p2dynamic.savePolicy()
        
        
        
        p1dynamic = player(stratnames+"1", explore_rate=exprate, learning_rate=alpha)
        p2dynamic = player(stratnames+"2", explore_rate=exprate, learning_rate=alpha)
        p1dynamic.loadPolicy("policy_"+stratnames+"1")
        p2dynamic.loadPolicy("policy_"+stratnames+"2")

    for i in range(0, numrounds*numexplores//100):
        x.append(i)
        ydynamic.append(np.mean(np.abs(winsdynamic[(i*100):(i+1)*100])))

    plt.scatter(x, ydynamic, s=4)
    plt.ylim(0, 1)
    plt.xlabel("Training round / 100")
    plt.ylabel("Rate of games with clear winner")
    plt.title(f"Rate of wins vs. training rounds with {numexplores} explore rates from {0.4} down to 0")
    for i in range(0, numexplores):
        plt.vlines(i*numrounds//100, 0, 1)
    plt.show()
    
    return p1dynamic, p2dynamic

