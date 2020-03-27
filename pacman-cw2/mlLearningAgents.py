# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

# Yilei Liang (K1764097)

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the

    # Note: I set the epsilon to 1 and decay this value during training
    def __init__(self, alpha=0.2, epsilon=1, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # A dictionary mapping the state(location) and Q-Value table
        # Key: State
        # Data: Mapping of {Q values} in this state (a dictionary as well)
        self.table = {}
        # For recording the reward of doing action A in state S
        # As we don't know the reward until we reach S'
        self.previous_state = None
        self.previous_score = 0
        self.lastAction = None
        # Init condition (We don't need learning at the first step)
        self.initial = True
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts



    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        alpha = self.getAlpha()
        gamma = self.getGamma()


        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # print "Legal moves: ", legal
        # print "Pacman position: ", state.getPacmanPosition()
        # print "Ghost positions:" , state.getGhostPositions()
        # print "Food locations: "
        # print state.getFood()
        # print "Score: ", state.getScore()

        location = state.getPacmanPosition()
        ghosts = state.getGhostPositions()


        # As I use dictionary (like hashmap), so we need to expand the dictionary for each state
        if state not in self.table:
            # Key: Action
            # Data: Q-Value of executing this action
            Q_table = {}
            for action in legal:
                # Init the Q-Value to 0
                Q_table[action] = 0
            self.table[state] = Q_table

        
        # Epsilon-Greedy
        current_table = self.table[state]
        # Random generate a number for explore/exploit
        choose = random.random()
        best_action = max(current_table, key=lambda k: current_table[k])
        #Exploitation
        if choose >= self.epsilon:
            #Choose the best action      
            pick = best_action
            
        #Exploration (random select)
        else:      
            pick = random.choice(legal)
        

        
        # As we don't know the reward value of taking action A in state S. we have to execute it first and update it later
        # And the beging state S0 has no previous state, we have to use this to prevent
        if self.initial == False and self.alpha != 0: 
            reward = state.getScore() - self.previous_score

            #print(reward)
            previous_state = self.previous_state
            previous_Q_Table = self.table[previous_state]
            previous_Q_Value = previous_Q_Table[self.lastAction]
            best_current_action = max(current_table, key=lambda k: current_table[k])
            best_current_Q = current_table[best_current_action]
            previous_Q_Table[self.lastAction] = previous_Q_Value + alpha * (reward + (gamma * best_current_Q) - previous_Q_Value)
            self.table[previous_state] = previous_Q_Table
        
        #Change this to false so that it's not the first training step now
        self.initial = False


        # Use these informations to calcualte the reward at next step (as we don't know the reward now)
        self.previous_score = state.getScore()
        self.previous_state = state
        self.lastAction = pick

        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"

        # Learning for game ending
        if (self.alpha != 0):
            alpha = self.getAlpha()
            gamma = self.getGamma()
            reward = state.getScore() - self.previous_score
            previous_state = self.previous_state
            previous_Q_Table = self.table[previous_state]
            previous_Q_Value = previous_Q_Table[self.lastAction]
            best_current_Q = 0
            if state in self.table:
                current_table = self.table[state]
                best_current_action = max(current_table, key=lambda k: current_table[k])
                best_current_Q = current_table[best_current_action]
            previous_Q_Table[self.lastAction] = (1 - alpha) * previous_Q_Value + alpha * (reward + (gamma * best_current_Q) - previous_Q_Value)
            self.table[previous_state] = previous_Q_Table



            # Reset the variables for recording the previous state
            self.previous_state = None
            self.lastAction = None
            self.initial = True
            self.previous_score = 0

            # Decay the epsilon rate
            if self.epsilon > 0.2:
                new_epsilon = self.epsilon - float(0.8)/self.numTraining
                self.setEpsilon(new_epsilon)
        
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


