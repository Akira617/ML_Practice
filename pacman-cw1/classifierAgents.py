# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # You may wish to create your classifier here.
        #
        # *********************************************
        self.random_forest = self.train_random_forest()
        self.evaluation_RF()
        
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST
    


    

    
    # Inner Class for A decision tree
    # Tree Data Structure
    class DecisionTree:
        def __init__(self,train_data, train_target, attribute_list):
            self.root = self.Node(train_data, train_target, attribute_list)
        def inference(self, input_data):
            return self.root.inference(input_data)

        # Inner Class for A node in Decision Tree
        # If value of current attribute = 1, go left
        # If value of current attribute = 0, go right 
        class Node:
            def __init__(self,train_data, train_target, attribute_list):
                self.data = train_data
                self.target = train_target
                self.attributes = attribute_list
                self.left = None
                self.right = None
                self.best_attribute = -1 # -1 stand for this node is leaf Node
                self.train(train_data, train_target, attribute_list)
            
            # Calculate the gini index
            def calculate_gini(self,train_data, train_target, attribute_list):
                gini_index_list = []
                for attribute in attribute_list:
                    #As we only have 4 classes
                    T_counter = [0,0,0,0]
                    F_counter = [0,0,0,0]
                    #As all attributes are binary attributes (i.e. have value 0 or 1)
                    gini_T = 1
                    gini_F = 1
                    for index in range(len(train_data)):
                        class_belong = train_target[index]
                        if train_data[index][attribute] == 1:
                            T_counter[class_belong] += 1
                        else:
                            F_counter[class_belong] += 1
                    T_sum = sum(T_counter)
                    F_sum = sum(F_counter)

                    if T_sum != 0:
                        #Calculate the GINI for T
                        for number in T_counter:
                            gini_T -= (number/T_sum) ** 2
                    if F_sum != 0:
                        for number in F_counter:
                            gini_F -= (number/F_sum) ** 2
                    total = T_sum + F_sum
                    gini_index = (T_sum/total * gini_T) + (F_sum/total * gini_F)
                    gini_index_list.append(gini_index)
                return gini_index_list.index(min(gini_index_list))

            # The training algorithm for CART Decision Tree
            def train(self,train_data, train_target, attribute_list):
                # Check if all the data belongs to the same class
                if train_target[1:] == train_target[:-1]:
                    self.left = train_target[0]
                    self.right = train_target[0]
                    return
                
                # Check if the attribute set is empty
                if len(attribute_list) == 0:
                    # Set the leave node to the greatest common class
                    values, freq = np.unique(train_target, return_counts=True)
                    node_output = values[np.argmax(freq)]
                    self.left = node_output
                    self.right = node_output
                    return 

                else:
                    #Find the best attribute by calculate the gini_index of each aatributes
                    training_data = self.data
                    training_target = self.target
                    attributes = self.attributes
                    best_attribute_index = self.calculate_gini(training_data, training_target, attributes)
                    best_attribute = attributes[best_attribute_index]
                    self.best_attribute = best_attribute
                    # Extract the dataset for the sub-tree
                    true_samples = []
                    true_target = []
                    false_samples = []
                    false_target = []
                    for index in range(len(training_data)):
                        if training_data[index][best_attribute] == 1:
                            true_samples.append(training_data[index])
                            true_target.append(training_target[index])
                        else:
                            false_samples.append(training_data[index])
                            false_target.append(training_target[index])
                    
                    attributes.remove(best_attribute)
                    if len(true_samples) == 0:
                        values, freq = np.unique(train_target, return_counts=True)
                        node_output = values[np.argmax(freq)]
                        self.left = node_output
                    else:
                        leftNode = self.__class__(true_samples, true_target, attributes)
                        self.left = leftNode

                    if len(false_samples) == 0:
                        values, freq = np.unique(train_target, return_counts=True)
                        node_output = values[np.argmax(freq)]
                        self.right = node_output
                    else:
                        rightNode = self.__class__(false_samples, false_target, attributes)
                        self.right = rightNode
            
            def inference(self, input_data):
                # print(self.best_attribute)
                # If the left/right node is NOT a inference result, then do inference on the Node
                # Else return the inference result
                value = input_data[self.best_attribute]
                results = [0,1,2,3]
                if value == 1:
                    if self.left in results:
                        return self.left
                    else:
                        return self.left.inference(input_data)
                else:
                    if self.right in results:
                        return self.right
                    else:
                        return self.right.inference(input_data)   
        
            

    # Use bootstrapping to split the dataset
    def bootstrapping(self):
        num_of_sample = len(self.data)
        sample_data = []
        sample_target = []
        for i in range (num_of_sample):
            random_number = np.random.randint(num_of_sample, size = 1)[0]
            sample_data.append(self.data[random_number])
            sample_target.append(self.target[random_number])
        return sample_data, sample_target
    
    # Train the random forest
    # Return the list of decision trees in the forest
    def train_random_forest(self):
        tree_list = []
        total_attributes = len(self.data[0])
        attributes_list = []
        for index in range(total_attributes):
            attributes_list.append(index)
        # The recommend number of attributes for each weak Decision Tree is k=log2(d) 
        num_of_attribute_weak = int(round(np.log(total_attributes)/np.log(2)))
        #num_of_attribute_weak = 25

        # Number of tree in random forest
        num_of_tree = 128
        for i in range(num_of_tree):
            sample_data, sample_target = self.bootstrapping()
            sample_attribute = random.sample(attributes_list, num_of_attribute_weak)
            weak_tree = self.DecisionTree(sample_data, sample_target, sample_attribute)
            tree_list.append(weak_tree)
        return tree_list

    # Inference by the random forest
    # Plurality Voting
    # Input: feature of current state
    # Return: Best Action number of current state
    def inference_RF(self,state_array):
        results = []
        for tree in self.random_forest:
            result = tree.inference(state_array)
            results.append(result)
        values, freq = np.unique(results, return_counts=True)
        best = values[np.argmax(freq)]
        return best
    
    # Perform the evaluation on random forest
    # As the size of dataset is limited, it's a good idea to use all the data as we don't have overfitting in RF
    def evaluation_RF(self):
        test_dataset = self.data
        test_target = self.target
        size_of_data = len(test_dataset)
        hit = 0
        for index in range(size_of_data):
            current_data = test_dataset[index]
            current_target = test_target[index]
            current_result = self.inference_RF(current_data)
            if current_result == current_target:
                hit += 1
        accuracy = float(hit)/float(size_of_data)
        print("The Accuracy is "+str(accuracy))


    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)
        #Get the best number by inference
        best_number = self.inference_RF(features)
        best_action = self.convertNumberToMove(best_number)
        


        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(best_action, legal)

