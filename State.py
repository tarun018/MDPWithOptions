import csv
from copy import deepcopy
from decimal import Decimal
import numpy as np
import cvxopt
import cvxopt.solvers
from cvxopt import matrix, log, spdiag
import cvxpy
class State:

    def __init__(self, ind, name, actions):
        self.index = ind
        self.name = name
        self.possibleActions = actions
        self.transition = []
        self.reward = []
        self.terminating = False
        self.utility = 0

    def __repr__(self):
        return "Name: " + self.name

    def modifyActions(self, actions):
        self.possibleActions = actions

    def setTransition(self, tran):
        self.transition = tran

    def getTransition(self):
        return self.transition

    def setReward(self, reward):
        self.reward = reward

    def getReward(self):
        return self.reward

    def getIndex(self):
        return self.index

    def getPossibleActions(self):
        return self.possibleActions

    def setPossibleActions(self, act):
        self.possibleActions = act

    def isTerminating(self):
        return self.terminating

    def setTerminating(self, term):
        self.terminating = term
        if term == True:
            self.possibleActions = []

    def setUtility(self, util):
        self.utility = util

    def getUtility(self):
        return self.utility

class Action:

    def __init__(self, ind, name):
        self.index = ind
        self.name = name

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + self.name

    def getIndex(self):
        return self.index


class MDP:

    def __init__(self, numberOfStates, numberOfActions):
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.numberOfOptions = 0
        self.states = []
        self.actions = []
        self.options = []

    # Define Action
    def initializeActions(self):
        for i in xrange(0, self.numberOfActions):
            a = Action(i, str("a" + str(i)))
            self.actions.append(a)

    # Define States
    def initializeStates(self):
        for i in xrange(0, self.numberOfStates):
            x = State(i, str("s" + str(i)), self.actions[0:self.numberOfActions-1])
            self.states.append(x)
        self.states[3].setTerminating(True)
        self.states[3].setUtility(1)
        self.states[3].setPossibleActions([self.actions[self.numberOfActions-1]])
        self.states[7].setTerminating(True)
        self.states[7].setUtility(-1)
        self.states[7].setPossibleActions([self.actions[self.numberOfActions-1]])

    # Leave one line space after each transition table for each action in the data file.
    def autoTransitionFunction(self, gamma=1):
        for s in self.states:
            s.setTransition([])
        stateIndex = 0
        actionIndex = 0
        with open('transitionData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 0:
                    stateIndex = 0
                    actionIndex = actionIndex + 1
                    continue
                for sp in xrange(0, self.numberOfStates):
                    triple = (actionIndex, sp, float(row[sp])*gamma)
                    self.states[stateIndex].getTransition().append(triple)
                stateIndex += 1

    # RewardFunctions For Actions
    def autoRewardFunction(self):

        tosend = []
        # if readFromFile == False:
        #     for x in self.states:
        #         if x.isTerminating():
        #             if x.getIndex() == 5:
        #                 triple = (a.getIndex(), -1)
        #             elif x.getIndex() == 8:
        #                 triple = (a.getIndex(), 1)
        #             tosend.append(triple)
        #         else:
        #             for a in self.actions:
        #                 triple = (a.getIndex(), -0.04)
        #                 tosend.append(triple)
        #         x.setReward(tosend)
        # else:
        stateIndex = 0
        with open('rewardData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row)==0:
                    continue
                for ap in xrange(0, self.numberOfActions):
                    triple = (ap, float(row[ap]))
                    tosend.append(triple)
                self.states[stateIndex].setReward(tosend)
                tosend = []
                stateIndex += 1

    # Leave one line space after each option in the data file.
    def setOptions(self, readFromFile=False):
        if readFromFile == False:
            return
        else:
            actionIndex = 0
            counter = 0
            tosend = []
            with open('OptionsData', 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    counter += 1

                    if len(row) == 0:

                        o.setPolicy(tosend)
                        self.numberOfOptions += 1

                        counter = 0
                        actionIndex = 0
                        continue

                    if counter == 1:
                        o = Option(self.numberOfOptions)
                        self.options.append(o)
                        #Initiation
                        initset = []
                        for x in row:
                            initset.append(int(x))
                        o.setInitiationSet(initset)

                    elif counter == 2:
                        #Beta
                        betaval = []
                        for x in row:
                            betaval.append(float(x))
                        o.setBeta(betaval)

                    elif counter >= 3:
                        #Policy
                        stateIndex = 0
                        for st in self.states:
                            triple = (st.getIndex(), actionIndex, float(row[st.getIndex()]))
                            tosend.append(triple)
                            stateIndex += 1
                        actionIndex += 1


    # def calculateRewardForOption(self, state, option, gamma, delta):
    #
    #     # print gamma
    #     sums = 0
    #     # print "State: " + str(state)
    #     # print "Option: " + str(option)
    #     if state not in self.options[option].getInitiationSet():
    #         # print "Not in Initiation Set."
    #         return  0
    #
    #     actionsavail = self.states[state].getPossibleActions()
    #
    #     if len(actionsavail) == 0:
    #         return 0
    #
    #     for act in actionsavail:
    #
    #         # print
    #         # print "State: " + str(state)
    #         # print "Action: " + str(act.getIndex())
    #         sumForAction = 0
    #         probOfAction = 0
    #         immReward = 0
    #         possibleStates = []
    #         actionSet = self.options[option].getPolicy()
    #
    #         for x in actionSet:
    #             if x[0] == state and x[1] == act.getIndex():
    #                 probOfAction = x[2]
    #                 break
    #
    #         # print "Prob Of Action: " + str(probOfAction)
    #
    #         # if probOfAction == 0:
    #         #     avail = self.states[state].getPossibleActions()
    #         #     avail = filter(lambda a: a != act.getIndex(), avail)
    #         #     self.states[state].setPossibleActions(avail)
    #         #     continue
    #
    #         rewards = self.states[state].getReward()
    #         for x in rewards:
    #             if x[0] == act.getIndex():
    #                 immReward = x[1]
    #                 break
    #
    #         # print "Imm Reward: " + str(immReward)
    #
    #         if probOfAction != 0 and gamma > delta:
    #
    #             transitions = self.states[state].getTransition()
    #
    #             # print "Transition Function for " + str(state) + " is: " + str(transitions)
    #             for x in transitions:
    #                 if x[0] == act.getIndex() and x[2] != 0:
    #                     possibleStates.append((x[1], x[2]))
    #
    #             # print "Possible States from " + str(state) + " are: " + str(possibleStates)
    #
    #             for x in possibleStates:
    #
    #                 product = 0
    #                 if len(possibleStates) == 0:
    #                     break
    #
    #                 # print "Possible State: " + str(x)
    #
    #
    #                 sdash = self.states[x[0]]
    #                 prob = float(x[1])
    #
    #                 # print "Sdash: " +str(sdash.getIndex())
    #                 # print "prob: " + str(prob)
    #
    #                 beta_sdash = float(self.options[option].getBeta()[x[0]])
    #
    #                 # print "Beta_sdash: " + str(1-beta_sdash)
    #
    #                 if prob != 0 and (1-beta_sdash) != 0:
    #                     product = prob * (1 - float(beta_sdash)) * self.calculateRewardForOption(sdash.getIndex(), option, gamma, delta)
    #
    #                 # print "Product for this state: " + str(product)
    #                 sumForAction += product
    #
    #
    #             # print "Sum for Action: " + str(sumForAction)
    #             sumForAction *= gamma
    #
    #             sumForAction += immReward
    #
    #             sumForAction *= probOfAction
    #
    #             # print "Final Sum For Action: " + str(sumForAction)
    #             # print
    #
    #         else:
    #             # print "Prob of action is 0."
    #             sumForAction = 0
    #
    #         # print immReward
    #         # print probOfAction
    #         # if sumForAction <= 0:
    #         #     avail = self.states[state].getPossibleActions()
    #         #     avail.remove(act.getIndex())
    #         #     self.states[state].setPossibleActions(avail)
    #
    #         sums += sumForAction
    #         # print "Sums: " + str(sums)
    #         # print
    #
    #     return sums
    #
    # def calculateTransitionForOption(self, state, option, statedash, gamma, delta):
    #
    #     sums = 0
    #     # print "State: " + str(state)
    #     # print "Option: " + str(option)
    #     # print "Statedash: " + str(statedash)
    #     if state not in self.options[option].getInitiationSet():
    #         # print "Not in Initiation Set."
    #         return 0
    #
    #     actionsavail = self.states[state].getPossibleActions()
    #
    #     if len(actionsavail) == 0:
    #         return 0
    #
    #     for act in actionsavail:
    #
    #         # print
    #         # print "State: " + str(state)
    #         # print "Action: " + str(act.getIndex())
    #         sumForAction = 0
    #         probOfAction = 0
    #         possibleStates = []
    #         actionSet = self.options[option].getPolicy()
    #
    #         for x in actionSet:
    #             if x[0] == state and x[1] == act.getIndex():
    #                 probOfAction = x[2]
    #                 break
    #
    #         # print "Prob Of Action: " + str(probOfAction)
    #
    #         # if probOfAction == 0:
    #         #     avail = self.states[state].getPossibleActions()
    #         #     avail = filter(lambda a: a != act.getIndex(), avail)
    #         #     self.states[state].setPossibleActions(avail)
    #         #     continue
    #
    #         if probOfAction != 0 and gamma > delta:
    #
    #             transitions = self.states[state].getTransition()
    #
    #             # print "Transition Function for " + str(state) + " is: " + str(transitions)
    #             for x in transitions:
    #                 if x[0] == act.getIndex() and x[2] != 0:
    #                     possibleStates.append((x[1], x[2]))
    #
    #             # print "Possible States from " + str(state) + " are: " + str(possibleStates)
    #
    #             for x in possibleStates:
    #
    #                 product = 0
    #                 if len(possibleStates) == 0:
    #                     break
    #
    #                 # print "Possible State: " + str(x)
    #
    #
    #                 sdash = self.states[x[0]]
    #                 prob = float(x[1])
    #
    #                 # print "Sdash: " +str(sdash.getIndex())
    #                 # print "prob: " + str(prob)
    #
    #                 beta_sdash = float(self.options[option].getBeta()[x[0]])
    #
    #                 # print "1-Beta_sdash: " + str(1-beta_sdash)
    #
    #                 if prob != 0 and float(1 - beta_sdash) != 0:
    #                     product += ((1 - float(beta_sdash)) * self.calculateTransitionForOption(sdash.getIndex(),
    #                                                                                              option, statedash, gamma, delta))
    #
    #                 if prob != 0 and float(beta_sdash) != 0:
    #                     product += (beta_sdash * self.delta(float(sdash.getIndex()), float(statedash)))
    #
    #                 product *= prob
    #
    #                 # print "Product for this state: " + str(product)
    #                 sumForAction += product
    #
    #             # print "Sum for Action: " + str(sumForAction)
    #
    #             sumForAction *= gamma
    #             sumForAction *= probOfAction
    #
    #             # print "Final Sum For Action: " + str(sumForAction)
    #             # print
    #
    #         else:
    #             # print "Prob of action is 0."
    #             sumForAction = 0
    #
    #         # print immReward
    #         # print probOfAction
    #         # if sumForAction <= 0:
    #         #     avail = self.states[state].getPossibleActions()
    #         #     avail.remove(act.getIndex())
    #         #     self.states[state].setPossibleActions(avail)
    #
    #         sums += sumForAction
    #         # print "Sums: " + str(sums)
    #         # print
    #
    #     return sums

    def modelActionAsOptions(self, action):
        global optionIndex
        initset = []
        beta = []
        policy = []

        for s in self.states:
            beta.append(float(1))
            if action in s.getPossibleActions():
                initset.append(s.getIndex())
                policy.append((s.getIndex(), action.getIndex(), float(1)))

        o = Option(self.numberOfOptions)
        self.options.append(o)
        o.setInitiationSet(initset)
        o.setBeta(beta)
        o.setPolicy(policy)
        self.numberOfOptions += 1

    def iterativeRewardCalculation(self, option, delta):

        values = [0]*len(self.states)
        # values[3] = 1
        # values[7] = -1
        while(True):

            # print "Values : " + str(values)
            prevValues = deepcopy(values)
            dummy = [0]*len(self.states)


            # print "Prev Values: " + str(prevValues)
            # print "Dummy Values: " + str(dummy)
            # print

            max_difference = 0
            for s in self.states:

                # print "Considering state: " + str(s.getIndex())
                # print
                # if s.getIndex() not in self.options[option].getInitiationSet():
                #     dummy[s.getIndex()] = 0.0
                #     continue

                sums = 0
                s = s.getIndex()
                actionsavail = self.states[s].getPossibleActions()

                for actions in actionsavail:

                    # print "Considering state: " + str(s)
                    # print "Considering Action: " + str(actions.getIndex())
                    possibleStates = []
                    valforaction = 0
                    immReward = 0
                    probOfAction = 0
                    actionSet = self.options[option].getPolicy()

                    for x in actionSet:
                        if x[0] == s and x[1] == actions.getIndex():
                            probOfAction = x[2]
                            break


                    rewards = self.states[s].getReward()
                    for x in rewards:
                        if x[0] == actions.getIndex():
                            immReward = x[1]
                            break
                    # print "Probofaction: " + str(probOfAction)
                    # print "immReward: " +str(immReward)
                    transitions = self.states[s].getTransition()

                    # print "Transition Function for " + str(s) + " is: " + str(transitions)
                    for x in transitions:
                        if x[0] == actions.getIndex() and x[2] != 0:
                            possibleStates.append((x[1], x[2]))

                    # print "Possible States: " + str(possibleStates)


                    for x in possibleStates:

                        product = 0
                        prob = float(x[1])
                        sdash = self.states[x[0]]

                        # print "sdash: " + str(sdash.getIndex())
                        beta_sdash = float(self.options[option].getBeta()[x[0]])

                        # print "prob " + str(prob)
                        product = prob * (1-float(beta_sdash)) * prevValues[sdash.getIndex()]

                        # print "product: " + str(product)
                        valforaction += product

                    # print "Value For Action: " + str(valforaction)
                    valforaction *= 1

                    valforaction += immReward
                    valforaction *= probOfAction


                    # print "Value For Action: " + str(valforaction)

                    sums += valforaction
                    sums = round(sums, 15)

                    # print "Sums: " + str(sums)

                # print "Overall Sums: " + str(sums)
                # print "Not Updated Dummy: " + str(dummy)
                dummy[s] = sums

                # print "Updated Dummy: "+ str(dummy)
                difference = abs(dummy[s] - values[s])

                if difference > max_difference:
                    max_difference = difference

                # print "max_diff: " + str(max_difference)

            if max_difference > delta:
                values = deepcopy(dummy)
            else:
                break

        return values

    def iterativeTransitionCalculation(self, option, statedash, delta):

        values = [0]*len(self.states)
        while (True):

            # print "Values : " + str(values)
            prevValues = deepcopy(values)
            dummy = [0]*len(self.states)

            # print "Prev Values: " + str(prevValues)
            # print "Dummy Values: " + str(dummy)
            # print

            max_difference = 0
            for s in self.states:

                # print "Considering state: " + str(s.getIndex())
                # print

                s = s.getIndex()
                sums = 0

                actionsavail = self.states[s].getPossibleActions()

                for actions in actionsavail:

                    # print "Considering state: " + str(s)
                    # print "Considering Action: " + str(actions.getIndex())
                    possibleStates = []
                    valforaction = 0
                    probOfAction = 0
                    actionSet = self.options[option].getPolicy()

                    for x in actionSet:
                        if x[0] == s and x[1] == actions.getIndex():
                            probOfAction = x[2]
                            break

                    # print "Probofaction: " + str(probOfAction)

                    transitions = self.states[s].getTransition()

                    # print "Transition Function for " + str(s) + " is: " + str(transitions)
                    for x in transitions:
                        if x[0] == actions.getIndex() and x[2] != 0:
                            possibleStates.append((x[1], x[2]))

                    # print "Possible States: " + str(possibleStates)


                    for x in possibleStates:
                        product = 0
                        prob = float(x[1])
                        sdash = self.states[x[0]]

                        # print "sdash: " + str(sdash.getIndex())
                        beta_sdash = float(self.options[option].getBeta()[x[0]])


                        product =  (1-float(beta_sdash)) * prevValues[sdash.getIndex()]

                        product += (float(beta_sdash) * self.delta(sdash.getIndex(), statedash))

                        product *= prob
                        # print "product: " + str(product)
                        valforaction += product

                    # print "Value For Action: " + str(valforaction)
                    # if option == 0:
                    valforaction *= 1
                    valforaction *= probOfAction

                    # print "Value For Action: " + str(valforaction)

                    sums += valforaction
                    sums = round(sums, 15)

                    # print "Sums: " + str(sums)

                # print "Overall Sums: " + str(sums)
                # print "Not Updated Dummy: " + str(dummy)
                dummy[s] = sums

                # print "Updated Dummy: "+ str(dummy)
                difference = abs(dummy[s] - values[s])

                if difference > max_difference:
                    max_difference = difference

                    # print "max_diff: " + str(max_difference)

            if max_difference > delta:
                values = deepcopy(dummy)
            else:
                break

        return values

    def actionsVI(self, delta):

        iter = 0
        values = [x.getUtility() for x in self.states]
        bestactions = [None]*len(self.states)

        while(True):
            iter += 1
            prevvalues = deepcopy(values)
            dummy = [0] * len(self.states)
            dummyActions = [None]*len(self.states)

            max_difference = 0
            for s in self.states:

                max_value = float('-INF')
                for a in s.getPossibleActions():

                    sum = 0
                    reward = [x[1] for x in self.states[s.getIndex()].getReward() if x[0] == a.getIndex()][0]
                    possiblestates = [(x[1], x[2]) for x in self.states[s.getIndex()].getTransition() if x[0] == a.getIndex() and x[2] != 0]
                    for x in possiblestates:

                        prob = float(x[1])
                        sdash = self.states[x[0]]
                        sum += (prob * prevvalues[x[0]])

                    sum *= 1
                    sum += float(reward)

                    if sum >= max_value:
                        max_value = sum
                        dummyActions[s.getIndex()] = a

                dummy[s.getIndex()] = max_value

                difference = abs(dummy[s.getIndex()] - values[s.getIndex()])
                if difference > max_difference:
                    max_difference = difference

            if max_difference > delta:
                print str(iter) + " " + str(values)
                values = deepcopy(dummy)
                bestactions = deepcopy(dummyActions)
            else:
                # print iter
                break

        return values, bestactions, iter

    def rewardOption(self, state, option, delta):
        x = self.iterativeRewardCalculation(option, delta)
        return x[state]

    def transitionOption(self, state, option, statedash):
        x = self.iterativeTransitionCalculation(option, statedash, 0.0001)
        return x[state]

    def optionsVI(self, delta):

        iterations = 0
        values = [0]*len(self.states)
        values[3] = 1
        values[7] = -1
        optionsbest = [None]*len(self.states)

        rso = []
        for o in self.options:
            x = self.iterativeRewardCalculation(o.getIndex(), delta)
            triple = (o.getIndex() , x)
            rso.append(triple)

        psxo = []
        for o in self.options:
            for s in self.states:
                x = self.iterativeTransitionCalculation(o.getIndex(), s.getIndex(), delta)
                triple = (o.getIndex(), s.getIndex(), x)
                psxo.append(triple)

        while(True):

            iterations += 1
            # print values
            prevvalues = deepcopy(values)
            dummy = [0] * len(self.states)
            dummyoptionsbest = [None]*len(self.states)

            max_difference = 0
            for s in self.states:

                # print
                # print "State: " + str(s.getIndex())

                max_value = float('-INF')
                optionsavailable = [o for o in self.options if s.getIndex() in o.getInitiationSet()]
                for o in optionsavailable:

                    # print
                    # print
                    # print "Avaialable Option: " + str(o.getIndex())
                    sum = 0
                    reward = [r[1][s.getIndex()] for r in rso if r[0] == o.getIndex()][0]
                    # if o.getIndex() == 0:
                    possible = [(self.states[p[1]], p[2][s.getIndex()]) for p in psxo if p[0] == o.getIndex() and p[2][s.getIndex()] != 0]
                    # else:
                    #     possible = [(self.states[x[1]], x[2]) for x in s.getTransition() if x[0] == o.getIndex() - 1 and x[2] != 0]
                    # for x in possible:
                    #     print "Possibilities: " + str(x[0].getIndex()) + " " + str(x[1])
                    # print "Reward: " + str(reward)

                    for x in possible:

                        prob = float(x[1])
                        sd = x[0]
                        # print "Prob: " + str(prob) + " Prev Value: " + str(prevvalues[sd.getIndex()])
                        sum += (prob * prevvalues[sd.getIndex()])

                    sum *= 1
                    sum += reward
                    # print "Final Value for This Option: " + str(sum)

                    if sum > max_value:
                        max_value = sum
                        dummyoptionsbest[s.getIndex()] = o.getIndex()


                dummy[s.getIndex()] = max_value
                difference = abs(dummy[s.getIndex()] - values[s.getIndex()])
                if difference > max_difference:
                    max_difference = difference

            # print "Values: " + str(values)
            # print "Dummy Values: " + str(dummy)

            if max_difference > delta:
                values = deepcopy(dummy)
                # print values
                optionsbest = deepcopy(dummyoptionsbest)
            else:
                # print iterations
                break
        return values, optionsbest, iterations

    def generateLPAc(self, gamma):
        decisionvar = []
        for x in self.states:
            triple = []
            for y in self.states:
                triplet = []
                for a in y.possibleActions:
                    if x.getIndex() == y.getIndex():
                        triplet.append(float(1))
                    else:
                        triplet.append(float(0))
                triple.append(triplet)
            decisionvar.append(triple)

        for x in self.states:
            incoming = []
            for s in self.states:
                for t in s.transition:
                    if t[1]==x.getIndex() and t[2]!=0:
                        incoming.append((s, t[0], t[2]))

            for h in incoming:
                decisionvar[x.getIndex()][h[0].getIndex()][h[1]] -= gamma*float(h[2])

        # for x in decisionvar:
        #     for y in x:
        #         for z in y:
        #             print str(z) + ",",
        #         print "",
        #     print
        #
        # print
        # print
        A_mat = []
        for x in decisionvar:
            lit = []
            for t in x:
                lit.extend(t)
            A_mat.append(lit)

        newA = A_mat

        # for x in A_mat:
        #     print x

        R_mat = []
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0]==y.getIndex():
                        R_mat.append(r[1])
        #print R_mat

        newR = []
        R_min = -1
        R_max = 1
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0]==y.getIndex():
                        newR.append((r[1]-R_min)/(R_max-R_min))

        #print decisionvar
        return A_mat, R_mat, newR, newA

    def lpsolve(self, gamma):
        a, R, newR, newA = self.generateLPAc(gamma)
        R_mat = -1 * np.array(R)[np.newaxis].T
        A_mat = np.array(a)
        alpha = np.zeros((np.shape(a)[0], 1))
        G = -1*np.identity(np.shape(R)[0])
        h = np.zeros((np.shape(R)[0], 1))
        alpha[8][0] = 1.0

        A = cvxopt.matrix(A_mat)
        b = cvxopt.matrix(alpha)
        c = cvxopt.matrix(R_mat)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        sol = cvxopt.solvers.lp(c, G, h, A, b)
        x_mat = np.array(sol['x'])
        print -1*np.dot(np.transpose(x_mat), R_mat)

    # def lpsolveEM(self, gamma):
    #     a, R, newR, newA = self.generateLPAc(gamma)
    #     R_mat = -1 * np.array(newR)[np.newaxis].T
    #     A_mat = np.array(newA)
    #     alpha = np.zeros((np.shape(a)[0]+1, 1))
    #     G = -1*np.identity(np.shape(R)[0])
    #     h = np.zeros((np.shape(R)[0], 1))
    #     alpha[8][0] = 1.0
    #     alpha[12][0] = 1/(1-gamma)
    #     x_mat = np.zeros((np.shape(R_mat)[0], 1))
    #     while(True):
    #         A = cvxopt.matrix(A_mat)
    #         b = cvxopt.matrix(alpha)
    #         G = cvxopt.matrix(G)
    #         h = cvxopt.matrix(h)
    #
    #         def F(x=None, z=None):
    #             if x is None: return 0, matrix(1.0, (42, 1))
    #             if min(x) <= 0.0: return None
    #             f = -sum(R_mat * x_mat * log(x))
    #             Df = -(x ** -1).T
    #             if z is None: return f, Df
    #             H = spdiag(z[0] * x ** -2)
    #             return f, Df, H
    #
    #         sol = cvxopt.solvers.cp(F=F, G=G, h=h, A=A, b=b)
    #         x_mat = np.array(sol['x'])
    #         print -1*np.dot(np.transpose(x_mat), R)

    def solveLP(self, gamma):
        A, R, newR, newA = self.generateLPAc(gamma)
        #print len(R)
        R_mat = np.array(R)[np.newaxis].T
        A_mat = np.array(A)
        alpha = np.zeros((np.shape(A)[0], 1))
        alpha[8][0] = 1.0

        x = cvxpy.Variable(42, 1)
        obj = cvxpy.Maximize(np.transpose(R_mat)*x)
        constraints = [A_mat*x == alpha, x >= 0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()
        #print "status:", prob.status
        print "LPsolver: optimal value", prob.value
        #print "optimal var", x.value

    def solveNewLP(self, gamma, x_mat_val=None, touse=0):
        print cvxpy.installed_solvers()
        A, R, newR, newA = self.generateLPAc(gamma)
        #print newA
        R_mat = np.array(newR)[np.newaxis].T
        A_mat = np.array(A)
        alpha = np.zeros((np.shape(A)[0], 1))
        alpha[8][0] = 1.0

        if(touse==0):
            x_mat = np.random.uniform(0,1 , size=(np.shape(A_mat)[1], 1))
        else:
            x_mat = x_mat_val

        #print x_mat
        rdiag = np.diag(R_mat[:,0])
        rdiagx = rdiag.dot(x_mat)

        xstar = cvxpy.Variable(42, 1)
        obj = cvxpy.Maximize(np.transpose(rdiagx)*cvxpy.log(xstar))
        constraints = [A_mat*xstar == alpha, xstar>0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False, max_iters=100)
        print np.transpose(R)*xstar.value
        #print cvxpy.sum_entries(xstar.value).value
        #prob.solve()

        return xstar.value, prob.value

    def generateLP(self, delta):
        decisionvar = []

        for x in self.states:

            triple = []
            for y in self.states:

                triplet = []
                for o in self.options:

                    if x.getIndex() == y.getIndex() and y.getIndex() in o.getInitiationSet():
                        triplet.append(float(1))
                    elif y.getIndex() in o.getInitiationSet():
                        triplet.append(float(0))

                triple.append(triplet)

            decisionvar.append(triple)


        rso = []
        for o in self.options:
            x = self.iterativeRewardCalculation(o.getIndex(), delta)
            triple = (o.getIndex() , x)
            rso.append(triple)

        psxo = []
        for o in self.options:
            for s in self.states:
                x = self.iterativeTransitionCalculation(o.getIndex(), s.getIndex(), delta)
                triple = (o.getIndex(), s.getIndex(), x)
                psxo.append(triple)

        for x in self.states:

            incoming = [ (t[0], t[2]) for t in psxo if t[1] == x.getIndex()]
            for h in incoming:

                opt = h[0]
                nonzeros = [ (l, h[1][l]) for l in xrange(0, len(h[1])) if h[1][l] != 0 ]
                for n in nonzeros:

                    decisionvar[x.getIndex()][n[0]][opt] -= (float(n[1]))

        for h in xrange(0,self.numberOfStates):
            for g in xrange(0, self.numberOfOptions):
                if h in self.options[g].getInitiationSet():
                    print "r" + str(h) + "(" + str(g) + ") ,",

        print
        print

        for h in self.states:
            for x in self.options:
                if h.getIndex() in x.getInitiationSet():
                    re = self.rewardOption(h.getIndex(), x.getIndex(), delta)
                    print str(re) + " ,",

        print
        print

        for h in xrange(0,self.numberOfStates):
            for g in xrange(0, self.numberOfOptions):
                if h in self.options[g].getInitiationSet():
                    print "x" + str(h) + "(" + str(g) + ") ,",

        print
        print

        ## have to be changed.
        for x in decisionvar:
            for y in x:
                for z in y:
                    print str(z) +",",
                print "",
            print

        print
        print

    def delta(self, state, statedash):
        if state == statedash:
            return 1
        else:
            return 0

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getOptions(self):
        return self.options


class Option:

    def __init__(self, ind):
        self.index = ind
        self.initiation = []
        self.beta = []
        self.policy = []

    def setInitiationSet(self, init):
        self.initiation = init

    def setBeta(self, beta):
        self.beta = beta

    def setPolicy(self, pol):
        self.policy = pol

    def getInitiationSet(self):
        return self.initiation

    def getIndex(self):
        return self.index

    def getPolicy(self):
        return self.policy

    def getBeta(self):
        return self.beta

    def __str__(self):
        print "Index: " + str(self.index) + " Initiation: " + str(self.initiation)
        print "Beta: " + str(self.beta) + " Policy: " + str(self.policy)

class Driver:
    a = MDP(12,5)
    a.initializeActions()
    a.initializeStates()
    a.autoTransitionFunction()
    a.autoRewardFunction()
    gamma = 1

    #a.generateLPAc(gamma)
    a.solveLP(gamma)
    # xv, v = a.solveNewLP(gamma)
    # for i in xrange(50):
    #     xvv, vv = a.solveNewLP(gamma, x_mat_val=xv, touse=1)
    #     xv = xvv

    # print xv1, v1
    # a.lpsolve(0.95)
    #a.setOptions(readFromFile=True)
    #a.generateLPAc()
    # for x in a.getActions():
    #     a.modelActionAsOptions(x)

    # for x in a.getStates():
    #     print x.transition

    # o = a.getOptions()
    # for x in o:
    #     for y in a.getStates():
    #         print a.iterativeTransitionCalculation(x.getIndex(),y.getIndex(),0.001)
    #     print
    #     print x.getPolicy()
    #     print x.getBeta()
    #     print x.getInitiationSet()
    # print a.iterativeRewardCalculation(0, 0.001)
    # print a.iterativeRewardCalculation(1, 0.001)
    # print a.iterativeRewardCalculation(2, 0.001)
    # print a.iterativeRewardCalculation(3, 0.001)
    # print a.iterativeRewardCalculation(4, 0.001)
    # print a.iterativeRewardCalculation(5, 0.001)


    # for x in xrange(0,12):
    #     print a.calculateRewardForOption(x, 0, absGamma, 0.001)
    # print a.iterativeTransitionCalculation(0, 1, absGamma, 0.0001)
    # s = a.getStates()
    # print s[4].getTransition()
    # print s[4].getReward()
    # val, act = a.actionsVI(absGamma, 0.0001)
    # print val
    # for x in act:
    #     if x is None:
    #         print None,
    #     else:
    #         print x.getIndex(),
    #
    # print
    # print
    # # #
    # v,o = a.optionsVI(absGamma, 0.0001)
    # print v
    # print o
    # gammas = [1.00, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    delta = 0.00001
    gammas = [0.99999]
    #print('{0:15} {1:15} {2:25} {3:15} {4:15}'.format('Gamma', 'Without Options', 'Iterations', 'With Options', 'Iterations'))
    for x in gammas:
        a.autoTransitionFunction(x)
        #a.lpsolve(x)
        #a.solveLP(x)
        #a.autoTransitionFunction(x)

        z, y, it = a.actionsVI(delta)
        #
        # t, u, iter = a.optionsVI(delta)
        # # print t
        # # print u
        # if u[8] == 0:
        #     strih = 'Selecting Option.'
        # else:
        #     strih = 'Selecting Action.'
        # print('{0:10f} {1:15f} {2:15d} {3:15f} {4:15d} {5:15}'.format(x, z[8], it, t[8], iter, strih))
        print "VI Solver: ", z[8], y, it
    # for x in gammas:
    #     a.autoTransitionFunction(x)
    #     val, act = a.actionsVI(delta)
    #     print
    #     print val
    #     for y in act:
    #         if y is None:
    #             print None,
    #         else:
    #             print y.getIndex(),
    #
    #     print
    #     print
    #
    #     v,o = a.optionsVI(delta)
    #     print v
    #     print o
    #
    # for x in gammas:
    #     print "Gamma: " + str(x)
    #     a.autoTransitionFunction(x)
    #     a.generateLP(delta)