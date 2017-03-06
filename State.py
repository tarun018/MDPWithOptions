import csv
from copy import deepcopy
import numpy as np
import cvxopt
import cvxopt.solvers
import cvxpy
import config
import random
import itertools

class State:

    def __init__(self, ind, location, time, dvals, dold, actions):
        self.index = ind
        self.location = location
        self.time = time
        self.dvals = dvals
        self.dold = dold
        self.possibleActions = actions
        self.transition = []
        self.reward = []

    def __repr__(self):
        return "Index: " + str(self.index) + " Location: " + str(self.location) + " Time: " + str(self.time) + " Dvals " + str(self.dvals) + " Dold: " + str(self.dold)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.location==other.location and self.time==other.time and self.dvals==other.dvals and self.dold==other.dold
        return False

class Action:
    def __init__(self, ind, name, gotox=None):
        self.index = ind
        self.gotox = gotox
        self.name = name

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + str(self.name) + " Goto: " + str(self.gotox)

class MDP:
    def __init__(self, T, nlocs, agent, collecTimes, transitTimes, alpha):
        self.T = T
        self.nlocs = nlocs
        self.agent = agent
        self.lst = list(itertools.product([0, 1], repeat=nlocs))
        self.states = []
        self.actions = []
        self.collectTimes = collecTimes
        self.transitTimes = transitTimes
        self.alpha = alpha
        self.terminal = State(0, -1, -1, [-1] * self.nlocs, -1, -1)
        self.states.append(self.terminal)
        self.initiateActions()
        self.initiateStates()
        self.waste()
        self.checkTransitionProbabilitySumTo1()
        self.writeTransitionsToFile()
        self.writeRewardsToFile()
        self.numberStates = len(self.states)
        self.numerActions = len(self.actions)

    def initiateActions(self):
        index = 0
        at = Action(index, "Collect")
        self.actions.append(at)
        index = index + 1
        # at1 = Action(index, "Dummy")
        # self.actions.append(at1)
        # index = index + 1
        for i in xrange(0, self.nlocs):
            at2 = Action(index, "Go to Site " + str(i), i)
            index = index + 1
            self.actions.append(at2)

    def initiateStates(self):
        index = 1
        for i in xrange(0, self.T+1):
            for j in xrange(0, self.nlocs):
                for k in self.lst:
                    for t in [0,1]:
                        st = State(index,j,i,k,t,self.actions)
                        self.states.append(st)
                        index = index + 1
        print self.states

    def transition(self, s, a, sd):
        if a.name == "Collect":
            return self.transitionCollect(s, sd)
        # elif a.name == "Dummy":
        #     return self.dummyTransition(s, sd)
        else:
            return self.transitionGoto(s, sd, a)

    def transitionCollect(self,s,sd):
        # print s
        # print "Collect"
        # print sd

        if s.time == -1 and sd == s:
            return 1

        # Terminal States
        if s.time == 0 and sd == self.terminal:
            return 1

        # Ensure that all d variables are same in s and sd other than d_l.
        sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs) if j != s.location])
        # print sameds

        # Location should be same
        if s.location != sd.location:
            return 0

        #Time should be subtracted
        if sd.time != s.time - self.collectTimes[s.location]:
            return 0

        # Time should not be negative after subtraction
        if sd.time < 0:
            return 0

        if sameds==False:
            return 0

        if sd.dold != s.dvals[s.location]:
            return 0

        if s.dold == 1 and s.dvals[s.location] == 0:
            return 0

        if s.dvals[s.location] == 0:
            return self.alpha if sd.dvals[sd.location] == 1 else (1-self.alpha)
        elif s.dvals[s.location] == 1:
            return 1 if sd.dvals[sd.location] == 1 else 0
        else:
            return 0

    def transitionGoto(self, s, sd, action):
        l = s.location
        ld = sd.location
        t = s.time
        td = sd.time

        dold = s.dold
        doldd = sd.dold
        dest = action.gotox
        sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs)])

        if s.time == -1 and sd == s:
            return 1

        if s.time == 0 and sd == self.terminal:
            return 1

        if ld != dest:
            return 0

        if td != t - self.transitTimes[l][ld]:
            return 0

        if td < 0:
            return 0

        if sameds == False:
            return 0

        if doldd != sd.dvals[ld]:
            return 0

        if dold == 1 and s.dvals[l] == 0:
            return 0

        if doldd == 1 and sd.dvals[ld] == 0:
            return 0

        return 1

    # def dummyTransition(self, s, sd):
    #
    #     sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs)])
    #     # print sameds
    #
    #     if s.time == 0 and s == sd:
    #         return 1
    #
    #     if s.location != sd.location:
    #         return 0
    #
    #     if sd.time != s.time - config.dummyTime:
    #         return 0
    #
    #     if sd.time < 0:
    #         return 0
    #
    #     if sameds==False:
    #         return 0
    #
    #     if sd.dold != s.dvals[s.location]:
    #         return 0
    #
    #     if s.dold == 1 and s.dvals[s.location] == 0:
    #         return 0
    #
    #     return 1

    def waste(self):
        removed = self.removeWasteStates()
        while removed != 0:
            removed = self.removeWasteStates()


    def removeWasteStates(self):
        wastestates = []
        for i in self.states:

            if i == self.terminal:
                continue

            if i.dold == 1 and i.dvals[i.location] == 0:
                wastestates.append(i)
                continue


            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == config.T and sameds==True and i.dold==0:
                continue

            flag = 0
            for j in self.states:
                for k in self.actions:
                    if self.transition(j, k, i) != 0:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 0:
                wastestates.append(i)

        # print len(wastestates)
        # print wastestates
        for x in wastestates:
            self.states.remove(x)
        # print len(self.states)
        # print

        return len(wastestates)

    def checkTransitionProbabilitySumTo1(self):
        fp = open('tds', 'w')
        for k in self.actions:
            for i in self.states:
                sum = 0
                for j in self.states:
                    fp.write(str(i))
                    fp.write("\n")
                    fp.write(str(j))
                    fp.write("\n")
                    tran = self.transition(i,k,j)
                    sum += tran
                    fp.write(str(tran))
                    fp.write("\n")
                if (sum != 1):
                    print "k: " + str(k) + " i: " + str(i) + " Sum: " + str(sum)
        fp.close()

    def rewardFunction(self, s, a):
        if a.name == "Collect":
            if s.dold == 0 and s.dvals[s.location] == 1:
                return config.rewardCollection
            else:
                return 0
        # elif a.name == "Dummy":
        #     if s.dold == 0 and s.dvals[s.location] == 1:
        #         return config.rewardCollection
        #     else:
        #         return 0
        else:
            return 0

    def writeTransitionsToFile(self):
        tran = open("DomainTransitionData"+str(self.agent)+".txt", 'w')
        for i in self.actions:
            for j in self.states:
                for k in self.states:
                    tran.write(str(self.transition(j,i,k)))
                    if k != self.states[len(self.states)-1]:
                        tran.write(",")
                tran.write("\n")
            tran.write("\n")
        tran.close()

    def writeRewardsToFile(self):
        rew = open("DomainRewardData"+str(self.agent)+".txt",'w')
        for i in self.states:
            for j in self.actions:
                rew.write(str(self.rewardFunction(i,j)))
                if j != self.actions[len(self.actions)-1]:
                    rew.write(",")
            rew.write("\n")
        rew.close()

class Driver:
    a = MDP(config.T, config.nloc, 1, config.collectTimes, config.transitTimes, config.alpha)
    print a.numberStates
    print a.numerActions
    # for i in a.states:
    #     for j in a.actions:
    #         print str(i) + " " + str(j) + "            " + str(a.rewardFunction(i,j))

    # Verification of incoming probabilities of a state
    # print a.states[65]
    # for i in a.states:
    #     for j in a.actions:
    #         if a.transition(i,j,a.states[65]) != 0:
    #             print i
    #             print j
    #             print

    # Verification of P summing to 1



