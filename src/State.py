import csv
import signal
import numpy as np
#import cvxpy
import config
import itertools
import copy_reg
import types
from copy import deepcopy
#import pyipopt
#import algopy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jnius
from jnius import autoclass
import time
import multiprocessing
import pickle
import math
import multiprocessing.pool
import traceback


#from pathos.multiprocessing import ProcessingPool as Pool

class State:

    def __init__(self, ind, location, actLocation, time, dvals, dold, actions):
        self.index = ind
        self.location = location
        self.actualLocation = actLocation
        self.time = time
        self.dvals = dvals
        self.dold = dold
        self.possibleActions = actions
        self.transition = []
        self.reward = []

    def setTransition(self, tran):
        self.transition = tran

    def setReward(self, rew):
        self.reward = rew

    def __repr__(self):
        return "Index: " + str(self.index) + " Location: " + str(self.location) + " Actual: " + str(self.actualLocation) + " Time: " + str(self.time) + " Dvals " + str(self.dvals) + " Dold: " + str(self.dold)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.location==other.location and self.actualLocation == other.actualLocation and self.time==other.time and self.dvals==other.dvals and self.dold==other.dold
        return False

class Action:
    def __init__(self, ind, name, gotox=None):
        self.index = ind
        self.gotox = gotox
        self.name = name

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + str(self.name) + " Goto: " + str(self.gotox)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.gotox==other.gotox and self.name==other.name
        return False

class MDP:
    def __init__(self, T, locs, agent, collecTimes, transitTimes, alpha, flag):
        self.manager = multiprocessing.Manager()
        self.T = T
        self.locs = locs
        self.nlocs = len(locs)
        self.agent = agent
        self.lst = list(itertools.product([0, 1], repeat=self.nlocs))
        self.states = []
        self.actions = []
        self.collectTimes = collecTimes
        self.transitTimes = transitTimes
        self.alpha = alpha
        self.start = []
        if(flag==0):
            self.states = self.manager.list()
            self.terminal = State(0, -1, -1, -1, [-1] * self.nlocs, -1, self.actions)
            self.states.append(self.terminal)
            self.initiateActions()
            self.initiateStates()
        else:
            self.readActions(config.workDir+"Data/DomainActionData"+str(self.agent)+"_exp_"+str(config.experiment)+".pickle")
            self.readStates(config.workDir+"Data/DomainStateData"+str(self.agent)+"_exp_"+str(config.experiment)+".pickle")
            self.terminal = self.states[0]
            self.writeTransitions()
            self.readRewards(config.workDir+"Data/DomainRewardData"+str(self.agent)+"_exp_"+str(config.experiment)+".pickle")
            self.defineStart()
            self.numberStates = len(self.states)
            self.numerActions = len(self.actions)
            self.numvariables = self.numberStates * self.numerActions

    def wasteRemovalParallel(self):
        self.waste()

    def AfterWasteRemoval(self):
        self.reindexStates()
        #self.checkTransitionProbabilitySumTo1()
        self.writeStatesToFile()
        self.writeActionsToFile()
        self.writeTransitions()
        self.writeRewards()
        self.defineStart()
        self.numberStates = len(self.states)
        self.numerActions = len(self.actions)
        self.numvariables = self.numberStates * self.numerActions

    def initiateActions(self):
        index = 0
        at = Action(index, "Collect")
        self.actions.append(at)
        index = index + 1
        for i in xrange(0, self.nlocs):
            at2 = Action(index, "Go to Site " + str(i), i)
            index = index + 1
            self.actions.append(at2)

    def initiateStates(self):
        index = 1
        for i in xrange(0, self.T+1, config.collectTimes[self.agent][0]):
            for j in xrange(0, self.nlocs):
                for k in self.lst:
                    if k[j]==0:
                       lyst = [0]
                    elif k[j]==1:
                       lyst = [0,1]
                    for t in lyst:
                        st = State(index,j,self.locs[j],i,k,t,self.actions)
                        self.states.append(st)
                        index = index + 1

    def transition(self, s, a, sd):
        if a.name == "Collect":
            return self.transitionCollect(s, sd)
        else:
            return self.transitionGoto(s, sd, a)

    def transitionCollect(self,s,sd):

        if s.time == -1 and sd == s:
            return 1

        # Terminal States
        if s.time == 0 and sd == self.terminal:
            return 1

        # Ensure that all d variables are same in s and sd other than d_l.
        sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs) if j != s.location])
        # print sameds

        if sameds==False:
            return 0

        # Location should be same
        if s.location != sd.location:
            return 0

        #Time should be subtracted
        if sd.time != s.time - self.collectTimes[s.location]:
            return 0

        # Time should not be negative after subtraction
        if sd.time < 0:
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

        if sameds == False:
            return 0

        if ld != dest and ld != l:
            return 0

        if td != t - self.transitTimes[l][ld]:
            return 0

        if td < 0:
            return 0

        if doldd != sd.dvals[ld]:
            return 0

        if dold == 1 and s.dvals[l] == 0:
            return 0

        if ld == l and ld == dest:
            return 1
        elif ld == l:
            return 1-config.beta
        elif ld == dest:
            return config.beta

    def waste(self):
        iter = 1
        removed, prevs1 = self.removeWasteStates(iter)
        while removed != 0:
            iter += 1
            removed, rst = self.succRemoval(prevs1, iter)
            prevs1 = rst

    def succRemoval(self, removedSt, iter):

        defRem = []
        mayeb = []
        sumd = 0
        val = 0
        tots = len(removedSt)
        start = time.time()
        offset = config.offset
        for rms in removedSt:

            sumd += 1
            if sumd % offset == 0:
                end = time.time()
                val += float(end - start)
                ntimes = float(sumd / offset)
                avg = float(val) / float(ntimes)
                timerem = (float(tots - sumd) / float(offset)) * avg
                print "[" + str(self.agent) + "," + str(iter) + "] Done. " + str(sumd) + " Out of: " + str(tots)+ " Avg: "+str(avg) + " Rem: "+str(timerem)
                start = time.time()

            for sts in self.states:

                if sts == self.terminal:
                    continue

                if sts.dold == 1 and sts.dvals[sts.location] == 0:
                    print "10 anomaly"
                    defRem.append(sts)
                    continue

                sameds = all([sts.dvals[j] == 0 for j in xrange(0, self.nlocs)])
                if sts.time == config.T[self.agent] and sameds == True and sts.dold == 0:
                    continue

                for a in self.actions:
                    if self.transition(rms, a, sts) != 0:
                        mayeb.append(sts)
                        break

        for i in mayeb:
            flag = 0
            for j in self.states:
                for k in self.actions:
                    if self.transition(j, k, i) != 0:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 0:
                defRem.append(i)

        prevs = []
        for sts in defRem:
            if sts in self.states:
                prevs.append(sts)
                self.states.remove(sts)

        print "For agent " + str(self.agent)+ " Iter "+str(iter)+" done and removed "+str(len(prevs))+"."
        return len(prevs), prevs

    def removeWasteStates(self, iter):
        wastestates = []
        sum = 0
        val = 0
        start = time.time()
        tots = len(self.states)
        removedSt = []
        offset = config.offset
        for i in self.states:
            sum += 1
            if sum%offset == 0:
                end = time.time()
                val += float(end-start)
                ntimes = float(sum / offset)
                avg = float(val) / float(ntimes)
                timerem = (float(tots - sum) / float(offset)) * avg
                print "["+str(self.agent)+","+str(iter)+"] Done. "+str(sum)+" Out of: "+str(tots)+ " Avg: "+str(avg) + " Rem: "+str(timerem)    
                start = time.time()
            if i == self.terminal:
                continue

            if i.dold == 1 and i.dvals[i.location] == 0:
                print "10 anomaly"
                wastestates.append(i)
                continue


            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == config.T[self.agent] and sameds==True and i.dold==0:
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
            removedSt.append(x)
            self.states.remove(x)
        # print len(self.states)
        # print
        print "For agent " + str(self.agent)+ " Iter "+str(iter)+" done and removed "+str(len(wastestates))+"."
        return len(wastestates), removedSt

    def reindexStates(self):
        index = 0
        lst = []
        for x in self.states:
            a = State(index, x.location, x.actualLocation, x.time, x.dvals, x.dold, self.actions)
            a.setReward(x.reward)
            a.setTransition(x.transition)
            lst.append(a)
            index = index + 1
        self.states = lst

    def checkTransitionProbabilitySumTo1(self):
        fp = open(config.workDir+'Data/tds'+str(config.experiment), 'w')
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
                    print "WARNING: k: " + str(k) + " i: " + str(i) + " Sum: " + str(sum)
        fp.close()

    def rewardFunction(self, s, a):
        import random
        if s.dold == 0 and s.dvals[s.location] == 1:
            return config.rewardCollection[self.agent][s.location] + random.random()
        else:
            return 0.1 + random.random()

    def writeTransitions(self):
        print "     Writing Transitions for Agent " +str(self.agent)
        for i in self.actions:
            for j in self.states:
                for k in self.states:
                    tt = self.transition(j,i,k)
                    if tt!=0:
                        j.transition.append((i.index, k.index, tt))

    def writeRewards(self):
        print "     Writing Rewards for Agent " +str(self.agent)
        rew = open(config.workDir + "Data/DomainRewardData" + str(self.agent) + "_exp_" + str(config.experiment) + ".pickle",'w')
        for i in self.states:
            for j in self.actions:
                re = self.rewardFunction(i,j)
                i.reward.append((j, re))
            pickle.dump(i.reward, rew)
        rew.close()

    def writeStatesToFile(self):
        print "     Writing States for Agent " +str(self.agent)
        stat = open(config.workDir+"Data/DomainStateData" + str(self.agent) +"_exp_"+str(config.experiment)+ ".pickle", 'w')
        pickle.dump(self.states, stat)
        stat.close()

    def writeActionsToFile(self):
        print "     Writing Actions for Agent " +str(self.agent)
        act = open(config.workDir+"Data/DomainActionData" + str(self.agent) +"_exp_"+str(config.experiment)+ ".pickle", 'w')
        pickle.dump(self.actions,act)
        act.close()

    def readRewards(self, filename):
        print "     Reading Rewards for Agent " +str(self.agent)
        f = open(filename, 'rb')
        for i in self.states:
            i.reward = pickle.load(f)
        f.close()

    def readActions(self, filename):
        print "     Reading Actions for Agent " +str(self.agent)
        f = open(filename, 'rb')
        self.actions = pickle.load(f)
        f.close()

    def readStates(self, filename):
        print "     Reading States for Agent " +str(self.agent)
        f = open(filename, 'rb')
        self.states = pickle.load(f)
        f.close()

    def defineStart(self):
        sum = 0
        for i in self.states:
            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == config.T[self.agent] and sameds==True and i.dold==0:
                sum += 1
        for i in self.states:
            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == config.T[self.agent] and sameds==True and i.dold==0:
                self.start.append(float(1/float(sum)))
            else:
                self.start.append(float(0))

    def generateLPAc(self, gamma, genA=False):
        print "Generating LP for "+str(self.agent)
        A_mat = []
        if genA is True:

            decisionvar = []
            for x in self.states:
                triple = []
                for y in self.states:
                    triplet = []
                    for a in y.possibleActions:
                        if x.index == y.index:
                            triplet.append(float(1))
                        else:
                            triplet.append(float(0))
                    triple.append(triplet)
                decisionvar.append(triple)

            for x in self.states:
                incoming = []
                for s in self.states:
                    for t in s.transition:
                        if t[1]==x.index and t[2]!=0:
                            incoming.append((s, t[0], t[2]))

                for h in incoming:
                    decisionvar[x.index][h[0].index][h[1]] -= gamma*float(h[2])

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

        # for x in A_mat:
        #     print x

        # for x in A_mat:
        #     print x

        R_mat = []
        for x in self.states:
            for y in x.possibleActions:
                assert len(x.reward) != 0
                for r in x.reward:
                    if r[0]==y:
                        R_mat.append(r[1])
        # print R_mat

        newR = []
        R_min = config.R_min
        R_max = config.R_max
        for x in self.states:
            for y in x.possibleActions:
                assert len(x.reward) != 0
                for r in x.reward:
                    if r[0]==y:
                        newR.append(float(r[1])/float(R_max-R_min))

        return A_mat, R_mat, newR

    def solveLP(self, gamma):
        A, R, newR = self.generateLPAc(gamma, genA=True)
        #print len(R)
        R_mat = np.array(R)[np.newaxis].T
        A_mat = np.array(A)

        alpha = self.start
        global num_vars
        x = cvxpy.Variable(self.numberStates*self.numerActions, 1)
        obj = cvxpy.Maximize(np.transpose(R_mat)*x)
        constraints = [A_mat*x == alpha, x >= 0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()
        #print "status:", prob.status
        print "LPsolver: optimal value", prob.value
        #print "Optimal x: ", x.value
        print "Sum of x values: ", cvxpy.sum_entries(x).value
        return prob.value

class PrimtiveEvent:
    def __init__(self, agent, state, action, statedash, index):
        self.agent = agent
        self.state = state
        self.action = action
        self.statedash = statedash
        self.index = index

    def __repr__(self):
        return "PE: Agent: " + str(self.agent) + " Index: " + str(self.index) + " State: " + str(self.state) + "\n" + " Action: " + str(self.action) + " Statedash: " + str(self.statedash)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.agent == other.agent and self.index==other.index and self.state==other.state and self.action==other.action and self.statedash==other.statedash
        return False

class Event:
    def __init__(self, agent, pevents, index, name, site):
        self.agent = agent
        self.pevents = pevents
        self.index = index
        self.name = name
        self.site = site

    def __repr__(self):
        return "E: ( " + str(self.agent) + " " + str(self.pevents) + " " + str(self.name) + " " + str(self.site) + " )"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.agent == other.agent and self.index==other.index and self.name==other.name and self.site==other.site
        return False

class Constraint:
    def __init__(self, num_agent, Events, rew, index):
        self.num_agent = num_agent
        self.Events = Events
        self.reward = rew
        self.index = index

    def __repr__(self):
        return "Cons: ( " + str(self.num_agent) + " " + str(self.Events) + " " + str(self.reward) + " )"

class JointReward:
    def __init__(self, num_cons, cons):
        self.num_cons = num_cons
        self.constraints = cons

    def __repr__(self):
        return "JointRew: ( " + str(self.num_cons) + " " + str(self.constraints) + " )"

class EMMDP:
    def __init__(self, n):
        self.num_agents = n
        self.mdps = []
        self.primitives = []
        self.events = []
        self.constraints = []
        self.generateMDPs()
        self.genPrimitiveEvents()
        self.genEvents()
        self.genConstraints()

    def generateMDPs(self):
        self.numvars = 0
        if config.flag == 0:
            prs = []
            for i in xrange(0, self.num_agents):
                print "Generating MDP for Agent"+str(i),
                a = MDP(config.T[i], config.locs[i], i, config.collectTimes[i], config.transitTimes[i], config.alpha, config.flag)
                print len(a.states)
                prs.append(multiprocessing.Process(target=a.wasteRemovalParallel))
                self.mdps.append(a)
            for pros in prs:
                pros.start()
            for pros in prs:
                pros.join()
            for i in xrange(0, self.num_agents):
                self.mdps[i].AfterWasteRemoval()
        else:
            for i in xrange(0, self.num_agents):
                print "Generating MDP for Agent"+str(i)
                a = MDP(config.T[i], config.locs[i], i, config.collectTimes[i], config.transitTimes[i], config.alpha, config.flag)
                self.mdps.append(a)
        for i in xrange(0, self.num_agents):
            self.numvars += len(self.mdps[i].states)*len(self.mdps[i].actions)
        print self.numvars

    def genPrimitiveEvents(self):
        print "Generating Primitive Events"
        index = 0
        for q in xrange(0,self.num_agents):
            a = self.mdps[q]
            for z in a.locs:
                for i in a.states:
                    if i.actualLocation == z and i.dvals[i.location] == 0 and i.dold == 0:
                        for k in a.states:
                            if k.actualLocation == z and k.dvals[k.location] == 1 and k.dold == 0:
                                if a.transition(i, a.actions[0], k) != 0:
                                    pe = PrimtiveEvent(q, i, a.actions[0], k, index)
                                    self.primitives.append(pe)
                                    index = index + 1

    def genEvents(self):
        print "Generating Events"
        index = 0
        for agent in xrange(0, self.num_agents):
            for j in self.mdps[agent].locs:
                arr = []
                for i in self.primitives:
                    if i.agent == agent and i.state.actualLocation == j:
                        arr.append(i)
                e = Event(agent, arr, index, "Agent " + str(agent) + " Collect at site "+str(j), j)
                index = index + 1
                self.events.append(e)

    def genConstraints(self):
        print "Generating Constraints"
        index = 0
        shared = config.sharedSites
        for x in xrange(0, len(shared)):
            local = []
            for y in self.events:
                if y.site == shared[x]:
                    local.append(y)
            c = Constraint(len(config.auction[shared[x]]), local, config.creward[x], index)
            index = index + 1
            self.constraints.append(c)

    def genAMPL(self):
        print "     Generating AMPL: ",
        ampl = open(config.workDir+'Data/nl2_exp_'+str(config.experiment)+'.dat', 'w')
        ampl.write("param n := " + str(self.num_agents) + ";\n")
        ampl.write("\n")
        for i in xrange(0, self.num_agents):
            ampl.write("set S[" + str(i+1) + "] := ")
            for x in self.mdps[i].states:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
            ampl.write("set A[" + str(i+1) + "] := ")
            for x in self.mdps[i].actions:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")
        ampl.write("param numcons := "+str(len(self.constraints))+";\n")
        ampl.write("param numprims := "+str(len(self.primitives))+";\n")
        ampl.write("param numevents := "+str(len(self.events))+";\n")
        ampl.write("param gamma := " + str(config.gamma) + ";\n")
        ampl.write("\n")

        ampl.write("param: sparseP: sparsePVal:= \n")
        for i in xrange(0, self.num_agents):
            for j in xrange(0, len(self.mdps[i].actions)):
                for k in xrange(0, len(self.mdps[i].states)):
                    h = self.mdps[i].states[k].transition
                    hh = [(x[1],x[2]) for x in h if x[0] == self.mdps[i].actions[j].index and x[2] != 0 ]
                    for valsac in hh:
                        ampl.write(str(i+1) + " " + str(self.mdps[i].actions[j].index+1) + " " + str(self.mdps[i].states[k].index+1) + " " +str(valsac[0]+1) + " " + str(valsac[1]) + "\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param R := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[i].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            for j in xrange(0, len(self.mdps[i].states)):
                ampl.write(str(j + 1) + " ")
                h = self.mdps[i].states[j].reward
                hh = [x[1] for x in h]
                for g in hh:
                    ampl.write(str(g) + " ")
                ampl.write("\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param alpha := \n")
        for i in xrange(0,self.num_agents):
            ampl.write("[" + str(i + 1) + ",*] := ")
            for gg in xrange(0,len(self.mdps[i].start)):
                ampl.write(str(gg+1) + " " + str(self.mdps[i].start[gg]) + " ")
            ampl.write("\n")
        ampl.write(";\n")

        ampl.write("param creward := ")
        for x in xrange(0, len(self.constraints)):
            ampl.write(str(x+1)+" "+str(self.constraints[x].reward)+" ")
        ampl.write(";\n")

        ampl.write("param primitives : ")
        for i in xrange(0, 4):
            ampl.write(str(i + 1) + " ")
        ampl.write(":= \n")
        for z in self.primitives:
                ampl.write(str(z.index + 1) + " " + str(z.agent + 1) + " " + str(z.state.index + 1) + " " +
                           str(z.action.index + 1) + " " + str(z.statedash.index + 1) + "\n")
        ampl.write(";\n")

        for i in xrange(0, len(self.events)):
            ampl.write("set events["+str(i+1)+"] := ")
            for x in self.events[i].pevents:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        for i in xrange(0, len(self.constraints)):
            ampl.write("set cons["+str(i+1)+"] := ")
            for x in self.constraints[i].Events:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        ampl.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }")
        ampl.close()
        print "Done"

    def genAMPLSingle(self, agent, initx=None):
        print "     Generating AMPL for Agent " + str(agent) + " : ",
        ampl = open(config.workDir+'Data/single'+str(agent)+'_exp_'+str(config.experiment)+'.dat', 'w')
        ampl.write("param n := " + str(self.num_agents) + ";\n")
        ampl.write("param agent := " + str(agent+1) + ";\n")
        ampl.write("param gamma := " + str(config.gamma) + ";\n")

        for i in xrange(0, self.num_agents):
            ampl.write("set S[" + str(i+1) + "] := ")
            for x in self.mdps[i].states:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
            ampl.write("set A[" + str(i+1) + "] := ")
            for x in self.mdps[i].actions:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        ampl.write("param: sparseP: sparsePVal:= \n")
        for i in xrange(0, self.num_agents):
            for j in xrange(0, len(self.mdps[i].actions)):
                for k in xrange(0, len(self.mdps[i].states)):
                    h = self.mdps[i].states[k].transition
                    hh = [(x[1],x[2]) for x in h if x[0] == self.mdps[i].actions[j].index and x[2] != 0 ]
                    for valsac in hh:
                        ampl.write(str(i+1) + " " + str(self.mdps[i].actions[j].index+1) + " " + str(self.mdps[i].states[k].index+1) + " " +str(valsac[0]+1) + " " + str(valsac[1]) + "\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param R := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[i].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            for j in xrange(0, len(self.mdps[i].states)):
                ampl.write(str(j + 1) + " ")
                h = self.mdps[i].states[j].reward
                hh = [x[1] for x in h]
                for g in hh:
                    ampl.write(str(g) + " ")
                ampl.write("\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param alpha := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*] := ")
            for gg in xrange(0, len(self.mdps[i].start)):
                ampl.write(str(gg + 1) + " " + str(self.mdps[i].start[gg]) + " ")
            ampl.write("\n")
        ampl.write(";\n")

        ampl.write("set all_numcons := ")
        for i in xrange(0, len(self.constraints)):
            ampl.write(str(i+1)+" ")
        ampl.write(";\n")

        ampl.write("set all_numprims := ")
        for i in xrange(0, len(self.primitives)):
            ampl.write(str(i + 1) + " ")
        ampl.write(";\n")

        ampl.write("set all_numevents := ")
        for i in xrange(0, len(self.events)):
            ampl.write(str(i + 1) + " ")
        ampl.write(";\n")

        ampl.write("param all_creward := ")
        for x in xrange(0, len(self.constraints)):
            ampl.write(str(x+1)+" "+str(self.constraints[x].reward)+" ")
        ampl.write(";\n")

        ampl.write("param all_primitives : ")
        for i in xrange(0, 4):
            ampl.write(str(i + 1) + " ")
        ampl.write(":= \n")
        for z in self.primitives:
                ampl.write(str(z.index + 1) + " " + str(z.agent + 1) + " " + str(z.state.index + 1) + " " +
                           str(z.action.index + 1) + " " + str(z.statedash.index + 1) + "\n")
        ampl.write(";\n")

        for i in xrange(0, len(self.events)):
            ampl.write("set all_events["+str(i+1)+"] := ")
            for x in self.events[i].pevents:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        for i in xrange(0, len(self.constraints)):
            ampl.write("set all_cons["+str(i+1)+"] := ")
            for x in self.constraints[i].Events:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        events = []
        consts = set()
        primits = []

        for cons in self.constraints:
            for eves in cons.Events:
                if eves.agent == agent:
                    events.append(eves)
                    primits.extend(eves.pevents)
                    consts.add(cons.index)

        ampl.write("set agent_numcons := ")
        for con in consts:
            ampl.write(str(con+1)+" ")
        ampl.write(";\n")

        ampl.write("set agent_numprims := ")
        for prims in primits:
            ampl.write(str(prims.index+1)+" ")
        ampl.write(";\n")

        ampl.write("set agent_numevents := ")
        for es in events:
            ampl.write(str(es.index+1)+" ")
        ampl.write(";\n")

        ampl.write("param agent_creward := ")
        for x in consts:
            ampl.write(str(x+1)+" "+str(self.constraints[x].reward)+" ")
        ampl.write(";\n\n")

        ampl.write("param agent_primitives : ")
        for i in xrange(0, 4):
            ampl.write(str(i + 1) + " ")
        ampl.write(":= \n")
        for z in primits:
                ampl.write(str(z.index + 1) + " " + str(z.agent + 1) + " " + str(z.state.index + 1) + " " +
                           str(z.action.index + 1) + " " + str(z.statedash.index + 1) + "\n")
        ampl.write(";\n")

        for es in events:
            ampl.write("set agent_events["+str(es.index+1)+"] := ")
            for x in es.pevents:
                ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n")

        for co in consts:
            ampl.write("set agent_cons["+str(co+1)+"] := ")
            for x in self.constraints[co].Events:
                if x.agent == agent:
                    ampl.write(str(x.index+1)+" ")
            ampl.write(";\n")
        ampl.write("\n\n")

        ampl.write("param theta := " + str(float(config.theta)) + ";\n")
        ampl.write("param Rmax := " + str(config.R_max) + ";\n")
        ampl.write("param Rmin := " + str(config.R_min) + ";\n\n")

        ampl.write("param x := \n")
        for k in xrange(0, self.num_agents):
            ampl.write("[" + str(k + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[k].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            if initx is None:
                for i in xrange(0, self.mdps[k].numberStates):
                    ampl.write(str(i+1)+" ")
                    for j in xrange(0, self.mdps[k].numerActions):
                        ampl.write(str(config.initialxval)+" ")
                    ampl.write("\n")
            else:
                for i in xrange(0, self.mdps[k].numberStates):
                    ampl.write(str(i+1)+" ")
                    for j in xrange(0, self.mdps[k].numerActions):
                        s = self.mdps[k].states[i]
                        a = self.mdps[k].actions[j]
                        val = initx[k][(s.index * self.mdps[k].numerActions) + a.index]
                        ampl.write(str(val)+" ")
                    ampl.write("\n")
        ampl.write(";\n")

        ampl.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }")

        ampl.close()
        print "Done"

    def runConfig(self, agent):
        print  "    Writting Running Config for Agent " + str(agent) + ": ",
        runf = open(config.workDir+'Data/single'+str(agent)+'_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("option solver '../ampl/"+str(config.solver)+"';\n")
        runf.write("option solver_msg 0;\n")
        #runf.write("option minos_options 'optimality_tolerance=1.0e-12';\n")
        runf.close()
        print "Done"

    def runConfigALLEM(self):
        runf = open(config.workDir+'Data/All2_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("option solver '../ampl/"+str(config.solver)+"';\n")
        runf.write("option solver_msg 0;\n")
        #runf.write("option minos_options 'optimality_tolerance=1.0e-12';\n")
        runf.close()
        print "Done"

    def runConfigNonLinear(self):
        print  "    Writting Running Config for Non Linear"
        runf = open(config.workDir+'Data/nl2_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("reset;\n")
        runf.write("model try.mod;\n")
        runf.write("data " + config.workDir+ 'Data/nl2_exp_'+str(config.experiment)+'.dat' + ";\n")
        runf.write("option solver_msg 0;\n")
        runf.write("write \"g/"+config.workDir+"Data/myfile"+str(config.experiment)+"\";\n")
        runf.close()
        print "Done"

    def updateRunConfigNonLinear(self):
        print  "    Writting Running Config for Non Linear"
        runf = open(config.workDir+'Data/nl2_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("reset;\n")
        runf.write("model try.mod;\n")
        runf.write("data " + config.workDir+ 'Data/nl2_exp_'+str(config.experiment)+'.dat' + ";\n")
        runf.write("option solver_msg 0;\n")
        runf.write("solution \""+config.workDir+"Data/myfile"+str(config.experiment)+".sol\";\n")
        runf.close()
        print "Done"

    def objective(self, xvals, Rs):
        sum = 0
        for i in xrange(0, self.num_agents):
            sum += np.dot(Rs[i], np.array(xvals[i]))
            if abs(np.sum(np.array(xvals[i])) - (float(1) / float(1-config.gamma))) > 1e-5:
                print "Warning", np.sum(xvals[i])

        for i in xrange(0, len(self.constraints)):
            cons = self.constraints[i]
            prod = cons.reward
            for eves in cons.Events:
                pesum = 0
                agent = eves.agent
                for peves in eves.pevents:
                    s = peves.state
                    a = peves.action
                    sd = peves.statedash
                    pesum += self.mdps[agent].transition(s,a,sd)*xvals[agent][(s.index*self.mdps[agent].numerActions)+a.index]
                prod *= pesum
            sum += prod
        return sum

    def genGraphAndSave(self, iters, results, nonLinear, times, nlptime):
        iterations = range(1, iters+1)
        line1, = plt.plot(iterations, results, marker='o', linestyle='-', color='r')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Objective')
        plt.title('EMMDP Experiment: ' + str(config.experiment))
        nonlx = [iters+1]
        nonly = [nonLinear]
        line2, = plt.plot(nonlx, nonly, marker='^', linestyle='-', color='g')
        #plt.xlim(0, 30)
        #plt.xticks(NumofTargets)
        #plt.legend([line1, line2, line3, line4, line5, line6, line7],
        #          ["Speed 0.2", "Speed 0.5", "Speed 0.8", "Speed 1.0", "Speed 1.2", "Speed 1.5", "Speed 1.8"], loc=2)
        #plt.show()
        plt.legend([line1, line2] , ["EM: "+str(max(results)), "NonLinear: "+str(nonLinear)], loc=4)

        plt.savefig('../Results/ObjGraph'+str(config.experiment)+'.png')
        plt.clf()
        plt.figure()

        iterations = range(1, len(times) + 1)
        line1, = plt.plot(iterations, times, marker='^', linestyle='-', color='b')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Time in seconds')
        plt.title('EMMDP Experiment: ' + str(config.experiment))
        nonlx = [iters + 1]
        nonly = [nlptime]
        line2, = plt.plot(nonlx, nonly, marker='o', linestyle='-', color='k')
        # plt.xlim(0, 30)
        # plt.xticks(NumofTargets)
        # plt.legend([line1, line2, line3, line4, line5, line6, line7],
        #          ["Speed 0.2", "Speed 0.5", "Speed 0.8", "Speed 1.0", "Speed 1.2", "Speed 1.5", "Speed 1.8"], loc=2)
        # plt.show()
        plt.legend([line1, line2], ["EM", "NonLinear"], loc=1)
        plt.savefig('../Results/TimeGraph' + str(config.experiment) + '.png')

    def timeout_handler(self, signum, frame):  # Custom signal handler
        raise TimeoutException

    def doIter(self, arg):
        i, ampl = arg
        ampl.solve()
        var = ampl.getVariable("xstar")
        var_vals = var.getValues()
        xstar_val = var_vals.getColumn('val')
        return xstar_val

    def doSuccIter(self, arg):
        i,ampl,greedy,oldX,A = arg
        ampl.solve()
        A = np.array(A)[np.newaxis].T
        A = np.transpose(A)
        var = ampl.getVariable("xstar")
        var_vals = var.getValues()
        xstar_val = var_vals.getColumn('val')
        alpha = np.array(self.mdps[i].start)
        if greedy:
            from decimal import Decimal
            pi_old = []
            pi_new = []
            ratios = []
            maxQ = [0]*self.mdps[i].numberStates
            for sts in self.mdps[i].states:
                sumProb_old = 0
                sumProb_new = 0
                maxQS = Decimal('-Infinity')
                for acts in self.mdps[i].actions:
                    sumProb_old += oldX[(sts.index * self.mdps[i].numerActions)+acts.index]
                    sumProb_new += xstar_val[(sts.index * self.mdps[i].numerActions)+acts.index]
                for acts in self.mdps[i].actions:
                    norm_val_old = oldX[(sts.index * self.mdps[i].numerActions)+acts.index] / float(sumProb_old)
                    norm_val_new = xstar_val[(sts.index * self.mdps[i].numerActions) + acts.index] / float(sumProb_new)
                    pi_old.append(norm_val_old)
                    pi_new.append(norm_val_new)
                    ratio = float(norm_val_new) / float(norm_val_old)
                    ratios.append(ratio)
                    if ratio > maxQS:
                        maxQS = ratio
                        maxQ[sts.index] = acts

            pi_new = []
            c = 0.1
            epsilon = self.PosNormal(0, 2)[0]
            for sts in self.mdps[i].states:
                ex_pi_new = []
                sum_new = 0
                for acts in self.mdps[i].actions:
                    if maxQ[sts.index]==acts:
                        indicator = 1
                    else:
                        indicator = 0
                    value = pi_old[(sts.index * self.mdps[i].numerActions)+acts.index]*(indicator+c+epsilon)
                    sum_new += value
                    ex_pi_new.append(value)
                for vals in ex_pi_new:
                    pi_new.append(float(vals)/float(sum_new))

            for sts in self.mdps[i].states:
                sum_new = 0
                for acts in self.mdps[i].actions:
                    sum_new += pi_new[(sts.index * self.mdps[i].numerActions)+acts.index]
                assert abs(float(1)-sum_new) < 0.0000001

            cons = []
            obj = cvxpy.Maximize(1)
            xnew = cvxpy.Variable(self.mdps[i].numberStates * self.mdps[i].numerActions, 1)
            for sts in self.mdps[i].states:
                sum = 0
                for acts in self.mdps[i].actions:
                    ind = (sts.index * self.mdps[i].numerActions)+acts.index
                    sum += xnew[ind]
                for acts in self.mdps[i].actions:
                    ind = (sts.index * self.mdps[i].numerActions)+acts.index
                    cons.append(xnew[ind] == sum*pi_new[ind])
            cons.append(A*xnew == alpha)
            cons.append(xnew >= 0)
            prob = cvxpy.Problem(obj, cons)
            prob.solve()
            #print abs(np.concatenate(np.array(xnew.value)) - xstar_val)
            return np.concatenate(np.array(xnew.value))
        else:
            return xstar_val

    def PosNormal(self, mean, sigma):
        x = np.random.normal(mean, sigma, 1)
        return (x if x >= 0 else self.PosNormal(mean, sigma))

    def NonLinear(self):
        AMPL = autoclass('com.ampl.AMPL')
        ampl = AMPL()
        ampl.reset()
        xstar_val = None
        nonlinearobj = 0
        nlptime = 0
        try:
            start = time.time()
            os.system("rm -rf "+config.workDir+"Data/myfile"+str(config.experiment)+".nl")
            os.system("rm -rf "+config.workDir+"Data/myfile"+str(config.experiment)+".sol")
            ampl.read(config.workDir+"Data/nl2_exp_"+str(config.experiment)+".run")
            os.system("pkill -f ampl")
            os.system("../ampl/"+config.solver+" -s "+config.workDir+"Data/myfile"+str(config.experiment)+".nl > nonLinearout")
            self.updateRunConfigNonLinear()
            ampl = AMPL()
            ampl.read(config.workDir + "Data/nl2_exp_" + str(config.experiment) + ".run")
            nonlinearobj = ampl.getObjective("ER").value()
            var = ampl.getVariable("x")
            var_vals = var.getValues()
            xstar_val = var_vals.getColumn('val')
            end = time.time()
            nlptime = end - start
            self.prGreen("Non Linear Took: "+str(nlptime))
            self.prYellow("Non Linear Obj: "+str(nonlinearobj))
        except Exception as e:
            print "Non Linear Not Able"
            print e
        except jnius.JavaException as e:
            print "Non Linear Not Able"
            print e
        return nonlinearobj, nlptime, xstar_val

    def resetAMPLs(self):
        ampls = []
        AMPL = autoclass('com.ampl.AMPL')
        for i in xrange(0, self.num_agents):
            ampl = AMPL()
            ampl.reset()
            ampl.read("single"+str(i)+".mod")
            ampl.readData(config.workDir + 'Data/single' + str(i) + '_exp_' + str(config.experiment) + '.dat')
            ampl.read(config.workDir + 'Data/single' + str(i) + '_exp_' + str(config.experiment) + '.run')
            ampls.append(ampl)
        return ampls

    def EMJavaAMPL(self):
        fails = 0
        AMPL = autoclass('com.ampl.AMPL')
        Double = autoclass('java.lang.Double')
        ampl = AMPL()
        ampl.reset()

        Rs = []
        As = []
        results = []
        deltas = []
        times = []
        iter = 1
        initial_x = []

        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=True)
            Rs.append(R)
            As.append(A)
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval]*numvar
            initial_x.append(lst)
        Rs = np.array(Rs)

        if config.GenRun == 1:
            for i in xrange(0, self.num_agents):
                self.genAMPLSingle(i, initial_x)
                self.runConfig(i)
            self.genAMPL()
            self.runConfigNonLinear()

        print "NonLinear:"
        pool = multiprocessing.pool.ThreadPool(processes=1)
        result = pool.apply_async(self.NonLinear)
        pool.close()
        nonlinearobj = 0
        nlptime = 0
        try:
            nonlinearobj, nlptime, Nonxv = result.get(timeout=config.timetorunsecN)
        except (Exception, multiprocessing.TimeoutError) as e:
            print e
            print("Process timed out")
        pool.terminate()
        print("Pool terminated")
        #os.system('pkill -f ampl')
        #os.system('pkill -f minos')

        assert len(Nonxv) == self.numvars
        NonVals = []
        sumbb = 0
        sumdd = 0
        for i in xrange(0, self.num_agents):
            for sts in self.mdps[i].states:
                for acts in self.mdps[i].actions:
                    sumdd += 1
            NonVals.append(Nonxv[sumbb:sumdd])
            sumbb = sumdd
        print "------------------Onjasdasd----------------"
        print self.objective(NonVals, Rs)

        print "EM-AMPL: "
        signal.signal(signal.SIGALRM, self.timeout_handler)
        sumIterTime = 0
        newobj = 0
        signal.alarm(config.timetorunsecE)

        if config.GenRun == 1:
            for i in xrange(0, self.num_agents):
                self.genAMPLSingle(i, NonVals)
                self.runConfig(i)
            self.genAMPL()
            self.runConfigNonLinear()

        try:
            ampls = []
            xvals = []
            args = []
            noOfProcess = multiprocessing.cpu_count() - 1
            pool = multiprocessing.pool.ThreadPool(noOfProcess)
            dastart = time.time()
            print "Iteration: " + str(iter)
            for i in xrange(0, self.num_agents):
                ampl = AMPL()
                ampl.reset()
                ampl.read("single.mod")
                ampl.readData(config.workDir + 'Data/single' + str(i) + '_exp_' + str(config.experiment) + '.dat')
                ampl.read(config.workDir + 'Data/single' + str(i) + '_exp_' + str(config.experiment) + '.run')
                ampls.append(ampl)
                args.append((i, ampl))
            daend = time.time()

            datot = daend - dastart
            sumIterTime += datot

            iterStartTime = time.time()
            iterEndTime = time.time()

            try:
                #print noOfProcess
                if self.num_agents <= noOfProcess:
                    pr = pool.map_async(self.doIter, args)
                    rss = pr.get(timeout=int(math.ceil(config.timetorunsecE - sumIterTime)))
                    xvals.extend(np.array(rss))
                    pool.close()
                else:
                    for i in xrange(0, self.num_agents, noOfProcess):
                        pr = pool.map_async(self.doIter, args[i:i+noOfProcess])
                        rss = pr.get(timeout=int(math.ceil(config.timetorunsecE - sumIterTime)))
                        xvals.extend(np.array(rss))
                    pool.close()
                iterEndTime = time.time()
            except multiprocessing.TimeoutError as e:
                print e
                print("Process timed out")
            pool.terminate()
            pool.join()
            print("Pool terminated")
            # os.system('pkill -f ampl')
            # os.system('pkill -f minos')
            newobj = self.objective(xvals, Rs)
            print "\nObjective: ", newobj
            iterTime = float(iterEndTime-iterStartTime)
            totTime = float(iterTime) + float(datot)
            times.append(totTime)
            print "Iteration %s time: %s\n\n" %(str(iter), str(totTime))
            sumIterTime += float(iterTime)
            greedy = False
            results.append(newobj)
            while(True):
                try:
                    iter += 1
                    print "Iteration: " + str(iter)
                    pool = multiprocessing.pool.ThreadPool(processes=self.num_agents)
                    args = []
                    xvalues = []
                    xvalsAsParam = np.concatenate(xvals)
                    xvalsParamFinal = [Double(k) for k in xvalsAsParam]
                    dastart = time.time()
                    for i in xrange(0, self.num_agents):
                        paramx = ampls[i].getParameter("x")
                        paramx_val = paramx.getValues()
                        paramx_val.setColumn('val', xvalsParamFinal)
                        paramx.setValues(paramx_val)
                        xvalsParamFinal = [Double(k) for k in xvalsAsParam]
                        args.append((i, ampls[i], greedy, xvals[i], As[i]))
                    daend = time.time()

                    datot = 0
                    sumIterTime += datot

                    iterStartTime = time.time()
                    iterEndTime = time.time()

                    try:
                        #print noOfProcess
                        if self.num_agents <= noOfProcess:
                            pr = pool.map_async(self.doSuccIter, args)
                            rss = pr.get(timeout=int(math.ceil(config.timetorunsecE - sumIterTime)))
                            xvalues.extend(np.array(rss))
                            pool.close()
                        else:
                            for i in xrange(0, self.num_agents, noOfProcess):
                                pr = pool.map_async(self.doSuccIter, args[i:i + noOfProcess])
                                rss = pr.get(timeout=int(math.ceil(config.timetorunsecE - sumIterTime)))
                                xvalues.extend(np.array(rss))
                            pool.close()
                        iterEndTime = time.time()
                    except multiprocessing.TimeoutError as e:
                        print e
                        print("Process timed out")
                    pool.terminate()
                    pool.join()
                    print("Pool terminated")

                    oldobj = self.objective(xvals, Rs)
                    print "\n\nOld Objective: ",oldobj
                    #print np.size(xvals), np.size(xvalues)
                    xvals = xvalues
                    newobj = self.objective(xvals, Rs)
                    results.append(newobj)
                    print "New Objective: ", newobj
                    iterTime = float(iterEndTime-iterStartTime)
                    totTime = float(iterTime) + float(datot)
                    times.append(totTime)
                    print "Iteration %s time: %s\n\n" %(str(iter), str(totTime))
                    sumIterTime += float(iterTime)

                    self.prRed("New: " + str(newobj))
                    self.prGreen("NonLinearObj: " + str(nonlinearobj))
                    if nonlinearobj != 0:
                        self.prYellow("PercentError: " + str(
                            (float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%")
                    print
                    print
                    #greedy = False

                    def signal_handler(signal, frame):
                        print('You pressed Ctrl+C!')
                        self.genGraphAndSave(len(results), results, nonlinearobj, times, nlptime)
                        import sys
                        os.system("pkill -f python")
                        sys.exit(0)

                    signal.signal(signal.SIGINT, signal_handler)

                    deltas.append(abs(newobj - oldobj))
                    lastX = deltas[-config.noiterConvergence:len(deltas)]
                    avgChange = sum(lastX) / float(len(lastX))
                    print "AvgChange: ", avgChange
                    if (abs(avgChange) < config.deltaFinal):
                        #self.genAMPLSingle(0,xvals)
                        #self.genAMPLSingle(1,xvals)
                        print "NonLinear Obj: ", nonlinearobj
                        print "EM Obj: ", newobj
                        print "AvgIterTime: ", sumIterTime/iter
                        print "NonLinearTime: ", nlptime
                        print "Overall EM Time: %s"%(sumIterTime)
                        print "PercentError: " + str((float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
                        if nonlinearobj != 0:
                            print "PercentError: " + str((float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
                        break
                    # elif newobj < oldobj:
                    #     greedy = False
                    #     print "---------------Setting Greedy to False---------------"
                    #elif abs(newobj - oldobj) < config.deltaIter:
                    #    greedy = True
                    #    print "---------------Setting Greedy to True---------------"
                except (jnius.JavaException) as e:
                    fails += 1
                    if fails > 5:
                        break
                    print e
                    pool.terminate()
                    pool.join()
                    #os.system('pkill -f ampl')
                    #os.system('pkill -f minos')
                    #ampls = self.resetAMPLs()
                    iter -= 1
                    print "Rerun"
                    traceback.print_exc()
                    traceback.print_stack()
                    continue

        except TimeoutException as e:
            print e
            print "\n\n EM Time's Up: "
            print "NonLinear Obj: ", nonlinearobj
            print "EM Obj: ", newobj
            print "AvgIterTime: ", sumIterTime / iter
            print "NonLinearTime: ", nlptime
            print "Overall EM Time: %s" % (sumIterTime)
            if nonlinearobj != 0:
                print "PercentError: " + str((float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
        else:
            signal.alarm(0)

        self.genGraphAndSave(len(results), results, nonlinearobj, times, nlptime)

    def prRed(self, prt):
        print("\033[91m {}\033[00m".format(prt))

    def prGreen(self, prt):
        print("\033[92m {}\033[00m".format(prt))

    def prYellow(self, prt):
        print("\033[93m {}\033[00m".format(prt))

    def prLightPurple(self, prt):
        print("\033[94m {}\033[00m".format(prt))

    def prPurple(self, prt):
        print("\033[95m {}\033[00m".format(prt))

    def prCyan(self, prt):
        print("\033[96m {}\033[00m".format(prt))

    def prLightGray(self, prt):
        print("\033[97m {}\033[00m".format(prt))

    def prBlack(self, prt):
        print("\033[98m {}\033[00m".format(prt))

    def writeALLEM(self, initx=None):
        print "     Generating AMPL: ",
        ampl = open(config.workDir + 'Data/All2_exp_' + str(config.experiment) + '.dat', 'w')
        ampl.write("param n := " + str(self.num_agents) + ";\n")
        ampl.write("param gamma := " + str(config.gamma) + ";\n")
        ampl.write("\n")

        for i in xrange(0, self.num_agents):
            ampl.write("set S[" + str(i + 1) + "] := ")
            for x in self.mdps[i].states:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
            ampl.write("set A[" + str(i + 1) + "] := ")
            for x in self.mdps[i].actions:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")

        ampl.write("param: sparseP: sparsePVal:= \n")
        for i in xrange(0, self.num_agents):
            for j in xrange(0, len(self.mdps[i].actions)):
                for k in xrange(0, len(self.mdps[i].states)):
                    h = self.mdps[i].states[k].transition
                    hh = [(x[1], x[2]) for x in h if x[0] == self.mdps[i].actions[j].index and x[2] != 0]
                    for valsac in hh:
                        ampl.write(str(i + 1) + " " + str(self.mdps[i].actions[j].index + 1) + " " + str(
                            self.mdps[i].states[k].index + 1) + " " + str(valsac[0] + 1) + " " + str(valsac[1]) + "\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param R := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[i].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            for j in xrange(0, len(self.mdps[i].states)):
                ampl.write(str(j + 1) + " ")
                h = self.mdps[i].states[j].reward
                hh = [x[1] for x in h]
                for g in hh:
                    ampl.write(str(g) + " ")
                ampl.write("\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param alpha := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*] := ")
            for gg in xrange(0, len(self.mdps[i].start)):
                ampl.write(str(gg + 1) + " " + str(self.mdps[i].start[gg]) + " ")
            ampl.write("\n")
        ampl.write(";\n")

        ampl.write("set numcons := ")
        for i in xrange(0, len(self.constraints)):
            ampl.write(str(i + 1) + " ")
        ampl.write(";\n")

        ampl.write("set numprims := ")
        for i in xrange(0, len(self.primitives)):
            ampl.write(str(i + 1) + " ")
        ampl.write(";\n")

        ampl.write("set numevents := ")
        for i in xrange(0, len(self.events)):
            ampl.write(str(i + 1) + " ")
        ampl.write(";\n")

        ampl.write("param creward := ")
        for x in xrange(0, len(self.constraints)):
            ampl.write(str(x + 1) + " " + str(self.constraints[x].reward) + " ")
        ampl.write(";\n")

        ampl.write("param primitives : ")
        for i in xrange(0, 4):
            ampl.write(str(i + 1) + " ")
        ampl.write(":= \n")
        for z in self.primitives:
            ampl.write(str(z.index + 1) + " " + str(z.agent + 1) + " " + str(z.state.index + 1) + " " +
                       str(z.action.index + 1) + " " + str(z.statedash.index + 1) + "\n")
        ampl.write(";\n")

        for i in xrange(0, len(self.events)):
            ampl.write("set events[" + str(i + 1) + "] := ")
            for x in self.events[i].pevents:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")

        for i in xrange(0, len(self.constraints)):
            ampl.write("set cons[" + str(i + 1) + "] := ")
            for x in self.constraints[i].Events:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")

        ampl.write("param theta := " + str(float(config.theta)) + ";\n")
        ampl.write("param Rmax := " + str(config.R_max) + ";\n")
        ampl.write("param Rmin := " + str(config.R_min) + ";\n\n")

        ampl.write("param x := \n")
        for k in xrange(0, self.num_agents):
            ampl.write("[" + str(k + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[k].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            if initx is None:
                for i in xrange(0, self.mdps[k].numberStates):
                    ampl.write(str(i + 1) + " ")
                    for j in xrange(0, self.mdps[k].numerActions):
                        ampl.write(str(config.initialxval) + " ")
                    ampl.write("\n")
            else:
                for i in xrange(0, self.mdps[k].numberStates):
                    ampl.write(str(i + 1) + " ")
                    for j in xrange(0, self.mdps[k].numerActions):
                        s = self.mdps[k].states[i]
                        a = self.mdps[k].actions[j]
                        val = initx[k][(s.index * self.mdps[k].numerActions) + a.index]
                        ampl.write(str(val) + " ")
                    ampl.write("\n")
        ampl.write(";\n")

        ampl.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }")

        ampl.close()
        print "Done"

    def AllEM(self):
        AMPL = autoclass('com.ampl.AMPL')
        Double = autoclass('java.lang.Double')
        ampl = AMPL()
        ampl.reset()

        Rs = []
        results = []
        times = []
        deltas = []
        times = []
        initial_x = []

        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=False)
            Rs.append(R)
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval] * numvar
            initial_x.append(lst)
        Rs = np.array(Rs)

        if config.GenRun == 1:
            self.writeALLEM(initial_x)
            self.genAMPL()
            self.runConfigALLEM()
            self.runConfigNonLinear()

        nonlinearobj, nlptime, Nonxv = self.NonLinear()
        assert len(Nonxv) == self.numvars
        NonVals = []
        sumbb = 0
        sumdd = 0
        for i in xrange(0, self.num_agents):
            for sts in self.mdps[i].states:
                for acts in self.mdps[i].actions:
                    sumdd += 1
            NonVals.append(Nonxv[sumbb:sumdd])
            sumbb = sumdd
        check_nlp_obj = self.objective(NonVals, Rs)
        assert abs(check_nlp_obj - nonlinearobj) < config.deltaError

        iter = 1
        print "Iteration: ", iter

        ampl = AMPL()
        ampl.reset()
        ampl.read("EMALL.mod")
        ampl.readData(config.workDir + 'Data/All2_exp_' + str(config.experiment) + '.dat')
        ampl.read(config.workDir + 'Data/All2_exp_' + str(config.experiment) + '.run')
        ampl.solve()

        # xvals = []
        # var = ampl.getVariable("xstar")
        # var_val = var.getValues()
        # var_vals = var_val.getColumn('val')
        # sumbb = 0
        # sumdd = 0
        # for i in xrange(0, self.num_agents):
        #     sumdd += self.mdps[i].numvariables
        #     xvals.append(var_vals[sumbb:sumdd])
        #     sumbb = sumdd
        # xvals = np.array(xvals)

        #oldobj = self.objective(xvals, Rs)
        ampl.eval("let iter_value := sum{i in 1..n, j in S[i], k in A[i]} xstar[i,j,k]*R[i,j,k] + sum{l in numcons} creward[l] * prod{m in cons[l]} zstar[l,m];")
        oldobj = ampl.getParameter("iter_value").get()
        results.append(oldobj)
        self.prYellow(str(oldobj))
        print
        print

        while (True):
            iter += 1
            ampl.eval("let iter_value := sum{i in 1..n, j in S[i], k in A[i]} xstar[i,j,k]*R[i,j,k] + sum{l in numcons} creward[l] * prod{m in cons[l]} zstar[l,m];")
            oldobj = ampl.getParameter("iter_value").get()

            print "Iteration: " + str(iter)
            ampl.eval("for {i in 1..n, j in S[i], k in A[i]} { let x[i,j,k] := xstar[i,j,k]; }")
            ampl.solve()
            ampl.eval("let iter_value := sum{i in 1..n, j in S[i], k in A[i]} xstar[i,j,k]*R[i,j,k] + sum{l in numcons} creward[l] * prod{m in cons[l]} zstar[l,m];")
            newobj = ampl.getParameter("iter_value").get()

            # var = ampl.getVariable("xstar")
            # var_val = var.getValues()
            # var_vals = var_val.getColumn('val')
            # sumbb = 0
            # sumdd = 0
            # for i in xrange(0, self.num_agents):
            #     sumdd += self.mdps[i].numvariables
            #     xvalues.append(var_vals[sumbb:sumdd])
            #     sumbb = sumdd
            # xvalues = np.array(xvalues)
            #oldobj = self.objective(xvals, Rs)

            results.append(newobj)
            self.prYellow("Old: "+str(oldobj))
            self.prCyan("New: "+str(newobj))
            self.prRed("Change: "+str(newobj-oldobj))
            self.prGreen("NonLinearObj: "+str(nonlinearobj))
            if nonlinearobj != 0:
                self.prYellow("PercentError: " + str(
                    (float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%")
            print
            print

            def signal_handler(signal, frame):
                print('You pressed Ctrl+C!')
                self.genGraphAndSave(len(results), results, nonlinearobj, times, nlptime)
                import sys
                os.system("pkill -f python")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            deltas.append(abs(newobj - oldobj))
            lastX = deltas[-config.noiterConvergence:len(deltas)]
            avgChange = sum(lastX)/float(len(lastX))
            print "AvgChange: ", avgChange
            if(abs(avgChange) < config.deltaFinal):
                print "NonLinear Obj: ", nonlinearobj
                print "EM Obj: ", newobj
                print "NonLinearTime: ", nlptime
                if nonlinearobj != 0:
                    print "PercentError: " + str(
                        (float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
                break

        self.genGraphAndSave(len(results), results, nonlinearobj, times, nlptime)

class TimeoutException(Exception):   # Custom exception class
    pass

class Driver:
    a = EMMDP(config.agents)
    #a.AllEM()
    a.EMJavaAMPL()