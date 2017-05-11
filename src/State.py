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
from jnius import autoclass
import time
import multiprocessing
import pickle
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
            self.writeRewards()
            self.defineStart()
            self.numberStates = len(self.states)
            self.numerActions = len(self.actions)

    def wasteRemovalParallel(self):
        self.waste()

    def AfterWasteRemoval(self):
        self.reindexStates()
        # self.checkTransitionProbabilitySumTo1()
        self.writeStatesToFile()
        self.writeActionsToFile()
        self.writeTransitions()
        self.writeRewards()
        self.defineStart()
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

        if s.dold == 0 and s.dvals[s.location] == 1:
            return config.rewardCollection[self.agent][s.location]
        else:
            return 0

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
        for i in self.states:
            for j in self.actions:
                re = self.rewardFunction(i,j)
                i.reward.append((j, re))

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

            pass

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

    def runConfigEMAMPL(self, agent):
        print  "    Writting Running Config for Agent " + str(agent) + ": ",
        runf = open('single'+str(agent)+'_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("reset;\n")
        runf.write("model single.mod;\n")
        runf.write("data "+config.workDir+'Data/single'+str(agent)+'_exp_'+str(config.experiment)+'.dat'+";\n")
        runf.write("option solver 'ampl/"+str(config.solver)+"';\n")
        #runf.write("option minos_options 'feasibility_tolerance=1.0e-8 optimality_tolerance=1.0e-8 scale=no Completion=full';\n")
        runf.write("solve;\n")
        runf.write("display {i in S[agent], j in A[agent]} : xstar[i,j] ;\n")
        runf.close()
        print "Done"

    def runConfig(self, agent):
        print  "    Writting Running Config for Agent " + str(agent) + ": ",
        runf = open('single'+str(agent)+'_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("option solver 'ampl/"+str(config.solver)+"';\n")
        runf.write("option solver_msg 0;\n")
        #runf.write("option minos_options 'feasibility_tolerance=1.0e-8 optimality_tolerance=1.0e-8 Completion=full';\n")
        runf.close()
        print "Done"

    def runConfigNonLinear(self):
        print  "    Writting Running Config for Non Linear"
        runf = open('try_exp_'+str(config.experiment)+'.run', 'w')
        runf.write("reset;\n")
        runf.write("model try.mod;\n")
        runf.write("data " + config.workDir+ 'Data/nl2_exp_'+str(config.experiment)+'.dat' + ";\n")
        runf.write("option solver 'ampl/"+str(config.solver)+"';\n")
        runf.write("option solver_msg 0;\n")
        #runf.write("option minos_options 'feasibility_tolerance=1.0e-8 optimality_tolerance=1.0e-8 Completion=full';\n")
        runf.write("solve > multi.out;\n")
        runf.close()
        print "Done"

    def objective(self, xvals, Rs):
        sum = 0
        for i in xrange(0, self.num_agents):
            sum += np.dot(Rs[i], np.array(xvals[i]))
            if abs(np.sum(np.array(xvals[i])) - (float(1) / float(1-config.gamma))) > 0.01:
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
            sum += np.asscalar(prod)
        #sum += config.theta * len(self.constraints)
        return sum

    def processFile(self, filename, agent):
        x = []
        #xsa = np.zeros((self.mdps[agent].numberStates,self.mdps[agent].numerActions))
        status = ""
        red = open(filename, 'r')
        for row in red:
            splitted = row.split('=')
            if len(splitted)==2 and splitted[0].find('xstar') != -1:
                try:
                    value = float(splitted[1])
                except(ValueError):
                    print "Error: " , splitted
                # indistr = splitted[0][splitted[0].find('[')+1 : splitted[0].find(']')]
                # indistr = indistr.split(',')
                # xsa[int(indistr[0])-1,int(indistr[1])-1] = value
                x.append(value)
            elif len(splitted) >= 1:
                status = status + row
        assert len(x) == self.mdps[agent].numberStates * self.mdps[agent].numerActions
        x = np.array(x)
        return x,status

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
        plt.legend([line1, line2] , ["EM", "NonLinear"], loc=4)

        plt.savefig('../Results/ObjGraph'+str(config.experiment)+'.png')
        plt.clf()
        plt.figure()

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

    def EMJavaAMPL(self):
        AMPL = autoclass('com.ampl.AMPL')
        DataFrame = autoclass('com.ampl.DataFrame')
        Objective = autoclass('com.ampl.Objective')
        Parameter = autoclass('com.ampl.Parameter')
        Variable = autoclass('com.ampl.Variable')
        Tuple = autoclass('com.ampl.Tuple')
        VarInstance = autoclass('com.ampl.VariableInstance')
        Double = autoclass('java.lang.Double')
        ampl = AMPL()
        ampl.reset()

        results = []
        times = []
        iter = 1
        print "NonLinear:"
        if config.flag == 0:
            self.genAMPL()

        nonlinearobj = 0.0001
        start_time_non = time.time()
        import multiprocessing.pool
        ampl.read("try.mod")
        ampl.readData(config.workDir+"Data/nl2_exp_"+str(config.experiment)+".dat")
        pool = multiprocessing.pool.ThreadPool(processes=1)
        result = pool.apply_async(ampl.solve)
        try:
            result.get(timeout=config.timetorunsec)
            obj = ampl.getObjective("ER")
            nonlinearobj = obj.value()
            print nonlinearobj
        except multiprocessing.TimeoutError:
            print("Process timed out")
        pool.terminate()
        print("Pool terminated")

        end_time_non = time.time()
        nlptime = end_time_non - start_time_non
        print "Non Linear Time:  %s seconds \n\n\n\n\n\n" % nlptime

        ampl = AMPL()
        ampl.reset()
        Rs = []
        print "EM-AMPL: "
        initial_x = []
        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=False)
            Rs.append(R)
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval]*numvar
            initial_x.append(lst)
        Rs = np.array(Rs)

        if config.flag == 0:
            for i in xrange(0, self.num_agents):
                self.genAMPLSingle(i, initial_x)
                self.runConfig(i)

        signal.signal(signal.SIGALRM, self.timeout_handler)
        iterStartTime = time.time()
        sumIterTime = 0
        newobj = 0
        signal.alarm(config.timetorunsec)
        try:
            xvals = []
            print "Iteration: " + str(iter)
            for i in xrange(0, self.num_agents):
                ampl.reset()
                ampl.read("single.mod")
                ampl.readData(config.workDir+'Data/single'+str(i)+'_exp_'+str(config.experiment)+'.dat')
                ampl.read('single'+str(i)+'_exp_'+str(config.experiment)+'.run')
                ampl.solve()

                var = ampl.getVariable("xstar")
                var_vals = var.getValues()
                xstar_val = var_vals.getColumn('val')
                xvals.append(np.array(xstar_val))

            iterEndTime = time.time()

            newobj = self.objective(xvals, Rs)
            print "\nObjective: ", newobj
            iterTime = float(iterEndTime-iterStartTime)/self.num_agents
            times.append(iterTime)
            print "Iteration %s time: %s\n\n" %(str(iter), str(iterTime))
            sumIterTime += float(iterEndTime-iterStartTime)/self.num_agents

            results.append(newobj)
            while(True):
                iter += 1
                print "Iteration: " + str(iter)
                iterStartTime = time.time()
                xvalues = []
                xvalsAsParam = np.concatenate(xvals)
                xvalsParamFinal = [Double(k) for k in xvalsAsParam]
                for i in xrange(0, self.num_agents):
                    ampl.reset()
                    ampl.read("single.mod")
                    ampl.readData(config.workDir+'Data/single' + str(i) + '_exp_' + str(config.experiment) + '.dat')
                    ampl.read('single' + str(i) + '_exp_' + str(config.experiment) + '.run')
                    paramx = ampl.getParameter("x")
                    paramx_val = paramx.getValues()
                    paramx_val.setColumn('val', xvalsParamFinal)
                    xvalsParamFinal = [Double(k) for k in xvalsAsParam]
                    paramx.setValues(paramx_val)

                    ampl.solve()
                    var = ampl.getVariable("xstar")
                    var_vals = var.getValues()
                    xstar_val = var_vals.getColumn('val')
                    xvalues.append(np.array(xstar_val))

                iterEndTime = time.time()
                oldobj = self.objective(xvals, Rs)
                print "\n\nOld Objective: ",oldobj
                xvals = xvalues
                newobj = self.objective(xvals, Rs)
                results.append(newobj)
                print "New Objective: ", newobj
                iterTime = float(iterEndTime - iterStartTime) / self.num_agents
                times.append(iterTime)
                print "Iteration %s time: %s\n\n" % (str(iter), str(iterTime))
                sumIterTime += float(iterEndTime-iterStartTime)/self.num_agents

                print "\n"
                if abs(newobj - oldobj) < config.delta:
                    print "NonLinear Obj: ", nonlinearobj
                    print "EM Obj: ", newobj
                    print "AvgIterTime: ", sumIterTime/iter
                    print "NonLinearTime: ", end_time_non - start_time_non
                    print "Overall EM Time: %s"%(sumIterTime)
                    print "PercentError: " + str((float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
                    break
        except TimeoutException as e:
            print e
            print "\n\n EM Time's Up: "
            print "NonLinear Obj: ", nonlinearobj
            print "EM Obj: ", newobj
            print "AvgIterTime: ", sumIterTime / iter
            print "NonLinearTime: ", end_time_non - start_time_non
            print "Overall EM Time: %s" % (sumIterTime)
            print "PercentError: " + str((float(abs(nonlinearobj - newobj)) / float(max(nonlinearobj, newobj))) * 100) + "%"
        else:
            signal.alarm(0)

        self.genGraphAndSave(len(results), results, nonlinearobj, times, nlptime)

    def EMAMPL(self):
        results = []
        iter = 1
        nonlinearobj = 0
        print "NonLinear:"
        self.genAMPL()
        self.runConfigNonLinear()
        os.system('../ampl/ampl try_exp_'+str(config.experiment)+'.run > '+config.workDir+'Data/NonLinearOut'+str(config.experiment)+'.txt')

        nonred = open(config.workDir+'Data/NonLinearOut'+str(config.experiment)+'.txt', 'r')
        for row in nonred:
            if row.find('objective') != -1:
                spli = row.split(' ')
                try:
                    nonlinearobj = float(spli[-1])
                except(ValueError):
                    print "Not able to parse.", spli[-1]
            print row,
        nonred.close()

        Rs = []
        print "EM-AMPL: "
        initial_x = []
        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=False)
            Rs.append(R)
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval]*numvar
            initial_x.append(lst)
        Rs = np.array(Rs)

        for i in xrange(0, self.num_agents):
            self.genAMPLSingle(i, initial_x)
            self.runConfigEMAMPL(i)

        print "Iteration: "+str(iter)
        xvals = []
        for i in xrange(0, self.num_agents):
            os.system('ampl/ampl single'+str(i)+'_exp_'+str(config.experiment)+'.run > '+config.workDir+'Data/EMOut' + str(i) +'_exp_'+str(config.experiment)+'.txt')
            x,status = self.processFile(filename=config.workDir+'Data/EMOut'+str(i)+'_exp_'+str(config.experiment)+'.txt', agent=i)
            assert (sum(x) - float(1) / float(1 - config.gamma)) < 0.01
            xvals.append(x)
            #print "Status for Agent " + str(i) + ": ", status.rstrip()

        initialobj = self.objective(xvals, Rs)
        print "Objective: ", initialobj
        results.append(initialobj)

        while(True):
            iter += 1
            print "Iteration: " + str(iter)
            xvalues = []
            for i in xrange(0, self.num_agents):
                self.genAMPLSingle(i, xvals)
            for i in xrange(0, self.num_agents):
                os.system('ampl/ampl single' + str(i) + '_exp_' + str(config.experiment) + '.run > '+config.workDir+'Data/EMOut' + str(i) + '_exp_' + str(config.experiment) + '.txt')
                x, status = self.processFile(filename=config.workDir+'Data/EMOut' + str(i) + '_exp_' + str(config.experiment) + '.txt', agent=i)
                assert (sum(x) - float(1) / float(1 - config.gamma)) < 0.01
                xvalues.append(x)
                #print "Status for Agent " + str(i) + ": ", status.rstrip()
            oldobj = self.objective(xvals, Rs)
            print "Old Objective: ",oldobj
            xvals = xvalues
            newobj = self.objective(xvals, Rs)
            results.append(newobj)
            print "New Objective: ", newobj;
            if abs(newobj - oldobj) < config.delta:
                break

        self.genGraphAndSave(iter, results, nonlinearobj)

    def EM(self, NonLinear=False):
        initial_x = []
        for i in xrange(0, self.num_agents):
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval]*numvar
            initial_x.append(lst)

        self.As = []
        self.Rs = []
        self.newRs = []
        self.alphas = []
        xvals = []
        pvals = []
        num_iter = 1

        print "Iteration: " + str(num_iter)
        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=True)
            self.As.append(A)
            self.Rs.append(R)
            self.newRs.append(newR)
            self.alphas.append(self.mdps[i].start)

        if NonLinear:
            x, obj = self.NonLinear(initial_x)
            print obj
            return

        sums, products = self.generateEstep(initial_x, self.newRs)
        for i in xrange(0, self.num_agents):
            if config.solver == 'ipopt':
                xstar_val, prob_val = self.Mstep1(sums, products, initial_x, i)
            elif config.solver == 'cvxpy':
                xstar_val, prob_val = self.Mstep(sums, products, initial_x, i)
            xvals.append(xstar_val)
            pvals.append(prob_val)
        o = self.objective(xvals, self.Rs)
        print "Objective: ", o

        while(True):
            num_iter += 1
            print "Iteration: " + str(num_iter)
            xvalues = []
            pvalues = []
            sums, products = self.generateEstep(xvals, self.newRs)
            for i in xrange(0, self.num_agents):
                if config.solver == 'ipopt':
                    xstar_val, prob_val = self.Mstep1(sums, products, xvals, i)
                elif config.solver == 'cvxpy':
                    xstar_val, prob_val = self.Mstep(sums, products, xvals, i)
                xvalues.append(xstar_val)
                pvalues.append(prob_val)
            prevobj = self.objective(xvals, self.Rs)
            print "PrevObj: ", prevobj
            xvals = xvalues
            pvals = pvalues
            newobj = self.objective(xvals, self.Rs)
            print "NewObj: ", newobj
            if abs(newobj - prevobj) < config.delta:
                print newobj
                break

    def generateEstep(self, x, newRs):
        print "Estep: "
        sums = []
        for i in xrange(0, self.num_agents):
            Rcap = np.array(newRs[i])[np.newaxis].T
            rdiag = np.diag(Rcap[:, 0])
            rdiagx = rdiag.dot(x[i])
            rdiagx = rdiagx * (1 - config.gamma)
            sums.append(rdiagx)

        products = []
        for i in xrange(0, len(self.constraints)):
            prod =  float(self.constraints[i].reward) / float(config.R_max - config.R_min)
            assert prod > 0
            for eves in self.constraints[i].Events:
                sum = 0
                agent = eves.agent
                for k in eves.pevents:
                    s = k.state
                    a = k.action
                    sd = k.statedash
                    sum += self.mdps[agent].transition(s,a,sd) * x[agent][(s.index*self.mdps[agent].numerActions)+a.index] * (1-config.gamma)
                prod *= sum
            products.append(prod)
        assert all(vals > 0 for vals in products)
        print "Done"
        return sums, np.array(products)

    def Mstep1(self, sums, products, initx, agent):
        print "Mstep: "
        nvar = self.mdps[agent].numberStates * self.mdps[agent].numerActions
        x_L = np.ones((nvar), dtype=np.float_) * 1e-7
        x_U = np.ones((nvar), dtype=np.float_) * 1

        ncon = self.mdps[agent].numberStates
        g_L = np.array(self.alphas[agent])
        g_U = np.array(self.alphas[agent])

        rdiagx = np.array(sums[agent])
        A_mat = np.array(self.As[agent])

        nnzj = ncon * nvar
        nnzh = 0

        def eval_f(x, user_data=None):
            thetahat = float(config.theta) / float(config.R_max - config.R_min)
            assert thetahat > 0
            obj = 0
            for i in xrange(0, len(self.constraints)):
                estepvalue = products[i]
                const = self.constraints[i]
                sumallevents = 0
                sumallevents1 = 0
                sumallevents2 = 0
                for eves in const.Events:
                    sumxstar = 0
                    sumx = 0
                    if eves.agent != agent:
                        continue
                    for pevens in eves.pevents:
                        s = pevens.state
                        a = pevens.action
                        sd = pevens.statedash

                        if type(sumxstar) is int:
                            sumxstar = self.mdps[agent].transition(s, a, sd) * x[(s.index * self.mdps[agent].numerActions) + a.index] * (1 - config.gamma)
                        else:
                            sumxstar += self.mdps[agent].transition(s, a, sd) * x[(s.index * self.mdps[agent].numerActions) + a.index] * (1 - config.gamma)

                        if type(sumx) is int:
                            sumx = self.mdps[agent].transition(s, a, sd) * initx[agent][(s.index * self.mdps[agent].numerActions) + a.index] * (1 - config.gamma)
                        else:
                            sumx += self.mdps[agent].transition(s, a, sd) * initx[agent][(s.index * self.mdps[agent].numerActions) + a.index] * (1 - config.gamma)

                    if type(sumallevents) is int:
                        sumallevents = np.log(sumxstar)
                    else:
                        sumallevents += np.log(sumxstar)

                    if type(sumallevents1) is int:
                        sumallevents1 = (1 - sumx) * np.log(1 - sumxstar)
                    else:
                        sumallevents1 = (1 - sumx) * np.log(1 - sumxstar)

                    if type(sumallevents2) is int:
                        sumallevents2 = sumx * np.log(sumxstar)
                    else:
                        sumallevents2 += sumx * np.log(sumxstar)

                if sumallevents != 0:
                    # print estepvalue * sumallevents
                    obj += estepvalue * sumallevents

                if sumallevents1 != 0:
                    # print thetahat * sumallevents1
                    obj += thetahat * sumallevents1

                if sumallevents2 != 0:
                    # print thetahat * sumallevents2
                    obj += thetahat * sumallevents2

            obj += np.dot(rdiagx, np.log(np.array(x)))
            return -1 * obj

        def eval_grad_f(x, user_data=None):
            x = algopy.UTPM.init_jacobian(x)
            return algopy.UTPM.extract_jacobian(eval_f(x))

        def eval_g(x, user_data=None):
            out = algopy.zeros(ncon, dtype=x)
            for i in xrange(0, ncon):
                out[i] = np.dot(A_mat[i, :], x)
            return out

        def eval_jac_g(x, flag, user_data=None):
            if flag:
                rows = []
                cols = []
                arr = range(0, nvar)
                for i in xrange(0, ncon):
                    rows.extend([i] * nvar)
                    cols.extend(arr)
                assert len(rows) == nnzj
                assert len(cols) == nnzj
                return (np.array(rows), np.array(cols))
            else:
                x = algopy.UTPM.init_jacobian(x)
                y = algopy.UTPM.extract_jacobian(eval_g(x))
                return np.concatenate(np.array(y))

        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
        #nlp.str_option('linear_solver', 'mumps')
        #nlp.num_option('tol', 1e-7)
        #nlp.int_option('print_level', 0)
        #nlp.str_option('mehrotra_algorithm', 'yes')
        #nlp.str_option('print_timing_statistics', 'yes')
        nlp.int_option('max_iter', 100)
        nlp.str_option('mu_strategy', 'adaptive')
        nlp.str_option('warm_start_init_point', 'yes')
        x0 = np.array(initx[agent])
        x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
        nlp.close()
        return x, -1*obj

    def Mstep(self, sums, products, x, agent):
        print "Mstep: ",
        num_of_var = self.mdps[agent].numberStates * self.mdps[agent].numerActions
        xstar = cvxpy.Variable(num_of_var, 1)
        obj = np.transpose(np.array(sums[agent]))*cvxpy.log(xstar)
        thetahat = float(config.theta) / float(config.R_max - config.R_min)
        assert thetahat > 0
        for i in xrange(0, len(self.constraints)):
            estepvalue = products[i]
            const = self.constraints[i]
            sumallevents = 0
            sumallevents1 = 0
            sumallevents2 = 0
            for eves in const.Events:
                sumxstar = 0
                sumx = 0
                if eves.agent != agent:
                    continue
                for pevens in eves.pevents:
                    s = pevens.state
                    a = pevens.action
                    sd = pevens.statedash

                    if type(sumxstar) is int:
                        sumxstar = self.mdps[agent].transition(s,a,sd) * xstar[(s.index*self.mdps[agent].numerActions)+a.index] * (1-config.gamma)
                    else:
                        sumxstar += self.mdps[agent].transition(s, a, sd) * xstar[
                            (s.index * self.mdps[agent].numerActions) + a.index] * (1-config.gamma)

                    if type(sumx) is int:
                        sumx = self.mdps[agent].transition(s,a,sd) * x[agent][(s.index*self.mdps[agent].numerActions)+a.index] * (1-config.gamma)
                    else:
                        sumx += self.mdps[agent].transition(s,a,sd) * x[agent][(s.index*self.mdps[agent].numerActions)+a.index] * (1-config.gamma)

                if type(sumallevents) is int:
                    sumallevents = cvxpy.log(sumxstar)
                else:
                    sumallevents += cvxpy.log(sumxstar)

                if type(sumallevents1) is int:
                    sumallevents1 = (1 - sumx) * cvxpy.log(1 - sumxstar)
                else:
                    sumallevents1 = (1 - sumx) * cvxpy.log(1 - sumxstar)

                if type(sumallevents2) is int:
                    sumallevents2 = sumx * cvxpy.log(sumxstar)
                else:
                    sumallevents2 += sumx * cvxpy.log(sumxstar)


            if sumallevents != 0:
                #print estepvalue * sumallevents
                obj += estepvalue*sumallevents

            if sumallevents1 != 0:
                #print thetahat * sumallevents1
                obj += thetahat * sumallevents1

            if sumallevents2 != 0:
                #print thetahat * sumallevents2
                obj += thetahat * sumallevents2

        obj = cvxpy.Maximize(obj)
        A_mat = np.array(self.As[agent])
        alpha = self.alphas[agent]
        cons = [A_mat * xstar == alpha, xstar > 0]
        # for evecon in xrange(0, len(self.agentwise[agent])):
        #     mt = np.zeros((num_of_var, 1))
        #     event = self.agentwise[agent][evecon][0]
        #     con = self.agentwise[agent][evecon][1]
        #     for peven in event.pevents:
        #         s = peven.state
        #         a = peven.action
        #         sd = peven.statedash
        #         ind = (s.index*self.mdps[agent].numerActions)+a.index
        #         mt[ind] = self.mdps[agent].transition(s,a,sd)
        #     cons.append(zstar[evecon] == np.transpose(mt)*xstar)
        #
        prob = cvxpy.Problem(objective=obj, constraints=cons)
        prob.solve(solver=cvxpy.ECOS, verbose=False, max_iters=10000000)
        print "Done"
        return xstar.value, prob.value

    def NonLinear(self, initx):

        ncon = 0
        nvar = 0
        vars = []
        vars.append(0)

        for i in xrange(0, self.num_agents):
            ncon += self.mdps[i].numberStates
            vars.append(vars[i] + self.mdps[i].numberStates * self.mdps[i].numerActions)
            nvar += self.mdps[i].numberStates * self.mdps[i].numerActions

        nnzj = sum([self.mdps[i].numberStates * self.mdps[i].numerActions * self.mdps[i].numberStates for i in xrange(0, self.num_agents)])
        #print vars, nvar, ncon, nnzj
        nnzh = 0

        x_L = np.ones((nvar), dtype=np.float_) * 0.000001
        x_U = np.ones((nvar), dtype=np.float_) * (float(1) / float(1-config.gamma))

        g_L = np.concatenate(np.array(self.alphas))
        g_U = np.concatenate(np.array(self.alphas))

        def eval_f(x, user_data=None):
            obj = 0
            x = np.array(x)
            for i in xrange(0, self.num_agents):
                obj += np.dot(np.array(self.Rs[i]), np.array(x[vars[i]:vars[i+1]]))

            for j in xrange(0, len(self.constraints)):
                cons = self.constraints[j]
                prod = cons.reward
                for eves in cons.Events:
                    pesum = 0
                    agent = eves.agent
                    for peves in eves.pevents:
                        s = peves.state
                        a = peves.action
                        sd = peves.statedash
                        pesum += self.mdps[agent].transition(s, a, sd) * np.array(x[(vars[agent] + ( s.index * self.mdps[agent].numerActions)) + a.index])
                    prod *= pesum
                obj += np.asscalar(np.array(prod))
            return -1*obj

        def eval_grad_f(x, user_data=None):
            x = algopy.UTPM.init_jacobian(x)
            return algopy.UTPM.extract_jacobian(eval_f(x))

        def eval_g(x, user_data=None):
            index = 0
            out = algopy.zeros(ncon, dtype=x)
            for i in xrange(0, self.num_agents):
                for j in xrange(0, self.mdps[i].numberStates):
                    out[index] = np.dot(np.array(self.As[i])[j,:], np.array(x[vars[i]:vars[i+1]]))
                    index += 1
            return out

        def eval_jac_g(x, flag, user_data=None):
            if flag:
                rows = []
                cols = []
                index = 0
                for i in xrange(0, self.num_agents):
                    for j in xrange(0, self.mdps[i].numberStates):
                        for k in xrange(vars[i],vars[i+1]):
                            rows.append(index)
                            cols.append(k)
                        index += 1
                assert len(rows) == nnzj
                assert len(cols) == nnzj
                return (np.array(rows), np.array(cols))
            else:
                x = algopy.UTPM.init_jacobian(x)
                y = algopy.UTPM.extract_jacobian(eval_g(x))
                return np.concatenate(np.array(y))


        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
        #nlp.int_option('max_iter', 150)
        nlp.str_option('mu_strategy', 'adaptive')
        nlp.str_option('warm_start_init_point', 'yes')
        x0 = np.concatenate(np.array(initx))
        x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
        nlp.close()
        return x, -1*obj

class TimeoutException(Exception):   # Custom exception class
    pass

class Driver:
    #print cvxpy.installed_solvers()
    a = EMMDP(config.agents)

    # for x in a.constraints:
    #     print x.num_agent
    #     for y in x.Events:
    #         print y.name, y.index
    #         for z in y.pevents:
    #             print z.index
    #             print z.state
    #             print z.action
    #             print z.statedash
    #     print x.reward
    #     print
    #print a.mdps[0].states[4].transition
    # sum = 0
    # for i in xrange(0, a.num_agents):
    #     sum += a.mdps[i].solveLP(config.gamma)
    # print sum
    #a.EMAMPL()
    a.EMJavaAMPL()
    #a.EMAMPL()
    #a.EM(NonLinear=False)
