import csv
import numpy as np
import cvxpy
import config
import itertools
import copy_reg
import types
import pyipopt
import algopy
from pathos.multiprocessing import ProcessingPool as Pool

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
            self.terminal = State(0, -1, -1, -1, [-1] * self.nlocs, -1, self.actions)
            self.states.append(self.terminal)
            self.initiateActions()
            self.initiateStates()
            self.waste()
            self.checkTransitionProbabilitySumTo1()
            self.writeStatesToFile()
            self.writeActionsToFile()
            self.writeTransitionsToFile()
            self.writeRewardsToFile()
        else:
            self.readActions("DomainActionData"+str(self.agent)+".txt")
            self.readStates("DomainStateData"+str(self.agent)+".txt")
            self.terminal = self.states[0]
            self.readTransition("DomainTransitionData"+str(self.agent)+".txt")
            #self.checkTransitionProbabilitySumTo1File()
            self.readRewards("DomainRewardData"+str(self.agent)+".txt")
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
                    for t in [0,1]:
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
            self.states.remove(x)
        # print len(self.states)
        # print

        index = 0
        for x in self.states:
            x.index = index
            index = index + 1

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
                    print "WARNING: k: " + str(k) + " i: " + str(i) + " Sum: " + str(sum)
        fp.close()

    def checkTransitionProbabilitySumTo1File(self):
        fp = open('tds', 'w')
        for k in self.actions:
            for i in self.states:
                sum = 0
                for j in self.states:
                    fp.write(str(i))
                    fp.write("\n")
                    fp.write(str(j))
                    fp.write("\n")
                    tran = [xx[2] for xx in i.transition if xx[0]==k and xx[1]==j]
                    tran = tran[0]
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

    def writeTransitionsToFile(self):
        print "     Writing Transitions for Agent " +str(self.agent)
        tran = open("DomainTransitionData"+str(self.agent)+".txt", 'w')
        for i in self.actions:
            for j in self.states:
                for k in self.states:
                    tt = self.transition(j,i,k)
                    j.transition.append((i, k, tt))
                    tran.write(str(tt))
                    if k != self.states[len(self.states)-1]:
                        tran.write(",")
                tran.write("\n")
            tran.write("\n")
        tran.close()

    def writeRewardsToFile(self):
        print "     Writing Rewards for Agent " +str(self.agent)
        rew = open("DomainRewardData"+str(self.agent)+".txt",'w')
        for i in self.states:
            for j in self.actions:
                re = self.rewardFunction(i,j)
                rew.write(str(re))
                i.reward.append((j, re))
                if j != self.actions[len(self.actions)-1]:
                    rew.write(",")
            rew.write("\n")
        rew.close()

    def writeStatesToFile(self):
        print "     Writing States for Agent " +str(self.agent)
        stat = open("DomainStateData" + str(self.agent) + ".txt", 'w')
        for j in self.states:
            stat.write(str(j.index)+","+str(j.location)+","+str(j.actualLocation)+","+str(j.time)+",")
            for x in j.dvals:
                stat.write(str(x)+",")
            stat.write(str(j.dold)+"\n")
        stat.close()

    def writeActionsToFile(self):
        print "     Writing Actions for Agent " +str(self.agent)
        act = open("DomainActionData" + str(self.agent) + ".txt", 'w')
        for j in self.actions:
            act.write(str(j.index)+","+str(j.gotox)+","+str(j.name)+"\n")

    def readActions(self, filename):
        print "     Reading Actions for Agent " +str(self.agent)
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                ind = int(row[0])
                if str(row[1])=="None":
                    gotox = None
                else:
                    gotox = int(row[1])
                name = row[2]
                a = Action(ind, name, gotox)
                self.actions.append(a)

    def readStates(self, filename):
        print "     Reading States for Agent " +str(self.agent)
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                ind = int(row[0])
                loc = int(row[1])
                acloc = int(row[2])
                tim = int(row[3])
                lst = []
                for x in xrange(0, self.nlocs):
                    lst.append(int(row[4+x]))
                dold = int(row[len(row)-1])
                s = State(ind, loc, acloc, tim, lst, dold, self.actions)
                self.states.append(s)

    def readTransition(self, filename):
        print "     Reading Transitions for Agent " +str(self.agent)
        for s in self.states:
            s.transition = []
        stateIndex = 0
        actionIndex = 0
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 0:
                    stateIndex = 0
                    actionIndex = actionIndex + 1
                    continue
                for sp in xrange(0, len(self.states)):
                    triple = (self.actions[actionIndex], self.states[sp], float(row[sp]))
                    self.states[stateIndex].transition.append(triple)
                stateIndex += 1

    def readRewards(self, filename):
        print "     Reading Rewards for Agent " +str(self.agent)
        tosend = []
        stateIndex = 0
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row)==0:
                    continue
                for ap in xrange(0, len(self.actions)):
                    triple = (self.actions[ap], float(row[ap]))
                    tosend.append(triple)
                self.states[stateIndex].reward = tosend
                tosend = []
                stateIndex += 1

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

    def generateLPAc(self, gamma):
        print "Generating LP for "+str(self.agent)
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
                    if t[1]==x and t[2]!=0:
                        incoming.append((s, t[0], t[2]))

            for h in incoming:
                decisionvar[x.index][h[0].index][h[1].index] -= gamma*float(h[2])

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
                    if r[0]==y:
                        R_mat.append(r[1])
        # print R_mat

        newR = []
        R_min = config.R_min
        R_max = config.R_max
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0]==y:
                        newR.append(float(r[1])/float(R_max-R_min))

        return A_mat, R_mat, newR

    def solveLP(self, gamma):
        A, R, newR = self.generateLPAc(gamma)
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
        print "Generating AMPL"
        ampl = open('nl2.dat', 'w')
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

        ampl.write("param P := \n")
        for i in xrange(0, self.num_agents):
            for j in xrange(0, len(self.mdps[i].actions)):
                ampl.write("[" + str(i + 1) + "," + str(j + 1) + ",*,*] : ")
                for k in xrange(0, len(self.mdps[i].states)):
                    ampl.write(str(k + 1) + " ")
                ampl.write(":= \n")
                for k in xrange(0, len(self.mdps[i].states)):
                    ampl.write(str(k + 1) + " ")
                    h = self.mdps[i].states[k].transition
                    hh = [x[2] for x in h if x[0] == self.mdps[i].actions[j]]
                    for g in hh:
                        ampl.write(str(g) + " ")
                    ampl.write("\n")
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
        ampl.close()

    def objective(self, xvals, Rs):
        sum = 0
        for i in xrange(0, self.num_agents):
            sum += np.dot(Rs[i], np.array(xvals[i]))
            if abs(np.sum(np.array(xvals[i])) - (float(1) / float(1-config.gamma))) > config.delta:
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

        print sum
        return sum

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
            A, R, newR = self.mdps[i].generateLPAc(config.gamma)
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
        self.objective(xvals, self.Rs)

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
            xvals = xvalues
            pvals = pvalues
            newobj = self.objective(xvals, self.Rs)
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

class Driver:
    print cvxpy.installed_solvers()
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
    a.genAMPL()
    a.EM(NonLinear=False)
