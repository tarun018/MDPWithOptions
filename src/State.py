import csv
import signal
import numpy as np
import cvxpy
import config
import itertools
import copy_reg
import types
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import multiprocessing
import pickle
import multiprocessing.pool
import madopt

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
            return 0.1

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
        numvars = 0
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
            numvars += len(self.mdps[i].states)*len(self.mdps[i].actions)
        print numvars

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
            sum += prod
        #sum += config.theta * len(self.constraints)
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

    def EM(self):
        initial_x = []
        for i in xrange(0, self.num_agents):
            numvar = self.mdps[i].numberStates * self.mdps[i].numerActions
            lst = [config.initialxval] * numvar
            initial_x.append(lst)

        self.As = []
        self.Rs = []
        self.newRs = []
        self.alphas = []
        xvals = []
        pvals = []
        num_iter = 1

        noOfProcess = multiprocessing.cpu_count()
        pool = multiprocessing.pool.ThreadPool(noOfProcess)
        args = []
        models = []

        print "Iteration: " + str(num_iter)
        for i in xrange(0, self.num_agents):
            A, R, newR = self.mdps[i].generateLPAc(config.gamma, genA=True)
            self.As.append(A)
            self.Rs.append(R)
            self.newRs.append(newR)
            self.alphas.append(self.mdps[i].start)

        sums, products = self.generateEstep(initial_x, self.newRs)
        for i in xrange(0, self.num_agents):
            model = madopt.BonminModel(show_solver=True)
            models.append(model)
            args.append((sums, products, initial_x, i, model))
            # if config.solver == 'ipopt':
            #     xstar_val = self.Mstep1(sums, products, initial_x, i)
            # elif config.solver == 'cvxpy':
            #     xstar_val = self.Mstep(sums, products, initial_x, i)
            # elif config.solver == 'bonmin':
            #     xstar_val = self.Mstep2(sums, products, initial_x, i)

        try:
            pr = pool.map_async(self.Mstep2, args)
            rss = pr.get(timeout=50)
            xvals.extend(rss)
        except multiprocessing.TimeoutError as e:
            print e
            print("Process timed out")

        o = self.objective(xvals, self.Rs)
        print "Objective: ", o

        while (True):
            num_iter += 1
            print "Iteration: " + str(num_iter)
            args = []
            xvalues = []
            pvalues = []
            sums, products = self.generateEstep(xvals, self.newRs)
            for i in xrange(0, self.num_agents):
                args.append((sums, products, xvals, i, models[i]))
                # if config.solver == 'ipopt':
                #     xstar_val = self.Mstep1(sums, products, xvals, i)
                # elif config.solver == 'cvxpy':
                #     xstar_val = self.Mstep(sums, products, xvals, i)
                # elif config.solver == 'bonmin':
                #     xstar_val = self.Mstep2(sums, products, xvals, i)

            try:
                pr = pool.map_async(self.Mstep2, args)
                rss = pr.get(timeout=50)
                xvalues.extend(rss)
            except multiprocessing.TimeoutError as e:
                print e
                print("Process timed out")
            prevobj = self.objective(xvals, self.Rs)
            print "PrevObj: ", prevobj
            xvals = xvalues
            pvals = pvalues
            newobj = self.objective(xvals, self.Rs)
            print "NewObj: ", newobj
            if abs(newobj - prevobj) < config.delta:
                print newobj
                pool.close()
                pool.terminate()
                break

    def generateEstep(self, x, newRs):
        print "Estep: "
        sums = []
        for i in xrange(0, self.num_agents):
            Rcap = np.array(newRs[i])[np.newaxis].T
            rdiag = np.diag(Rcap[:, 0])
            rdiagx = rdiag.dot(x[i])
            sums.append(rdiagx)

        products = []
        for i in xrange(0, len(self.constraints)):
            prod = float(self.constraints[i].reward) / float(config.R_max - config.R_min)
            assert prod > 0
            for eves in self.constraints[i].Events:
                sum = 0
                agent = eves.agent
                for k in eves.pevents:
                    s = k.state
                    a = k.action
                    sd = k.statedash
                    sum += self.mdps[agent].transition(s, a, sd) * x[agent][(s.index * self.mdps[agent].numerActions) + a.index]
                prod *= sum
            products.append(prod)
        assert all(vals > 0 for vals in products)
        print "Done"
        return sums, np.array(products)

    def Mstep1(self, sums, products, initx, agent):
        print "Mstep: "
        nvar = self.mdps[agent].numberStates * self.mdps[agent].numerActions
        x_L = np.ones((nvar), dtype=np.float_) * 0.000001
        x_U = np.ones((nvar), dtype=np.float_) * (float(1) / float(1-config.gamma))

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
                            sumxstar = self.mdps[agent].transition(s, a, sd) * x[(s.index * self.mdps[agent].numerActions) + a.index]
                        else:
                            sumxstar += self.mdps[agent].transition(s, a, sd) * x[(s.index * self.mdps[agent].numerActions) + a.index]

                        if type(sumx) is int:
                            sumx = self.mdps[agent].transition(s, a, sd) * initx[agent][(s.index * self.mdps[agent].numerActions) + a.index]
                        else:
                            sumx += self.mdps[agent].transition(s, a, sd) * initx[agent][(s.index * self.mdps[agent].numerActions) + a.index]

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

                    #print sumxstar

                if sumallevents != 0:
                    #print estepvalue * sumallevents
                    obj += estepvalue * sumallevents

                if sumallevents1 != 0:
                    #print thetahat * sumallevents1
                    obj += thetahat * sumallevents1

                if sumallevents2 != 0:
                    #print thetahat * sumallevents2
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
        # nlp.str_option('linear_solver', 'mumps')
        # nlp.num_option('tol', 1e-7)
        # nlp.int_option('print_level', 0)
        # nlp.str_option('mehrotra_algorithm', 'yes')
        # nlp.str_option('print_timing_statistics', 'yes')
        #nlp.int_option('max_iter', 100)
        nlp.str_option('mu_strategy', 'adaptive')
        nlp.str_option('warm_start_init_point', 'yes')
        x0 = np.array(initx[agent])
        x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
        nlp.close()
        return x, -1 * obj

    def Mstep2(self, args):
        sums, products, x, agent, model = args
        print "Mstep-----------------------------------: ",
        num_of_var = self.mdps[agent].numberStates * self.mdps[agent].numerActions

        rdiagx = np.array(sums[agent])
        A_mat = np.array(self.As[agent])

        xstar = dict()
        for i in range(num_of_var):
            xstar[i] = model.addVar(lb=0, init=config.initialxval, name="x" + str(i))
        obj = madopt.Expr(0)

        for i in range(num_of_var):
            obj += sums[agent][i] * madopt.log2(xstar[i])

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
                        sumxstar = self.mdps[agent].transition(s, a, sd) * xstar[
                            (s.index * self.mdps[agent].numerActions) + a.index]
                    else:
                        sumxstar += self.mdps[agent].transition(s, a, sd) * xstar[
                            (s.index * self.mdps[agent].numerActions) + a.index]

                    if type(sumx) is int:
                        sumx = self.mdps[agent].transition(s, a, sd) * x[agent][
                            (s.index * self.mdps[agent].numerActions) + a.index]
                    else:
                        sumx += self.mdps[agent].transition(s, a, sd) * x[agent][
                            (s.index * self.mdps[agent].numerActions) + a.index]

                if type(sumallevents) is int:
                    sumallevents = madopt.log2(sumxstar)
                else:
                    sumallevents += madopt.log2(sumxstar)

                if type(sumallevents1) is int:
                    sumallevents1 = (1 - sumx) * madopt.log2(1 - sumxstar)
                else:
                    sumallevents1 = (1 - sumx) * madopt.log2(1 - sumxstar)

                if type(sumallevents2) is int:
                    sumallevents2 = sumx * madopt.log2(sumxstar)
                else:
                    sumallevents2 += sumx * madopt.log2(sumxstar)

            if sumallevents != 0:
                # print estepvalue * sumallevents
                obj += estepvalue * sumallevents

            if sumallevents1 != 0:
                # print thetahat * sumallevents1
                obj += thetahat * sumallevents1

            if sumallevents2 != 0:
                # print thetahat * sumallevents2
                obj += thetahat * sumallevents2
        obj = -1*obj
        model.setObj(obj)
        ncon = self.mdps[agent].numberStates

        for i in xrange(0, ncon):
            dot_prod = madopt.Expr(0)
            for j in xrange(0, num_of_var):
                dot_prod += A_mat[i][j] * xstar[j]
            model.addConstr(dot_prod, lb=self.alphas[agent][i], ub=self.alphas[agent][i])

        for i in xrange(0, num_of_var):
            model.addConstr(xstar[i], lb=0.000001)

        model.solve()
        ret = []
        for key, value in xstar.iteritems():
            temp = [key, value]
            ret.append(temp[1].x)
        return ret

    def Mstep(self, sums, products, x, agent):
        print "Mstep: ",
        num_of_var = self.mdps[agent].numberStates * self.mdps[agent].numerActions
        xstar = cvxpy.Variable(num_of_var, 1)
        obj = np.transpose(np.array(sums[agent])) * cvxpy.log(xstar)
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
                        sumxstar = self.mdps[agent].transition(s, a, sd) * xstar[
                            (s.index * self.mdps[agent].numerActions) + a.index]
                    else:
                        sumxstar += self.mdps[agent].transition(s, a, sd) * xstar[
                            (s.index * self.mdps[agent].numerActions) + a.index]

                    if type(sumx) is int:
                        sumx = self.mdps[agent].transition(s, a, sd) * x[agent][
                            (s.index * self.mdps[agent].numerActions) + a.index]
                    else:
                        sumx += self.mdps[agent].transition(s, a, sd) * x[agent][
                            (s.index * self.mdps[agent].numerActions) + a.index]

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
                # print estepvalue * sumallevents
                obj += estepvalue * sumallevents

            if sumallevents1 != 0:
                # print thetahat * sumallevents1
                obj += thetahat * sumallevents1

            if sumallevents2 != 0:
                # print thetahat * sumallevents2
                obj += thetahat * sumallevents2

        obj = cvxpy.Maximize(obj)
        A_mat = np.array(self.As[agent])
        alpha = self.alphas[agent]
        cons = [A_mat * xstar == alpha, xstar >= 0.000001]
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

class TimeoutException(Exception):   # Custom exception class
    pass

class Driver:
    #print cvxpy.installed_solvers()
    a = EMMDP(config.agents)
    a.EM()
