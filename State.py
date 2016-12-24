import csv
from copy import deepcopy
import numpy as np
import cvxopt
import cvxopt.solvers
import cvxpy
import config

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
            x = State(i, str("s" + str(i)), self.actions)
            self.states.append(x)
        #self.states[1].setPossibleActions([self.actions[0]])

    # Leave one line space after each transition table for each action in the data file.
    def autoTransitionFunction(self, filename, gamma=1):
        for s in self.states:
            s.setTransition([])
        stateIndex = 0
        actionIndex = 0
        with open(filename, 'rb') as csvfile:
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
    def autoRewardFunction(self, filename):

        tosend = []
        stateIndex = 0
        with open(filename, 'rb') as csvfile:
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

    def actionsVI(self, delta, gamma):

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
        R_min = config.R_min
        R_max = config.R_max
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0]==y.getIndex():
                        newR.append((r[1]-R_min)/(R_max-R_min))

        return A_mat, R_mat, newR

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

    def solveLP(self, gamma):
        A, R, newR = self.generateLPAc(gamma)
        #print len(R)
        R_mat = np.array(R)[np.newaxis].T
        A_mat = np.array(A)

        alpha = config.alpha
        global num_vars
        x = cvxpy.Variable(config.num_of_var, 1)
        obj = cvxpy.Maximize(np.transpose(R_mat)*x)
        constraints = [A_mat*x == alpha, x >= 0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()
        #print "status:", prob.status
        print "LPsolver: optimal value", prob.value
        #print "Optimal x: ", x.value
        print "Sum of x values: ", cvxpy.sum_entries(x).value
        return prob.value

        #print "optimal var", x.value

    def EM(self, gamma, delta):
        num_iter = 1
        A, R, newR = self.generateLPAc(gamma)
        dualLP = self.solveLP(gamma)
        newR = np.array(newR)[np.newaxis].T
        A_mat = np.array(A)
        alpha = config.alpha
        global num_vars
        initial_x = [0.5] * config.num_of_var
        initial_x = np.array(initial_x)
        rdiagx,total = self.Estep(newR, initial_x, gamma)
        xstar_val = self.Mstep(rdiagx, initial_x, total, A_mat, alpha)
        expectedRew = np.asscalar(np.transpose(R) * xstar_val)
        print "Expected Reward from EM: ", expectedRew

        #prevExpecRew = expectedRew
        while(True):
            num_iter += 1
            xstar_val = np.array(xstar_val)
            rdiagx, total = self.Estep(newR, xstar_val, gamma)
            xstar_val = self.Mstep(rdiagx, xstar_val, total, A_mat, alpha)
            expectedRew = np.asscalar(np.transpose(R)*xstar_val)
            print "Expected Reward from EM: ", expectedRew
            if (abs(dualLP - expectedRew) < delta) :
                break

        print "Optimal x: ", xstar_val
        print "Sum of x values: ", cvxpy.sum_entries(xstar_val).value
        print "Number of iterations: ", num_iter
            #prevExpecRew = expectedRew

    def Estep(self, Rcap, x, gamma):
        print "Estep: "
        rdiag = np.diag(Rcap[:,0])
        rdiagx = rdiag.dot(x)
        rdiagx = rdiagx*(1-gamma)
        total = np.sum(rdiagx)
        rdiagx = rdiagx/total
        #print np.transpose(rdiagx), total
        return rdiagx, total

    def Mstep(self, E, x, c, A_mat, alpha):
        print "Mstep: "
        xstar = cvxpy.Variable(config.num_of_var, 1)
        obj = cvxpy.Maximize(np.transpose(E)*(cvxpy.log(xstar) - cvxpy.log(x) + cvxpy.log(c)))
        cons = [A_mat*xstar == alpha, xstar>0]
        prob = cvxpy.Problem(objective=obj, constraints=cons)
        prob.solve(solver=cvxpy.ECOS, verbose=False, max_iters=100)
        #print np.transpose(xstar.value)
        return xstar.value

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
    a = MDP(config.states,config.actions)
    a.initializeActions()
    a.initializeStates()
    a.autoTransitionFunction(filename=config.tranFile)
    a.autoRewardFunction(filename=config.rewFile)
    gamma = config.gamma

    #a.generateLPAc(gamma)
    a.solveLP(gamma)
    a.EM(gamma, config.delta)
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

    gammas = [gamma]
    #print('{0:15} {1:15} {2:25} {3:15} {4:15}'.format('Gamma', 'Without Options', 'Iterations', 'With Options', 'Iterations'))
    for x in gammas:
        a.autoTransitionFunction(filename=config.tranFile, gamma=x)
        #a.lpsolve(x)
        #a.solveLP(x)
        #a.autoTransitionFunction(x)

        z, y, it = a.actionsVI(config.delta, x)
        #
        # t, u, iter = a.optionsVI(delta)
        # # print t
        # # print u
        # if u[8] == 0:
        #     strih = 'Selecting Option.'
        # else:
        #     strih = 'Selecting Action.'
        # print('{0:10f} {1:15f} {2:15d} {3:15f} {4:15d} {5:15}'.format(x, z[8], it, t[8], iter, strih))
        print "VI Solver: ", z, it
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