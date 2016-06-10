import csv

class State:

    def __init__(self, ind, name, actions):
        self.index = ind
        self.name = name
        self.possibleActions = actions
        self.transition = []
        self.reward = []

    def __str__(self):
        print "Index: " + str(self.index) + " Name: " + self.name + " Actions: " + str(self.possibleActions)

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

class Action:

    def __init__(self, ind, name):
        self.index = ind
        self.name = name

    def __str__(self):
        print "Index: " + str(self.index) + " Name: " + self.name

    def getIndex(self):
        return self.index


class MDP:

    def __init__(self, numberOfStates, numberOfActions):
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
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
        self.states[3].modifyActions([])

    # TransitionFunction For Actions
    def autoTransitionFunction(self):
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
                    triple = (actionIndex, sp, float(row[sp]))
                    self.states[stateIndex].getTransition().append(triple)
                stateIndex += 1


    # RewardFunctions For Actions
    def autoRewardFunction(self):
        tosend = []
        stateIndex = 0
        with open('rewardData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                for ap in xrange(0, self.numberOfActions):
                    triple = (ap, float(row[ap]))
                    tosend.append(triple)
                self.states[stateIndex].setReward(tosend)
                tosend = []
                stateIndex += 1

    def setOptions(self):
        optionIndex = 0
        actionIndex = 0
        counter = 0
        tosend = []
        with open('OptionsData', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                counter += 1

                if len(row) == 0:
                    o.setPolicy(tosend)
                    optionIndex = optionIndex + 1
                    counter = 0
                    actionIndex = 0
                    continue

                if counter == 1:
                    o = Option(optionIndex)
                    self.options.append(o)
                    #Initiation
                    # for x in xrange(0, len(row)):
                    #     pass
                    initset = []
                    for x in row:
                        initset.append(float(x))
                    o.setInitiationSet(initset)

                elif counter == 2:
                    #Beta
                    o.setBeta(row)

                elif counter >= 3:
                    #Policy
                    stateIndex = 0
                    for st in self.states:
                        triple = (st.getIndex(), actionIndex, float(row[st.getIndex()]))
                        tosend.append(triple)
                        stateIndex += 1
                    actionIndex += 1
            o.setPolicy(tosend)


    def calculateRewardForOption(self, state, option):

        sums = 0
        print "State: " + str(state)
        print "Option: " + str(option)
        if state not in self.options[option].getInitiationSet():
            print "Not in Initiation Set."
            return  0

        actionsavail = self.states[state].getPossibleActions()

        if len(actionsavail) == 0:
            return 0

        for act in actionsavail:

            print
            print "State: " + str(state)
            print "Action: " + str(act.getIndex())
            sumForAction = 0
            probOfAction = 0
            immReward = 0
            possibleStates = []
            actionSet = self.options[option].getPolicy()

            for x in actionSet:
                if x[0] == state and x[1] == act.getIndex():
                    probOfAction = x[2]
                    break

            print "Prob Of Action: " + str(probOfAction)

            # if probOfAction == 0:
            #     avail = self.states[state].getPossibleActions()
            #     avail = filter(lambda a: a != act.getIndex(), avail)
            #     self.states[state].setPossibleActions(avail)
            #     continue

            rewards = self.states[state].getReward()
            for x in rewards:
                if x[0] == act.getIndex():
                    immReward = x[1]
                    break

            print "Imm Reward: " + str(immReward)

            if probOfAction != 0:

                transitions = self.states[state].getTransition()
                for x in transitions:
                    if x[0] == act.getIndex() and x[2] != 0:
                        possibleStates.append((x[1], x[2]))

                for x in possibleStates:

                    product = 0
                    if len(possibleStates) == 0:
                        product = 0
                        break

                    print "Possible State: " + str(x)


                    sdash = self.states[x[0]]
                    prob = float(x[1])

                    print "Sdash: " +str(sdash.getIndex())
                    print "prob: " + str(prob)

                    beta_sdash = float(self.options[option].getBeta()[x[0]])

                    print "Beta_sdash: " + str(1-beta_sdash)

                    if prob != 0 and (1-beta_sdash) != 0:
                        product = prob * (1 - float(beta_sdash)) * self.calculateRewardForOption(sdash.getIndex(), option)

                    print "Product for this state: " + str(product)
                    sumForAction += product


                print "Sum for Action: " + str(sumForAction)

                sumForAction += immReward

                sumForAction *= probOfAction

                print "Final Sum For Action: " + str(sumForAction)
                print

            else:
                print "Prob of action is 0."
                sumForAction = 0

            # print immReward
            # print probOfAction
            # if sumForAction <= 0:
            #     avail = self.states[state].getPossibleActions()
            #     avail.remove(act.getIndex())
            #     self.states[state].setPossibleActions(avail)

            sums += sumForAction
            print "Sums: " + str(sums)
            print

        return sums


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

class Driver:
    a = MDP(4, 2)
    a.initializeActions()
    a.initializeStates()
    a.autoTransitionFunction()
    a.autoRewardFunction()
    a.setOptions()
    print a.calculateRewardForOption(0,0)