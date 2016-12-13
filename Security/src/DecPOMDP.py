class Agent:
    def __init__(self, states, actions, observations):
        self.States = states
        self.Actions = actions
        self.Observations = observations


class Action:
    def __init__(self, index, name):
        self.name = name
        self.index = index

    def __repr__(self):
        return self.name


class State:
    def __init__(self, index, name):
        self.name = name
        self.index = index

    def __repr__(self):
        return self.name



class Observation:
    def __init__(self, index, name):
        self.name = name
        self.index = index

    def __repr__(self):
        return self.name

class Utils:

    def obsExtended(self, ext1, a, o):
        return self.getIndObservationProb(ext1[0], a, o)

    def getIndObservationProb(self, sd, a, o):
        if a[0].index==3 and a[1].index==3:
            if sd.index == 1:
                if o.index==1:
                    return 0.85
                elif o.index==2:
                    return 0.15
            elif sd.index==2:
                if o.index == 1:
                    return 0.15
                elif o.index == 2:
                    return 0.85
        else:
            return 0.5

    def getTransitionProb(self, s, a, sd):
        if a[0].index == 3 and a[1].index == 3:
            if s.index==1 and sd.index==1:
                return 1.0
            elif s.index==1 and sd.index==2:
                return 0.0
            elif s.index==2 and sd.index==2:
                return 1.0
            elif s.index==2 and sd.index==1:
                return 0.0
        else:
            return 0.5


    def transitionExtended(self, ext, ext1, a1):
        return (self.getTransitionProb(ext[0], (a1, ext[1][0][0]), ext1[0]) * self.getIndObservationProb(ext1[0],
                                                                                                     (a1, ext[1][0][0]),
                                                                                                     ext1[1][0][1]))
    def probac(self, a2, obhistory):
        if a2.index == 3:
            return 1.0
        else:
            return 0.0

class DecPOMDP:
    States = []
    States.append(State(1, 'Left'))
    States.append(State(2, 'Right'))

    Actions = []
    Actions.append(Action(1, 'OpenLeft'))
    Actions.append(Action(2, 'OpenRight'))
    Actions.append(Action(3, 'Listen'))

    Observations = []
    Observations.append(Observation(1, 'HearLeft'))
    Observations.append(Observation(2, 'HearRight'))

    Agents = []
    Agents.append(Agent(States, Actions, Observations))
    Agents.append(Agent(States, Actions, Observations))

    JointActions = []
    JointActions.append((Actions[0], Actions[0]))
    JointActions.append((Actions[0], Actions[1]))
    JointActions.append((Actions[0], Actions[2]))

    JointActions.append((Actions[1], Actions[0]))
    JointActions.append((Actions[1], Actions[1]))
    JointActions.append((Actions[1], Actions[2]))

    JointActions.append((Actions[2], Actions[0]))
    JointActions.append((Actions[2], Actions[1]))
    JointActions.append((Actions[2], Actions[2]))

    JointObservations = []
    JointObservations.append((Observations[0], Observations[0]))
    JointObservations.append((Observations[0], Observations[1]))

    JointObservations.append((Observations[1], Observations[0]))
    JointObservations.append((Observations[1], Observations[1]))

    actobsHistory1 = []
    actobsHistory2 = []

    extendedStates = []
    extendedStates.append((States[0], actobsHistory2))
    extendedStates.append((States[1], actobsHistory2))
    oldExtended = extendedStates

    ut = Utils()

    print "Initial Extended States: ", extendedStates[0], extendedStates[1]

    belief = []
    belief.append((extendedStates[0], 0.5))
    belief.append((extendedStates[1], 0.5))

    print "Initial Belief State: ", belief[0], belief[1]

# ------------------------------------------------------------
    print "Agent 1 takes action Listen and Observes HearLeft"
    print
    # Agent 1 takes action Listen and observes HearLeft.
    a1 = Actions[2]
    w1 = Observations[0]
    actobsHistory1.append((a1, w1))

    extendedStates = []
    for s in States:
        for a in Actions:
            for o in Observations:
                actobsHistory2.append((a, o))
                ext = (s, actobsHistory2)
                extendedStates.append(ext)
                #print "New Extended State: ", ext
                actobsHistory2 = []

    for ex in extendedStates:
        for ex1 in extendedStates:
            # pass
            print ex, '-> ', ex1
            print ut.transitionExtended(ex, ex1, a1)

    newbelief = []
    over = 0
    for ex in extendedStates:
        sum = 0
        for b in belief:
            sum += (b[1] * ut.getTransitionProb(b[0][0], (a1, ex[1][0][0]), ex[0]))
        sum *= ut.getIndObservationProb(ex[0], (a1, ex[1][0][0]), ex[1][0][1])
        sum *= ut.getIndObservationProb(ex[0], (a1, ex[1][0][0]), w1)
        sum *= ut.probac(ex[1][0][0], actobsHistory2)
        newbelief.append((ex, sum))
        over += sum

    #print over
    finalnew = []
    print
    print "New Belief States: "
    for x in newbelief:
        finalnew.append((x[0], x[1]/over))
        print "New Belief of ", x[0], ' is -> ', x[1]/over


#--------------------------------------------------------
    print
    print "Agent 1 takes action Listen and Observes HearRight"
    # Agent 1 takes action Listen and observes HearLeft.
    a1 = Actions[2]
    w1 = Observations[1]
    actobsHistory1.append((a1, w1))

    newbelief = []
    over = 0
    for ex in extendedStates:
        sum = 0
        for b in belief:
            sum += (b[1] * ut.getTransitionProb(b[0][0], (a1, ex[1][0][0]), ex[0]))
        sum *= ut.getIndObservationProb(ex[0], (a1, ex[1][0][0]), ex[1][0][1])
        sum *= ut.getIndObservationProb(ex[0], (a1, ex[1][0][0]), w1)
        sum *= ut.probac(ex[1][0][0], actobsHistory2)
        newbelief.append((ex, sum))
        over += sum

    #print over
    finalnew = []
    print
    print "New Belief States: "
    for x in newbelief:
        finalnew.append((x[0], x[1]/over))
        print "New Belief of ", x[0], ' is -> ', x[1]/over
