import random, csv
import pickle
import math
#flag=1 fileread
solver = 'minos'
flag = 1

timetorunsec = 5

experiment = 1002

offset = 500
GenRun = 1
workDir = "/home/tarun/PycharmProjects/MDPWithOptions/"

theta = 0.1
gamma = 0.8
initialxval = 0.1
alpha = 0.8
delta = 0.00001
print "theta: ", theta
print "gamma: ", gamma
print "initialx: ", initialxval
print "alpha: ", alpha
print "delta: ", delta

if flag == 0:

    agents = 2
    nPrivatePerAgent = 1
    nShared = 1
    minSharing = 2
    maxSharing = 2
    minT = 6
    maxT = 6
    minTaction = 3
    maxTaction = 3
    agentMax = [math.ceil(float(nShared*maxSharing)/float(agents))]*agents
    nLocs = (agents*nPrivatePerAgent) + nShared
    auction = [-1]*nLocs
    locs = []
    sharedSites = []
    for i in xrange(0, agents):
        lst = []
        for j in xrange(0, nPrivatePerAgent):
            num = random.randint(0, nLocs-1)
            while auction[num] != -1:
                num = random.randint(0, nLocs - 1)
            auction[num] = i
            lst.append(num)
        locs.append(lst)

    allowedvalues = list(range(0, agents))
    for i in xrange(0, nLocs):
        if auction[i] != -1:
            continue
        tobesharedbetween = random.randint(minSharing, maxSharing)
        setOfAgents = set()
        while len(setOfAgents) < tobesharedbetween:
            selectedAgent = random.choice(allowedvalues)
            setOfAgents.add(selectedAgent)
        for vals in list(setOfAgents):
            agentMax[vals] -= 1
            if agentMax[vals] == 0:
                allowedvalues.remove(vals)
        auction[i] = list(setOfAgents)
        for j in list(setOfAgents):
            locs[j].append(i)
        sharedSites.append(i)

    for vals in locs:
        vals = vals.sort()

    print "Agents: ", agents
    print "PrivatePer: ", nPrivatePerAgent
    print "nShared: ", nShared
    print "nLocs", nLocs
    print "Auctioned: ", auction
    print "AgentWise: ", locs
    print "SharedSites: ", sharedSites

    collectTimes = []
    transitTimes = []
    totalPow = random.randint(minT, maxT)
    T = [2**totalPow]*agents

    nloc = [len(lo) for lo in locs]
    for i in xrange(0, agents):
        t = 2**random.randint(minTaction, maxTaction)
        collectTimes.append([t] * nloc[i])
        transitTimes.append([[t] * nloc[i]] * nloc[i])

    print "NoLocPerAgent: ", nloc
    print "TotalTime: ", T
    print "Collect: ", collectTimes
    print "Transit: ", transitTimes

    rewardCollection = []
    creward = []
    rmi = 5
    rma = 10
    for i in xrange(0, agents):
        lst = []
        for j in xrange(0, nloc[i]):
            rew = random.randint(rmi, rma)
            lst.append(rew)
        rewardCollection.append(lst)

    for x in sharedSites:
        rew = random.randint(rma, 1.5*rma)
        creward.append(rew)

    print "MDPRew: ", rewardCollection
    print "ConsReward: ", creward

    R_min = min(creward)
    for i in xrange(0, agents):
        mm = min(rewardCollection[i])
        if mm < R_min:
            R_min = mm
    R_min -= 0.0
    R_max = max(creward)
    for i in xrange(0, agents):
        mm = max(rewardCollection[i])
        if R_max < mm:
            R_max = mm
    R_max += 0.0
    print "Rmin: ", R_min
    print "Rmax: ", R_max

    def writeConfig1():
        with open(workDir+'Data/objs'+str(experiment)+'.pickle', 'w') as f:
            pickle.dump([agents, nPrivatePerAgent, nShared, nLocs, auction, locs, sharedSites, nloc, T, collectTimes, transitTimes,
                         rewardCollection, creward, R_min, R_max ], f)
    writeConfig1()

else:
    with open(workDir+'Data/objs'+str(experiment)+'.pickle') as f:  # Python 3: open(..., 'rb')
        agents, nPrivatePerAgent, nShared, nLocs, auction, locs, sharedSites, nloc, T, collectTimes, transitTimes, \
        rewardCollection, creward, R_min, R_max = pickle.load(f)

        print "Agents: ", agents
        print "PrivatePer: ", nPrivatePerAgent
        print "nShared: ", nShared
        print "nLocs", nLocs
        print "Auctioned: ", auction
        print "AgentWise: ", locs
        print "SharedSites: ", sharedSites
        print "NoLocPerAgent: ", nloc
        print "TotalTime: ", T
        print "Collect: ", collectTimes
        print "Transit: ", transitTimes
        print "MDPRew: ", rewardCollection
        print "ConsReward: ", creward
        print "Rmin: ", R_min
        print "Rmax: ", R_max
