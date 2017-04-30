import random, csv
import pickle
#flag=1 fileread
solver = 'minos'
flag = 1

experiment = 1

theta = 0.4
gamma = 0.8
initialxval = 0.01
alpha = 0.8
delta = 0.000001
print "theta: ", theta
print "gamma: ", gamma
print "initialx: ", initialxval
print "alpha: ", alpha
print "delta: ", delta

if flag == 0:

    agents = 3
    nPrivatePerAgent = 2
    nShared = 2
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

    for i in xrange(0, nLocs):
        if auction[i] != -1:
            continue
        tobesharedbetween = random.randint(2, agents)
        setOfAgents = set()
        while len(setOfAgents) < tobesharedbetween:
            setOfAgents.add(random.randint(0, agents-1))
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
    totalPow = random.randint(4,6)
    T = [2**totalPow]*agents

    nloc = [len(lo) for lo in locs]
    for i in xrange(0, agents):
        t = 2**random.randint(totalPow-2, totalPow)
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
        with open('../Data/objs'+str(experiment)+'.pickle', 'w') as f:
            pickle.dump([agents, nPrivatePerAgent, nShared, nLocs, auction, locs, sharedSites, nloc, T, collectTimes, transitTimes,
                         rewardCollection, creward, R_min, R_max ], f)
    writeConfig1()

else:
    with open('../Data/objs'+str(experiment)+'.pickle') as f:  # Python 3: open(..., 'rb')
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
