import numpy as np
import mdptoolbox.example

agents = 4
const = 4
R_min = []
R_max = []
tranFile = []
rewFile = []
delta = 0.001
gamma = 0.5
states = 4
actions = 2
num_of_var = states * actions
a = float(1.0 / float(states))

alpha = [a] * states
alpha = np.array(alpha)

for i in xrange(0,agents):

    #P, R = mdptoolbox.example.small()
    # P, R = mdptoolbox.example.forest(S=states,p=0.45)
    # print R

    mask = np.ones((actions, states, states))
    P, R = mdptoolbox.example.rand(states, actions, mask=mask)
    newR = []
    for x in R:
        sums = np.sum(x, axis=1)
        newR.append(sums)
    newR = np.array(newR)
    newR = np.transpose(newR)
    R = newR

    tran = open('tdata'+str(i),'w')
    rew = open('rdata'+str(i),'w')
    for x in P:
        for y in x:
            tran.write(','.join(map(str, y)))
            tran.write("\n")
        tran.write("\n")
    tran.close()

    mi = float('INF')
    ma = float('-INF')
    for x in R:
        mini = min(x)
        if mini < mi:
            mi = mini

        maxi = max(x)
        if maxi > ma:
            ma = maxi

        rew.write(','.join(map(str, x)))
        rew.write("\n")
    rew.close()

    tranFile.append('tdata'+str(i))
    rewFile.append('rdata'+str(i))
    R_min.append(mi)
    R_max.append(ma)