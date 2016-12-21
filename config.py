import numpy as np
import mdptoolbox.example

states = 4
actions = 2
num_of_var = states*actions
#num_of_var = 3
gamma = 0.25
a = float(1.0/float(states))
print a
alpha = [a]*states
alpha = np.array(alpha)

#P, R = mdptoolbox.example.small()
P, R = mdptoolbox.example.forest(S=states,p=0.45)
# print R

# mask = np.zeros((states, states))
# P, R = mdptoolbox.example.rand(states, actions, mask=mask)
# newR = []
# for x in R:
#     for y in x:
#         newR.append([sum(y)])
# newR = np.array(newR)
# R = newR

tran = open('tdata1','w')
rew = open('rdata1','w')
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
tranFile = 'tdata1'
rewFile = 'rdata1'
R_min = mi
R_max = ma
delta = 0.00001
