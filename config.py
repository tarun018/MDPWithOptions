import numpy as np
import mdptoolbox.example

states = 2
actions = 2
num_of_var = states*actions
#num_of_var = 3
gamma = 0.95
alpha = [0.5]*states
alpha = np.array(alpha)

P, R = mdptoolbox.example.small()
#P, R = mdptoolbox.example.forest()
tran = open('tdata1','w')
rew = open('rdata1','w')
for x in P:
    for y in x:
        tran.write(','.join(map(str, y)))
        tran.write("\n")
    tran.write("\n")
tran.close()
for x in R:
    rew.write(','.join(map(str, x)))
    rew.write("\n")
rew.close()
tranFile = 'tdata1'
rewFile = 'rdata1'
R_min = -1
R_max = 10
delta = 0.00001
