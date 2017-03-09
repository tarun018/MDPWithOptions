#flag=1 fileread
flag = 0
agents = 2
collectTimes = []
transitTimes = []
T = [4]*agents
nloc = [2,2]
shared = [0,1]
collectTimes.append([2]*nloc[0])
collectTimes.append([2]*nloc[1])
transitTimes.append([[2]*nloc[0]]*nloc[0])
transitTimes.append([[2]*nloc[1]]*nloc[1])
gamma = 0.8
rewardCollection = [2]*agents
alpha = 0.8
delta = 0.001
R_min = [0,0]
R_max = [2,2]
#Rmax Rmin in normalizing cks