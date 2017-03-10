#flag=1 fileread
flag = 1
agents = 3
collectTimes = []
transitTimes = []
T = [4]*agents
nloc = [3,3,4]
shared = [[0,1], [0,1,2], [1,2]]
nc = len(shared)
creward = [0,0,0]
collectTimes.append([2]*nloc[0])
collectTimes.append([2]*nloc[1])
collectTimes.append([2]*nloc[2])
transitTimes.append([[2]*nloc[0]]*nloc[0])
transitTimes.append([[2]*nloc[1]]*nloc[1])
transitTimes.append([[2]*nloc[2]]*nloc[2])
gamma = 0.8
rewardCollection = []
rewardCollection.append([1, 1, 1])
rewardCollection.append([1.5, 1.5, 1.5])
rewardCollection.append([2, 2, 2, 2])
alpha = 0.8
delta = 0.001
R_min = min(creward)
R_max = 0
for i in xrange(0, agents):
    mm = max(rewardCollection[i])
    if R_max < mm:
        R_max = mm
#Rmax Rmin in normalizing cks