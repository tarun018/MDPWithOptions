#flag=1 fileread
flag = 1
agents = 3
collectTimes = []
transitTimes = []
T = [8]*agents
nloc = [3,3,3]
shared = [[0,1,2],[0,1]]
nc = len(shared)
creward = [-1,-1,-1]
collectTimes.append([2]*nloc[0])
collectTimes.append([2]*nloc[1])
collectTimes.append([2]*nloc[2])
transitTimes.append([[2]*nloc[0]]*nloc[0])
transitTimes.append([[2]*nloc[1]]*nloc[1])
transitTimes.append([[2]*nloc[2]]*nloc[2])
gamma = 0.8
rewardCollection = []
rewardCollection.append([1, 1, 1, 1, 1])
rewardCollection.append([2, 2, 2, 2, 2])
rewardCollection.append([3, 3, 3, 3, 3])
alpha = 0.8
delta = 0.01
R_min = min(creward)
R_max = 0
for i in xrange(0, agents):
    mm = max(rewardCollection[i])
    if R_max < mm:
        R_max = mm
R_max -= 0.0
#Rmax Rmin in normalizing cks