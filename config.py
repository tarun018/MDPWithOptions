#flag=1 fileread
flag = 0
agents = 1
collectTimes = []
transitTimes = []
T = [8]*agents
nloc = [3]*agents
shared = [[0],[0],[0]]
nc = len(shared)
creward = [1,0,1]
theta = 0.00001
collectTimes.append([4]*nloc[0])
#collectTimes.append([8]*nloc[1])
#collectTimes.append([8]*nloc[2])
transitTimes.append([[4]*nloc[0]]*nloc[0])
#transitTimes.append([[8]*nloc[1]]*nloc[1])
#transitTimes.append([[8]*nloc[2]]*nloc[2])
gamma = 0.8
rewardCollection = []
rewardCollection.append([1,3,7])
#rewardCollection.append([4,3,6])
#rewardCollection.append([3,4,2])
alpha = 0.8
delta = 0.000001
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
print R_min, R_max
#Rmax Rmin in normalizing cks