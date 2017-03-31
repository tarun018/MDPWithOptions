#flag=1 fileread
flag = 0
agents = 1
collectTimes = []
transitTimes = []
T = [8]*agents
nloc = [3,3]
shared = [[0],[0],[0]]
nc = len(shared)
creward = [-5,-2,-1]
thetahat = 0.5
collectTimes.append([4]*nloc[0])
collectTimes.append([2]*nloc[1])
transitTimes.append([[4]*nloc[0]]*nloc[0])
transitTimes.append([[2]*nloc[1]]*nloc[1])
gamma = 0.8
rewardCollection = []
rewardCollection.append([3,3,3])
rewardCollection.append([4,4,4])
#rewardCollection.append([10, 5, 9])
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
R_max -= 0.0
#print R_min, R_max
#Rmax Rmin in normalizing cks