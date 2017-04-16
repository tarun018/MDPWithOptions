#flag=1 fileread
solver = 'ipopt'
flag = 1
agents = 5
collectTimes = []
transitTimes = []
T = [8]*agents
nloc = [3]*agents
shared = [[0],[1,4],[2,3]]
nc = len(shared)
creward = [1,1,1]
theta = 0.00001
collectTimes.append([4]*nloc[0])
collectTimes.append([8]*nloc[1])
collectTimes.append([8]*nloc[2])
collectTimes.append([8]*nloc[3])
collectTimes.append([8]*nloc[4])
transitTimes.append([[4]*nloc[0]]*nloc[0])
transitTimes.append([[8]*nloc[1]]*nloc[1])
transitTimes.append([[8]*nloc[2]]*nloc[2])
transitTimes.append([[8]*nloc[3]]*nloc[3])
transitTimes.append([[8]*nloc[4]]*nloc[4])
gamma = 0.8
rewardCollection = []
rewardCollection.append([1,3,1])
rewardCollection.append([4,4,5])
rewardCollection.append([3,4,2])
rewardCollection.append([3,4,2])
rewardCollection.append([3,4,2])
initialxval = 0.00001
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