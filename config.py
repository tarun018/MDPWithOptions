#flag=1 fileread
flag = 1
agents = 2
collectTimes = []
transitTimes = []
T = [8]*agents
nloc = [5,5]
shared = [0,1,2]
collectTimes.append([2]*nloc[0])
collectTimes.append([4]*nloc[1])
transitTimes.append([[2]*nloc[0]]*nloc[0])
transitTimes.append([[4]*nloc[1]]*nloc[1])
gamma = 0.8
rewardCollection = [2]*agents
alpha = 0.8
