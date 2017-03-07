param n;
param cardS;
param cardA;
param gamma default 0.9;

param num_cons;
param num_prim_events;
param num_events;
param num_events_one_cons;
param num_prim_events_one_event;

set N := 1..n;
set cS := 1..cardS;
set cA := 1..cardA;

#param S{i in N, j in cS};
#param A{i in N, j in cA};
param P{i in N, j in cA, k in cS, l in cS};
param R{i in N, j in cS, k in cA};
param alpha{i in N, j in cS};

param prim_event{i in 1..num_prim_events, j in 1..4};
param event{i in 1..num_events, j in 1..num_prim_events_one_event};
param cons{i in 1..num_cons, j in 1..num_events_one_cons};
param creward{i in 1..num_cons, j in 1..1};

var x{i in N, j in cS, k in cA};
var z{i in 1..num_cons, j in 1..num_events_one_cons};

maximize EReward: 
(sum{i in N, j in cS, k in cA} x[i,j,k]*R[i,j,k]) + sum{l in 1..num_cons} (creward[l,1] * prod{m in 1..num_events_one_cons} z[l,m]);
subject to Flow {i in N, j in cS} : 
sum{k in cA} x[i,j,k] - gamma * sum {l in cS, m in cA} x[i,l,m]*P[i,m,l,j] = alpha[i,j];
subject to Positive {i in N, j in cS, k in cA}: x[i,j,k] >= 0;
#subject to DefineZ {i in 1..num_cons, j in 1..num_events_one_cons} : 
#z[i,j] = sum{k in 1..num_prim_events_one_event} x[prim_event[event[cons[i,j], k], 1], prim_event[event[cons[i,j], k], 2], prim_event[event[cons[i,j], k], 3]] * P[prim_event[event[cons[i,j], k], 1], prim_event[event[cons[i,j], k], 3], prim_event[event[cons[i,j], k], 2], prim_event[event[cons[i,j], k], 4]];

