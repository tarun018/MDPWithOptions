param n;
param cardS1;
param cardA1;
param cardS2;
param cardA2;
param gamma default 0.8;

param num_cons;
param num_prim_events;
param num_events;
param num_events_one_cons;
param num_prim_events_one_event;

set cS1 := 1..cardS1;
set cA1 := 1..cardA1;
set cS2 := 1..cardS2;
set cA2 := 1..cardA2;

param P1{j in cA1, k in cS1, l in cS1};
param R1{j in cS1, k in cA1};
param alpha1{j in cS1};
param P2{j in cA2, k in cS2, l in cS2};
param R2{j in cS2, k in cA2};
param alpha2{j in cS2};

param prim_event{i in 1..num_prim_events, j in 1..4};
param event{i in 1..num_events, j in 1..num_prim_events_one_event};
param cons{i in 1..num_cons, j in 1..num_events_one_cons};
param creward{i in 1..num_cons, j in 1..1};

var x1{j in cS1, k in cA1};
var x2{j in cS2, k in cA2};

var z{i in 1..num_cons, j in 1..num_events_one_cons};

maximize EReward: 
(sum{j in cS1, k in cA1} x1[j,k]*R1[j,k]) + sum{j in cS2, k in cA2} x2[j,k]*R2[j,k]) +

sum{l in 1..num_cons} (creward[l,1] * prod{m in 1..n} z[l,m]);
subject to Flow {i in N, j in cS} : 
sum{k in cA} x[i,j,k] - gamma * sum {l in cS, m in cA} x[i,l,m]*P[i,m,l,j] = alpha[i,j];
subject to Positive {i in N, j in cS, k in cA}: x[i,j,k] >= 0;
subject to DefineZ {i in 1..num_cons, j in 1..num_events_one_cons} : 
z[i,j] = sum{k in 1..num_prim_events_one_event} x[prim_event[event[cons[i,j], k], 1], prim_event[event[cons[i,j], k], 2], prim_event[event[cons[i,j], k], 3]] * P[prim_event[event[cons[i,j], k], 1], prim_event[event[cons[i,j], k], 3], prim_event[event[cons[i,j], k], 2], prim_event[event[cons[i,j], k], 4]];
