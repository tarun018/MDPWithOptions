param n;
param gamma default 0.8;
set S{1..n} ordered;
set A{1..n} ordered;
param P{i in 1..n, j in 1..card(A[i]), k in 1..card(S[i]), l in 1..card(S[i])};
param R{i in 1..n, j in 1..card(S[i]), k in 1..card(A[i])};
param alpha{i in 1..n, j in 1..card(S[i])};

param numcons;
param numprims;
param numevents;
param primitives{i in 1..numprims, j in 0..3};
param creward{i in 1..numcons};
set events{1..numevents} ordered;
set cons{1..numcons} ordered;

var x{i in 1..n, j in 1..card(S[i]), k in 1..card(A[i])};
var z{i in 1..numcons, j in cons[i]};


maximize ER: 
sum{i in 1..n, j in 1..card(S[i]), k in 1..card(A[i])} x[i,j,k]*R[i,j,k] +
sum{l in 1..numcons} creward[l] * prod{m in cons[l]} z[l,m];

subject to Flow {i in 1..n, j in 1..card(S[i])}:
sum{k in 1..card(A[i])} x[i,j,k] - 
gamma * sum{l in 1..card(S[i]), m in 1..card(A[i])} x[i,l,m]*P[i,m,l,j] = alpha[i,j];
subject to Positive {i in 1..n, j in 1..card(S[i]), k in 1..card(A[i])}: x[i,j,k] >= 0;
subject to defineZ{i in 1..numcons, j in cons[i]}:
z[i,j] = sum{k in events[j]} x[primitives[k,0],primitives[k,1],primitives[k,2]]*
P[primitives[k,0], primitives[k,2], primitives[k,1], primitives[k,3]];
