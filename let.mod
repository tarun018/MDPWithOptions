set S;
set events{S} ordered;

data;
set S := 1 2;
set events[1] := 1 2 ;
set events[2] := 2 3 ;