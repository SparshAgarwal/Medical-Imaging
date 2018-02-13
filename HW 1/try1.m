new = normrnd(0,1,[4 4])
new_mod = new( new>=0)
whos new_mod

[i,j,v] = find(new>0) 
new(new > 0) = 1;
new(new < 0) = 0

