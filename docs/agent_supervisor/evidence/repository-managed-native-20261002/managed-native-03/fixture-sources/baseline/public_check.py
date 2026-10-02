from calc import increment
values = [increment(n) for n in [-2,-1,0,1,2]]
assert all(type(v) is int for v in values)
assert values == [0,1,2,3,4], values
