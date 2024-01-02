from copy import copy

a = 1
b = [copy(a)]
a = 2
# b[0] = 0
print('b', b)
print('a', a)
