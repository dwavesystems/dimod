\ Test LP file
\ this test implements a quadratic problem with pure quadratic variable on the objective
Minimize
 obj: x0 + [x0^2]/2

Subject To
c1: x0 + [x0^2] <= 1

Binary
 x0

End