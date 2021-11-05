\ Test LP file
\ this test implements a quadratic problem with linear constraint
Minimize
 obj: [x0*x1 + x0*x2 + x0*x3 + x1*x2 + x1*x3 + x2*x3] / 2
Subject To
 c1: x0 + x3 <= 1
 c2: x2 + x1 >= 2
Binary
 x0 x1 x2 x3

End