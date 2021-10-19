\ Test LP file
\ this test implements a quadratic problem with
\ quadratic constraint and integer variables
Minimize
 obj:  -4 x0 -9 x1 + 6 i0 + i1 +[x0*x1 + x0*x2]/2
Subject To
 c1: x2 + x1 >= 2
 c2: [x0*x2] >= 1
 c3: 3 x1 + [6 x0*x2] >= -9
 c4: 5 x0 - 9 i0 <= 1
 c5: -34 i0 + 26 i1 = 2
 c6: [i0*i1] >= 0

Bounds
 3 <= i0 <= 15
 i1 <= 10

Binary
 x0 x1 x2

General
 i0 i1

End