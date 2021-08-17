\ Test LP file
\ this test implements a problem with binary and integer variable with
\ quadratic objective
Minimize
 obj: [x0^2 + i0^2]/2
Subject To
 c1: [x0^2] >= 1
 c2: [i0^2] >= 25

Binary
 x0

General
 i0

End