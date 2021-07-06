#! /usr/bin/env python
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

from __future__ import print_function
import numpy as np
from scipy.optimize import leastsq


def exp(p, x):
  t = x * 1.44269504088896341
  i = np.floor(t)
  f = t - i
  c = p[0]
  for _i in range(1, len(p)):
    c = c * f + p[_i]
  return c * np.power(2, i)


def exp_rel_error(p, x):
  t = x * 1.44269504088896341
  i = np.floor(t)
  return np.abs(np.exp(x) - exp(p, x)) / np.abs(np.exp(x))


x = np.linspace(-88, 88, 100000)
p0 = [
    8.2886153525e-14,
    7.7822959126e-02,
    2.2586729288e-01,
    6.9617327373e-01,
    9.9986347636e-01,
]
p, _ = leastsq(exp_rel_error, p0, args=x)
for i in range(len(p)):
  print('%.10e,' % p[i])
for i in range(len(p)):
  print('static const float EXP_P%d = %.10ef;' % (i, p[i]))
for i in range(len(p)):
  print('EXP_P%d_V2:' % i)
  print('.float %.10e' % p[i])

e = exp_rel_error(p, x)
print('exp_rel_error=%e/%e/%e' % (np.max(e), np.min(e), np.mean(e)))
