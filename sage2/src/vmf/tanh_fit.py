#! /usr/bin/env python
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

from __future__ import print_function
import numpy as np
from scipy.optimize import leastsq


def tanh(p, x):
  t = x * 2 * 1.44269504088896341
  i = np.floor(t)
  f = t - i
  c = p[0]
  for _i in range(1, len(p)):
    c = c * f + p[_i]
  c = c * np.power(2, i)
  return (c - 1) / (c + 1)


def tanh_rel_error(p, x):
  return np.abs(np.tanh(x) - tanh(p, x)) / np.abs(np.tanh(x))


x = np.linspace(-44.0, 44.0, 100000)
p0 = [
    1.3537703155e-02,
    5.2170695889e-02,
    2.4121210200e-01,
    6.9307905933e-01,
    1.0000001462e+00,
]
p, _ = leastsq(tanh_rel_error, p0, args=x)
for i in range(len(p)):
  print('%.10e,' % p[i])
for i in range(len(p)):
  print('static const float EXP_P%d = %.10ef;' % (i, p[i]))
for i in range(len(p)):
  print('EXP_P%d_V2:' % i)
  print('.float %.10e' % p[i])

e = tanh_rel_error(p, x)
print('tanh_rel_error=%e/%e/%e' % (np.max(e), np.min(e), np.mean(e)))
