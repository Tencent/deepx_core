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
  x = x + i * -0.693359375
  x = x + i * 2.12194440e-4
  b = x * x
  c = p[0]
  for _i in range(1, len(p)):
    c = c * x + p[_i]
  c = c * b + x + 1
  return c * np.power(2, i)


def exp_rel_error(p, x):
  t = x * 1.44269504088896341
  i = np.floor(t)
  return np.abs(np.exp(x) - exp(p, x)) / np.abs(np.exp(x))


x = np.linspace(-88, 88, 100000)
p0 = [
    2.7565422393e-04,
    1.3038713518e-03,
    8.3795212816e-03,
    4.1653515712e-02,
    1.6666851064e-01,
    4.9999990238e-01,
]
p, _ = leastsq(exp_rel_error, p0, args=x)
for i in range(len(p)):
  print('%.10e,' % p[i])
for i in range(len(p)):
  print('static const float EXP_P%d = %.10ef;' % (i, p[i]))
for i in range(len(p)):
  print('EXP_P%d_V1:' % i)
  print('.float %.10e' % p[i])

e = exp_rel_error(p, x)
print('exp_rel_error=%e/%e/%e' % (np.max(e), np.min(e), np.mean(e)))
