#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

from george_ess import GP
from george import kernels


def lnlike(y):
    return -0.5 * np.sum(y ** 2)


x = np.random.randn(100)
gp = GP(0.1 * kernels.ExpSquaredKernel(3))
gp.compute(x, 1e-6)

# for y, ll in gp.elliptical_slice_sampling(lnlike, nstep=100):
#     print(ll)

for y, h, lp, acc in gp.elliptical_slice_sampling(lnlike, nstep=1000,
                                                  hyper_update=5,
                                                  stepsize=0.2):
    print(h, acc)

print(gp.kernel.pars)

pl.plot(x, y, ".k")
pl.savefig("demo.png")
