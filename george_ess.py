# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import math
import logging
import numpy as np

# Total hack to allow install without George installed.
try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False
if __GEORGE_SETUP__:
    _GP = object
else:
    from george import GP as _GP

__version__ = "0.0.1"


class GP(_GP):

    def _ess_step(self, lnlike, xx, ll0, angle_range=None, maxiter=np.inf,
                  quiet=False):
        # Make sure that the factorization is up to date.
        self.recompute()

        # Change of variables for non-zero mean.
        mu = self.mean(self._x)
        xx -= mu

        # Set up the ellipse and the slice threshold.
        D = len(xx)
        nu = np.dot(self._factor[0], np.random.randn(D))
        hh = ll0 + math.log(np.random.rand())

        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range is None or angle_range <= 0.:
            # Bracket whole ellipse with both edges at first proposed point.
            phi = np.random.rand() * 2. * math.pi
            phi_min = phi - 2. * math.pi
            phi_max = phi
        else:
            # Randomly center bracket on current point
            phi_min = -angle_range * np.random.rand()
            phi_max = phi_min + angle_range
            phi = np.random.uniform(phi_min, phi_max)

        i = 0
        while True:
            xx_prop = xx * math.cos(phi) + nu * math.sin(phi)
            cur_log_like = lnlike(xx_prop + mu)
            if cur_log_like > hh:
                # New point is on slice.
                break

            # Shrink slice to rejected point.
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError("Shrunk to current position and still not "
                                   "acceptable.")

            # Propose new angle difference
            phi = np.random.uniform(phi_min, phi_max)
            phi = np.random.rand() * (phi_max - phi_min) + phi_min

            # Stopping condition.
            i += 1
            if i >= maxiter:
                msg = ("Elliptical slice sampling didn't converge in the "
                       "maximum number of allowed iterations. Try increasing "
                       "maxiter")
                if quiet:
                    logging.warn(msg)
                    break
                raise RuntimeError(msg)

        return xx_prop + mu, cur_log_like

    def _metropolis_step(self, w, step):
        # Compute the initial vector and ln-likelihood.
        v0 = np.append(self.mean.vector, self.kernel.vector)
        lp0 = self.lnlikelihood(w, quiet=True)
        if not np.isfinite(lp0):
            raise RuntimeError("The initial ln-likelihood in the Metropolis "
                               "step has zero probability.")

        # Propose a new position and compute the updated ln-likelihood.
        q = v0 + step * np.random.randn(len(v0))
        self.mean.vector = q[:len(self.mean)]
        self.kernel.vector = q[len(self.mean):]
        lp1 = self.lnlikelihood(w, quiet=True)

        # Accept or reject the update.
        diff = lp1 - lp0
        if np.isfinite(lp1) and np.exp(diff) >= np.random.rand():
            return q, lp1, True

        # The step was rejected, revert the parameters.
        self.mean.vector = v0[:len(self.mean)]
        self.kernel.vector = v0[len(self.mean):]
        return v0, lp0, False

    def elliptical_slice_sampling(self, lnlike, y, nstep=np.inf,
                                  angle_range=None, maxiter=np.inf,
                                  hyper_update=0, step=0.1):
        ll = lnlike(y)
        step = 0
        accepted, total = 1, 1
        while True:
            # Do an iteration of elliptical slice sampling.
            y, ll = self._ess_step(lnlike, y, ll, angle_range=angle_range,
                                   maxiter=maxiter)

            # Update the hyperparameters using a Metropolis step if requested.
            if hyper_update > 0:
                if step % hyper_update == 0:
                    h, lp, a = self._metropolis_step(y, step)
                    accepted += int(a)
                    total += 1
                yield y, h, ll + lp, accepted / total
            else:
                yield y, ll

            # Stopping criterion.
            step += 1
            if step >= nstep:
                break
