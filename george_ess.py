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
        xx = np.array(xx - mu)

        # Set up the ellipse and the slice threshold.
        D = len(xx)
        nu = np.dot(np.random.randn(D), self._factor[0])
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

    def _metropolis_step(self, w, step, getter, setter):
        # Compute the initial vector and ln-likelihood.
        v0 = getter()
        lp0 = self.lnlikelihood(w, quiet=True) + self.lnprior()
        if not np.isfinite(lp0):
            raise RuntimeError("The initial ln-likelihood in the Metropolis "
                               "step has zero probability.")

        # Propose a new position and compute the updated ln-likelihood.
        q = v0 + step * np.random.randn(len(v0))
        setter(q)
        lp1 = self.lnlikelihood(w, quiet=True) + self.lnprior()

        # Accept or reject the update.
        diff = lp1 - lp0
        if np.isfinite(lp1) and np.exp(diff) >= np.random.rand():
            return q, lp1, True

        # The step was rejected, revert the parameters.
        setter(v0)
        return v0, lp0, False

    def lnprior(self):
        return self.kernel.lnprior() + self.mean.lnprior()

    def elliptical_slice_sampling(self, lnlike, y=None, nstep=np.inf,
                                  angle_range=None, maxiter=np.inf,
                                  hyper_update=0, update_mean=True,
                                  stepsize=None):
        # Check that a step size is given if required.
        if hyper_update > 0 and stepsize is None:
            raise ValueError("If you want to update the hyperparameters, you "
                             "need to provide a step size")

        # If no initial guess is given, sample from the prior.
        if y is None:
            y = self.sample()
        y = y[self.inds]
        yret = np.empty_like(y)

        # Set up the parameter space that we'll be sampling in and pre-compute
        # some stuff.
        if hyper_update > 0:
            if update_mean:
                def setter(v):
                    self.mean.vector = v[:len(self.mean)]
                    self.kernel.vector = v[len(self.mean):]
                getter = lambda: np.append(self.mean.vector,
                                           self.kernel.vector)
            else:
                def setter(v):
                    self.kernel.vector = v
                getter = lambda: self.kernel.vector

            # Check the step size.
            h = getter()
            try:
                len(stepsize)
            except TypeError:
                pass
            else:
                if len(stepsize) != len(h):
                    raise ValueError("The step size must be a float or have "
                                     "the same dimensions as the parameter "
                                     "space ({0})".format(len(h)))

            # Compute the initial ln-probability.
            lp = self.lnlikelihood(y, quiet=True) + self.lnprior()

        # Define a new ln-likelihood function to deal with sample ordering.
        def _lnlike(y0):
            yret[self.inds] = y0
            return lnlike(yret)

        # Compute the initial ln-likelihood.
        ll = _lnlike(y)

        # Run the ESS iterations.
        step = 0
        accepted, total = 1, 1
        while True:
            # Do an iteration of elliptical slice sampling.
            y, ll = self._ess_step(_lnlike, y, ll, angle_range=angle_range,
                                   maxiter=maxiter)
            yret[self.inds] = y

            # Update the hyperparameters using a Metropolis step if requested.
            if hyper_update > 0:
                if (step + 1) % hyper_update == 0:
                    h, lp, a = self._metropolis_step(np.array(yret), stepsize,
                                                     getter, setter)
                    accepted += int(a)
                    total += 1
                yield yret, h, ll + lp, accepted / total
            else:
                yield yret, ll

            # Stopping criterion.
            step += 1
            if step >= nstep:
                break
