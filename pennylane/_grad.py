# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the autograd wrappers :class:`grad` and :func:`jacobian`
"""
from collections.abc import Iterable

import numpy as _np

from autograd.core import make_vjp as _make_vjp
from autograd.differential_operators import make_jvp_reversemode
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace
from autograd import jacobian as _jacobian

make_vjp = unary_to_nary(_make_vjp)


class grad:
    """Returns the gradient as a callable function of (functions of) QNodes.

    Function arguments with the property ``requires_grad`` set to ``False``
    will automatically be excluded from the gradient computation, unless
    the ``argnum`` keyword argument is passed.

    When the output gradient function is executed, both the forward pass
    *and* the backward pass will be performed in order to
    compute the gradient. The value of the forward pass is available via the
    :attr:`~.forward` property.

    Args:
        func (function): a plain QNode, or a Python function that contains
            a combination of quantum and classical nodes
        argnum (int, list(int), None): Which argument(s) to take the gradient
            with respect to. By default, the arguments themselves are used
            to determine differentiability, by examining the ``requires_grad``
            property. Providing this keyword argument overrides this behaviour,
            allowing argument differentiability to be defined manually for the returned gradient function.

    Returns:
        function: The function that returns the gradient of the input
        function with respect to the differentiable arguments, or, if specified,
        the arguments in ``argnum``.
    """

    def __init__(self, fun, argnum=None):
        self._forward = None
        self._grad_fn = None

        self._fun = fun
        self._argnum = argnum

        if self._argnum is not None:
            # If the differentiable argnum is provided, we can construct
            # the gradient function at once during initialization
            self._grad_fn = self._grad_with_forward(fun, argnum=argnum)

    def _get_grad_fn(self, args):
        """Get the required gradient function.

        * If the differentiable argnum was provided on initialization,
          this has been pre-computed and is available via self._grad_fn

        * Otherwise, we must dynamically construct the gradient function by
          inspecting as to which of the parameter arguments are marked
          as differentiable.
        """
        if self._grad_fn is not None:
            return self._grad_fn

        # Inspect the arguments for differentiability, and
        # compute the autograd gradient function with required argnums
        # dynamically.
        argnum = []

        for idx, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                argnum.append(idx)

        return self._grad_with_forward(
            self._fun,
            argnum=argnum,
        )

    def __call__(self, *args, **kwargs):
        """Evaluates the gradient function, and saves the function value
        calculated during the forward pass in :attr:`.forward`."""
        grad_value, ans = self._get_grad_fn(args)(*args, **kwargs)
        self._forward = ans
        return grad_value

    @property
    def forward(self):
        """float: The result of the forward pass calculated while performing
        backpropagation. Will return ``None`` if the backpropagation has not yet
        been performed."""
        return self._forward

    @staticmethod
    @unary_to_nary
    def _grad_with_forward(fun, x):
        """This function is a replica of ``autograd.grad``, with the only
        difference being that it returns both the gradient *and* the forward pass
        value."""
        vjp, ans = _make_vjp(fun, x)

        if not vspace(ans).size == 1:
            raise TypeError(
                "Grad only applies to real scalar-output functions. "
                "Try jacobian, elementwise_grad or holomorphic_grad."
            )

        grad_value = vjp(vspace(ans).ones())
        return grad_value, ans


class jacobian:
    """Returns the Jacobian as a callable function of vector-valued
    (functions of) QNodes.

    This is a wrapper around the :mod:`autograd.jacobian` function.

    Args:
        func (function): A vector-valued Python function or QNode that contains
            a combination of quantum and classical nodes. The output of the computation
            must consist of a single NumPy array (if classical) or a tuple of
            expectation values (if a quantum node)
        argnum (int or Sequence[int]): Which argument to take the gradient
            with respect to. If a sequence is given, the Jacobian matrix
            corresponding to all input elements and all output elements is returned.

    Returns:
        function: the function that returns the Jacobian of the input
        function with respect to the arguments in argnum
    """

    def __init__(self, fun, argnum=None):
        self._jac_fn = None
        self._fun = fun
        self._argnum = argnum

        if self._argnum is not None:
            # If the differentiable argnum is provided, we can construct
            # the gradient function at once during initialization
            self._jac_fn = self._jac(fun, argnum=argnum)

    def __call__(self, *args, **kwargs):
        """Evaluates the gradient function, and saves the function value
        calculated during the forward pass in :attr:`.forward`."""
        grad_value = self._get_jac_fn(args)(*args, **kwargs)
        return grad_value

    def _get_jac_fn(self, args):
        """Get the required jacoian function.

        * If the differentiable argnum was provided on initialization,
          this has been pre-computed and is available via self._jac_fn

        * Otherwise, we must dynamically construct the gradient function by
          inspecting as to which of the parameter arguments are marked
          as differentiable.
        """
        if self._jac_fn is not None:
            return self._jac_fn

        # Inspect the arguments for differentiability, and
        # compute the autograd gradient function with required argnums
        # dynamically.
        argnum = []

        for idx, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                argnum.append(idx)

        return self._jac(self._fun, argnum=argnum)

    @staticmethod
    @unary_to_nary
    def _jac(fun, x):
        from pennylane import numpy as np

        vjp, y = _make_vjp(fun, x)
        jvp, _ = _make_vjp(vjp, np.array(vspace(y).zeros(), requires_grad=True))

        jac = []

        if not isinstance(x, Iterable) or not getattr(x, "ndim", 1):
            num_vectors = 1
        else:
            num_vectors = len(x)

        # loop through each basis
        for basis_idx in range(num_vectors):
            basis_vector = _np.eye(1, num_vectors, basis_idx, dtype=_np.int)[0]
            jac.append(jvp(basis_vector))

        if num_vectors == 1:
            return _np.array(jac).T.flatten()

        return _np.array(jac).T
