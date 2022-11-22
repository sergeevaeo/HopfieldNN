import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor import slinalg
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import (BoundedProperty, ChoiceProperty,
                                   WithdrawProperty)
from neupy.algorithms.gd import StepSelectionBuiltIn, errors
from neupy.algorithms.utils import parameter_values, setup_parameter_updates




def compute_jacobian(errors, parameters):
    """
    Compute jacobian.

    Parameters
    ----------
    errors : Theano variable
        Computed MSE for each sample separetly.

    parameters : list of Theano variable
        Neural network parameters (e.g. weights, biases).

    Returns
    -------
    Theano variable
    """
    n_samples = errors.shape[0]
    J = T.jacobian(errors, wrt=parameters)

    jacobians = []
    for jacobian, parameter in zip(J, parameters):
        jacobian = jacobian.reshape((n_samples, parameter.size))
        jacobians.append(jacobian)

    return T.concatenate(jacobians, axis=1)


