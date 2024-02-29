import functools

import jax
import numpy as np
import jax.numpy as jnp
import jax.lax as lax

from typing import Any, Callable, Iterable, List, Mapping, Optional, TypeVar, Union

T = TypeVar("T")
Tree = Union[T, Iterable["Tree[T]"], Mapping[Any, "Tree[T]"]]

def build_target_sdf(boundary, alpha):
    def target(x):
        distance_to_boundaries = jnp.array([x - boundary[0, 0], boundary[0, 1] - x, boundary[1, 1] - x, boundary[1, 0] - x])
        min_distance = jnp.min(distance_to_boundaries, axis=0) #If min_dist is negative, outside boundary
        def outside_target(_):
            max_dist_dim1 = jnp.max(jnp.array([boundary[0,0] - x, x - boundary[0,1]]), axis=0)
            max_dist_dim1 = jnp.maximum(max_dist_dim1, 0)
            max_dist_dim2 = jnp.max(jnp.array([boundary[1,0] - x, x - boundary[1,1]]), axis=0)
            max_dist_dim2 = alpha*jnp.maximum(max_dist_dim2, 0)
            vector = jnp.array([max_dist_dim1, max_dist_dim2])
            return jnp.linalg.norm(vector)
        
        def inside_target(_):
            return -1*jnp.min(min_distance)
    
        is_outside = jnp.any(min_distance < 0)
        sdf = lax.cond(is_outside, outside_target, inside_target, operand=None)
        return sdf
    return target

def param_solve(solver_settings, dyn_hjr, grid, times, init_values):
    """
    Extension of hj_solve for parametric uncertainty.
    Args:
        dynamics: hj_reachability.Dynamics object
        grid: hj_reachability.Grid object
        target_values: [n x n x n x n] array of target values
        sdf: signed distance function
    Returns:
        [n x n x n x n] array of value function values
    """
    import hj_reachability as hj
    extrema = dyn_hjr.get_param_combinations(type="extrema")
    values = []
    for i, extremum in enumerate(extrema):
        import copy
        dyn_hjr_alt = copy.deepcopy(dyn_hjr)
        dyn_hjr_alt.params = extremum
        values_i = hj.solve(solver_settings, dyn_hjr_alt, grid, times, init_values)
        values.append(copy.deepcopy(values_i))
        print(jnp.array(values_i).shape)
    return jnp.max(jnp.array(values), axis=0)

