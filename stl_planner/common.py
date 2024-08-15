import time
import torch
import einops
import mlflow
from math import ceil
from abc import abstractmethod

import numpy as np


from corallab_lib import MotionPlanningProblem
from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, freeze_torch_model_params

from wip_trajectory_generator.stl_method import stl
from .stl import *

import scipy.interpolate

import gurobipy as gp
from gurobipy import GRB


def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]


def _add(x1, x2):
    return [x1[i] + x2[i] for i in range(len(x1))]


def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == gp.abs_(xvar[i]))
    return sum(abs_x)


class AbstractSTLPlanner(PlannerInterface):
    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,

            n_segments : int = None,
            min_n_segments : int = None,
            max_n_segments : int = None,

            **kwargs
    ):
        assert problem.has_linear_constraint_obstacles()

        self.problem = problem

        if n_segments is None:
            assert max_n_segments is not None
            self.min_n_segments = 1
            self.max_n_segments = max_n_segments
        else:
            self.min_n_segments = n_segments
            self.max_n_segments = n_segments

        m = 1e3
        # a large M causes numerical issues and make the model infeasible to Gurobi
        self.t_min_sep = 1e-2
        # see comments in GreaterThanZero
        self.int_feas_tol  = 1e-1 * self.t_min_sep / m
        self.mip_gap = 1e-4

        self.t_max : int = 64
        self.v_max = 0.2

        self.bloat = 0.005
        self.size = 0.005

    def _add_space_constraints(self, model, points):
        q_min = self.problem.get_q_min()
        q_max = self.problem.get_q_max()

        for p in points:
            for i, v in p.items():
                model.addConstr(v >= (q_min[i] + self.bloat))
                model.addConstr(v <= (q_max[i] - self.bloat))

    def _add_time_constraints(self, model, PWL):
        if self.t_max is not None:
            model.addConstr(PWL[-1][1] <= self.t_max - self.t_min_sep)

        for i in range(len(PWL)-1):
            x1, t1 = PWL[i]
            x2, t2 = PWL[i+1]
            model.addConstr(t2 - t1 >= self.t_min_sep)

    def _add_velocity_constraints(self, model, PWL):
        for i in range(len(PWL)-1):
            x1, t1 = PWL[i]
            x2, t2 = PWL[i+1]
            # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
            # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
            L1_dist = L1Norm(model, _sub(x1,x2))
            model.addConstr(L1_dist <= self.v_max * (t2 - t1))

    def _clear_lcf_vars(self, expression):
        """Remove variables created for LCF of expression."""

        for child in expression.subformulas:
            self._clear_lcf_vars(child)

        expression.props.zs = []

    def _get_time_bounds(self, expression):
        # TODO: Settle the semantics of unbounded time intervals in the
        # expression class
        if expression.unbound:
            left_time_bound = 0
        else:
            left_time_bound = expression.left_time_bound

        if expression.right_unbound or expression.unbound:
            right_time_bound = self.t_max
        else:
            right_time_bound = expression.right_time_bound

        return left_time_bound, right_time_bound

    def _construct_lcf_from_stl_expression(self, expression, PWL):
        """This function takes an STL expression and inductively constructs a
        linear constraint formula (LCF), which is a sentence of atomic formulas
        connected by disjunction and conjuction operators. Each atomic formula
        states some linear expression of several variables (taken from PWL) is
        greater than or equal to zero. The LCF is stored in a property of every
        node in the STL expression tree.
        """

        # post order traversal
        for node in expression.subformulas:
            self._construct_lcf_from_stl_expression(node, PWL)

        # ??
        if len(expression.props.zs) == len(PWL)-1:
            return
        elif len(expression.props.zs) > 0:
            raise ValueError('incomplete zs')

        if isinstance(expression, stl.LinearExp):
            A = expression.A
            b = expression.b
            expression.props.zs = [mu(i, PWL, self.bloat + self.size, A, b) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.NegLinearExp):
            A = expression.A
            b = expression.b
            expression.props.zs = [negmu(i, PWL, self.bloat + self.size, A, b) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.Conjunction) or isinstance(expression, stl.And):
            expression.props.zs = [Conjunction([c.props.zs[i] for c in expression.subformulas]) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.Disjunction) or isinstance(expression, stl.Or):
            expression.props.zs = [Disjunction([c.props.zs[i] for c in expression.subformulas]) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.Eventually):
            left_b, right_b = self._get_time_bounds(expression)
            expression.props.zs = [eventually(i, left_b, right_b, \
                                              expression.subformula.props.zs, \
                                              PWL) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.Always):
            left_b, right_b = self._get_time_bounds(expression)
            expression.props.zs = [always(i, left_b, right_b, \
                                          expression.subformula.props.zs, \
                                          PWL) for i in range(len(PWL)-1)]
        elif isinstance(expression, stl.Until):
            left_b, right_b = self._get_time_bounds(expression)
            expression.props.zs = [until(i, left_b, right_b, \
                                         expression.left_subformula.props.zs, \
                                         expression.right_subformula.props.zs, \
                                         PWL) for i in range(len(PWL)-1)]
        # elif spec.op == 'mu':
        #     spec.zs = [mu(i, PWL, 0.1, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
        # elif spec.op == 'negmu':
        #     spec.zs = [negmu(i, PWL, bloat_factor + size, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
        # elif spec.op == 'and':
        #     spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
        # elif spec.op == 'or':
        #     spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
        # elif spec.op == 'U':
        #     spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, spec.deps[1].zs, PWL) for i in range(len(PWL)-1)]
        # elif spec.op == 'F':
        #     spec.zs = [eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
        # elif spec.op == 'BF':
        #     spec.zs = [bounded_eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL, spec.info['tmax']) for i in range(len(PWL)-1)]
        # elif spec.op == 'A':
        #     spec.zs = [always(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
        else:
            raise NotImplementedError(f"Support for {expression} not implemented.")

        # ValueError('Unable to make wrong op code')

    def _gen_cd_tree_constraints(self, model, root):
        if not hasattr(root, 'deps'):
            return [root,]
        else:
            if len(root.constraints)>0:
                # TODO: more check here
                return root.constraints
            dep_constraints = []
            for dep in root.deps:
                dep_constraints.append(self._gen_cd_tree_constraints(model, dep))
            zs = []
            for dep_con in dep_constraints:
                if isinstance(root, Disjunction):
                    z = model.addVar(vtype=GRB.BINARY)
                    zs.append(z)
                    dep_con = [con + M * (1 - z) for con in dep_con]
                root.constraints += dep_con
            if len(zs)>0:
                root.constraints.append(sum(zs)-1)
            model.update()
            return root.constraints

    def _add_cd_tree_constraints(self, model, root):
        constrs = self._gen_cd_tree_constraints(model, root)
        for con in constrs:
            model.addConstr(con >= 0)


    def _create_integer_time_solution(self, PWL):
        reference_times = np.array([seg[1] for seg in PWL])
        reference_points = np.array([seg[0] for seg in PWL])
        start, end = reference_points[0], reference_points[-1]

        interpolator = scipy.interpolate.interp1d(reference_times, \
                                                  reference_points, \
                                                  axis=0, bounds_error=False, \
                                                  fill_value=(start, end))
        target_times = np.arange(self.t_max)
        solution = interpolator(target_times)

        return solution


    def solve(
            self,
            start,
            goal,
            stl_expression=None,
            n_trajectories=1,
            **kwargs
    ):
        q_mins, q_maxs = self.problem.get_linear_constraint_obstacles()
        not_in_box_exps = [stl.NotInBox(stl.Var("q", dim=2), q_min, q_max) for q_min, q_max in zip(q_mins, q_maxs)]
        not_in_boxes = stl.Conjunction(not_in_box_exps)
        collision_free = stl.Always(not_in_boxes, right_time_bound=self.t_max)

        if stl_expression is None:
            stl_expression = collision_free
        else:
            stl_expression = stl.Conjunction([collision_free, stl_expression])

        start = start.cpu().numpy()
        goal = goal.cpu().numpy()

        sol_l = []

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            for i in range(n_trajectories):
                PWL = self._get_single_solution(
                    start,
                    goal,
                    stl_expression=stl_expression,
                    grb_env=env,
                    **kwargs
                )

                if PWL is not None:
                    sol = self._create_integer_time_solution(PWL)
                else:
                    sol = None

                sol_l.append(sol)

        failed = any([sol is None for sol in sol_l])

        return sol_l, { "failed": failed }

    @abstractmethod
    def _get_single_solution(
            self,
            start,
            goal,
            stl_expression=None,
            **kwargs
    ):
        raise NotImplementedError()

    def reset(self):
        pass
