import time
import torch
import einops
import mlflow
from math import ceil

import numpy as np


from corallab_lib import MotionPlanningProblem
from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, freeze_torch_model_params

from wip_trajectory_generator import stl
from ..common import *
from ..stl import *
from ..exceptions import *

import gurobipy as gp
from gurobipy import GRB


class STLPlanner(AbstractSTLPlanner):
    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,
            **kwargs
    ):
        super().__init__(planner_name, problem=problem, **kwargs)

    @property
    def name(self):
        return "stl_planner"

    def _get_single_solution(
            self,
            start,
            goal,
            stl_expression=None,
            n_trajectories=1,
            seed=0,
            grb_env=None,
            **kwargs
    ):
        var = stl.Var("q", dim=self.problem.get_q_dim())
        collision_free = self._create_collision_avoidance_expression(var)

        if stl_expression is None:
            stl_expression = collision_free
        else:
            stl_expression = stl.Conjunction([collision_free, stl_expression])

        model_infeasible = False

        for n_segments in range(self.min_n_segments, self.max_n_segments + 1):
            self._clear_lcf_vars(stl_expression)

            m = gp.Model("xref", env=grb_env)
            # m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.IntFeasTol, self.int_feas_tol)
            m.setParam(GRB.Param.MIPGap, self.mip_gap)
            # m.setParam(GRB.Param.NonConvex, 2)
            # m.getEnv().set(GRB_IntParam_OutputFlag, 0)

            # m.setParam(GRB.Param.PoolSolutions, n_trajectories)
            # m.setParam(GRB.Param.PoolSearchMode, 1)
            # m.setParam(GRB.Param.SolutionNumber, n_trajectories)

            x0 = start
            x0 = np.array(x0).reshape(-1).tolist()

            dims = len(x0)

            PWL = []
            for i in range(n_segments + 1):
                # point coordinates and time
                point_vars = [
                    m.addVars(dims, lb=-GRB.INFINITY),
                    m.addVar()
                ]

                PWL.append(point_vars)

            m.update()

            # the initial constriant
            m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
            m.addConstr(PWL[0][1] == 0)

            # the goal constraint
            m.addConstrs(PWL[-1][0][i] == goal[i] for i in range(dims))

            self._add_space_constraints(m, [P[0] for P in PWL])
            self._add_velocity_constraints(m, PWL)
            self._add_time_constraints(m, PWL)

            self._construct_lcf_from_stl_expression(stl_expression, PWL)
            self._add_cd_tree_constraints(m, stl_expression.props.zs[0])

            # Minimize final time
            obj = PWL[-1][1]
            m.setObjective(obj, GRB.MINIMIZE)

            try:
                start_time = time.time()
                m.optimize()
                end_time = time.time()
                # print('solving it takes %.3f s'%(end_time - start_time))

                if m.status == GRB.Status.INFEASIBLE:
                    raise InfeasibleModelError()

                PWL_output = []
                for P in PWL:
                    PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])

                m.dispose()

                solution = self._create_integer_time_solution(PWL_output)
                return solution
            except AttributeError as e:
                m.dispose()
            except InfeasibleModelError as e:
                model_infeasible = True
                m.dispose()
            except Exception as e:
                m.dispose()

        if model_infeasible:
            print(f"Model is infeasible for the expression \"{stl_expression}\"")

        return None

    def reset(self):
        pass
