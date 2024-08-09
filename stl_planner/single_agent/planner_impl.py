import time
import torch
import einops
import mlflow
from math import ceil

import numpy as np


from corallab_lib import MotionPlanningProblem
from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, freeze_torch_model_params

from wip_trajectory_generator.stl_method import stl
from ..common import *
from ..stl import *

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

    def solve(
            self,
            start,
            goal,
            stl_expression=None,
            # n_trajectories=1,
            **kwargs
    ):

        start = torch.tensor([-1.0, -1.0])
        start = start.cpu().numpy()
        goal = torch.tensor([1.0, 1.0])
        goal = goal.cpu().numpy()

        for n_segments in range(self.min_n_segments, self.max_n_segments + 1):
            self._clear_lcf_vars(stl_expression)

            m = gp.Model("xref")
            # m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.IntFeasTol, self.int_feas_tol)
            m.setParam(GRB.Param.MIPGap, self.mip_gap)
            # m.setParam(GRB.Param.NonConvex, 2)
            # m.getEnv().set(GRB_IntParam_OutputFlag, 0)

            bloat = 0.05
            size = 0.11/2

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

            # self._add_space_constraints(m, [P[0] for P in PWL])
            self._add_velocity_constraints(m, PWL)
            self._add_time_constraints(m, PWL)

            self._construct_lcf_from_stl_expression(stl_expression, PWL, bloat, size)
            self._add_cd_tree_constraints(m, stl_expression.props.zs[0])

            # Minimize final time
            obj = PWL[-1][1]
            m.setObjective(obj, GRB.MINIMIZE)

            try:
                start_time = time.time()
                m.optimize()
                end_time = time.time()
                print('solving it takes %.3f s'%(end_time - start_time))

                PWL_output = []
                for P in PWL:
                    PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])

                m.dispose()

                solution = np.stack([p for p, _ in PWL_output])
                return solution, {}

            except Exception as e:
                m.dispose()

        return None, {}

    def reset(self):
        pass
