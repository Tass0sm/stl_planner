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


def add_mutual_clearance_constraints(model, PWLs, bloat):
    for i in range(len(PWLs)):
        for j in range(i+1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            for k in range(len(PWL1)-1):
                for l in range(len(PWL2)-1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k+1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l+1]
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    z_disjoint_segments = disjoint_segments(model, [x11, x12], [x21, x22], bloat)
                    z = Disjunction([z_noIntersection, z_disjoint_segments])
                    add_CDTree_Constraints(model, z)


class MASTLPlanner(AbstractSTLPlanner):
    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,
            **kwargs
    ):
        super().__init__(planner_name, problem=problem, **kwargs)

        assert problem.robot.is_multi_agent()
        self.subrobots = self.problem.robot.get_subrobots()

    def _get_subrobot_states(self, joint_state):
        states = []

        for i, r in enumerate(self.subrobots):
            subrobot_state = r.get_position(joint_state)
            states.append(subrobot_state)
            joint_state = joint_state[r.get_n_dof():]

        return states

    @property
    def name(self):
        return "ma_stl_planner"

    def solve(
            self,
            start,
            goal,
            ma_stl_expression=None,
            # n_trajectories=1,
            **kwargs
    ):

        start = torch.tensor([-1.0, -1.0, -0.8, -0.8])
        goal = torch.tensor([1.0, 1.0, -0.8, -0.5])

        subrobot_starts = self._get_subrobot_states(start)
        subrobot_starts = [s.cpu().numpy() for s in subrobot_starts]
        subrobot_goals = self._get_subrobot_states(goal)
        subrobot_goals = [s.cpu().numpy() for s in subrobot_goals]

        for n_segments in range(self.min_n_segments, self.max_n_segments + 1):
            for _, stl_expression in ma_stl_expression.items():
                self._clear_lcf_vars(stl_expression)

            m = gp.Model("xref")
            # m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.IntFeasTol, self.int_feas_tol)
            m.setParam(GRB.Param.MIPGap, self.mip_gap)
            # m.setParam(GRB.Param.NonConvex, 2)
            # m.getEnv().set(GRB_IntParam_OutputFlag, 0)

            bloat = 0.05
            size = 0.11/2

            PWLs = []

            for i, (start_i, goal_i) in enumerate(zip(subrobot_starts, subrobot_goals)):
                dims = len(start_i)

                stl_expression = ma_stl_expression.get(i, None)

                PWL = []

                for i in range(n_segments + 1):
                    # point coordinates and time
                    point_vars = [
                        m.addVars(dims, lb=-GRB.INFINITY),
                        m.addVar()
                    ]

                    PWL.append(point_vars)

                PWLs.append(PWL)

                m.update()

                # the initial constriant
                m.addConstrs(PWL[0][0][i] == start_i[i] for i in range(dims))
                m.addConstr(PWL[0][1] == 0)

                # the goal constraint
                m.addConstrs(PWL[-1][0][i] == goal_i[i] for i in range(dims))

                # self._add_space_constraints(m, [P[0] for P in PWL])
                self._add_velocity_constraints(m, PWL)
                self._add_time_constraints(m, PWL)

                if stl_expression is not None:
                    self._construct_lcf_from_stl_expression(stl_expression, PWL, bloat, size)
                    self._add_cd_tree_constraints(m, stl_expression.props.zs[0])

            # TODO:
            # add_mutual_clearance_constraints(m, PWLs, bloat)

            # Minimize final time
            obj = sum([PWL[-1][1] for PWL in PWLs])
            m.setObjective(obj, GRB.MINIMIZE)

            try:
                start_time = time.time()
                m.optimize()
                end_time = time.time()
                print('solving it takes %.3f s'%(end_time - start_time))

                PWLs_output = []
                for PWL in PWLs:
                    PWL_output = []
                    for P in PWL:
                        PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])
                    PWLs_output.append(PWL_output)

                # PWL_output = []
                # for P in PWL:
                #     PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])

                solution = {}
                for i, PWL_output in enumerate(PWLs_output):
                    sol_i = np.stack([p for p, _ in PWL_output])
                    solution[i] = sol_i

                m.dispose()

                solution = np.concatenate([s for k, s in solution.items()], axis=-1)

                # solution = PWLs_output # np.stack([p for p, _ in PWL_output])
                return solution, {}

            except Exception as e:
                m.dispose()

        return None, {}

    def reset(self):
        pass
