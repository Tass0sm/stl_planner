import torch
import einops
import mlflow
from math import ceil

from corallab_lib import MotionPlanningProblem
from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, freeze_torch_model_params

import gurobipy as grb


class STLPlanner(PlannerInterface):
    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,

            n_segments : int = None,
            min_n_segments : int = None,
            max_n_segments : int = None,

            **kwargs
    ):
        self.problem = problem

        if n_segments is None:
            assert max_n_segments is not None
            self.min_n_segments = 1
            self.max_n_segments = max_n_segments
        else:
            self.min_n_segments = n_segments
            self.max_n_segments = n_segments

    @property
    def name(self):
        return "stl_planner"

    def _add_space_constraints(model, vs, bloat=0.):
        xlim, ylim = limits
        breakpoint()

        for x in xlist:
            model.addConstr(x[0] >= (xlim[0] + bloat))
            model.addConstr(x[1] >= (ylim[0] + bloat))
            model.addConstr(x[0] <= (xlim[1] - bloat))
            model.addConstr(x[1] <= (ylim[1] - bloat))

        return None


    def solve(
            self,
            start,
            goal,
            stl_expression=None,
            # n_trajectories=1,
            **kwargs
    ):

        for n_segments in range(self.min_n_segments, self.max_n_segments + 1):
            for spec in specs:
                clearSpecTree(spec)

            if tasks:
                for task in tasks:
                    for t in task:
                        clearSpecTree(t)

            print('----------------------------')
            print('num_segs', num_segs)

            PWLs = []
            m = grb.Model("xref")

            # m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
            m.setParam(GRB.Param.MIPGap, MIPGap)
            # m.setParam(GRB.Param.NonConvex, 2)
            # m.getEnv().set(GRB_IntParam_OutputFlag, 0)

            q_dim = self.problem.q_dim

            solution_vars = []

            # Add vars for every point in the trajectory, plus another one for
            # time?
            for i in range(n_segments + 1):
                point_i_vars = [
                    m.addVars(q_dim, lb=-GRB.INFINITY),
                    m.addVar()
                ]

                solution_vars.append(point_i_vars)

                if i == 0:
                    # the initial constriant
                    m.addConstrs(point_i_vars[0][i] == start[i] for i in range(q_dim))
                    m.addConstr(point_i_vars[1] == 0)

                if i == n_segments:
                    m.addConstrs(point_i_vars[0][i] == goal[i] for i in range(q_dim))

            # if limits is not None:
            #     add_space_constraints(m, [P[0] for P in PWL], limits)

            # add_velocity_constraints(m, PWL, vmax=vmax)
            # add_time_constraints(m, PWL, tmax)

            # handleSpecTree(spec, PWL, bloat, size)
            # add_CDTree_Constraints(m, spec.zs[0])

            m.update()


            # if tasks is not None:
            #     for idx_agent in range(len(tasks)):
            #         for idx_task in range(len(tasks[idx_agent])):
            #             handleSpecTree(tasks[idx_agent][idx_task], PWLs[idx_agent], bloat, size)

            #     conjunctions = []
            #     for idx_task in range(len(tasks[0])):
            #         disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
            #         conjunctions.append(Disjunction(disjunctions))
            #     z = Conjunction(conjunctions)
            #     add_CDTree_Constraints(m, z)

            # add_mutual_clearance_constraints(m, PWLs, bloat)

            # # obj = sum([L1Norm(m, _sub(PWL[i][0], PWL[i+1][0])) for PWL in PWLs for i in range(len(PWL)-1)])
            # obj = sum([PWL[-1][1] for PWL in PWLs])
            # m.setObjective(obj, GRB.MINIMIZE)

            # m.write("test.lp")
            # print('NumBinVars: %d'%m.getAttr('NumBinVars'))

            # m.computeIIS()
            # import ipdb;ipdb.set_treace()
            # try:
            #     start = time.time()
            #     m.optimize()
            #     end = time.time()
            #     print('sovling it takes %.3f s'%(end - start))
            #     PWLs_output = []
            #     for PWL in PWLs:
            #         PWL_output = []
            #         for P in PWL:
            #             PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])
            #         PWLs_output.append(PWL_output)
            #     m.dispose()
            #     return PWLs_output

            # except Exception as e:
            #     m.dispose()

        # hard_conds = self.dataset.get_hard_conditions(
        #     torch.vstack((start_state_pos, goal_state_pos)), normalize=True
        # )
        # start_state = torch.hstack((start_state_pos, torch.zeros_like(start_state_pos))).unsqueeze(0)
        # context = None

        # trajs_normalized_iters = self.model.run_inference(
        #     context, hard_conds,
        #     n_samples=n_trajectories, horizon=self.n_support_points,
        #     return_chain=True,
        #     **self.sample_fn_kwargs,
        #     n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
        #     # ddim=True
        # )

        # if self.run_prior_then_guidance:
        #     n_post_diffusion_guide_steps = (self.t_start_guide + self.n_diffusion_steps_without_noise) * self.n_guide_steps
        #     trajs = trajs_normalized_iters[-1]

        #     trajs_post_diff_l = []

        #     for i in range(n_post_diffusion_guide_steps):
        #         trajs = guide_gradient_steps(
        #             trajs,
        #             hard_conds=hard_conds,
        #             guide=self.guide,
        #             n_guide_steps=1,
        #             unnormalize_data=False,
        #         )

        #         trajs_post_diff_l.append(trajs)

        #     chain = torch.stack(trajs_post_diff_l, dim=1)
        #     chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
        #     trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))

        # trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)
        # solution = trajs_iters[-1]
        # info = { "solution_iters": trajs_iters }

        # return solution, info
        return None, {}

    def reset(self):
        pass
