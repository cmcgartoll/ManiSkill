import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.so100.motionplanner import \
    SO100MotionPlanningSolver
from mani_skill.examples.motionplanning.so100.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = SO100MotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.01
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([-0.05, 0, -0.02])  # Smaller reach distance
    result = planner.move_to_pose_with_screw(reach_pose) 
    if result == -1:
        print("Failed to reach pre-grasp position")
        planner.close()
        return -1
    print("Reached pre-grasp position")

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        print("Failed to reach grasp position")
        planner.close()
        return -1
    print("Reached grasp position")
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift slightly to ensure stable grasp
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, 0.02])
    result = planner.move_to_pose_with_screw(lift_pose)
    if result == -1:
        print("Failed to lift cube")
    print("Lifted cube")
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_position = env.goal_site.pose.sp.p
    goal_position[2] = min(goal_position[2], 0.06)
    goal_pose = sapien.Pose(goal_position, grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        print("Failed to reach goal position")
        planner.close()
        return -1
    print("Reached goal position")
    planner.close()
    return res
