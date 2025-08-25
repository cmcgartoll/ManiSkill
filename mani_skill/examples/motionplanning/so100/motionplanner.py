import numpy as np
import mplib
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from mani_skill.envs.sapien_env import BaseEnv
import sapien.core as sapien

from mani_skill.utils.structs.pose import to_sapien_pose

class SO100MotionPlanningSolver(BaseMotionPlanningSolver):
    CLOSED = -1.1
    OPEN = 1.1
    NUM_LINKS = 5

    def __init__(self, env: BaseEnv, debug: bool = False, vis: bool = True, base_pose: sapien.Pose = None, visualize_target_grasp_pose: bool = True, print_env_info: bool = True, joint_vel_limits=0.9, joint_acc_limits=0.9):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits, num_links=self.NUM_LINKS)
        self.gripper_state = self.OPEN

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="Fixed_Jaw_tip",
            joint_vel_limits=np.ones(self.num_links) * self.joint_vel_limits,
            joint_acc_limits=np.ones(self.num_links) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner
    
    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
        ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            # mask=[False, False, False, False, False, True]
        )
        goal_pose = self.planner.transform_goal_to_wrt_base(np.concatenate([pose.p, pose.q]))   # New pose in world frame
        print(f"New pose: {result['new_pose']}")
        print(f"Goal pose: {goal_pose}")
        new_pose = result["new_pose"]
        if new_pose is not None and goal_pose is not None:
            print("6D Distance between goal and new pose: ", self.planner.distance_6D(goal_pose[:3], goal_pose[3:], new_pose[:3], new_pose[3:]))
        
        builder = self.base_env.scene.create_actor_builder()
        # Create or reuse visualization sphere
        if not hasattr(self, 'new_pose_visual') or self.new_pose_visual is None:
            # Remove existing visualization if it exists
            if "new_pose_visual" in self.base_env.scene.actors:
                self.base_env.scene.remove_actor(self.base_env.scene.actors["new_pose_visual"])
            
            
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0]),  # Relative to actor origin
                radius=0.02,
                material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])  # Bright red, fully opaque
            )
            self.new_pose_visual = builder.build_kinematic(name="new_pose_visual")
        
        # Update the pose
        if self.new_pose_visual is not None and new_pose is not None:
            self.new_pose_visual.set_pose(sapien.Pose(p=[new_pose[0]-0.725, new_pose[1], new_pose[2]], q=new_pose[3:]))

        if not hasattr(self, 'goal_pose_visual') or self.goal_pose_visual is None:
            builder.add_sphere_visual(
                    pose=sapien.Pose(p=[0, 0, 0]),  # Relative to actor origin
                    radius=0.01,
                    material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1])  # Bright blue, fully opaque
                )
            self.goal_pose_visual = builder.build_kinematic(name="goal_pose_visual")
        self.goal_pose_visual.set_pose(sapien.Pose(p=[goal_pose[0]-0.725, goal_pose[1], goal_pose[2]], q=goal_pose[3:]))
        # Update scene and force a render update
        self.base_env.scene.update_render()
        self.render_wait()
        self.base_env.render_human()
        self.render_wait()

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)