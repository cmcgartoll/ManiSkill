import numpy as np
import mplib
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.envs.scene import ManiSkillScene
import sapien.core as sapien
from transforms3d import quaternions
from mani_skill.utils.structs.pose import to_sapien_pose


class XArm6RobotiqMotionPlanningSolver(BaseMotionPlanningSolver):
    CLOSED = 0.81
    OPEN = 0
    NUM_LINKS = 6

    def __init__(self, env: BaseEnv, debug: bool = False, vis: bool = True, base_pose: sapien.Pose = None, visualize_target_grasp_pose: bool = True, print_env_info: bool = True, joint_vel_limits=0.9, joint_acc_limits=0.9):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits, num_links=self.NUM_LINKS)
        self.gripper_state = self.OPEN
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = self.build_robotiq_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)

    def build_robotiq_gripper_grasp_pose_visual(self, scene: ManiSkillScene):
        builder = scene.create_actor_builder()
        grasp_pose_visual_width = 0.01
        grasp_width = 0.05

        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0.0]),
            radius=grasp_pose_visual_width,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
        )

        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.08]),
            half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.05]),
            half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    grasp_width + grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    -grasp_width - grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
        return grasp_pose_visual


class XArm6PandaGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    NUM_LINKS = 6
    def __init__(self, env: BaseEnv, debug: bool = False, vis: bool = True, base_pose: sapien.Pose = None, visualize_target_grasp_pose: bool = True, print_env_info: bool = True, joint_vel_limits=0.9, joint_acc_limits=0.9):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits, num_links=self.NUM_LINKS)
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = self.build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(self.num_links) * self.joint_vel_limits,
            joint_acc_limits=np.ones(self.num_links) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    
    def build_panda_gripper_grasp_pose_visual(self, scene: ManiSkillScene):
        builder = scene.create_actor_builder()
        grasp_pose_visual_width = 0.01
        grasp_width = 0.05

        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0.0]),
            radius=grasp_pose_visual_width,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
        )

        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.08]),
            half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.05]),
            half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    grasp_width + grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    -grasp_width - grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
        return grasp_pose_visual
    
