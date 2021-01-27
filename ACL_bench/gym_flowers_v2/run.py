import os
import gym
import ACL_bench.gym_flowers_v2
import ACL_bench.gym_flowers
import time
import numpy as np

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
#
# random_state = np.random.RandomState(43)
# env = gym.make('parametric-continuous-parkour-v0', agent_body_type='profile_chimpanzee',
#                ceiling_offset=150)#, ceiling_clip_offset = 80)
# # input_vector = np.random.normal(0.0, 1.0, size=(3)).astype(np.float32)
# # print(input_vector)
# # env.set_environment(input_vector=input_vector) # Stump Tracks
# env.set_environment(input_vector=np.zeros(3), water_level = 0.1) # Stump Tracks
# env.reset()
#
# while True:
#     _, _, d, _ = env.step(env.action_space.sample())
#     #print("Done ? :" + str(d))
#     env.render()
#     time.sleep(0.1)


# env = gym.make('parametric-continuous-climber-v0', body_type='back_chimpanzee')
# # env.set_environment(splitting_height=2,
# #                     circle_radius=0.2,
# #                     space_between_grips=[2, 0.5],
# #                     y_variations=[0, 0.5],
# #                     group_sections=[2, 1],
# #                     group_x_variations=[0, 1.5])
# # env.set_environment(splitting_height=2,
# #                     circle_radius=0.2,
# #                     space_between_grips=[2, 0.5],
# #                     y_variations=[0, 0],
# #                     group_sections=[2, 1],
# #                     group_x_variations=[0, 0])
# # ESAY WALL
# # env.set_environment(splitting_height=1.5,
# #                     circle_radius=0.18,
# #                     space_between_grips=[3.3, 0],
# #                     y_variations=[0, 0],
# #                     group_sections=[2, 0],
# #                     group_x_variations=[0, 0])
# env.set_environment(splitting_height=1.5,
#                     circle_radius=0.18,
#                     space_between_grips=3.3,
#                     y_variations_std=0,
#                     group_sections=2,
#                     group_x_variations_std=0)
# state = env.reset()
# while True:
#     #_, _, d, _ = env.step([14, 6, 1])
#     #print("Done ? :" + str(d))
#     env.render(draw_lidars=True)
#     time.sleep(0.01)

'''
env = gym.make('parametric-continuous-drawing-path-v0')
video_recorder = VideoRecorder(env, "D:\draw_3.mp4", enabled=True)
env.set_environment(basket_x=16, basket_width=30, balls_gravity=0.3)
state = env.reset()
_, _, d, _ = env.step([19, 10, 0])
_, _, d, _ = env.step([17, 4, 1])
i = 0
while True:
    #_, _, d, _ = env.step([14, 6, 1])
    #print("Done ? :" + str(d))
    env.render(debug=True)
    video_recorder.capture_frame()
    time.sleep(0.01)
    i +=1
    if i >= 100:
        video_recorder.close()
        video_recorder.enabled = False
        break
'''

video_recorder = None
folder_video = "D:\documents\PRO\Etudes\MSc_U-Bordeaux\StageInriaFlowers\TestbedProject\DebugMorphology\Water_Test"
env = gym.make('parametric-continuous-walker-v0', walker_type='climbing_profile_chimpanzee')
# env = gym.make('parametric-continuous-flat-parkour-v0', water_level = 0, agent_body_type='old_classic_bipedal')
# env = gym.make('parametric-continuous-flat-parkour-v0', water_level = 4, agent_body_type='fish')
# video_recorder = VideoRecorder(env, folder_video + "\\amphibious_bipedal_11-06.mp4", enabled=True)
env.set_environment(stump_height=0.0)
                    # input_vector=np.zeros(3),
                    # water_level = 0.1,
                    # creepers_width=0.25,
                    # creepers_height=3,
                    # creepers_spacing=3,#4.407788,
                    # terrain_cppn_scale=10)
# env.set_environment(creepers_width=0.1,
#                     creepers_height=0.1,
#                     creepers_spacing=1.0)
env._SET_RENDERING_VIEWPORT_SIZE(450, 400, keep_ratio=False)
# env.set_environment(creepers_size=2)
env.reset()

repeat = 30
i = 0
j = 0
actions = ""
seq = [[0, 0, 0, 0.01] for _ in range(10)] + [[0, 0, 0, 0] for _ in range(10)] + [[0, 0, 0, i*-0.01] for i in range(100, 0, -1 )]
while True:
    if actions == "Random":
        actions_to_play = env.action_space.sample()
        if hasattr(env.agent_body, "sensors"):
            for i in range(len(env.agent_body.sensors)):#
                actions_to_play[len(actions_to_play) - i - 1] = actions_to_play[len(actions_to_play) - i - 1] + 1.0
        # if j >= 30 and j % 30 == 0:
        #     actions_to_play[-1] = -1
        _, _, d, _ = env.step(actions_to_play)
    elif actions == "Nothing":
        nothing_action = np.zeros(env.action_space.shape[0])
        if hasattr(env.agent_body, "sensors"):
            nb_sensors = len(env.agent_body.sensors)
            for i in range(nb_sensors):
                nothing_action[len(nothing_action) - i - 1] = 1
        _, _, d, _ = env.step(nothing_action)
    elif actions == "Forward":
        if i <= repeat:
            _, _, d, _ = env.step(np.array([0, 0, 0, -1]))
            i+=1
        elif i <= 2*repeat:
            _, _, d, _ = env.step(np.array([0, 0, 0, 1]))
            i+=1
        else:
            i = 0
    elif actions == "Seq":
        if j < len(seq):
            _, _, d, _ = env.step(np.array(seq[j]))
        else:
            _, _, d, _ = env.step(np.array([0, 0, 0, 0]))
    else:
        d = False
    j+=1
    # print("Done ? :" + str(d))
    if video_recorder is not None:
        video_recorder.capture_frame()
    else:
        env.render(draw_lidars=True)
    time.sleep(0.1)
    if j >= 100 or d:
        j = 0
        env.reset()
        if video_recorder is not None:
            video_recorder.close()
            video_recorder.enabled = False
        # break


# env = gym.make('bipedal-walker-continuous-v0')
# # walker_types = ['short', 'default', 'quadru', 'quadru']
# walker_types = ['quadru']
# for i,w_type in enumerate(walker_types):
#     env.env.my_init({'leg_size': w_type})  # set walker type
#     env.set_environment(stump_height=np.random.uniform(0,3), obstacle_spacing=np.random.uniform(0,6)) # Stump Tracks
#     if i == len(walker_types)-1:
#         env.set_environment(poly_shape=np.random.uniform(0,4,12))  # Hexagon Tracks
#     env.reset()
#     for i in range(250):
#         #env.step(env.env.action_space.sample())
#         env.render()