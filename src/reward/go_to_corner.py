# import random
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
#
# from typing import Dict, Tuple
# from collections import deque, namedtuple
# from kesslergame import KesslerController, KesslerGame, Scenario
#
# # See: https://github.com/ThalesGroup/kessler-game/blob/main/examples/test_controller.py
# # class CornerNet(nn.Module):
# #     def __init__(self, input_dim, output_dim)
# #         super().__init__()
# #
#
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#
# # class CornerNet(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.online = self.dqn(4, 4)
# #         self.target = self.dqn(4, 4)
# #         self.target.load_state_dict(self.online.state_dict())
# #         for p in self.target.parameters():
# #             p.requires_grad = False
# #
# #     def forward(self, inp, model):
# #         if model == "online":
# #             return self.online(inp)
# #         elif model == "target":
# #             return self.target(inp)
# #
# #     def dqn(self, input_dim, output_dim):
# #         return
#
# class CornerController(KesslerController):
#     def __init__(self, exploration_rate=0.2):
#         self.network = nn.Sequential(
#             nn.Linear(4, 24, dtype=torch.float64),
#             nn.ReLU(),
#             nn.Linear(24, 16, dtype=torch.float64),
#             nn.ReLU(),
#             nn.Linear(16, 4, dtype=torch.float64)
#         )
#
#         self.replay = deque(maxlen=3000)
#         self.exploration_rate=exploration_rate
#
#         self.loss_fn = torch.nn.SmoothL1Loss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
#
#
#     def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
#             x, y = list(ship_state['position'])
#             heading, speed = ship_state['heading'], ship_state['speed']
#             brain_input = [x, y, heading, speed]
#             brain_input = torch.tensor(brain_input, dtype=torch.float64)
#
#             if np.random.random() < self.exploration_rate:
#                 action = np.random.randint(4)
#             else:
#                 with torch.no_grad():
#                     brain_output = self.model(brain_input)
#                 action = torch.argmax(brain_output).item()
#
#             if action == 0:
#                 return 180, 0, False, False
#             if action == 1:
#                 return 0, -135, False, False
#             if action == 2:
#                 return 0, 135, False, False
#             else:
#                 return 0, 0, False, False
#
#     @property
#     def name(self) -> str:
#         return "I Love The Corner"
#
# def main():
#     game = KesslerGame()
#     scenario = Scenario(num_asteroids=0, map_size=(800, 800), time_limit=60)
#     controller = CornerController()
#
#     run_step = True
#     for score, perf_list, game_state in game.run(scenario=scenario, controllers=[controller], run_step=run_step, stop_on_no_asteroids=False):
#         print(game_state['ships'][0]['position'])
#         pass
#
#
# if __name__ == '__main__':
#     main()
#
#
# # from typing import Dict, Tuple
# # from kesslergame import KesslerController
# #
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import numpy as np
# # from collections import deque
# #
#
# # class TestController(KesslerController):
# #     def __init__(self, exploration_rate=0.2):
# #         print("Hello, there!")
# #         self.replay = deque(maxlen=3000)
# #         self.exploration_rate = exploration_rate
# #
# #         # Inputs:
# #         #   Ship: X, Y, Heading, Speed
# #         #   For each asteroid: X, Y, Vx, Vy, Size
# #         self.n_asteroids = 5
# #         input_dim = 4 + self.n_asteroids * 5
# #
# #         # Outputs:
# #         #   Trust: Full forward, half forward, neutral, half backward, full backward
# #         #   Turn:  Full left, half left, neutral, half right, full right
# #         output_dim = 5
# #
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, 32, dtype=torch.float64),
# #             nn.ReLU(),
# #             nn.Linear(32, 24, dtype=torch.float64),
# #             nn.ReLU(),
# #             nn.Linear(24, 16, dtype=torch.float64),
# #             nn.ReLU(),
# #             nn.Linear(16, output_dim, dtype=torch.float64),
# #             nn.Softmax(dim=0),
# #         )
# #
# #         self.loss_fn = torch.nn.SmoothL1Loss()
# #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
# #
# #     def reset(self):
# #         self.replay.clear()
# #
# #     def train(self):
# #         max_frames = len(self.replay)
# #         for frame in range(0, max_frames - 61, 4):
# #             state, thrust, turn = self.replay[frame]['state'], self.replay[frame]['thrust'], self.replay[frame]['turn_rate']
# #             next_state = self.replay[frame + 1]['state']
# #
# #             brain_output = self.model(state)
# #             # expected_reward = (brain_output[thrust] + brain_output[turn]) / 2
# #             expected_reward = brain_output[turn]
# #
# #             next_output = self.model(next_state)
# #             # next_thrust = torch.argmax(next_output[:5])
# #             # next_turn_rate = torch.argmax(next_output[5:])
# #             next_turn_rate = torch.argmax(next_output)
# #             # actual_reward = (next_output[next_thrust] + next_output[next_turn_rate]) / 2
# #             actual_reward = next_output[next_turn_rate]
# #             actual_reward = 0.99 * actual_reward
# #             if frame + 30 < max_frames:
# #                 actual_reward += 1
# #             else:
# #                 actual_reward -= 50
# #
# #             loss = self.loss_fn(expected_reward, actual_reward)
# #             self.optimizer.zero_grad()
# #             loss.backward()
# #             self.optimizer.step()
# #
# #     def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
# #         x, y = list(ship_state['position'])
# #         heading, speed = ship_state['heading'], ship_state['speed']
# #         brain_input = [x, y, heading, speed]
# #         for i in range(self.n_asteroids):
# #             asteroid = game_state['asteroids'][i]
# #             asteroid_x, asteroid_y = list(asteroid['position'])
# #             asteroid_vx, asteroid_vy = list(asteroid['velocity'])
# #             asteroid_info = [asteroid_x, asteroid_y, asteroid_vx, asteroid_vy, asteroid['size']]
# #             brain_input += asteroid_info
# #         brain_input = self.normalize(brain_input)
# #         brain_input = torch.tensor(brain_input, dtype=torch.float64)
# #
# #         if np.random.random() < self.exploration_rate:
# #             # thrust = torch.tensor(np.random.randint(0, 5))
# #             turn_rate = torch.tensor(np.random.randint(0, 5))
# #         else:
# #             with torch.no_grad():
# #                 brain_output = self.model(brain_input)
# #             # thrust = torch.argmax(brain_output[:5])
# #             turn_rate = torch.argmax(brain_output)
# #
# #         thrust = torch.tensor([3])
# #         fire = False
# #         drop_mine = True
# #
# #         self.replay.append({
# #             'state': brain_input,
# #             'thrust': thrust,
# #             'turn_rate': turn_rate,
# #         })
# #         return (thrust.item() - 2) * 240, (turn_rate.item() - 2) * 90, fire, drop_mine
# #
# #     def normalize(self, brain_input):
# #         brain_input[0] = (brain_input[0] - 500) / 500
# #         brain_input[1] = (brain_input[1] - 400) / 400
# #         brain_input[2] = (brain_input[2]) / 360
# #         brain_input[3] = (brain_input[3]) / 240
# #         for i in range(self.n_asteroids):
# #             idx = (i * 5) + 4
# #             brain_input[idx] = (brain_input[idx] - 500) / 500
# #             brain_input[idx + 1] = (brain_input[idx + 1] - 400) / 400
# #             brain_input[idx + 2] = (brain_input[idx + 2]) / 120
# #             brain_input[idx + 3] = (brain_input[idx + 3]) / 120
# #             brain_input[idx + 4] = (brain_input[idx + 4]) / 4
# #         return brain_input
# #
# #     @property
# #     def name(self) -> str:
# #         return "Test Controller"
