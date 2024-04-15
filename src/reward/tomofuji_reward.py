import numpy as np

def tomofuji_reward(game_state, prev_state, obs):
    prev_ship = prev_state['ships'][0]
    prev_ast_list = np.array(prev_state["asteroids"])
    prev_astnum = len(prev_ast_list)
    ast_list = np.array(game_state["asteroids"])
    ship = game_state['ships'][0]
    prev_bullet_list = np.array(prev_state["bullets"])
    bullet_list = np.array(game_state["bullets"])
    hit_ast = False
    collision = False

    # collision detection by is_respawning status
    if not prev_ship['is_respawning'] and ship['is_respawning']:
        collision = True

    # compare the numbers of asteroids in the current and previous steps
    # and detect the asteroids that have been destroyed
    if (len(ast_list) == prev_astnum + 2 or len(ast_list) == prev_astnum - 1) and not collision:
        hit_ast = True


    reward = 0.1

    if len(prev_bullet_list) < len(bullet_list) and not hit_ast:
        # check if fire a bullet, and give a negative reward to improve accuracy
        reward -= 0.1

    if hit_ast:
        reward += 1.0

    min = np.argmin(obs['asteroid_dist'])

    if obs['asteroid_dist'][min] < 50:
        reward -= 100.0

    elif obs['asteroid_dist'][min] < 100:
        reward -= 50.0

    if 350 <obs['asteroid_angle'][min] or obs['asteroid_angle'][min] < 10:
        reward += 10.0



    if collision:
        reward -= 1000.0

    return reward