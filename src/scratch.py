from typing import Dict, Tuple

import numpy as np
import time

from kesslergame import KesslerController, Scenario, KesslerGame

from center_coords import center_coords
from src.envs.radar_env import get_radar
from src.lib import parse_game_state
from src.navigation_scenario import benchmark

# Fuzzy sets -- For now (probably not optimal)
# 0. Don't care,
# 1. Low (0 - 0.125)
# 2. Medium (0.125 - 0.375)
# 3. High (0.25 - 0.5+)
# Inputs (again, for now...)
# 12 point radar
# Outputs:
# Turn Left, Don't Turn, Turn Right
# Forward, Neutral, Backwards
# So this very simple scenario... 4**12 * 3 * 3 possible rules (just under 151 million)
# Encoding....
# One idea: [2 bits - first input] [2 bits - second input] ... etc. 28 bit encoding (just store in 32-bit integer)

FUZZY_SETS = [
    # Triangular fuzzy sets can be defined as functions of their center and radius
    lambda x: np.ones_like(x),  # Don't care
    lambda x: np.maximum(1 - np.abs((0.125 - x) / 0.25), 0),
    lambda x: np.maximum(1 - np.abs((0.250 - x) / 0.25), 0),
    lambda x: np.maximum(1 - np.abs((0.375 - x) / 0.25), 0),
]
N_ATTR = 12
N_POSSIBLE_RULES = 4 ** 14
N_RULESETS = 20
STARTING_RULES_PER_RULESET = 5
MAX_RULES_PER_RULESET = 100

def ga():
    population = []
    for i in range(N_RULESETS):
        population.append(
            np.random.randint(low=0, high=N_POSSIBLE_RULES + 1, size=(STARTING_RULES_PER_RULESET,), dtype=np.int32)
        )
    for generation in range(100):
        population_fitness = np.array([
            fitness(ruleset) for ruleset in population
        ])
        print(f'{np.min(population_fitness)} {np.mean(population_fitness)} {np.max(population_fitness)}')
        parent_A_idx = binary_tournament(population_fitness)
        parent_B_idx = binary_tournament(population_fitness)
        next_generation = []
        for child_idx in range(N_RULESETS):
            child = crossover(population[parent_A_idx], population[parent_B_idx])
            child = mutate(child)
            child = trim(child)
            next_generation.append(child)
        population = next_generation
    population_fitness = np.array([
        fitness(ruleset) for ruleset in population
    ])
    print(f'{np.min(population_fitness)} {np.mean(population_fitness)} {np.max(population_fitness)}')

def fitness(ruleset):
    controller = FuzzyController(decode_rules(ruleset))
    return np.mean(benchmark(controller, n_trials=5))

def binary_tournament(fitness_values):
    n = len(fitness_values)
    contestant_A = np.random.randint(low=0, high=n)
    contestant_B = np.random.randint(low=0, high=n)
    if fitness_values[contestant_A] > fitness_values[contestant_B]:
        return contestant_A
    return contestant_B

def crossover(parent_A, parent_B):
    new_rules = []
    # Rules in both will always be in the child. Likewise, rules in neither will never be in the child.
    new_rules.extend(np.intersect1d(parent_A, parent_B))

    # There is *probably* a more efficient way!
    # Take 90% of the rules which are only in parent A (pick A's "1" bit 90% of the time, else B's "0" bit)
    in_only_parent_A = np.setdiff1d(parent_A, parent_B)
    rands = np.random.random(size=in_only_parent_A.shape) > 0.1
    new_rules.extend(in_only_parent_A[rands])

    in_only_parent_B = np.setdiff1d(parent_B, parent_A)
    rands = np.random.random(size=in_only_parent_B.shape) > 0.9
    new_rules.extend(in_only_parent_B[rands])

    return np.array(new_rules, dtype=np.int32)

def mutate(subject):
    # Pick 10 fuzzy rules and add them to the ruleset.
    # Theoretically, we should remove the ones which already exist, but that is a very rare case...
    mutations = np.random.randint(low=0, high=N_POSSIBLE_RULES + 1, size=(10,))
    return np.union1d(subject, mutations)

def trim(ruleset):
    if ruleset.shape[0] > MAX_RULES_PER_RULESET:
        return np.random.choice(ruleset, size=(MAX_RULES_PER_RULESET,), replace=False)
    return ruleset

def rule_weights(rules):
    # In this case, all fuzzy sets have the same area
    nonzero = np.count_nonzero(rules, axis=-1)
    a = 0.5 ** nonzero
    return 1 - a

def get_compatibility(decoded_rule, observation):
    antecedents = decoded_rule[:, :-2]
    ob2 = np.broadcast_to(observation, shape=antecedents.shape)
    compat = np.ones_like(antecedents, dtype=np.float64)
    for i in range(4):
        mask = antecedents == i
        lmb = FUZZY_SETS[i]
        compat[mask] = lmb(ob2[mask])
    return np.prod(compat, axis=-1)

def get_output(ruleset, observation):
    compatibility = get_compatibility(ruleset, observation)
    weights = rule_weights(ruleset)
    weighted_compatibility = compatibility * weights
    winner_idx = np.argmax(weighted_compatibility)
    return ruleset[winner_idx, -2:]

def encode_rules(rule):
    rule_length = rule.shape[-1]
    encoder = 4 ** np.arange(rule_length)
    return np.sum(rule * encoder, axis=-1)

def decode_rules(rule_idx):
    rule_length = 14
    n_rules = rule_idx.shape[0]
    rule = np.empty(shape=(n_rules, rule_length), dtype=np.int8)
    for i in range(rule_length):
        val = rule_idx & 3
        rule[:, i] = val
        rule_idx = rule_idx >> 2
    return rule

def parse_output(fuzzy_output):
    thrust, turn = None, None

    if fuzzy_output[0] == 0:
        thrust = -480
    elif fuzzy_output[1] == 1:
        thrust = 480
    else:
        thrust = 0

    if fuzzy_output[1] == 0:
        turn = -180
    elif fuzzy_output[1] == 1:
        turn = 180
    else:
        turn = 0

    return thrust, turn



class FuzzyController(KesslerController):
    def __init__(self, ruleset):
        self.rules = ruleset

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        state = parse_game_state(ship_state, game_state)
        radar = get_radar(state['asteroids']['polar_positions'], state['asteroids']['radii'], [100, 250, 400])
        radar[radar > 0.5] = 0.5
        fuzzy_output = get_output(self.rules, radar)
        thrust, turn = parse_output(fuzzy_output)

        return thrust, turn, False, False

    @property
    def name(self) -> str:
        return "Please Work"
    pass

def main():
    ga()

if __name__ == '__main__':
    main()
