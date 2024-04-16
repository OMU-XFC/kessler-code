import time
import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from src.fuzzy_rule_extractor import *
from src.fuzzy import *

class TestRuleExtraction(unittest.TestCase):
    def test_fuzzy_sets(self):
        assert_almost_equal(get_membership((0.0, 1.0), 0.0), 1.0)
        assert_almost_equal(get_membership((0.0, 1.0), 0.5), 0.5)
        assert_almost_equal(get_membership((0.0, 1.0), 1.0), 0.0)

        assert_almost_equal(get_membership((0.7, 0.4), 0.0), 0.0)
        assert_almost_equal(get_membership((0.7, 0.4), 0.3), 0.0)
        assert_almost_equal(get_membership((0.7, 0.4), 0.5), 0.5)
        assert_almost_equal(get_membership((0.7, 0.4), 0.7), 1.0)
        assert_almost_equal(get_membership((0.7, 0.4), 1.0), 0.25)

    def test_compatibility_rule(self):
        np.random.seed(0)
        rule = np.array([0, 0, 2, 4])
        patterns = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Result: 0
            [1.0, 1.0, 1.0, 1.0],  # Result: 0
            [0.5, 0.5, 0.5, 0.5],  # Result: 1 * 1 * 0.5 * 1.0 = 0.5
            [0.3, 0.4, 0.8, 0.8],  # Result: 1 * 1 * 0.8 * 0.4 = 0.32
        ])
        assert_allclose(get_compatibility_rule(rule, patterns), np.array([0, 0, 0.5, 0.32]))

        rule = np.array([1, 2, 3, 0])
        patterns = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Result: 0
            [0.3, 0.6, 0.0, 1.0],  # Result: 0.7 * 0.6 * 1.0 = 0.42
            [0.4, 1.0, 0.2, 0.5],  # Result: 0.6 * 1.0 * 0.6 = 0.36
            [0.7, 0.5, 0.6, 0.6],  # Result: 0
        ])
        assert_allclose(get_compatibility_rule(rule, patterns), np.array([0, 0.42, 0.36, 0]))


    def test_compatibility_ruleset(self):
        ruleset = np.array([
            [0, 0, 2, 4, 0],
            [1, 2, 3, 0, 0],
        ])

        patterns = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.3],
            [0.3, 0.6, 0.0, 1.0, 0.5],
            [0.4, 1.0, 0.2, 0.5, 0.7],
            [0.7, 0.5, 0.6, 0.6, 0.4],
        ])
        assert_allclose(get_compatibility_ruleset(ruleset, patterns), np.array([
            [0, 0, 0.2, 0.48],
            [0, 0.42, 0.36, 0],
        ]))

        for _ in range(20):
            ruleset = np.random.randint(low=0, high=6, size=(10, 4))
            patterns = np.random.uniform(size=(100, 4))
            compat = np.zeros(shape=(10, 100))
            for i in range(10):
                compat[i] = get_compatibility_rule(ruleset[i], patterns)
            assert_allclose(compat, get_compatibility_ruleset(ruleset, patterns))

    def test_compatibility_population(self):
        for _ in range(20):
            population = np.random.randint(low=0, high=6, size=(10, 20, 4))
            patterns = np.random.uniform(size=(100, 4))
            compat = np.zeros(shape=(10, 20, 100))
            for i in range(10):
                for j in range(20):
                    compat[i, j] = get_compatibility_rule(population[i, j], patterns)
            assert_allclose(compat, get_compatibility_population(population, patterns))

    def test_heuristic_weights(self):
        population = np.array([
            [
                [0, 0, 0, 0, 0],  # 0 points
                [0, 0, 1, 0, 1],  # 2 points
                [0, 2, 1, 1, 1],  # 4 points
                [1, 3, 3, 3, 5],  # 9 points
            ],
            [
                [6, 0, 0, 0, 0],  # 3 points
                [7, 8, 9, 0, 0],  # 9 points
                [10, 0, 0, 0, 10],  # 8 points
                [10, 11, 12, 13, 14],  # 20 points
            ]
        ])
        expected = np.array([
            [[0], [2], [4], [9]],
            [[3], [9], [8], [20]],
        ])
        actual = heuristic_weights(population)
        assert_allclose(expected, heuristic_weights(population))

    def test_compatibility(self):
        test = np.random.random(size=(5,))
        print(test)
        print(mutate_floats(test))
    pass
