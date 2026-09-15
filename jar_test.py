import unittest
import numpy as np

from jar import Jar 

class TestJarEnvironment(unittest.TestCase):

    def setUp(self):
        # Initialize Jar with deterministic parameters for testing
        self.jar = Jar(size=1024, num_morphogenes=2, threshold=0.01)
        
        # Override random rates to ensure predictable behavior
        self.jar.diffusion_speeds = np.array([2.0, 5.0])
        self.jar.decay_rates = np.array([0.1, 0.5])

    def test_initialization(self):
        self.assertEqual(self.jar.size, 1024)
        self.assertEqual(self.jar.num_morphogenes, 2)
        self.assertEqual(self.jar.current_tick, 0)
        self.assertTrue(0 in self.jar.releases)
        self.assertTrue(1 in self.jar.releases)

    def test_add_morphogene(self):
        self.jar.add_morphogene((500, 500, 500), morph_id=0, amount=100.0)
        
        sources = self.jar.releases[0]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["amount"], 100.0)
        self.assertEqual(sources[0]["release_tick"], 0)
        np.testing.assert_array_equal(sources[0]["pos"], np.array([500, 500, 500]))

    def test_tick_decay_and_pruning(self):
        self.jar.add_morphogene((10, 10, 10), morph_id=0, amount=1.0)
        self.jar.add_morphogene((20, 20, 20), morph_id=1, amount=0.015) # Barely above threshold
        
        self.jar.tick()
        
        # Morph 0: 1.0 * (1 - 0.1) = 0.9 (kept)
        self.assertEqual(len(self.jar.releases[0]), 1)
        self.assertAlmostEqual(self.jar.releases[0][0]["amount"], 0.9)
        
        # Morph 1: 0.015 * (1 - 0.5) = 0.0075 (below 0.01 threshold, pruned)
        self.assertEqual(len(self.jar.releases[1]), 0)
        self.assertEqual(self.jar.current_tick, 1)

    def test_gradient_no_sources_or_below_threshold(self):
        # No sources
        grad = self.jar.get_gradient((100, 100, 100), morph_id=0)
        self.assertEqual(grad, (1, 1, 1))
        
        # Source exists but distance makes concentration extremely low (below threshold)
        self.jar.add_morphogene((0, 0, 0), morph_id=0, amount=0.1)
        self.jar.tick()
        grad_far = self.jar.get_gradient((1000, 1000, 1000), morph_id=0)
        self.assertEqual(grad_far, (1, 1, 1))

    def test_gradient_direction(self):
        # Place a massive source so concentration stays above threshold
        self.jar.add_morphogene((50, 50, 50), morph_id=0, amount=10000.0)
        for _ in range(10):
            self.jar.tick()
        
        # Target at (40, 50, 50): To reach the source (50), we must move +10 in X.
        # So the gradient along X should be positive (+1).
        grad_left = self.jar.get_gradient((40, 50, 50), morph_id=0)
        print(f"{grad_left = }")
        self.assertEqual(grad_left[0], 1)
        self.assertEqual(grad_left[1], -1)
        
        # Target at (60, 50, 50): To reach the source (50), we must move -10 in X.
        # So the gradient along X should be negative (-1).
        grad_right = self.jar.get_gradient((60, 50, 50), morph_id=0)
        print(f"{grad_right = }")
        self.assertEqual(grad_right[0], -1)
        self.assertEqual(grad_right[1], 1)
        
        # Target at (50, 30, 80): 
        # Y must move +20 (positive gradient)
        # Z must move -30 (negative gradient)
        grad_mixed = self.jar.get_gradient((50, 30, 80), morph_id=0)
        print(f"{grad_mixed = }")
        self.assertEqual(grad_mixed[1], 1)
        self.assertEqual(grad_mixed[2], -1)

if __name__ == '__main__':
    unittest.main()