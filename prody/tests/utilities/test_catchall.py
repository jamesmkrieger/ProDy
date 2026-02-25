"""This module contains some unit tests for :mod:`prody.utilities.catchall` module,
starting with calcTree."""

import numpy as np

from prody.tests import unittest
from prody.utilities import calcTree

class TestCalcTree(unittest.TestCase):

    def testCalcTreeUPGMA(self):
        """Test calcTree with UPGMA method."""
        names = ['A', 'B', 'C', 'D']
        distance_matrix = np.array([[0,   1,   2, 1],
                                    [1,   0, 1.5, 2],
                                    [2, 1.5,   0, 2],
                                    [1,   2,   2, 0]])
        tree = calcTree(names, distance_matrix, method='upgma')
        self.assertIsNotNone(tree)
        # Check that tree has 4 leaves and they include the names
        leaves = tree.get_terminals()
        self.assertEqual(len(leaves), 4)
        self.assertEqual(set([leaf.name for leaf in leaves]), set(names))
        # Check that the tree has split evenly as expected for UPGMA
        self.assertEqual(len(tree.root.clades), 2)

    def testCalcTreeNJ(self):
        """Test calcTree with NJ method."""
        names = ['A', 'B', 'C', 'D']
        distance_matrix = np.array([[0,   1,   2, 1],
                                    [1,   0, 1.5, 2],
                                    [2, 1.5,   0, 2],
                                    [1,   2,   2, 0]])
        tree = calcTree(names, distance_matrix, method='nj')
        self.assertIsNotNone(tree)
        leaves = tree.get_terminals()
        # Check that tree has 4 leaves and they include the names
        self.assertEqual(len(leaves), 4)
        self.assertEqual(set([leaf.name for leaf in leaves]), set(names))
        # Check that the tree has split unevenly as expected for NJ
        self.assertEqual(len(tree.root.clades), 3)

    def testCalcTreeMismatchSize(self):
        """Test calcTree with mismatched names and matrix sizes."""
        names = ['A', 'B']
        distance_matrix = np.array([[0, 1, 2],
                                    [1, 0, 1.5],
                                    [2, 1.5, 0]])
        with self.assertRaises(ValueError):
            calcTree(names, distance_matrix)

if __name__ == '__main__':
    unittest.main()
