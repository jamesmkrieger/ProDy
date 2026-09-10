"""This module contains unit tests for :mod:`~prody.trajectory.psffile`."""

from os.path import join

from numpy.testing import assert_equal

from prody import parsePSF, writePSF
from prody.tests import TEMPDIR, TestCase
from prody.tests.datafiles import DATA_FILES, pathDatafile


PSF = pathDatafile('topology_psf')
COUNTS = DATA_FILES['topology_psf']

# the topology of the fixture, in the order it appears in the file and using
# 1-based indices as the file does
BONDS = [[2, 1], [3, 1], [4, 3], [5, 3], [6, 5]]
ANGLES = [[2, 1, 3], [1, 3, 5], [4, 3, 5], [3, 5, 6]]
DIHEDRALS = [[2, 1, 3, 5], [1, 3, 5, 6], [4, 3, 5, 6]]
IMPROPERS = [[5, 3, 7, 6]]
DONORS = [[1, 2]]
ACCEPTORS = [[6, 5]]
CROSSTERMS = [[5, 7, 9, 10, 1, 3, 5, 7]]

SECTIONS = ('bonds', 'angles', 'dihedrals', 'impropers',
            'donors', 'acceptors', 'crossterms')


def topology(ag):
    """Returns the topology of *ag* section by section, as 1-based lists."""

    return {name: (getattr(ag, '_' + name) + 1).tolist() for name in SECTIONS}


class TestPSFFile(TestCase):

    def setUp(self):

        self.ag = parsePSF(PSF)
        self.top = topology(self.ag)

    def testCounts(self):
        """Every section is read, and read whole."""

        self.assertEqual(self.ag.numAtoms(), COUNTS['atoms'])
        for name in SECTIONS:
            self.assertEqual(len(self.top[name]), COUNTS[name],
                             'wrong number of {0}'.format(name))

    def testCrosstermWidth(self):
        """A CMAP cross-term is the eight atoms of two coupled dihedrals."""

        assert_equal(self.ag._crossterms.shape, (COUNTS['crossterms'], 8))

    def testAtomOrderWithinTerms(self):
        """The atom order within a term identifies it and must be preserved.

        An angle's middle index is its vertex, a torsion's four indices are a
        sequence, and a cross-term's eight are two dihedrals; sorting them
        turns each into a different term."""

        assert_equal(self.top['angles'], ANGLES)
        assert_equal(self.top['dihedrals'], DIHEDRALS)
        assert_equal(self.top['impropers'], IMPROPERS)
        assert_equal(self.top['crossterms'], CROSSTERMS)
        # donors and acceptors are ordered pairs too
        assert_equal(self.top['donors'], DONORS)
        assert_equal(self.top['acceptors'], ACCEPTORS)

    def testBondsAreCanonicalised(self):
        """A bond is symmetric, so its pair is sorted and deduplicated."""

        assert_equal(self.top['bonds'], sorted(sorted(b) for b in BONDS))

    def testAngleVertex(self):
        """The vertex an Angle reports is the one the file gives it."""

        vertices = [angle.getAtoms()[1].getIndex() + 1
                    for angle in self.ag.iterAngles()]
        assert_equal(vertices, [a[1] for a in ANGLES])

    def testRoundTrip(self):
        """Writing a parsed topology and reading it back changes nothing."""

        out = writePSF(join(TEMPDIR, 'topology_roundtrip.psf'), self.ag)
        again = topology(parsePSF(out))
        for name in SECTIONS:
            assert_equal(again[name], self.top[name],
                         '{0} did not round-trip'.format(name))

    def testRoundTripIsIdempotent(self):
        """A second write/read pass is a no-op, not a further drift."""

        first = parsePSF(writePSF(join(TEMPDIR, 'topology_rt1.psf'), self.ag))
        second = parsePSF(writePSF(join(TEMPDIR, 'topology_rt2.psf'), first))
        assert_equal(topology(second), topology(first))
