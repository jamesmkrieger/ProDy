# -*- coding: utf-8 -*-
"""This module defines classes for handling polypeptide/nucleic acid chains."""

from numpy import arange, unique

from .subset import AtomSubset
from .atomic import AAMAP

__all__ = ['Chain']

def getSequence(resnames, **kwargs):
    """Returns polypeptide sequence as from list of *resnames* (residue
    name abbreviations)."""

    threeLetter = kwargs.get('threeLetter', False)
    if threeLetter:
        return ' '.join(resnames)

    get = AAMAP.get
    return ''.join([get(rn, 'X') for rn in resnames])


class Chain(AtomSubset):

    """Instances of this class point to atoms with same chain identifiers and
    are generated by :class:`.HierView` class.  Following built-in functions
    are customized for this class:

    * :func:`len` returns the number of residues in the chain
    * :func:`iter` yields :class:`.Residue` instances

    Indexing :class:`Chain` instances by:

         - *residue number [, insertion code]* (:func:`tuple`),
           e.g. ``10`` or  ``10, "B"``, returns a :class:`.Residue`
         - *slice* (:func:`slice`), e.g, ``10:20``, returns a list of
           :class:`.Residue` instances"""

    __slots__ = ['_ag', '_indices', '_hv', '_acsi', '_selstr', '_seq']

    def __init__(self, ag, indices, hv, acsi=None, **kwargs):

        AtomSubset.__init__(self, ag, indices, acsi, **kwargs)
        self._hv = hv
        self._seq = None

    def __repr__(self):

        n_csets = self._ag.numCoordsets()
        segment = self.getSegment()
        if segment is None:
            segment = ''
        else:
            segment = ' from ' + str(segment)
        if n_csets == 1:
            return ('<Chain: {0}{1} from {2} ({3} residues, {4} atoms)>'
                    ).format(self.getChid(), segment, self._ag.getTitle(),
                             self.numResidues(), self.numAtoms())
        elif n_csets > 1:
            return ('<Chain: {0}{1} from {2} ({3} residues, {4} '
                    'atoms; active #{5} of {6} coordsets)>'
                    ).format(self.getChid(), segment, self._ag.getTitle(),
                             self.numResidues(), self.numAtoms(),
                             self.getACSIndex(), n_csets)
        else:
            return ('<Chain: {0}{1} from {2} ({3} residues, '
                    '{4} atoms; no coordinates)>'
                    ).format(self.getChid(), segment, self._ag.getTitle(),
                             self.numResidues(), self.numAtoms())

    def __str__(self):

        return 'Chain ' + self.getChid()

    def __getitem__(self, key):

        if isinstance(key, tuple):
            return self.getResidue(*key)
        else:
            return AtomSubset.__getitem__(self, key)

    def getSegment(self):
        """Returns segment of the chain."""

        segname = self.getSegname()
        if segname is not None:
            return self._hv.getSegment(segname)

    def getSegname(self):
        """Returns segment name."""

        segnames = self._ag._getSegnames()
        if segnames is not None:
            return segnames[self._indices[0]]

    def getResidue(self, resnum, icode=None):
        """Returns residue with number *resnum* and insertion code *icode*."""

        return self._hv.getResidue(self.getChid(), resnum, icode,
                                   self.getSegname())

    def iterResidues(self):
        """Yield residues."""

        get = self._hv._getResidue
        for index in unique(self._getResindices()):
            yield get(index)

    __iter__ = iterResidues

    def numResidues(self):
        """Returns number of residues."""

        return len(set(self._getResindices()))

    __len__ = numResidues

    def getChid(self):
        """Returns chain identifier."""

        return self._ag._getChids()[self._indices[0]]

    def setChid(self, chid):
        """Set chain identifier."""

        self.setChids(chid)

    def getChindex(self):
        """Returns chain index."""

        return self._ag._getChindices()[self._indices[0]]

    def getSequence(self, **kwargs):
        """Returns one-letter sequence string for amino acids in the chain.
        When *allres* keyword argument is **True**, sequence will include all
        residues (e.g. water molecules) in the chain and **X** will be used for
        non-standard residue names."""

        if kwargs.get('allres', False):
            get = AAMAP.get
            seq = ''.join([get(res.getResname(), 'X') for res in self])
        elif self._seq:
            seq = self._seq
        else:
            calpha = self.calpha
            if calpha:
                seq = getSequence(calpha.getResnames(), **kwargs)
            else:
                seq = ''
            self._seq = seq
        return seq

    def getSelstr(self):
        """Returns selection string that selects atoms in this chain."""

        segment = self.getSegment()
        if segment is None:
            if self._selstr:
                return 'chain {0} and ({1})'.format(self.getChid(),
                                                    self._selstr)
            else:
                return 'chain {0}'.format(self.getChid())
        else:
            return 'chain {0} and ({1})'.format(self.getChid(),
                                                segment.getSelstr())

    def getHierView(self, **kwargs):
        """Returns a hierarchical view of the this chain."""

        return HierView(self, **kwargs)
        