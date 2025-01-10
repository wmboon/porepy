""" Ad representation of grid-related quantities needed to write equations. The classes
defined here are mainly wrappers that constructs Ad matrices based on grid information.

"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .operators import Operator, SparseArray

__all__ = [
    "BoundaryProjection",
    "MortarProjections",
    "Divergence",
    "Trace",
    "SubdomainProjections",
]


class SubdomainProjections:
    """Wrapper class for generating projection to and from subdomains.

    The class should be used through the methods {cell, face}_{projection, restriction}.

    Parameters:
        subdomains: List of grids for which the projections should map to and from.
        dim: Dimension of the quantities to be mapped. Will typically be 1 (for scalar
            quantities) or Nd (the ambient dimension, for vector quantities).

    Raises:
        ValueError: If a subdomain occur more than once in the input list.

    See also:
        MortarProjections for projections to and from mortar subdomains.

    """

    def __init__(
        self,
        subdomains: Sequence[pp.Grid],
        dim: int = 1,
    ) -> None:
        """Construct subdomain restrictions and prolongations for a set of subdomains.

        The projections will be ordered according to the ordering in subdomains. It is
        critical that the same ordering is used by other operators.

        Parameters:
            subdomains: List of subdomains. The order of the subdomains in
                the list will establish the ordering of the subdomain projections.
            dim: Dimension of the quantities to be projected.

        """

        self._name = "SubdomainProjection"
        self.dim = dim
        self._is_scalar: bool = dim == 1
        
        # Store the list of subdomains. This will be needed to construct the projection
        # matrices.
        self._all_subdomains = subdomains

        # Uniquify the list of subdomains. There is no need to have the same subdomain
        # represented several times.
        if len(set(subdomains)) < len(subdomains):
            # The problem here is that the subdomain projections are stored in a dict,
            # with the subdomanis as keys. If the same subdomain is represented twice,
            # the first projection will be overwritten by the second, thus the order
            # of the subdomains will be lost. There is no easy way to handle this, the
            # only option is to fix the error on the caller side. An internal fix would
            # entail changing the storage format for the projection, potentially needing
            # a lot of memory.
            raise ValueError("Subdomains must be unique")

        self._num_grids: int = len(subdomains)

        # Store total number of faces and cells in the list of subdomains. This will be
        # needed to handle projections to and from empty lists (see usage below).
        self._tot_num_cells: int = sum([g.num_cells for g in subdomains])
        self._tot_num_faces: int = sum([g.num_faces for g in subdomains])

        # Initialize storage for the projection matrices. These will be constructed
        # lazily, when the projection is requested, and then stored for later use.
        self._cell_projections: Optional[dict[pp.Grid, sps.spmatrix]] = None
        self._face_projections: Optional[dict[pp.Grid, sps.spmatrix]] = None

    def cell_restriction(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct restrictions from global to subdomain cell quantities.

        Parameters:
            subdomains: One or several subdomains to which the projection should apply.

        Raises:
            ValueError: If subdomains is not a list.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represents the
                projection.

        """
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

        if self._cell_projections is None:
            # Construct and store projection matrices for cells.
            self._cell_projections = _cell_projections(self._all_subdomains, self.dim)        

        if len(subdomains) > 0:
            # A key error will be raised if a grid in g is not known to
            # self._cell_projection
            # IMPLEMENTATION NOTE: Use csr format, since the number of rows can
            # be much less than the number of columns.
            mat = sps.bmat([[self._cell_projections[g].T] for g in subdomains]).tocsr()
        else:
            # If the grid list is empty, we project from the full set of cells to
            # nothing.
            mat = sps.csr_matrix((0, self._tot_num_cells * self.dim))
        return pp.ad.SparseArray(mat, name="CellRestriction")

    def cell_prolongation(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct prolongation from subdomain to global cell quantities.

        Parameters:
            subdomains: One or several subdomains to which the prolongation should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
            prolongation.

        """
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

        if self._cell_projections is None:
            # Construct and store projection matrices for cells.
            self._cell_projections = _cell_projections(self._all_subdomains, self.dim)        

        if len(subdomains) > 0:
            # A key error will be raised if a grid in g is not known to
            # self._cell_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be much less than the number of rows.
            mat = sps.bmat([[self._cell_projections[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # cells. CSC format is used for efficiency.
            mat = sps.csc_matrix((self._tot_num_cells * self.dim, 0))
        return pp.ad.SparseArray(mat, name="CellProlongation")

    def face_restriction(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct restrictions from global to subdomain face quantities.

        Parameters:
            subdomains: One or several subdomains to which
                the projection should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
                projection.

        """
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

        if self._face_projections is None:
            # Construct and store projection matrices for faces.
            self._face_projections = _face_projections(self._all_subdomains, self.dim)        

        if len(subdomains) > 0:
            # A key error will be raised if a grid in subdomains is not known to
            # self._face_projection
            # IMPLEMENTATION NOTE: Use csr format, since the number of rows can
            # be much less than the number of columns.
            mat = sps.bmat([[self._face_projections[g].T] for g in subdomains]).tocsr()
        else:
            # If the grid list is empty, we project from the full set of faces to
            # nothing.
            mat = sps.csr_matrix((0, self._tot_num_faces * self.dim))
        return pp.ad.SparseArray(mat, name="FaceRestriction")

    def face_prolongation(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct prolongation from subdomain to global face quantities.

        Parameters:
            subdomains: One or several subdomains to which the prolongation should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
            prolongation.

        """
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

        if self._face_projections is None:
            # Construct and store projection matrices for faces.
            self._face_projections = _face_projections(self._all_subdomains, self.dim)        

        if len(subdomains) > 0:
            # A key error will be raised if a grid in subdomains is not known to
            # self._face_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be far smaller than the number of rows.
            mat = sps.bmat([[self._face_projections[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # faces.
            mat = sps.csc_matrix((self._tot_num_faces * self.dim, 0))
        return pp.ad.SparseArray(mat, name="FaceProlongation")

    def __repr__(self) -> str:
        s = (
            f"Restriction and prolongation operators for {self._num_grids}"
            " unique subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
        )
        return s


class MortarProjections:
    """Wrapper class to generate projections to and from MortarGrids.

    Attributes:
        mortar_to_primary_int (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the primary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_primary_avg (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the primary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        primary_to_mortar_int (pp.ad.SparseArray): Matrix of projections from the primary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        primary_to_mortar_avg (pp.ad.SparseArray): Matrix of projections from the primary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_int (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the secondary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_avg (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the secondary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_int (pp.ad.SparseArray): Matrix of projections from the secondary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_avg (pp.ad.SparseArray): Matrix of projections from the secondary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        sign_of_mortar_sides (pp.ad.SparseArray): Matrix representation that assigns signs
            to two mortar sides. Needed to implement a jump operator in contact
            mechanics.

    """

    def __init__(
        self,
        mdg: pp.MixedDimensionalGrid,
        subdomains: Sequence[pp.Grid],
        interfaces: Sequence[pp.MortarGrid],
        dim: int = 1,
    ) -> None:
        """Construct mortar projection object.

        The projections will be ordered according to the ordering in grids, or the order
        of the MixedDimensionalGrid iteration over grids. It is critical that the same
        ordering is used by other operators.

        Parameters:
            mdg: Mixed-dimensional grid.
            subdomains: List of grids for which the projections should apply. The order
                of the grids in the list establishes the ordering of the subdomain
                projections.
            interfaces: List of edges for which the projections should apply. The order
                of the grids in the list establishes the ordering of the subdomain
                projections.
            dim: Dimension of the quantities to be projected.

        """
        self._name = "MortarProjection"
        self._num_edges: int = len(interfaces)
        self.dim: int = dim

        # Store the list of subdomains and interfaces. This will be needed to construct
        # the projection matrices.
        self._subdomains = subdomains
        self._interfaces = interfaces
        self._mdg = mdg

        # Check if all interfaces are conforming. If so, we can use the same projection
        # for intensive and extensive quantities.
        is_conforming_primary = True
        is_conforming_secondary = True
        for intf in interfaces:
            for proj in [intf.mortar_to_primary_int(), intf.primary_to_mortar_int()]:
                if not np.allclose(proj.data, 1, atol=1e-10):
                    is_conforming_primary = False
            for proj in [intf.mortar_to_secondary_int(), intf.secondary_to_mortar_int()]:
                if not np.allclose(proj.data, 1, atol=1e-10):
                    is_conforming_secondary = False

        self._is_conforming_primary = is_conforming_primary
        self._is_conforming_secondary = is_conforming_secondary  

        # Storage for the projection matrices. These will be constructed lazily, when the
        # projection is requested, and then stored for later use. Depending on whether
        # the interfaces are conforming or not, we need to store different projections.
        if self._is_conforming_primary:
            self._mortar_to_primary: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._primary_to_mortar: Optional[dict[pp.Grid, sps.spmatrix]] = None
        else:
            # Here we need to store both intensive and extensive projections.
            self._mortar_to_primary_int: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._mortar_to_primary_avg: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._primary_to_mortar_int: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._primary_to_mortar_avg: Optional[dict[pp.Grid, sps.spmatrix]] = None

        if self._is_conforming_secondary:
            self._mortar_to_secondary: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._secondary_to_mortar: Optional[dict[pp.Grid, sps.spmatrix]] = None
        else:
            self._mortar_to_secondary_int: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._mortar_to_secondary_avg: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._secondary_to_mortar_int: Optional[dict[pp.Grid, sps.spmatrix]] = None
            self._secondary_to_mortar_avg: Optional[dict[pp.Grid, sps.spmatrix]] = None

    def sign_of_mortar_sides(self) -> SparseArray:
        if len(self._interfaces) == 0:
            return SparseArray(
                sps.bmat([[None]]), name="SignOfMortarSides"
            )
        mats = []
        for intf in self._interfaces:
            assert isinstance(intf, pp.MortarGrid)  # Appease mypy
            mats.append(intf.sign_of_mortar_sides(self.dim))
        else:
            return SparseArray(
                sps.block_diag(mats), name="SignOfMortarSides"
            )

    def _construct_projection(self, proj_func, to_mortar, is_primary, name):

        if is_primary:
            non_mortar_size = self.dim * np.sum([sd.num_faces for sd in self._subdomains], dtype=int)
        else:
            non_mortar_size = self.dim * np.sum([sd.num_cells for sd in self._subdomains], dtype=int)
        # Shortcut for the case where there are no interfaces. We then just need to return
        # a zero matrix of the right size.
        if len(self._interfaces) == 0:
            if to_mortar:
                return SparseArray(
                    sps.csr_matrix((0, non_mortar_size)), name=name)
            else:
                return SparseArray(
                    sps.csc_matrix((non_mortar_size, 0)), name=name
                )                

        codim = np.unique([intf.codim for intf in self._interfaces])
        if codim.size > 1:
            raise ValueError("All interfaces must have the same codimension")
        if codim[0] not in (1, 2):
            raise ValueError("Unsupported codimension")

        if codim[0] == 1 and is_primary:
            projections = _face_projections(self._subdomains, self.dim)
        else:  # codim[0] == 2 or projection to or from secondary
            projections = _cell_projections(self._subdomains, self.dim)

        proj_mats = []
        for intf in self._interfaces:
            if is_primary:
                sd, _ = self._mdg.interface_to_subdomain_pair(intf)
            else:
                _, sd = self._mdg.interface_to_subdomain_pair(intf)
            if sd in self._subdomains:
                if to_mortar:
                    loc_mat = getattr(intf, proj_func)(self.dim) @ projections[sd].T
                else:
                    loc_mat = projections[sd] @ getattr(intf, proj_func)(self.dim)



                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                    loc_mat 
                ))
            else:
                # Can we use a better format than csr/s here?
                if to_mortar:
                    mat = sps.csr_matrix((intf.num_cells * self.dim, non_mortar_size))
                else:
                    mat = sps.csc_matrix((non_mortar_size, intf.num_cells * self.dim))

                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(mat))

        if to_mortar:
            return self._bmat([[m] for m in proj_mats], name=name)
        else:
            return self._bmat([proj_mats], name=name)

    def mortar_to_primary_int(self) -> Operator:
        # Retrieved cached projection if it exists.
        if self._is_conforming_primary and self._mortar_to_primary is not None:
            return self._mortar_to_primary
        elif not self._is_conforming_primary and self._mortar_to_primary_int is not None:
            return self._mortar_to_primary_int

        mat = self._construct_projection("mortar_to_primary_int", False, True, "MortarToPrimaryInt")
        if self._is_conforming_primary:
            self._mortar_to_primary = mat
        else:
            self._mortar_to_primary_int = mat
        return mat

    
    def mortar_to_primary_avg(self) -> Operator:

        # Retrieve cached projection if it exists.
        if self._is_conforming_primary and self._mortar_to_primary is not None:
            return self._mortar_to_primary
        elif not self._is_conforming_primary and self._mortar_to_primary_avg is not None:
            return self._mortar_to_primary_avg

        mat = self._construct_projection("mortar_to_primary_avg", False, True, "MortarToPrimaryAvg")
        if self._is_conforming_primary:
            self._mortar_to_primary = mat
        else:
            self._mortar_to_primary_avg = mat
        return mat

    def primary_to_mortar_int(self) -> Operator:

        # Retrieve cached projection if it exists.
        if self._is_conforming_primary and self._primary_to_mortar is not None:
            return self._primary_to_mortar
        elif not self._is_conforming_primary and self._primary_to_mortar_int is not None:
            return self._primary_to_mortar_int

        mat = self._construct_projection("primary_to_mortar_int", True, True, "PrimaryToMortarInt")
        if self._is_conforming_primary:
            self._primary_to_mortar = mat
        else:
            self._primary_to_mortar_int = mat
        return mat
    
    def primary_to_mortar_avg(self) -> Operator:
        # Retrieve cached projection if it exists.
        if self._is_conforming_primary and self._primary_to_mortar is not None:
            return self._primary_to_mortar
        elif not self._is_conforming_primary and self._primary_to_mortar_avg is not None:
            return self._primary_to_mortar_avg

        mat = self._construct_projection("primary_to_mortar_avg", True, True, "PrimaryToMortarAvg")

        if self._is_conforming_primary:
            self._primary_to_mortar = mat
        else:
            self._primary_to_mortar_avg = mat
        return mat

    def mortar_to_secondary_int(self) -> Operator:
        # Retrieve cached projection if it exists.
        if self._is_conforming_secondary and self._mortar_to_secondary is not None:
            return self._mortar_to_secondary
        elif not self._is_conforming_secondary and self._mortar_to_secondary_int is not None:
            return self._mortar_to_secondary_int

        mat = self._construct_projection("mortar_to_secondary_int", False, False, "MortarToSecondaryInt")
        
        if self._is_conforming_secondary:
            self._mortar_to_secondary = mat
        else:
            self._mortar_to_secondary_int = mat
        return mat

    def mortar_to_secondary_avg(self) -> Operator:
        # Retrieve cached projection if it exists.
        if self._is_conforming_secondary and self._mortar_to_secondary is not None:
            return self._mortar_to_secondary
        elif not self._is_conforming_secondary and self._mortar_to_secondary_avg is not None:
            return self._mortar_to_secondary_avg

        mat = self._construct_projection("mortar_to_secondary_avg", False, False, "MortarToSecondaryAvg")
        if self._is_conforming_secondary:
            self._mortar_to_secondary = mat
        else:
            self._mortar_to_secondary_avg = mat
        return mat

    def secondary_to_mortar_int(self) -> Operator:
        # Retrieve cached projection if it exists.
        if self._is_conforming_secondary and self._secondary_to_mortar is not None:
            return self._secondary_to_mortar
        elif not self._is_conforming_secondary and self._secondary_to_mortar_int is not None:
            return self._secondary_to_mortar_int

        mat = self._construct_projection("secondary_to_mortar_int", True, False, "SecondaryToMortarInt")
        if self._is_conforming_secondary:
            self._secondary_to_mortar = mat
        else:
            self._secondary_to_mortar_int = mat
        return mat

    def secondary_to_mortar_avg(self) -> Operator:
        # Retrieve cached projection if it exists.
        if self._is_conforming_secondary and self._secondary_to_mortar is not None:
            return self._secondary_to_mortar
        elif not self._is_conforming_secondary and self._secondary_to_mortar_avg is not None:
            return self._secondary_to_mortar_avg

        mat = self._construct_projection("secondary_to_mortar_avg", True, False, "SecondaryToMortarAvg")
        if self._is_conforming_secondary:
            self._secondary_to_mortar = mat
        else:
            self._secondary_to_mortar_avg = mat
        return mat

    def _bmat(self, matrices, name):
        # Create block matrix, convert it to optimized storage format.
        if len(matrices[0]) == 0:
            block_matrix = sps.csr_matrix((0, 0))
        else:
            block_matrix = pp.matrix_operations.optimized_compressed_storage(
                sps.bmat(matrices)
            )
        return SparseArray(block_matrix, name=name)                                    

    def __repr__(self) -> str:
        s = (
            f"Mortar projection for {self._num_edges} interfaces\n"
            f"Aimed at variables with dimension {self.dim}\n"
            f"Projections to primary have dimensions {self.mortar_to_primary_avg.shape}\n"
            f"Projections to secondary have dimensions {self.mortar_to_secondary_avg.shape}\n"
        )
        return s


class BoundaryProjection:
    """A projection operator between boundary grids and subdomains."""

    def __init__(
        self, mdg: pp.MixedDimensionalGrid, subdomains: Sequence[pp.Grid], dim: int = 1
    ) -> None:
        face_projections = _face_projections(subdomains, dim)

        # Size for the matrix, used for 0d subdomains.
        tot_num_faces = np.sum([sd.num_faces for sd in subdomains]) * dim

        mat = []
        for sd in subdomains:
            if sd.dim > 0:
                bg = mdg.subdomain_to_boundary_grid(sd)
                if bg is not None:
                    mat_loc = bg.projection(dim)
                    mat_loc = mat_loc * face_projections[sd].T
            else:
                # The subdomain has no faces, so the projection does not exist.
                mat_loc = sps.csr_matrix((0, tot_num_faces))
            mat.append(mat_loc)

        self._projection: sps.spmatrix
        """Projection from subdomain faces to boundary grid cells."""
        if len(mat) > 0:
            self._projection = sps.bmat([[m] for m in mat], format="csr")
        else:
            self._projection = sps.csr_matrix((0, 0))

    @property
    def subdomain_to_boundary(self) -> Operator:
        return SparseArray(self._projection, name="subdomains to boundaries projection")

    @property
    def boundary_to_subdomain(self) -> Operator:
        return SparseArray(
            self._projection.transpose().tocsc(),
            name="boundaries to subdomains projection",
        )


class Trace:
    """Wrapper class for Ad representations of trace operators and their inverse,
    that is, mappings between grid cells and faces.

    NOTE: The mapping will hit both boundary and interior faces, so the values
    to be mapped should be carefully filtered (e.g. by combining it with a
    mortar mapping).

    The mapping does not alter signs of variables, that is, the direction
    of face normal vectors is not accounted for.

    """

    def __init__(
        self,
        subdomains: list[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct trace operators and their inverse for a given set of subdomains.

        The operators will be ordered according to the ordering in subdomains. It is
        critical that the same ordering is used by other operators.

        Parameters:
            subdomains: List of grids. The order of the grids in the list sets the
                ordering of the trace operators.
            dim: Dimension of the quantities to be projected. Defaults to 1.
            name: Name of the operator. Default is None.

        """
        self.subdomains: list[pp.Grid] = subdomains
        self.dim: int = dim
        self._name: Optional[str] = name
        self._is_scalar: bool = dim == 1
        self._num_grids: int = len(subdomains)

        cell_projections = _cell_projections(subdomains, self.dim)

        trace: list[sps.spmatrix] = []

        if len(subdomains) > 0:
            for sd in subdomains:
                if self._is_scalar:
                    # Local trace operator.
                    sd_trace = sd.trace(dim=self.dim)
                    # Restrict global cell values to the local grid.
                    trace.append(sd_trace * cell_projections[sd].T)

                else:
                    raise NotImplementedError("kronecker")
        else:
            trace = [sps.csr_matrix((0, 0))]
        # Stack trace vertically to make them into mappings to global quantities. Wrap
        # the stacked matrices into an AD object.
        self.trace = SparseArray(sps.bmat([[m] for m in trace]).tocsr())
        """ Matrix of trace projections from cells to faces."""

    def __repr__(self) -> str:
        s = (
            f"Trace operator for {self._num_grids} subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
            f"Projection from grid to mortar has dimensions {self.trace}\n"
        )
        return s

    def __str__(self) -> str:
        s = "Trace"
        if self._name is not None:
            s += f" named {self._name}"
        return s


class Divergence(Operator):
    """Wrapper class for Ad representations of divergence operators."""

    def _key(self) -> str:
        subdomain_ids = [sd.id for sd in self.subdomains]
        return f"(divergence, dim={self.dim}, subdomains={subdomain_ids})"

    def __init__(
        self,
        subdomains: list[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct divergence operators for a set of subdomains.

        The operators will be ordered according to the ordering in subdomains, or the
        order of the MixedDimensionalGrid iteration over subdomains. It is critical that
        the same ordering is used by other operators.

        IMPLEMENTATION NOTE: Only scalar quantities so far; vector operators will be
        added in due course.

        Parameters:
            subdomains: List of grids. The order of the subdomains in
                the list sets the ordering of the divergence operators.
            dim: Dimension of vector field. Defaults to 1.
            name: Name to be assigned to the operator. Default is None.

        """
        super().__init__(domains=subdomains, name=name)

        self.dim: int = dim

    def __repr__(self) -> str:
        s = (
            f"Divergence for vector field of size {self.dim}"
            f" defined on {len(self.subdomains)} subdomains\n"
        )

        num_faces = 0
        num_cells = 0
        for g in self.subdomains:
            num_faces += g.num_faces * self.dim
            num_cells += g.num_cells * self.dim
        s += f"The total size of the matrix is ({num_cells}, {num_faces}).\n"

        return s

    def __str__(self) -> str:
        s = "Divergence "
        if self._name is not None:
            s += f"named {self._name}"
        return s

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """Convert the Ad expression into a divergence operators on all relevant
        subdomains, represented as a sparse block matrix.

        Parameters:
            mdg: Not used, but needed for compatibility with the general parsing method
                for Operators.

        Returns:
            sps.spmatrix: Block matrix representation of a divergence operator on
            multiple subdomains.

        """
        mat = [sd.divergence(dim=self.dim) for sd in self.subdomains]
        matrix = sps.block_diag(mat)
        return matrix


def _cell_projections(subdomains: Sequence[pp.Grid], dim: int) -> dict[pp.Grid, sps.spmatrix]:
    """Construct prolongation matrices for cell-based quantities from individual subdomains to a set of subdomains.

    Parameters:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to
            a vector of two components etc.

    Returns:
        cell_projection: Dictionary with the individual subdomains as keys and
            projection matrices for cell-based quantities as items.

    The global cell numbering is set according to the order of the input subdomains.

    """
    cell_projection: dict[pp.Grid, np.ndarray] = {}
    if len(subdomains) == 0:
        return cell_projection

    tot_num_cells = np.sum([g.num_cells for g in subdomains]) * dim
    cell_offset = 0

    for sd in subdomains:
        cell_ind = cell_offset + pp.fvutils.expand_indices_nd(np.arange(sd.num_cells), dim)
        cell_sz = sd.num_cells * dim

        cell_projection[sd] = sps.coo_matrix(
            (np.ones(cell_sz), (cell_ind, np.arange(cell_sz))),
            shape=(tot_num_cells, cell_sz),
        ).tocsc()
        cell_offset = cell_ind[-1] + 1

    return cell_projection

def _face_projections(subdomains: Sequence[pp.Grid], dim: int) -> dict[pp.Grid, sps.spmatrix]:
    """Construct prolongation matrices for face-based quantities from individual subdomains to a set of subdomains.

    Parameters:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to
            a vector of two components etc.

    Returns:
        face_projection: Dictionary with the individual subdomains as keys and
        projection matrices for face-based quantities as items.

    The global face numbering is set according to the order of the input subdomains.

    """
    face_projection: dict[pp.Grid, np.ndarray] = {}
    if len(subdomains) == 0:
        return face_projection

    tot_num_faces = np.sum([g.num_faces for g in subdomains]) * dim
    face_offset = 0

    for sd in subdomains:
        face_ind = face_offset + pp.fvutils.expand_indices_nd(np.arange(sd.num_faces), dim)
        face_sz = sd.num_faces * dim

        face_projection[sd] = sps.coo_matrix(
            (np.ones(face_sz), (face_ind, np.arange(face_sz))),
            shape=(tot_num_faces, face_sz),
        ).tocsc()

        if sd.dim > 0:
            face_offset = face_ind[-1] + 1

    return face_projection
