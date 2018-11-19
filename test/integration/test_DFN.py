import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp

class TestDFN(unittest.TestCase):
    def test_mvem_0(self):
        dfn_dim = 1
        f1 = np.array([[1.0, 1.0], [0.0, 2.0]])
        f2 = np.array([[0.0, 2.0], [1.0, 1.0]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2], [2, 2])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_mvem(gb)

        A, b, _, _ = assembler.assemble_matrix_rhs(gb)

        A_known = np.matrix(\
           [[0.75, 0.  , 0.  ,-0.25, 1, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0, 0.  ,-0.25, 0.  ,0.   ],
            [0.  , 0.75, 0.  , 0.  , 0, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0, 0.  , 0.  , 0.  ,0.   ],
            [0.  ,-0.25, 0.75, 0.  , 0,-1,0.  , 0.  , 0.  , 0.  ,0, 0, 0, 0.25, 0.  , 0.  ,0.   ],
            [0.  , 0.  , 0.  , 0.75, 0, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0, 0.  , 0.  , 0.  ,0.   ],
            [1.  , 0.  , 0.  ,-1.  , 0, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0, 0.  ,-1.  , 0.  ,0.   ],
            [0.  , 1.  ,-1.  , 0.  , 0, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0,-1.  , 0.  , 0.  ,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.75, 0.  , 0.  ,-0.25,1, 0, 0, 0.  , 0.  , 0.  ,-0.25],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  , 0.75, 0.  , 0.  ,0, 0, 0, 0.  , 0.  , 0.  ,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  ,-0.25, 0.75, 0.  ,0,-1, 0, 0.  , 0.  , 0.25,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  , 0.  , 0.  , 0.75,0, 0, 0, 0.  , 0.  , 0.  ,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,1.  , 0.  , 0.  ,-1.  ,0, 0, 0, 0.  , 0.  , 0.  ,-1.  ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  , 1.  ,-1.  , 0.  ,0, 0, 0, 0.  , 0.  ,-1.  ,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  , 0.  , 0.  , 0.  ,0, 0, 0,-1.  ,-1.  ,-1.  ,-1.  ],
            [0.  , 0.75,-0.25, 0.  , 0, 1,0.  , 0.  , 0.  , 0.  ,0, 0,-1,-0.75, 0.  , 0.  ,0.   ],
            [0.25, 0.  , 0.  ,-0.75, 1, 0,0.  , 0.  , 0.  , 0.  ,0, 0,-1, 0.  ,-0.75, 0.  ,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.  , 0.75,-0.25, 0.  ,0, 1,-1, 0.  , 0.  ,-0.75,0.   ],
            [0.  , 0.  , 0.  , 0.  , 0, 0,0.25, 0.  , 0.  ,-0.75,1, 0,-1, 0.  , 0.  , 0.  ,-0.75]])

        b_known = np.array([2,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0])

        self.assertTrue(np.allclose(A.todense(), A_known))
        self.assertTrue(np.allclose(b, b_known))

    def test_tpfa_0(self):
        dfn_dim = 1
        f1 = np.array([[1.0, 1.0], [0.0, 2.0]])
        f2 = np.array([[0.0, 2.0], [1.0, 1.0]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2], [2, 2])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_tpfa(gb)

        A, b, _, _ = assembler.assemble_matrix_rhs(gb)

        A_known = np.matrix(\
           [[ 2. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ],
            [ 0. ,  2. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  2. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
            [ 0. ,  0. ,  0. ,  2. ,  0. ,  0. ,  0. ,  1. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ,  0. , -1. , -1. , -1. , -1. ],
            [ 0. ,  1. ,  0. ,  0. , -1. , -0.5,  0. ,  0. ,  0. ],
            [ 1. ,  0. ,  0. ,  0. , -1. ,  0. , -0.5,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  1. , -1. ,  0. ,  0. , -0.5,  0. ],
            [ 0. ,  0. ,  1. ,  0. , -1. ,  0. ,  0. ,  0. , -0.5]])

        b_known = np.array([4., 0., 2., 2., 0., 0., 0., 0., 0.])

        self.assertTrue(np.allclose(A.todense(), A_known))
        self.assertTrue(np.allclose(b, b_known))

    def test_mvem_1(self):
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, (discr, p_trace) = setup_discr_mvem(gb)

        A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(gb, x, block_dof, full_dof)
        for g, d in gb:
            discr = d["discretization"]["flow"]["flux"]
            d["pressure"] = discr.extract_pressure(g, d["flow"], d)

        for g, d in gb:

            if g.dim == 1:
                if np.all(g.cell_centers[1] == 0.5 * N): #f1
                    known = np.array([4., 4., 4., 4., 4., 4., 4., 4.])
                elif np.all(g.cell_centers[0] == 0.5 * N): #f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(g.cell_centers[0] == 0.625 * N): #f3
                    known = np.array([4, 4])
                elif np.all(g.cell_centers[1] == 0.25 * N): #f4
                    known = np.array([2, 2, 2, 2])
                elif np.all(g.cell_centers[0] == 0.75 * N): #f5
                    known = np.array([2, 2])
                else:
                    raise ValueError

            else: #g.dim == 0
                if np.allclose(g.cell_centers, np.array([[0.5], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.625], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.5], [0.25], [0]]) * N):
                    known = np.array([2])
                elif np.allclose(g.cell_centers, np.array([[0.75], [0.25], [0]]) * N):
                    known = np.array([2])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(d["pressure"], known))


    def test_tpfa_1(self):
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, (discr, p_trace) = setup_discr_tpfa(gb)

        A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(gb, x, block_dof, full_dof)

        for g, d in gb:

            if g.dim == 1:
                if np.all(g.cell_centers[1] == 0.5 * N): #f1
                    known = np.array([4., 4., 4., 4., 4., 4., 4., 4.])
                elif np.all(g.cell_centers[0] == 0.5 * N): #f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(g.cell_centers[0] == 0.625 * N): #f3
                    known = np.array([4, 4])
                elif np.all(g.cell_centers[1] == 0.25 * N): #f4
                    known = np.array([2, 2, 2, 2])
                elif np.all(g.cell_centers[0] == 0.75 * N): #f5
                    known = np.array([2, 2])
                else:
                    raise ValueError

            else: #g.dim == 0
                if np.allclose(g.cell_centers, np.array([[0.5], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.625], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.5], [0.25], [0]]) * N):
                    known = np.array([2])
                elif np.allclose(g.cell_centers, np.array([[0.75], [0.25], [0]]) * N):
                    known = np.array([2])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(d["flow"], known))

    def test_mvem_2(self):
        dfn_dim = 2
        f1 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [0.0, 2.0, 2.0, 0.0],
                       [0.0, 0.0, 2.0, 2.0]])
        f2 = np.array([[0.0, 2.0, 2.0, 0.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [0.0, 0.0, 2.0, 2.0]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2], [2, 2, 2])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_mvem(gb)

        A, b, _, _ = assembler.assemble_matrix_rhs(gb)

        #np.savetxt('matrix.txt', A.todense(), fmt="%1.2f", delimiter=',', newline='],\n[')
        A_known = test_mvem_2_matrix()
        b_known = np.array([0.5, 0, -0.5, 1.5, 0, -1.5, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, \
                            0, 0, 1, 0, -1, 1, 0, -1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertTrue(np.allclose(A.todense(), A_known))
        self.assertTrue(np.allclose(b, b_known))

    def test_tpfa_2(self):
        dfn_dim = 2
        f1 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [0.0, 2.0, 2.0, 0.0],
                       [0.0, 0.0, 2.0, 2.0]])
        f2 = np.array([[0.0, 2.0, 2.0, 0.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [0.0, 0.0, 2.0, 2.0]])

        # create the grid bucket
        gb = pp.meshing.cart_grid([f1, f2], [2, 2, 2])
        gb.compute_geometry()
        create_dfn(gb, dfn_dim)

        # setup data and assembler
        setup_data(gb)
        assembler, _ = setup_discr_tpfa(gb)

        A, b, _, _ = assembler.assemble_matrix_rhs(gb)

        A_known = np.matrix([\
        [ 5. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
        [-1. ,  5. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  5. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. , -1. ,  5. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  5. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  5. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  5. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  5. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. , -1. ,  0. , -1. ,  0. , -1. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. , -1. ,  0. , -1. ,  0. , -1. ],
        [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. , -0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. , -0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
        [ 1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ],
        [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5]])

        b_known = np.array([1., 1., 7., 7., 4., 4., 4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        self.assertTrue(np.allclose(A.todense(), A_known))
        self.assertTrue(np.allclose(b, b_known))

#------------------------- HELP FUNCTIONS --------------------------------#

def setup_data(gb, key="flow"):
    """ Setup the data
    """
    gb.add_node_props(["param", "is_tangential"])
    for g, d in gb:
        param = pp.Parameters(g)
        kxx = np.ones(g.num_cells)
        param.set_tensor("flow", pp.SecondOrderTensor(g.dim, kxx))

        if g.dim == gb.dim_max():
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            param.set_bc("flow", bound)
            param.set_bc_val("flow", bc_val)
        d["param"] = param
        d["is_tangential"] = True

def setup_discr_mvem(gb, key="flow"):
    """ Setup the discretization. """
    discr = pp.MVEM(key)
    p_trace = pp.PressureTrace(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {"flux": p_trace}}

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.keywords.COUPLING_DISCRETIZATION] = {
            "flux": {
                g_slave: (key, "flux"),
                g_master: (key, "flux"),
                e: (key, interface)
            }
        }

    return pp.Assembler(), (discr, p_trace)

def setup_discr_tpfa(gb, key="flow"):
    """ Setup the discretization. """
    discr = pp.Tpfa(key)
    p_trace = pp.PressureTrace(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {"flux": p_trace}}

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.keywords.COUPLING_DISCRETIZATION] = {
            "flux": {
                g_slave: (key, "flux"),
                g_master: (key, "flux"),
                e: (key, interface)
            }
        }

    return pp.Assembler(), (discr, p_trace)

def create_dfn(gb, dim):
    """ given a GridBucket remove the higher dimensional node and
    fix the internal mapping. """
    # remove the +1 and -2 dimensional grids with respect to the
    # considered dfn, and re-write the node number
    gd = np.hstack((gb.grids_of_dimension(dim + 1),
                    gb.grids_of_dimension(dim - 2)))

    for g in gd:
        node_number = gb.node_props(g, "node_number")
        gb.remove_node(g)
        gb.update_node_ordering(node_number)

def test_mvem_2_matrix():
    return np.array([\
    [0.75,-0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [-0.25,1.50,-0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,-1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,-0.25,0.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.75,-0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,-0.25,1.50,-0.25,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,-1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,-0.25,0.75,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,-0.00,-0.00,0.00,0.00,0.00,-0.25,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [1.00,-1.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00],
    [0.00,1.00,-1.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,1.00,-1.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,1.00,-1.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.75,0.00,0.00,0.00,0.00,-0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,-0.25,0.00,0.00,0.00,-0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,0.75,0.00,-0.25,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,1.50,0.00,-0.25,0.00,-0.00,0.00,-1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00,0.00,-0.25,0.00,1.50,0.00,-0.25,0.00,0.00,0.00,-1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.25,0.00,0.75,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,-1.00,0.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,-1.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,-1.00,0.00,-1.00,0.00,-1.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,-1.00,0.00,-1.00,0.00,-1.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,-0.25,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,-0.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,0.00,-0.25,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,-0.75,0.00,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,-0.75,0.00,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,-0.75,0.00,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,-0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.75,-0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,1.00,0.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.75,0.00],
    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.75,0.00,0.00,1.00,0.00,0.00,-1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,-0.75]])


if __name__ == "__main__":
    TestDFN().test_tpfa_2()
