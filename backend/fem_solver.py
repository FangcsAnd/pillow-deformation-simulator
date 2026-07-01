import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def compute_element_stiffness(verts, E, nu):
    """
    Compute 12x12 stiffness matrix for a linear tetrahedral element.
    verts: 4x3 array of vertex positions
    """
    J = np.column_stack([
        verts[1] - verts[0],
        verts[2] - verts[0],
        verts[3] - verts[0],
    ])
    detJ = np.linalg.det(J)
    V = abs(detJ) / 6.0

    if V < 1e-15:
        return np.zeros((12, 12)), 0.0

    J_inv = np.linalg.inv(J)
    J_inv_T = J_inv.T

    grad_N_ref = np.array([
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])  # 4x3

    betas = grad_N_ref @ J_inv_T  # 4x3, physical gradients

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    D = np.zeros((6, 6))
    a = lam + 2.0 * mu
    D[0, 0] = D[1, 1] = D[2, 2] = a
    D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    B = np.zeros((6, 12))
    for i in range(4):
        bx, by, bz = betas[i]
        c = 3 * i
        B[0, c] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz
        B[3, c] = by
        B[3, c + 1] = bx
        B[4, c + 1] = bz
        B[4, c + 2] = by
        B[5, c] = bz
        B[5, c + 2] = bx

    Ke = B.T @ D @ B * V
    return Ke, V


def assemble_global_stiffness(vertices, tetrahedra, E, nu):
    """
    Assemble global stiffness matrix.
    vertices: Nx3
    tetrahedra: Mx4
    """
    n_nodes = len(vertices)
    K = lil_matrix((3 * n_nodes, 3 * n_nodes))

    for tet in tetrahedra:
        verts = vertices[tet]
        Ke, _ = compute_element_stiffness(verts, E, nu)

        for i in range(4):
            for j in range(4):
                for di in range(3):
                    for dj in range(3):
                        gi = 3 * int(tet[i]) + di
                        gj = 3 * int(tet[j]) + dj
                        K[gi, gj] += Ke[3 * i + di, 3 * j + dj]

    return K.tocsr()


def solve_fem(K, f, fixed_dofs):
    """Solve Ku = f with Dirichlet BC at fixed_dofs."""
    n_dofs = K.shape[0]
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[free_dofs][:, free_dofs]
    f_f = f[free_dofs]

    u_f = spsolve(K_ff, f_f)

    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    return u


class FEMSimulator:
    """Linear elastic FEM simulator for tetrahedral meshes."""

    def __init__(self, vertices, tetrahedra, youngs_modulus=50000.0, poisson_ratio=0.3):
        self.vertices = np.asarray(vertices, dtype=float)
        self.tetrahedra = np.asarray(tetrahedra, dtype=int)
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.n_nodes = len(self.vertices)
        self._assemble()

    def _assemble(self):
        self.K = assemble_global_stiffness(
            self.vertices, self.tetrahedra, self.E, self.nu
        )

    def set_material(self, youngs_modulus, poisson_ratio=0.3):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self._assemble()

    def solve(self, fixed_nodes, node_forces):
        """
        fixed_nodes: array of node indices with zero displacement
        node_forces: dict {node_idx: [fx, fy, fz]}
        Returns deformed vertices (Nx3).
        """
        n_dofs = 3 * self.n_nodes

        f = np.zeros(n_dofs)
        for node_idx, force in node_forces.items():
            node_idx = int(node_idx)
            force = np.asarray(force, dtype=float)
            f[3 * node_idx: 3 * node_idx + 3] = force

        fixed_dofs = []
        for n in fixed_nodes:
            n = int(n)
            fixed_dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])

        u = solve_fem(self.K, f, fixed_dofs)
        u = u.reshape(-1, 3)
        return self.vertices + u
