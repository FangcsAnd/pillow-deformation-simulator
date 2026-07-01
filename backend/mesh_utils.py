import os
import subprocess
import tempfile
import numpy as np
from scipy.spatial import Delaunay


def stp_to_surface_mesh(stp_path, deflection=1.0):
    """
    Convert STEP file to triangle surface mesh.
    Tries multiple backends: pythonocc-core, gmsh, pure-Python parser.
    Returns vertices (Nx3) and faces (Mx3).
    """
    errors = []

    try:
        return _stp_via_pythonocc(stp_path, deflection)
    except ImportError:
        errors.append("pythonocc-core not installed")
    except Exception as e:
        errors.append(f"pythonocc-core: {e}")

    try:
        return _stp_via_gmsh(stp_path)
    except Exception as e:
        errors.append(f"gmsh: {e}")

    try:
        return _stp_via_pure_python(stp_path)
    except Exception as e:
        errors.append(f"pure-python: {e}")

    msg = "Cannot read STEP file. Tried:\n"
    for err in errors:
        msg += f"  - {err}\n"
    msg += "\nSolutions:\n"
    msg += "  1. Install gmsh: brew install gmsh\n"
    msg += "  2. Install pythonocc-core via conda: conda install -c conda-forge pythonocc-core\n"
    msg += "  3. Re-export the CAD file as STL (triangle mesh) and upload the STL"
    raise RuntimeError(msg)


def _stp_via_pythonocc(stp_path, deflection=1.0):
    """Convert STEP using pythonocc-core."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    status = reader.ReadFile(stp_path)
    if status != 1:
        raise RuntimeError(f"Failed to read STEP file: {stp_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    mesh = BRepMesh_IncrementalMesh(shape, deflection, False, 1.0, False)
    mesh.Perform()

    vertices = []
    faces_out = []
    vertex_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is not None:
            nodes = triangulation.Nodes()
            num_nodes = triangulation.NbNodes()
            for i in range(1, num_nodes + 1):
                p = nodes.Value(i)
                vertices.append([p.X(), p.Y(), p.Z()])

            tris = triangulation.Triangles()
            num_tris = triangulation.NbTriangles()
            for i in range(1, num_tris + 1):
                tri = tris.Value(i)
                n1, n2, n3 = tri.Value(1), tri.Value(2), tri.Value(3)
                faces_out.append([
                    vertex_offset + n1 - 1,
                    vertex_offset + n2 - 1,
                    vertex_offset + n3 - 1,
                ])

            vertex_offset += num_nodes

        explorer.Next()

    if len(vertices) == 0:
        raise RuntimeError("No mesh data extracted from STEP file")

    return np.array(vertices, dtype=float), np.array(faces_out, dtype=int)


def _stp_via_gmsh(stp_path):
    """Convert STEP to mesh using Gmsh command line (if installed)."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name

    try:
        result = subprocess.run(
            ["gmsh", stp_path, "-0", "-o", stl_path, "-format", "stl", "-2"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0 or not os.path.exists(stl_path):
            raise RuntimeError(f"gmsh conversion failed: {result.stderr.strip()}")

        import trimesh
        mesh = trimesh.load(stl_path)
        return np.array(mesh.vertices, dtype=float), np.array(mesh.faces, dtype=int)
    finally:
        if os.path.exists(stl_path):
            os.unlink(stl_path)


def _stp_via_pure_python(stp_path):
    """Pure Python STEP parser for tessellated STEP files."""
    import re

    with open(stp_path, 'r', errors='replace') as f:
        content = f.read()

    points = {}
    tris = []

    entities = content.split(';')
    for line in entities:
        line = line.strip()
        if not line or line.startswith('/*') or line.startswith('*'):
            continue

        eq = line.find('=')
        if eq < 0:
            continue
        eid = line[:eq].strip()
        if not eid.startswith('#'):
            continue
        rest = line[eq + 1:].strip()

        if rest.startswith('CARTESIAN_POINT('):
            try:
                p1 = rest.find('(')
                p2 = rest.rfind(')')
                if p1 >= 0 and p2 > p1:
                    inner = rest[p1 + 1:p2]
                    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', inner)
                    if len(nums) >= 3:
                        points[eid] = [float(nums[0]), float(nums[1]), float(nums[2])]
            except Exception:
                pass

        elif rest.startswith('POLY_LOOP('):
            refs = re.findall(r'#\d+', rest)
            if len(refs) == 3:
                tris.append(refs)
            elif len(refs) == 4:
                tris.append([refs[0], refs[1], refs[2]])
                tris.append([refs[0], refs[2], refs[3]])

        elif 'TRIANGLE(' in rest and 'TRIANGULATED' not in rest:
            refs = re.findall(r'#\d+', rest)
            if len(refs) >= 3:
                tris.append(refs[:3])

    if not points or not tris:
        raise RuntimeError(
            "Pure Python parser could not extract mesh from this STEP file. "
            "The file may contain only NURBS surfaces (no tessellation data). "
            "Please re-export from your CAD tool with tessellation enabled, "
            "or export as STL directly."
        )

    vertex_map = {}
    vertex_list = []
    faces_out = []

    for tri_refs in tris:
        face = []
        for ref in tri_refs:
            if ref not in vertex_map:
                if ref not in points:
                    break
                vertex_map[ref] = len(vertex_list)
                vertex_list.append(points[ref])
            face.append(vertex_map[ref])
        if len(face) == 3:
            faces_out.append(face)

    if len(faces_out) == 0 or len(vertex_list) == 0:
        raise RuntimeError("Could not resolve face-to-vertex references")

    return np.array(vertex_list, dtype=float), np.array(faces_out, dtype=int)


def surface_to_tetmesh(vertices, faces, num_interior=800):
    """
    Generate tetrahedral mesh from surface triangle mesh.
    Uses interior sampling + Delaunay triangulation + containment filter.
    """
    import trimesh as tm

    mesh = tm.Trimesh(vertices=vertices, faces=faces)
    if not mesh.is_watertight:
        mesh.fill_holes()
        mesh.merge_vertices()

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    margin = 0.05 * (bbox_max - bbox_min)
    bbox_min -= margin
    bbox_max += margin

    interior_pts = []
    max_attempts = num_interior * 12
    attempts = 0
    while len(interior_pts) < num_interior and attempts < max_attempts:
        p = np.random.uniform(bbox_min, bbox_max)
        try:
            if mesh.contains([p])[0]:
                interior_pts.append(p)
        except Exception:
            pass
        attempts += 1

    if len(interior_pts) < num_interior * 0.3:
        pass  # Low interior points, but continue anyway

    interior_pts = np.array(interior_pts) if interior_pts else np.zeros((0, 3))
    n_surface = len(vertices)

    all_pts = np.vstack([vertices, interior_pts])
    tri = Delaunay(all_pts)

    centroids = all_pts[tri.simplices].mean(axis=1)
    centroid_inside = np.zeros(len(tri.simplices), dtype=bool)
    for j in range(len(tri.simplices)):
        try:
            centroid_inside[j] = mesh.contains([centroids[j]])[0]
        except Exception:
            pass

    n_surface_per_tet = np.sum(tri.simplices < n_surface, axis=1)
    valid_mask = centroid_inside | (n_surface_per_tet >= 3)
    valid_tets = tri.simplices[valid_mask]
    return all_pts, valid_tets


def find_bottom_nodes(vertices, fraction=0.08):
    """Find node indices on the bottom of the pillow (to fix)."""
    z_vals = vertices[:, 2]
    z_min = z_vals.min()
    z_max = z_vals.max()
    z_range = z_max - z_min
    threshold = z_min + fraction * z_range
    return np.where(z_vals <= threshold)[0]


def find_contact_nodes(vertices, weight_pos, rx, ry=None, top_fraction=0.4):
    """
    Find nodes in contact area of an elliptical weight.
    weight_pos: [x, y, z]
    rx, ry: elliptical radii
    Returns dict of {node_idx: [fx, fy, fz]}
    """
    if ry is None:
        ry = rx

    z_vals = vertices[:, 2]
    z_max = z_vals.max()
    z_min = z_vals.min()
    z_range = z_max - z_min

    top_mask = z_vals >= z_min + top_fraction * z_range
    top_indices = np.where(top_mask)[0]

    if len(top_indices) == 0:
        return {}

    top_verts = vertices[top_indices]
    dx = (top_verts[:, 0] - weight_pos[0]) / max(rx, 1e-6)
    dy = (top_verts[:, 1] - weight_pos[1]) / max(ry, 1e-6)
    ellip_dists = np.sqrt(dx**2 + dy**2)

    forces = {}
    for search_scale in [1.0, 1.5, 2.0]:
        in_range = ellip_dists <= search_scale
        if in_range.sum() > 0:
            for j in np.where(in_range)[0]:
                i = top_indices[j]
                d = ellip_dists[j]
                wf = max(0.0, 1.0 - d / search_scale)
                forces[i] = np.array([0.0, 0.0, -wf])
            break

    return forces


def find_contact_xy_projection(vertices, model_verts, model_pos, expand=1.3):
    """Find pillow nodes under head model XY projection, weighted by head shape."""
    from scipy.spatial import ConvexHull, cKDTree

    mv_xy = model_verts[:, :2]
    hull = ConvexHull(mv_xy)
    hull_verts = mv_xy[hull.vertices]

    top_mask = vertices[:, 2] >= vertices[:, 2].min() + 0.15 * (vertices[:, 2].max() - vertices[:, 2].min())
    top_idx = np.where(top_mask)[0]
    if len(top_idx) == 0:
        return {}

    top_v = vertices[top_idx]
    shifted = top_v[:, :2] - model_pos[:2] + np.mean(hull_verts, axis=0)
    inside = _points_in_hull(shifted / expand, hull_verts)

    if inside.sum() == 0:
        return {}

    # Build thickness map from head model
    model_xy = mv_xy - model_pos[:2] + np.mean(hull_verts, axis=0)
    tree = cKDTree(model_xy)
    
    forces = {}
    for j in np.where(inside)[0]:
        idx = top_idx[j]
        px, py = shifted[j]
        # Find nearby head model points and compute thickness
        dists, nn = tree.query([px, py], k=min(20, len(model_verts)))
        if hasattr(dists, '__len__'):
            nearby_z = model_verts[nn, 2]
            thickness = nearby_z.max() - nearby_z.min() if len(nearby_z) > 1 else 0.05
        else:
            thickness = 0.05
        # Heavier force where head is thicker (back of head)
        weight = max(thickness, 0.02)
        forces[idx] = np.array([0.0, 0.0, -weight])
    
    return forces


def _points_in_3d_hull(points, hull_verts):
    """Check if points are inside a 3D convex hull."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(hull_verts)
    except Exception:
        return np.zeros(len(points), dtype=bool)
    inside = np.ones(len(points), dtype=bool)
    for eq in hull.equations:
        a, b, c, d = eq
        norm = np.sqrt(a*a + b*b + c*c)
        if norm < 1e-12:
            continue
        inside &= (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d <= 1e-7 * norm)
    return inside


def _points_in_3d_hull(points, hull_verts):
    """Check if points are inside a 3D convex hull."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(hull_verts)
    # Use the hull equations: for each facet, dot(normal, point) + offset <= 0 for interior
    inside = np.ones(len(points), dtype=bool)
    for eq in hull.equations:
        a, b, c, d = eq
        inside &= (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d <= 1e-8)
    return inside


def _points_in_hull(points, hull_verts):
    """Pure numpy point-in-convex-polygon test."""
    n = len(hull_verts)
    if n < 3:
        return np.zeros(len(points), dtype=bool)
    inside = np.ones(len(points), dtype=bool)
    for i in range(n):
        a = hull_verts[i]
        b = hull_verts[(i + 1) % n]
        edge = b - a
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        perp = np.array([-edge[1], edge[0]])
        normal = perp / edge_len
        proj = (points - a) @ normal
        inside &= (proj >= -1e-8)
    return inside


def generate_default_pillow_mesh(length=0.5, width=0.3, height=0.1,
                                  subdivisions=5):
    """
    Generate a default pillow mesh (rounded rectangular prism) for testing.
    Uses loop subdivision to keep the mesh watertight.
    Returns surface vertices and faces.
    """
    try:
        import trimesh as tm
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    box = tm.creation.box(extents=(length, width, height))

    mesh = box
    for _ in range(subdivisions):
        mesh = mesh.subdivide()

    verts = mesh.vertices.copy()
    hx = length / 2.0
    hy = width / 2.0
    for i, v in enumerate(verts):
        if v[2] > height * 0.3:
            dx = (v[0]) / hx
            dy = (v[1]) / hy
            r = np.sqrt(dx**2 + dy**2)
            if r < 1.15:
                dome = np.cos(min(r, 1.0) * np.pi * 0.5) * 0.025
                v[2] += dome
        if v[2] < height * 0.15:
            v[2] = max(0.0, v[2] - 0.002)

    mesh.vertices = verts
    mesh.merge_vertices()

    return np.array(mesh.vertices, dtype=float), np.array(mesh.faces, dtype=int)


def normalize_mesh(vertices, target_height=0.15):
    """Center and scale mesh so bottom is at z=0 and height = target_height."""
    z_min = vertices[:, 2].min()
    vertices = vertices.copy()
    vertices[:, 2] -= z_min

    # Center XY
    for axis in [0, 1]:
        mid = (vertices[:, axis].min() + vertices[:, axis].max()) / 2
        vertices[:, axis] -= mid

    # Scale to target height
    z_max = vertices[:, 2].max()
    if z_max > 1e-6:
        scale = target_height / z_max
        vertices *= scale

    return vertices
