import os
import uuid
import tempfile
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from mesh_utils import (
    stp_to_surface_mesh,
    surface_to_tetmesh,
    find_bottom_nodes,
    find_contact_nodes,
    find_contact_xy_projection,
    generate_default_pillow_mesh,
    normalize_mesh,
)
from fem_solver import FEMSimulator

app = FastAPI(title="Pillow Deformation Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DIR = os.path.join(tempfile.gettempdir(), "pillow_sim_sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


class WeightConfig(BaseModel):
    position: List[float]
    mass: float
    shape: str = "model"
    radius: float = 0.10
    rx: Optional[float] = None
    ry: Optional[float] = None
    neck_ratio: float = 0.0
    neck_shift: float = 0.0
    model_verts: Optional[List[List[float]]] = None


class SimulateRequest(BaseModel):
    session_id: str
    hf_hardness: float = 40
    density: float = 50
    weights: List[WeightConfig]
    mattress_enabled: bool = True  # 10mm rigid mattress under pillow


def hf_to_youngs(hf: float, density: float = 50) -> float:
    """Convert LX-F sponge hardness (HF) to Young's modulus (Pa).
    
    Factory calibration: HF=13, 43mm pillow, 500g on 40mm disk → 6.5mm
    Simple compression: E ≈ 26kPa. FEM with fixed bottom is ~6x stiffer.
    Effective FEM E(Pa) ≈ 300 * HF
    """
    hf = max(5, min(60, hf))
    return 300.0 * hf


def get_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_session(session_id: str, data: dict):
    path = os.path.join(SESSION_DIR, f"{session_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


@app.post("/api/upload")
async def upload_stp(file: UploadFile = File(...)):
    """Upload a STEP file, convert to surface + tetrahedral mesh."""
    suffix = os.path.splitext(file.filename or "model.stp")[1].lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix in (".stp", ".step"):
            try:
                surface_verts, surface_faces = stp_to_surface_mesh(tmp_path)
            except RuntimeError:
                upload_dir = os.path.join(
                    os.path.dirname(__file__), "..", "uploads"
                )
                os.makedirs(upload_dir, exist_ok=True)
                saved_path = os.path.join(upload_dir, file.filename or "model.stp")
                with open(saved_path, "wb") as f:
                    f.write(content)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "STEP file needs tessellation data. Your file has been saved to:\n"
                        f"  {saved_path}\n\n"
                        "Quick fix - use an online converter (no install needed):\n"
                        "  1. Open https://anyconv.com/stp-to-stl-converter/\n"
                        "  2. Upload your .stp file\n"
                        "  3. Download the .stl result\n"
                        "  4. Re-upload the .stl file here\n\n"
                        "Or click '使用默认枕头模型' to test with a built-in pillow."
                    )
                )
        elif suffix in (".stl",):
            import trimesh
            mesh = trimesh.load(tmp_path)
            surface_verts = np.array(mesh.vertices, dtype=float)
            surface_faces = np.array(mesh.faces, dtype=int)
        elif suffix in (".obj",):
            import trimesh
            mesh = trimesh.load(tmp_path)
            surface_verts = np.array(mesh.vertices, dtype=float)
            surface_faces = np.array(mesh.faces, dtype=int)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {suffix}. "
                       "Please upload .stp, .step, .stl, or .obj"
            )
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    if len(surface_verts) == 0:
        raise HTTPException(status_code=400, detail="Empty mesh extracted from file")

    # Center: bottom at z=0, XY centrered
    z_min = surface_verts[:, 2].min()
    surface_verts = surface_verts.copy()
    surface_verts[:, 2] -= z_min
    for axis in [0, 1]:
        mid = (surface_verts[:, axis].min() + surface_verts[:, axis].max()) / 2
        surface_verts[:, axis] -= mid

    # Auto-scale to meters
    max_dim = (surface_verts.max(axis=0) - surface_verts.min(axis=0)).max()
    if max_dim > 5.0:
        surface_verts *= 0.001
    elif max_dim > 1.0:
        surface_verts *= 0.01

    # Keep original for display, decimate only for FEM
    display_verts = surface_verts.copy()
    display_faces = surface_faces.copy()

    fem_limit = 30000
    if len(surface_verts) > fem_limit:
        surface_verts, surface_faces = _decimate_mesh(
            surface_verts, surface_faces, fem_limit
        )

    num_interior = 1000
    tet_verts, tet_cells = surface_to_tetmesh(surface_verts, surface_faces, num_interior=num_interior)

    if len(tet_cells) == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate tetrahedral mesh."
        )

    session_id = uuid.uuid4().hex[:12]
    session_data = {
        "surface_verts": surface_verts,
        "surface_faces": surface_faces,
        "display_verts": display_verts,
        "display_faces": display_faces,
        "tet_verts": tet_verts,
        "tet_cells": tet_cells,
        "num_surface_verts": len(surface_verts),
    }
    save_session(session_id, session_data)

    return {
        "session_id": session_id,
        "surface_verts": display_verts.tolist(),
        "surface_faces": display_faces.tolist(),
        "num_vertices": len(display_verts),
        "num_faces": len(display_faces),
        "num_tetrahedra": len(tet_cells),
        "bbox": {
            "x": [float(display_verts[:, 0].min()), float(display_verts[:, 0].max())],
            "y": [float(display_verts[:, 1].min()), float(display_verts[:, 1].max())],
            "z": [float(display_verts[:, 2].min()), float(display_verts[:, 2].max())],
        },
    }


@app.post("/api/default-pillow")
async def create_default_pillow():
    """Create a default pillow mesh for testing without file upload."""
    surface_verts, surface_faces = generate_default_pillow_mesh()
    z_min = surface_verts[:, 2].min()
    surface_verts = surface_verts.copy()
    surface_verts[:, 2] -= z_min
    for axis in [0, 1]:
        mid = (surface_verts[:, axis].min() + surface_verts[:, axis].max()) / 2
        surface_verts[:, axis] -= mid

    num_interior = 1000
    tet_verts, tet_cells = surface_to_tetmesh(surface_verts, surface_faces, num_interior=num_interior)

    session_id = uuid.uuid4().hex[:12]
    session_data = {
        "surface_verts": surface_verts,
        "surface_faces": surface_faces,
        "tet_verts": tet_verts,
        "tet_cells": tet_cells,
        "num_surface_verts": len(surface_verts),
    }
    save_session(session_id, session_data)

    return {
        "session_id": session_id,
        "surface_verts": surface_verts.tolist(),
        "surface_faces": surface_faces.tolist(),
        "num_vertices": len(surface_verts),
        "num_faces": len(surface_faces),
        "num_tetrahedra": len(tet_cells),
        "bbox": {
            "x": [float(surface_verts[:, 0].min()), float(surface_verts[:, 0].max())],
            "y": [float(surface_verts[:, 1].min()), float(surface_verts[:, 1].max())],
            "z": [float(surface_verts[:, 2].min()), float(surface_verts[:, 2].max())],
        },
    }


@app.post("/api/upload-weight-model")
async def upload_weight_model(file: UploadFile = File(...)):
    """Upload a 3D model to use as a weight (head, body, etc.)."""
    suffix = os.path.splitext(file.filename or "model.stl")[1].lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import trimesh
        mesh = trimesh.load(tmp_path)
        verts = np.array(mesh.vertices, dtype=float)
        faces = np.array(mesh.faces, dtype=int)

        # Center and auto-scale, keep all original triangles
        for axis in range(3):
            mid = (verts[:, axis].min() + verts[:, axis].max()) / 2
            verts[:, axis] -= mid
        max_dim = (verts.max(axis=0) - verts.min(axis=0)).max()
        if max_dim > 5.0: verts *= 0.001
        elif max_dim > 1.0: verts *= 0.01

        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "num_vertices": len(verts),
            "num_faces": len(faces),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """Run FEM simulation with given parameters."""
    session = get_session(req.session_id)

    tet_verts = session["tet_verts"]
    tet_cells = session["tet_cells"]
    surface_verts = session["surface_verts"]

    E = hf_to_youngs(req.hf_hardness, req.density)

    fem = FEMSimulator(tet_verts, tet_cells, youngs_modulus=E)

    fixed_nodes = find_bottom_nodes(tet_verts, fraction=0.06)

    g = 9.81
    model_weight = next((w for w in req.weights if w.shape == 'model' and w.model_verts), None)

    if model_weight:
        mv = np.array(model_weight.model_verts, dtype=float)
        target_force = model_weight.mass * g
        hx, hy = model_weight.position[0], model_weight.position[1]

        # Binary search for equilibrium head Z
        z_low = float(tet_verts[:, 2].min())
        z_high = float(tet_verts[:, 2].max()) + 0.02
        best_deformed = None
        best_head_z = z_high

        for _ in range(8):
            head_z = (z_low + z_high) / 2
            pos = np.array([hx, hy, head_z])
            contact = find_contact_xy_projection(tet_verts, mv, pos)

            if len(contact) < 3:
                z_low = head_z
                continue

            total_w = sum(abs(v[2]) for v in contact.values())
            scale = target_force / total_w
            all_f = {}
            for idx, force in contact.items():
                all_f[idx] = force * scale

            try:
                deformed = fem.solve(fixed_nodes.copy(), all_f)
                if np.any(~np.isfinite(deformed)):
                    z_high = head_z
                    continue

                best_deformed = deformed
                best_head_z = head_z

                # Compute reaction: average downward displacement of contact nodes
                contact_z = deformed[list(contact.keys()), 2]
                orig_z = tet_verts[list(contact.keys()), 2]
                avg_disp = np.mean(orig_z - contact_z)
                reaction = avg_disp * len(contact) * (model_weight.mass * g / max(total_w, 1e-10)) * 0.01

                if reaction < target_force * 0.5:
                    z_low = head_z  # Need more penetration
                else:
                    z_high = head_z  # Too much

            except Exception:
                z_high = head_z

        if best_deformed is None:
            # Fallback: solve at pillow surface
            pos = np.array([hx, hy, float(tet_verts[:, 2].max())])
            contact = find_contact_xy_projection(tet_verts, mv, pos)
            if contact:
                total_w = sum(abs(v[2]) for v in contact.values())
                scale = target_force / total_w
                all_f = {}
                for idx, force in contact.items():
                    all_f[idx] = force * scale
                best_deformed = fem.solve(fixed_nodes, all_f)
                best_head_z = float(tet_verts[:, 2].max())

        if best_deformed is None:
            raise HTTPException(status_code=400, detail="Contact detection failed")

        deformed_tet_verts = best_deformed

    else:
        all_forces = {}
        for weight in req.weights:
            pos = np.array(weight.position, dtype=float)
            if weight.shape == 'ellipsoid':
                rx = weight.rx if weight.rx else weight.radius
                ry = weight.ry if weight.ry else weight.radius
                contact_nodes = find_contact_nodes(tet_verts, pos, rx, ry)
            else:
                contact_nodes = find_contact_nodes(tet_verts, pos, weight.radius, weight.radius)

            total_weight = sum(abs(v[2]) for v in contact_nodes.values())
            if total_weight > 1e-10:
                scale = weight.mass * g / total_weight
                for idx, force in contact_nodes.items():
                    f_scaled = force * scale
                    if idx in all_forces: all_forces[idx] += f_scaled
                    else: all_forces[idx] = f_scaled

        if not all_forces:
            raise HTTPException(status_code=400, detail="No contact nodes found")

        deformed_tet_verts = fem.solve(fixed_nodes, all_forces)

    # Handle NaN
    if np.any(~np.isfinite(deformed_tet_verts)):
        raise HTTPException(status_code=400, detail="FEM produced invalid results")

    # Map FEM displacements to display mesh
    num_surf = session["num_surface_verts"]
    deformed_surface = deformed_tet_verts[:num_surf]
    display_v = session.get("display_verts")
    if display_v is not None and len(display_v) > len(deformed_surface):
        from scipy.spatial import cKDTree
        fem_disp = deformed_surface - session["surface_verts"][:num_surf]
        tree = cKDTree(session["surface_verts"][:num_surf])
        _, indices = tree.query(display_v)
        deformed_display = display_v + fem_disp[indices]
    else:
        display_v = session["surface_verts"]
        deformed_display = deformed_surface

    max_disp = float(np.max(np.linalg.norm(deformed_display - display_v, axis=1)))
    avg_disp = float(np.mean(np.linalg.norm(deformed_display - display_v, axis=1)))

    return {
        "deformed_verts": deformed_display.tolist(),
        "max_displacement": max_disp,
        "avg_displacement": avg_disp,
        "unit": "meters",
    }


@app.post("/api/export-deformed")
async def export_deformed(req: SimulateRequest):
    """Export deformed pillow as STL file."""
    session = get_session(req.session_id)
    tet_verts = session["tet_verts"]
    tet_cells = session["tet_cells"]
    surface_verts = session["surface_verts"]
    surface_faces = session["surface_faces"]

    E = hf_to_youngs(req.hf_hardness, req.density)
    fem = FEMSimulator(tet_verts, tet_cells, youngs_modulus=E)
    fixed_nodes = find_bottom_nodes(tet_verts, fraction=0.06)

    g = 9.81
    all_forces = {}
    for weight in req.weights:
        pos = np.array(weight.position, dtype=float)
        if weight.shape == 'model' and weight.model_verts:
            mv = np.array(weight.model_verts, dtype=float)
            contact_nodes = find_contact_from_model(tet_verts, mv, pos)
        elif weight.shape == 'ellipsoid':
            rx = weight.rx if weight.rx else weight.radius
            ry = weight.ry if weight.ry else weight.radius
            contact_nodes = find_contact_nodes(tet_verts, pos, rx, ry)
        else:
            contact_nodes = find_contact_nodes(tet_verts, pos, weight.radius, weight.radius)

        total_weight = sum(abs(v[2]) for v in contact_nodes.values())
        if total_weight > 1e-10:
            scale = weight.mass * g / total_weight
            for idx, force in contact_nodes.items():
                f_scaled = force * scale
                if idx in all_forces: all_forces[idx] += f_scaled
                else: all_forces[idx] = f_scaled

    if not all_forces:
        raise HTTPException(status_code=400, detail="No contact nodes found")

    deformed_tet_verts = fem.solve(fixed_nodes, all_forces)
    if np.any(~np.isfinite(deformed_tet_verts)):
        raise HTTPException(status_code=400, detail="FEM produced invalid results")

    num_surf = session["num_surface_verts"]
    deformed_surface = deformed_tet_verts[:num_surf]

    import trimesh, io
    mesh = trimesh.Trimesh(vertices=deformed_surface, faces=surface_faces)
    buf = io.BytesIO()
    mesh.export(buf, file_type='stl')
    buf.seek(0)

    return Response(content=buf.read(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=deformed_pillow.stl"})


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


def _decimate_mesh(vertices, faces, target_vertices):
    """Voxel-grid decimation with light smoothing for FEM mesh."""
    bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
    vol = np.prod(bbox_size[bbox_size > 1e-10])
    voxel_size = (vol / target_vertices) ** (1.0 / 3.0) * 0.6
    voxel_size = max(voxel_size, 1e-6)

    quantized = np.round(vertices / voxel_size).astype(np.int64)
    uq, inv = np.unique(quantized, axis=0, return_inverse=True)
    new_verts = uq.astype(float) * voxel_size
    new_faces = inv[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1]) &
        (new_faces[:, 1] != new_faces[:, 2]) &
        (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
        mesh.remove_unreferenced_vertices()
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=4, lamb=0.5, mu=-0.53)
        mesh.remove_unreferenced_vertices()
        return np.array(mesh.vertices, dtype=float), np.array(mesh.faces, dtype=int)
    except Exception:
        pass

    return np.array(new_verts, dtype=float), np.array(new_faces, dtype=int)


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
