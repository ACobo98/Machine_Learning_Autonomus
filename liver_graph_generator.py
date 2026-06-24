"""
liver_graph_generator.py
========================

Reusable graph generator for single-cell-resolved 3D liver tissue segmentations.

The library is designed to reorganize the original liver-regeneration notebook
pipeline into an importable Python module. It keeps the same core computations:

- optional loading/cropping of segmented TIFF volumes;
- master-label construction from hepatocytes, BC, sinusoids, CV and PV;
- hepatocyte and nucleus mesh generation with marching cubes;
- nucleus-to-hepatocyte assignment using raw labels first and expanded labels second;
- face-wise topology probing along mesh normals;
- apical/basal nematic tensors using spherical projection of selected mesh faces;
- cell and nucleus inertia axes;
- PV/CV reference vector field;
- local coarse-grained nematic fields;
- pairwise absolute nematic cosine alignments;
- NetworkX graph construction and diagnostic exports.

Coordinates are assumed to be in z-y-x order, consistent with volumetric TIFF data.

Typical usage
-------------

from liver_graph_generator import LiverGraphGenerator, GraphGeneratorConfig

config = GraphGeneratorConfig(output_dir="outputs/sample_001")
generator = LiverGraphGenerator(config)

result = generator.run_from_paths(
    data_dir="/path/to/sample",
    cells_file="Cells.tif",
    bc_file="BC.tif",
    sinusoids_file="Sinusoids.tif",
    nuclei_file="Nuclei.tif",
    vcvp_file="CV-PV.tif",
    sample_id="sample_001",
)

G = result.graph
node_features = result.node_features
edge_features = result.edge_features
scalar_columns = result.feature_registry.get(level="node", kind="scalar")
vector_groups = result.feature_registry.get(level="node", kind="vector")

Notes
-----
This module operates on segmented/label images, not raw microscopy channels.
Raw segmentation is intentionally outside the scope of this graph generator.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Limit internal numerical-library threads. This follows the original pipeline idea:
# if we parallelize across cells, each worker should not spawn many extra threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import networkx as nx
import numpy as np
import pandas as pd
import tifffile
import trimesh
from numpy.linalg import eigh, norm
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.segmentation import expand_labels

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -----------------------------------------------------------------------------
# Configuration and result containers
# -----------------------------------------------------------------------------


@dataclass
class GraphGeneratorConfig:
    """Configuration for :class:`LiverGraphGenerator`.

    Label conventions follow the original notebook by default.
    """

    # Raw labels in the CV/PV image.
    raw_id_cv: int = 100
    raw_id_pv: int = 200

    # Special labels inside master_labels.
    id_cv: int = 65531
    id_pv: int = 65532
    id_border: int = 65533
    id_bile: int = 65534
    id_sinu: int = 65535

    # Biological labels in nuclei image.
    nucleus_type_labels: Dict[int, str] = field(
        default_factory=lambda: {
            100: "normal_hepatocyte_nucleus",
            200: "metaphase",
            300: "anaphase_telophase",
            400: "cytokinesis",
        }
    )

    # How to convert nucleus type names to feature prefixes.
    nucleus_feature_prefix: Dict[int, str] = field(
        default_factory=lambda: {
            100: "normal",
            200: "metaphase",
            300: "anaphase_telophase",
            400: "cytokinesis",
        }
    )

    # Input preprocessing.
    expansion_dist: int = 20
    output_dtype: Any = np.uint16

    # Nuclear pre-filtering.
    # If enabled, connected nuclear components below the selected voxel-count
    # threshold are removed before mesh generation and nucleus-cell assignment.
    filter_small_nuclei: bool = False
    min_nucleus_voxels: int = 0
    min_nucleus_voxels_by_type: Dict[int, int] = field(default_factory=dict)

    # Mesh generation.
    cell_mesh_pad: int = 2
    nucleus_mesh_pad: int = 2
    vessel_mesh_pad: int = 2
    n_jobs: int = 16

    # Face-wise probing.
    probe_start_dist: float = 1.0
    probe_end_dist: float = 30.0
    probe_step_dist: float = 0.5
    border_margin: int = 5
    exclude_border_cells: bool = True

    # Diagnostics.
    collect_void_face_diagnostics: bool = True
    max_void_faces_per_cell: Optional[int] = None  # None = keep all void faces.

    # Physics.
    compute_reference_frame: bool = True
    compute_coarse_graining: bool = True
    compute_pairwise_cosines: bool = True
    coarse_sigma_vox: float = 80.0
    coarse_exclude_self: bool = True
    max_reference_faces: int = 20000
    reference_min_distance: float = 2.0

    # Output.
    output_dir: Optional[Union[str, Path]] = None
    save_outputs: bool = True
    save_graph_pickle: bool = True
    prefer_parquet: bool = True
    verbose: bool = True


@dataclass
class GraphGeneratorResult:
    """Structured output from :meth:`LiverGraphGenerator.run_from_arrays`."""

    graph: nx.Graph
    node_features: pd.DataFrame
    edge_features: pd.DataFrame
    feature_registry: "FeatureRegistry"
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]
    cell_data: Optional[Dict[int, Dict[str, Any]]] = None
    cell_meshes: Optional[Dict[int, trimesh.Trimesh]] = None
    nucleus_meshes: Optional[Dict[int, trimesh.Trimesh]] = None
    nucleus_data: Optional[Dict[int, Dict[str, Any]]] = None


# -----------------------------------------------------------------------------
# Feature registry
# -----------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Metadata describing one feature or one vector feature group."""

    name: str
    level: str  # node, edge, diagnostic
    kind: str  # scalar, vector, boolean, categorical
    group: str  # geometric, topological, nuclear, polarity, coarse_grained, reference, alignment, identity, etc.
    columns: List[str]
    description: str = ""


class FeatureRegistry:
    """Registry that lets you query features by level, kind, and semantic group."""

    def __init__(self) -> None:
        self._features: Dict[str, FeatureSpec] = {}

    def add(
        self,
        name: str,
        *,
        level: str,
        kind: str,
        group: str,
        columns: Optional[Sequence[str]] = None,
        description: str = "",
    ) -> None:
        if columns is None:
            columns = [name]
        self._features[name] = FeatureSpec(
            name=name,
            level=level,
            kind=kind,
            group=group,
            columns=list(columns),
            description=description,
        )

    def add_scalar(self, name: str, *, level: str, group: str, description: str = "") -> None:
        self.add(name, level=level, kind="scalar", group=group, description=description)

    def add_boolean(self, name: str, *, level: str, group: str, description: str = "") -> None:
        self.add(name, level=level, kind="boolean", group=group, description=description)

    def add_categorical(self, name: str, *, level: str, group: str, description: str = "") -> None:
        self.add(name, level=level, kind="categorical", group=group, description=description)

    def add_vector(self, name: str, *, level: str, group: str, columns: Sequence[str], description: str = "") -> None:
        self.add(name, level=level, kind="vector", group=group, columns=columns, description=description)

    def get(
        self,
        *,
        level: Optional[str] = None,
        kind: Optional[str] = None,
        group: Optional[str] = None,
        names_only: bool = True,
        expand_vectors: bool = False,
    ) -> List[Union[str, FeatureSpec]]:
        """Return features matching filters.

        If ``names_only=True`` and ``expand_vectors=False``, vector features return
        their group name, e.g. ``axis_apical_bipolar``. If ``expand_vectors=True``,
        vectors return their component columns.
        """
        out: List[Union[str, FeatureSpec]] = []
        for spec in self._features.values():
            if level is not None and spec.level != level:
                continue
            if kind is not None and spec.kind != kind:
                continue
            if group is not None and spec.group != group:
                continue
            if not names_only:
                out.append(spec)
            elif expand_vectors and spec.kind == "vector":
                out.extend(spec.columns)
            else:
                out.append(spec.name)
        return out

    def columns(
        self,
        *,
        level: Optional[str] = None,
        kind: Optional[str] = None,
        group: Optional[str] = None,
    ) -> List[str]:
        cols: List[str] = []
        for spec in self.get(level=level, kind=kind, group=group, names_only=False):
            assert isinstance(spec, FeatureSpec)
            cols.extend(spec.columns)
        # Keep order but remove duplicates.
        seen = set()
        unique = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def to_dict(self) -> Dict[str, Any]:
        return {name: asdict(spec) for name, spec in self._features.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureRegistry":
        reg = cls()
        for name, spec in data.items():
            reg._features[name] = FeatureSpec(**spec)
        return reg


# -----------------------------------------------------------------------------
# Basic I/O helpers
# -----------------------------------------------------------------------------


def get_tiff_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Read TIFF metadata without loading the full image into RAM."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape
        dtype = series.dtype
    return {
        "path": str(path),
        "shape": tuple(shape),
        "dtype": str(dtype),
        "file_size_gib": path.stat().st_size / 1024**3,
        "estimated_ram_gib": np.prod(shape) * np.dtype(dtype).itemsize / 1024**3,
    }


def open_memmap(path: Union[str, Path]) -> np.ndarray:
    """Open a TIFF as a NumPy memmap."""
    return tifffile.memmap(Path(path))


def sanitize_crop_bounds(bounds: Tuple[int, int, int, int, int, int], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    """Validate crop bounds given as (z0, z1, y0, y1, x0, x1)."""
    z0, z1, y0, y1, x0, x1 = bounds
    max_z, max_y, max_x = shape
    z0 = max(0, min(int(z0), max_z))
    z1 = max(0, min(int(z1), max_z))
    y0 = max(0, min(int(y0), max_y))
    y1 = max(0, min(int(y1), max_y))
    x0 = max(0, min(int(x0), max_x))
    x1 = max(0, min(int(x1), max_x))
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop bounds {(z0, z1, y0, y1, x0, x1)} for shape {shape}")
    return z0, z1, y0, y1, x0, x1


def read_crop_zyx(path: Union[str, Path], bounds: Optional[Tuple[int, int, int, int, int, int]] = None, output_dtype: Any = np.uint16) -> np.ndarray:
    """Read a z-y-x crop from a TIFF using memmap."""
    arr = open_memmap(path)
    if bounds is None:
        crop = np.asarray(arr[:])
    else:
        z0, z1, y0, y1, x0, x1 = sanitize_crop_bounds(bounds, arr.shape)
        crop = np.asarray(arr[z0:z1, y0:y1, x0:x1])
    if output_dtype is not None:
        crop = crop.astype(output_dtype, copy=False)
    return crop


def safe_write_table(df: pd.DataFrame, path_base: Union[str, Path], prefer_parquet: bool = True, index: bool = False) -> Path:
    """Write DataFrame to parquet if available, otherwise CSV."""
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet:
        try:
            path = path_base.with_suffix(".parquet")
            df.to_parquet(path, index=index)
            return path
        except Exception:
            pass
    path = path_base.with_suffix(".csv")
    df.to_csv(path, index=index)
    return path


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (Path,)):
        return str(x)
    return str(x)


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def normalize_vector(v: Any, eps: float = 1e-12) -> np.ndarray:
    """Normalize a 3D vector. Invalid vectors return NaNs."""
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        return np.full(3, np.nan)
    n = norm(v)
    if n < eps or not np.isfinite(n):
        return np.full(3, np.nan)
    return v / n


def cos_alignment(u: Any, v: Any, eps: float = 1e-12) -> float:
    """Absolute nematic cosine alignment between two axes."""
    u = normalize_vector(u, eps=eps)
    v = normalize_vector(v, eps=eps)
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return float("nan")
    return float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))


def sorted_eigh(tensor: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition sorted from smallest to largest eigenvalue."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        return np.full(3, np.nan), np.full((3, 3), np.nan)
    eigvals, eigvecs = eigh(tensor)
    idx = np.argsort(eigvals)
    return eigvals[idx], eigvecs[:, idx]


def vector_columns(prefix: str) -> List[str]:
    return [f"{prefix}_z", f"{prefix}_y", f"{prefix}_x"]


def add_vector_columns(row: Dict[str, Any], prefix: str, vec: Any) -> None:
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (3,):
        vec = np.full(3, np.nan)
    row[f"{prefix}_z"] = float(vec[0]) if np.isfinite(vec[0]) else np.nan
    row[f"{prefix}_y"] = float(vec[1]) if np.isfinite(vec[1]) else np.nan
    row[f"{prefix}_x"] = float(vec[2]) if np.isfinite(vec[2]) else np.nan


def add_eigval_columns(row: Dict[str, Any], prefix: str, eigvals: Any) -> None:
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.shape != (3,):
        eigvals = np.full(3, np.nan)
    for i in range(3):
        val = eigvals[i]
        row[f"{prefix}_eigval_{i+1}"] = float(val) if np.isfinite(val) else np.nan


def empty_tensor_output() -> Dict[str, Any]:
    return {
        "valid": False,
        "tensor": np.full((3, 3), np.nan),
        "eigvals": np.full(3, np.nan),
        "bipolar_axis": np.full(3, np.nan),
        "ring_axis": np.full(3, np.nan),
        "spherical_area": np.nan,
    }


def empty_axis_field_output() -> Dict[str, Any]:
    return {
        "valid": False,
        "tensor": np.full((3, 3), np.nan),
        "eigvals": np.full(3, np.nan),
        "vec1": np.full(3, np.nan),
        "vec3": np.full(3, np.nan),
        "total_weight": np.nan,
        "n_neighbors_used": 0,
    }


def spherical_triangle_area(A: np.ndarray, B: np.ndarray, C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Area/solid angle of spherical triangles with unit vertices A, B, C.

    Vectorized implementation of the robust oriented-solid-angle formula:

        area = 2 atan2(|det(A,B,C)|, 1 + A·B + B·C + C·A)

    Returns non-negative spherical areas.
    """
    det = np.einsum("ij,ij->i", A, np.cross(B, C))
    denom = 1.0 + np.einsum("ij,ij->i", A, B) + np.einsum("ij,ij->i", B, C) + np.einsum("ij,ij->i", C, A)
    area = 2.0 * np.arctan2(np.abs(det), np.maximum(denom, eps))
    area[~np.isfinite(area)] = 0.0
    return area


def compute_surface_center(mesh: trimesh.Trimesh, eps: float = 1e-12) -> np.ndarray:
    """Area-weighted surface center from all mesh faces."""
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = float(np.sum(face_areas))
    if total_area < eps:
        return np.asarray(mesh.centroid, dtype=float)
    return np.sum(face_centers * face_areas[:, None], axis=0) / total_area


def compute_surface_nematic_tensor(
    mesh: trimesh.Trimesh,
    face_indices: Sequence[int],
    center: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute apical/basal nematic tensor using spherical projection.

    This follows the original pipeline logic:
    selected mesh faces are projected from the area-weighted surface center
    onto the unit sphere. Spherical triangle areas weight the tensor

        Q = 1/(2 A_Ω) sum_f A_f^Ω (3 n_f n_f^T - I)

    where n_f is the spherical face direction approximated by the normalized
    sum of the three projected triangle vertices. The ring axis is the smallest
    eigenvalue eigenvector and the bipolar axis is the largest eigenvalue
    eigenvector.
    """
    output = empty_tensor_output()
    face_indices = np.asarray(face_indices, dtype=int)
    if len(face_indices) == 0:
        return output

    triangles = np.asarray(mesh.triangles, dtype=float)[face_indices]
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        return output

    A = triangles[:, 0, :] - center[None, :]
    B = triangles[:, 1, :] - center[None, :]
    C = triangles[:, 2, :] - center[None, :]

    A = A / (norm(A, axis=1, keepdims=True) + eps)
    B = B / (norm(B, axis=1, keepdims=True) + eps)
    C = C / (norm(C, axis=1, keepdims=True) + eps)

    valid = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1) & np.all(np.isfinite(C), axis=1)
    if not np.any(valid):
        return output
    A, B, C = A[valid], B[valid], C[valid]

    spherical_areas = spherical_triangle_area(A, B, C, eps=eps)
    valid_area = spherical_areas > eps
    if not np.any(valid_area):
        return output
    A, B, C = A[valid_area], B[valid_area], C[valid_area]
    spherical_areas = spherical_areas[valid_area]

    spherical_area_total = float(np.sum(spherical_areas))
    if spherical_area_total < eps:
        return output

    directions = A + B + C
    directions = directions / (norm(directions, axis=1, keepdims=True) + eps)

    Q = np.zeros((3, 3), dtype=float)
    I = np.eye(3)
    for n_i, area_i in zip(directions, spherical_areas):
        Q += float(area_i) * (3.0 * np.outer(n_i, n_i) - I)
    Q *= 0.5 / spherical_area_total

    eigvals, eigvecs = sorted_eigh(Q)
    if not np.all(np.isfinite(eigvals)):
        return output

    output["valid"] = True
    output["tensor"] = Q
    output["eigvals"] = eigvals
    output["ring_axis"] = normalize_vector(eigvecs[:, 0])
    output["bipolar_axis"] = normalize_vector(eigvecs[:, 2])
    output["spherical_area"] = spherical_area_total
    return output


def compute_inertia_axes(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Compute inertia tensor and first/third principal axes from a mesh."""
    output = {
        "valid": False,
        "tensor": np.full((3, 3), np.nan),
        "eigvals": np.full(3, np.nan),
        "vec1": np.full(3, np.nan),
        "vec3": np.full(3, np.nan),
    }
    try:
        tensor = np.asarray(mesh.moment_inertia, dtype=float)
    except Exception:
        return output
    eigvals, eigvecs = sorted_eigh(tensor)
    if not np.all(np.isfinite(eigvals)):
        return output
    output["valid"] = True
    output["tensor"] = tensor
    output["eigvals"] = eigvals
    output["vec1"] = normalize_vector(eigvecs[:, 0])
    output["vec3"] = normalize_vector(eigvecs[:, 2])
    return output


def aggregate_nematic_axes(axes: Sequence[Any], eps: float = 1e-12) -> Dict[str, Any]:
    """Aggregate a collection of nematic axes with equal weights.

    The returned vec3 is the dominant nematic axis. This avoids invalid simple
    vector averaging, because nematic axes have no sign.
    """
    valid_axes = []
    for a in axes:
        a = normalize_vector(a)
        if not np.any(np.isnan(a)):
            valid_axes.append(a)
    if len(valid_axes) == 0:
        return empty_axis_field_output()
    I = np.eye(3)
    Q = np.zeros((3, 3), dtype=float)
    for n in valid_axes:
        Q += np.outer(n, n) - (1.0 / 3.0) * I
    Q = (3.0 / 2.0) * Q / len(valid_axes)
    eigvals, eigvecs = sorted_eigh(Q)
    return {
        "valid": True,
        "tensor": Q,
        "eigvals": eigvals,
        "vec1": normalize_vector(eigvecs[:, 0]),
        "vec3": normalize_vector(eigvecs[:, 2]),
        "total_weight": float(len(valid_axes)),
        "n_neighbors_used": int(len(valid_axes)),
    }


# -----------------------------------------------------------------------------
# Mesh helpers
# -----------------------------------------------------------------------------


def expand_local_slices(local_slices: Tuple[slice, slice, slice], pad: int, volume_shape: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    expanded = []
    for slc, max_size in zip(local_slices, volume_shape):
        start = max(0, int(slc.start) - pad)
        stop = min(int(max_size), int(slc.stop) + pad)
        expanded.append(slice(start, stop))
    return tuple(expanded)  # type: ignore[return-value]


def generate_single_label_mesh(
    label_volume: np.ndarray,
    object_id: int,
    label_slices: Sequence[Optional[Tuple[slice, slice, slice]]],
    pad: int = 2,
) -> Optional[Tuple[int, trimesh.Trimesh]]:
    """Generate mesh for one object label using a local crop."""
    object_id = int(object_id)
    if object_id <= 0 or object_id > len(label_slices):
        return None
    local_slices = label_slices[object_id - 1]
    if local_slices is None:
        return None
    local_slices = expand_local_slices(local_slices, pad, label_volume.shape)
    z_slice, y_slice, x_slice = local_slices
    local_labels = label_volume[z_slice, y_slice, x_slice]
    local_mask = local_labels == object_id
    if not np.any(local_mask):
        return None
    local_mask_padded = np.pad(local_mask, pad_width=1, mode="constant", constant_values=False)
    try:
        verts, faces, _, _ = marching_cubes(local_mask_padded.astype(np.uint8), level=0.5)
    except Exception:
        return None
    verts -= 1.0
    offset = np.array([z_slice.start, y_slice.start, x_slice.start], dtype=np.float32)
    verts = verts + offset
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        if mesh.volume < 0:
            mesh.invert()
    except Exception:
        pass
    return object_id, mesh


def mesh_binary_mask(mask: np.ndarray, pad: int = 2) -> Optional[trimesh.Trimesh]:
    """Mesh one binary mask in global coordinates."""
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    z0, y0, x0 = np.maximum(coords.min(axis=0) - pad, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    local = mask[z0:z1, y0:y1, x0:x1]
    local = np.pad(local, pad_width=1, mode="constant", constant_values=False)
    try:
        verts, faces, _, _ = marching_cubes(local.astype(np.uint8), level=0.5)
    except Exception:
        return None
    verts -= 1.0
    verts += np.array([z0, y0, x0], dtype=float)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        if mesh.volume < 0:
            mesh.invert()
    except Exception:
        pass
    return mesh


def get_surface_charges(mesh: trimesh.Trimesh, max_faces: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a mesh surface into area-weighted charge points."""
    centers = np.asarray(mesh.triangles_center, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    if len(centers) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    if len(centers) > max_faces:
        idx = np.linspace(0, len(centers) - 1, max_faces).astype(int)
        centers = centers[idx]
        areas = areas[idx]
    total_area = float(np.sum(areas))
    if total_area <= 0:
        weights = np.ones(len(areas), dtype=float) / max(len(areas), 1)
    else:
        weights = areas / total_area
    return centers, weights


def field_from_surface(point: np.ndarray, centers: np.ndarray, weights: np.ndarray, sign: float = 1.0, min_distance: float = 2.0) -> np.ndarray:
    """Coulomb-like field contribution from area-weighted surface points."""
    point = np.asarray(point, dtype=float)
    if len(centers) == 0:
        return np.full(3, np.nan)
    diff = point[None, :] - centers
    r2 = np.sum(diff**2, axis=1)
    r2 = np.maximum(r2, min_distance**2)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    return sign * np.sum(weights[:, None] * diff * inv_r3[:, None], axis=0)


def build_local_reference_frame(J: Any, eps: float = 1e-12) -> Dict[str, Any]:
    """Build local orthonormal frame from vector field J.

    v is aligned with J. w is the component of the z-axis orthogonal to v;
    if this degenerates, the y-axis is used. u completes the right-handed frame.
    """
    J = np.asarray(J, dtype=float)
    v = normalize_vector(J, eps=eps)
    if np.any(np.isnan(v)):
        return {"valid": False, "J": J, "J_norm": float(norm(J)) if J.shape == (3,) else np.nan, "u": np.full(3, np.nan), "v": v, "w": np.full(3, np.nan)}

    ez = np.array([1.0, 0.0, 0.0])  # z-axis in z-y-x coordinates.
    ey = np.array([0.0, 1.0, 0.0])
    w = ez - np.dot(ez, v) * v
    w = normalize_vector(w, eps=eps)
    if np.any(np.isnan(w)):
        w = ey - np.dot(ey, v) * v
        w = normalize_vector(w, eps=eps)
    if np.any(np.isnan(w)):
        return {"valid": False, "J": J, "J_norm": float(norm(J)), "u": np.full(3, np.nan), "v": v, "w": w}
    u = normalize_vector(np.cross(v, w), eps=eps)
    if np.any(np.isnan(u)):
        return {"valid": False, "J": J, "J_norm": float(norm(J)), "u": u, "v": v, "w": w}
    w = normalize_vector(np.cross(u, v), eps=eps)
    return {"valid": True, "J": J, "J_norm": float(norm(J)), "u": u, "v": v, "w": w}


# -----------------------------------------------------------------------------
# Main graph generator
# -----------------------------------------------------------------------------


class LiverGraphGenerator:
    """Importable graph generator for segmented 3D liver tissue images."""

    def __init__(self, config: Optional[GraphGeneratorConfig] = None) -> None:
        self.config = config or GraphGeneratorConfig()
        self.registry = FeatureRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_from_paths(
        self,
        *,
        data_dir: Union[str, Path],
        cells_file: str = "Cells.tif",
        bc_file: str = "BC.tif",
        sinusoids_file: str = "Sinusoids.tif",
        nuclei_file: str = "Nuclei.tif",
        vcvp_file: str = "CV-PV.tif",
        crop_bounds: Optional[Tuple[int, int, int, int, int, int]] = None,
        sample_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GraphGeneratorResult:
        """Load segmented TIFFs and generate the graph.

        The input files must be segmented/label images in z-y-x order.
        """
        data_dir = Path(data_dir)
        paths = {
            "cells": data_dir / cells_file,
            "bc": data_dir / bc_file,
            "sinusoids": data_dir / sinusoids_file,
            "nuclei": data_dir / nuclei_file,
            "vcvp": data_dir / vcvp_file,
        }
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} file: {path}")

        self._log("Reading segmented volumes...")
        cells_raw = read_crop_zyx(paths["cells"], crop_bounds, output_dtype=self.config.output_dtype)
        bc_img = read_crop_zyx(paths["bc"], crop_bounds, output_dtype=self.config.output_dtype)
        sinusoids_img = read_crop_zyx(paths["sinusoids"], crop_bounds, output_dtype=self.config.output_dtype)
        nuclei_img = read_crop_zyx(paths["nuclei"], crop_bounds, output_dtype=self.config.output_dtype)
        vcvp_img = read_crop_zyx(paths["vcvp"], crop_bounds, output_dtype=self.config.output_dtype)

        if sample_id is None:
            sample_id = data_dir.name

        return self.run_from_arrays(
            cells_raw=cells_raw,
            bc_img=bc_img,
            sinusoids_img=sinusoids_img,
            nuclei_img=nuclei_img,
            vcvp_img=vcvp_img,
            cells_expanded=None,
            master_labels=None,
            cell_meshes=None,
            nucleus_meshes=None,
            nucleus_data=None,
            sample_id=sample_id,
            output_dir=output_dir,
            crop_bounds=crop_bounds,
        )

    def run_from_arrays(
        self,
        *,
        cells_raw: Optional[np.ndarray] = None,
        cells_expanded: Optional[np.ndarray] = None,
        bc_img: Optional[np.ndarray] = None,
        sinusoids_img: Optional[np.ndarray] = None,
        nuclei_img: Optional[np.ndarray] = None,
        vcvp_img: Optional[np.ndarray] = None,
        master_labels: Optional[np.ndarray] = None,
        cell_meshes: Optional[Dict[int, trimesh.Trimesh]] = None,
        nucleus_meshes: Optional[Dict[int, trimesh.Trimesh]] = None,
        nucleus_data: Optional[Dict[int, Dict[str, Any]]] = None,
        sample_id: str = "sample",
        output_dir: Optional[Union[str, Path]] = None,
        crop_bounds: Optional[Tuple[int, int, int, int, int, int]] = None,
    ) -> GraphGeneratorResult:
        """Generate graph from arrays and/or precomputed meshes.

        For a full graph with all features, provide at least segmented cells, BC,
        sinusoids, nuclei, and CV/PV information. If precomputed meshes are given,
        they are reused.
        """
        cfg = self.config
        self.registry = FeatureRegistry()

        if cells_raw is None and cells_expanded is None and cell_meshes is None:
            raise ValueError("Provide cells_raw, cells_expanded, or precomputed cell_meshes.")

        if cells_expanded is None:
            if cells_raw is None:
                raise ValueError("cells_expanded is missing and cannot be generated without cells_raw.")
            self._log("Expanding hepatocyte labels...")
            cells_expanded = expand_labels(cells_raw, distance=cfg.expansion_dist).astype(np.uint32, copy=False)

        if master_labels is None:
            if bc_img is None or sinusoids_img is None or vcvp_img is None:
                raise ValueError("master_labels is missing. Provide master_labels or bc_img, sinusoids_img, and vcvp_img.")
            self._log("Building master_labels...")
            master_labels = self.build_master_labels(cells_expanded, bc_img, sinusoids_img, vcvp_img)

        self._validate_shapes(cells_raw, cells_expanded, nuclei_img, vcvp_img, master_labels)

        if cell_meshes is None:
            self._log("Generating hepatocyte meshes...")
            cell_meshes, failed_cells = self.generate_cell_meshes(cells_expanded)
        else:
            failed_cells = []
            cell_meshes = {int(k): v for k, v in cell_meshes.items()}

        if nuclei_img is not None and (nucleus_meshes is None or nucleus_data is None):
            self._log("Generating nucleus meshes...")
            nucleus_meshes, nucleus_data, failed_nuclei, filtered_small_nuclei = (
                self.generate_nucleus_meshes(nuclei_img)
            )
        else:
            failed_nuclei = []
            filtered_small_nuclei = []
            nucleus_meshes = nucleus_meshes or {}
            nucleus_data = nucleus_data or {}
            nucleus_meshes = {int(k): v for k, v in nucleus_meshes.items()}
            nucleus_data = {int(k): v for k, v in nucleus_data.items()}
            self._augment_nucleus_inertia(nucleus_meshes, nucleus_data)

        self._log("Assigning nuclei to hepatocytes...")
        cell_nuclei_data, assigned_nuclei_df, unassigned_nuclei_df = self.assign_nuclei_to_cells(
            cell_meshes=cell_meshes,
            nucleus_data=nucleus_data,
            cells_raw=cells_raw,
            cells_expanded=cells_expanded,
        )

        self._log("Mapping face-wise topology...")
        cell_data, void_faces_df = self.map_surface_topology(
            label_volume=master_labels,
            cell_meshes=cell_meshes,
            cell_nuclei_data=cell_nuclei_data,
        )

        valid_analysis_ids, border_cells_df, cells_gt2_df = self.select_valid_cells(cell_data)

        self._log("Computing raw physical tensors...")
        self.compute_raw_physics(cell_data, cell_meshes, valid_analysis_ids)

        if cfg.compute_reference_frame:
            self._log("Computing PV/CV reference frame...")
            self.compute_reference_frame(cell_data, valid_analysis_ids, master_labels=master_labels, vcvp_img=vcvp_img)

        if cfg.compute_coarse_graining:
            self._log("Computing coarse-grained nematic fields...")
            self.compute_coarse_grained_fields(cell_data, valid_analysis_ids)

        self._log("Assembling node features...")
        node_features = self.assemble_node_features(cell_data, cell_meshes, valid_analysis_ids)

        if cfg.compute_pairwise_cosines:
            self._log("Computing pairwise cosine alignments...")
            node_features = self.add_pairwise_cosine_features(node_features)

        self._log("Building graph edges and NetworkX graph...")
        graph, edge_features = self.build_graph(cell_data, node_features, valid_analysis_ids)

        diagnostics = {
            "failed_cell_meshes": pd.DataFrame({"cell_id": [int(x) for x in failed_cells]}),
            "failed_nucleus_meshes": pd.DataFrame(failed_nuclei),
            "filtered_small_nuclei": pd.DataFrame(filtered_small_nuclei),
            "assigned_nuclei": assigned_nuclei_df,
            "unassigned_nuclei": unassigned_nuclei_df,
            "cells_with_more_than_two_nuclei": cells_gt2_df,
            "border_cells": border_cells_df,
            "void_faces": void_faces_df,
            "probing_summary": self._build_probing_summary(cell_data, valid_analysis_ids),
        }

        metadata = {
            "sample_id": sample_id,
            "crop_bounds": crop_bounds,
            "n_cell_meshes": len(cell_meshes),
            "n_nucleus_meshes": len(nucleus_meshes),
            "n_filtered_small_nuclei": len(filtered_small_nuclei),
            "n_valid_cells": len(valid_analysis_ids),
            "n_graph_nodes": graph.number_of_nodes(),
            "n_graph_edges": graph.number_of_edges(),
            "config": asdict(cfg),
        }
        graph.graph.update(metadata)
        graph.graph["feature_registry"] = self.registry.to_dict()

        result = GraphGeneratorResult(
            graph=graph,
            node_features=node_features,
            edge_features=edge_features,
            feature_registry=self.registry,
            diagnostics=diagnostics,
            metadata=metadata,
            cell_data=cell_data,
            cell_meshes=cell_meshes,
            nucleus_meshes=nucleus_meshes,
            nucleus_data=nucleus_data,
        )

        out_dir = Path(output_dir) if output_dir is not None else (Path(cfg.output_dir) if cfg.output_dir is not None else None)
        if cfg.save_outputs and out_dir is not None:
            self.export_result(result, out_dir)

        self._log("Graph generation complete.")
        return result

    # ------------------------------------------------------------------
    # Core pipeline pieces
    # ------------------------------------------------------------------

    def build_master_labels(
        self,
        cells_expanded: np.ndarray,
        bc_img: np.ndarray,
        sinusoids_img: np.ndarray,
        vcvp_img: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        master = cells_expanded.astype(np.uint32, copy=True)
        master[sinusoids_img > 0] = cfg.id_sinu
        master[bc_img > 0] = cfg.id_bile
        master[vcvp_img == cfg.raw_id_cv] = cfg.id_cv
        master[vcvp_img == cfg.raw_id_pv] = cfg.id_pv
        return master

    def generate_cell_meshes(self, cells_expanded: np.ndarray) -> Tuple[Dict[int, trimesh.Trimesh], List[int]]:
        ids = np.unique(cells_expanded)
        ids = ids[ids > 0].astype(int)
        label_slices = ndi.find_objects(cells_expanded)
        meshes: Dict[int, trimesh.Trimesh] = {}
        failed: List[int] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = {
                executor.submit(generate_single_label_mesh, cells_expanded, int(cid), label_slices, self.config.cell_mesh_pad): int(cid)
                for cid in ids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Meshing hepatocytes"):
                cid = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if result is None:
                    failed.append(cid)
                else:
                    out_id, mesh = result
                    meshes[int(out_id)] = mesh
        return meshes, failed

    def generate_nucleus_meshes(
        self,
        nuclei_img: np.ndarray,
    ) -> Tuple[
        Dict[int, trimesh.Trimesh],
        Dict[int, Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Generate nucleus meshes from connected components of the nuclei image.

        Optional small-component filtering happens here, before marching cubes.
        This prevents tiny segmentation artefacts from becoming nucleus meshes,
        being assigned to hepatocytes, and entering the final graph features.
        """
        cfg = self.config
        nucleus_meshes: Dict[int, trimesh.Trimesh] = {}
        nucleus_data: Dict[int, Dict[str, Any]] = {}
        failed: List[Dict[str, Any]] = []
        filtered_small: List[Dict[str, Any]] = []
        next_nucleus_id = 1

        from concurrent.futures import ThreadPoolExecutor, as_completed

        for type_label, type_name in cfg.nucleus_type_labels.items():
            type_label = int(type_label)
            mask = nuclei_img == type_label
            if not np.any(mask):
                continue

            labeled, n_components = ndi.label(mask)
            component_slices = ndi.find_objects(labeled)
            component_ids = np.arange(1, n_components + 1, dtype=int)

            # Count voxels per connected component directly from the original
            # type-specific label image. Component 0 is background.
            component_voxel_counts = np.bincount(
                labeled.ravel(),
                minlength=n_components + 1,
            )

            min_voxels = int(
                cfg.min_nucleus_voxels_by_type.get(
                    type_label,
                    cfg.min_nucleus_voxels,
                )
            )

            component_ids_to_mesh: List[int] = []

            for comp_id in component_ids:
                comp_id = int(comp_id)
                voxel_count = int(component_voxel_counts[comp_id])
                local_slices = component_slices[comp_id - 1]

                if local_slices is None:
                    filtered_small.append({
                        "type_label": type_label,
                        "type_name": str(type_name),
                        "component_id_within_type": comp_id,
                        "voxel_count": voxel_count,
                        "min_nucleus_voxels_used": min_voxels,
                        "bbox_size_z": np.nan,
                        "bbox_size_y": np.nan,
                        "bbox_size_x": np.nan,
                        "bbox_max_size": np.nan,
                        "reason": "missing_component_slice",
                    })
                    continue

                bbox_size_z = int(local_slices[0].stop - local_slices[0].start)
                bbox_size_y = int(local_slices[1].stop - local_slices[1].start)
                bbox_size_x = int(local_slices[2].stop - local_slices[2].start)
                bbox_max_size = int(max(bbox_size_z, bbox_size_y, bbox_size_x))

                if cfg.filter_small_nuclei and voxel_count < min_voxels:
                    filtered_small.append({
                        "type_label": type_label,
                        "type_name": str(type_name),
                        "component_id_within_type": comp_id,
                        "voxel_count": voxel_count,
                        "min_nucleus_voxels_used": min_voxels,
                        "bbox_size_z": bbox_size_z,
                        "bbox_size_y": bbox_size_y,
                        "bbox_size_x": bbox_size_x,
                        "bbox_max_size": bbox_max_size,
                        "reason": "voxel_count_below_threshold",
                    })
                    continue

                component_ids_to_mesh.append(comp_id)

            if cfg.filter_small_nuclei:
                self._log(
                    f"Nucleus type {type_label}: "
                    f"{len(component_ids_to_mesh)} retained, "
                    f"{n_components - len(component_ids_to_mesh)} filtered "
                    f"(min_voxels={min_voxels})."
                )

            with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
                futures = {
                    executor.submit(
                        generate_single_label_mesh,
                        labeled,
                        int(comp_id),
                        component_slices,
                        cfg.nucleus_mesh_pad,
                    ): int(comp_id)
                    for comp_id in component_ids_to_mesh
                }

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Meshing nuclei {type_label}",
                ):
                    comp_id = int(futures[fut])

                    try:
                        result = fut.result()
                    except Exception as e:
                        failed.append({
                            "type_label": type_label,
                            "type_name": str(type_name),
                            "component_id": comp_id,
                            "voxel_count": int(component_voxel_counts[comp_id]),
                            "reason": str(e),
                        })
                        continue

                    if result is None:
                        failed.append({
                            "type_label": type_label,
                            "type_name": str(type_name),
                            "component_id": comp_id,
                            "voxel_count": int(component_voxel_counts[comp_id]),
                            "reason": "mesh_failed",
                        })
                        continue

                    _, mesh = result
                    nid = next_nucleus_id
                    next_nucleus_id += 1

                    centroid = np.asarray(mesh.centroid, dtype=float)
                    try:
                        center_mass = np.asarray(mesh.center_mass, dtype=float)
                    except Exception:
                        center_mass = centroid

                    local_slices = component_slices[comp_id - 1]
                    if local_slices is not None:
                        bbox_size_z = int(local_slices[0].stop - local_slices[0].start)
                        bbox_size_y = int(local_slices[1].stop - local_slices[1].start)
                        bbox_size_x = int(local_slices[2].stop - local_slices[2].start)
                        bbox_max_size = int(max(bbox_size_z, bbox_size_y, bbox_size_x))
                    else:
                        bbox_size_z = np.nan
                        bbox_size_y = np.nan
                        bbox_size_x = np.nan
                        bbox_max_size = np.nan

                    nucleus_meshes[nid] = mesh
                    inertia = compute_inertia_axes(mesh)
                    nucleus_data[nid] = {
                        "nucleus_id": int(nid),
                        "type_label": int(type_label),
                        "type_name": str(type_name),
                        "component_id_within_type": int(comp_id),
                        "voxel_count": int(component_voxel_counts[comp_id]),
                        "bbox_size_z": bbox_size_z,
                        "bbox_size_y": bbox_size_y,
                        "bbox_size_x": bbox_size_x,
                        "bbox_max_size": bbox_max_size,
                        "mesh_volume": float(mesh.volume),
                        "mesh_area": float(mesh.area),
                        "mesh_centroid_zyx": centroid.tolist(),
                        "mesh_center_mass_zyx": center_mass.tolist(),
                        "inertia": inertia,
                    }

            del mask, labeled, component_slices
            gc.collect()

        return nucleus_meshes, nucleus_data, failed, filtered_small

    def assign_nuclei_to_cells(
        self,
        *,
        cell_meshes: Dict[int, trimesh.Trimesh],
        nucleus_data: Dict[int, Dict[str, Any]],
        cells_raw: Optional[np.ndarray],
        cells_expanded: Optional[np.ndarray],
    ) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
        cell_nuclei_data: Dict[int, Dict[str, Any]] = {}
        for cid in cell_meshes.keys():
            cell_nuclei_data[int(cid)] = {
                "cell_id": int(cid),
                "n_nuclei": 0,
                "nuclei": [],
                "has_nucleus_pair_vector": False,
                "nucleus_pair_vector_zyx": None,
                "nucleus_pair_unit_vector_zyx": None,
                "nucleus_pair_distance": np.nan,
            }

        assigned: List[Dict[str, Any]] = []
        unassigned: List[Dict[str, Any]] = []
        if len(nucleus_data) == 0:
            return cell_nuclei_data, pd.DataFrame(), pd.DataFrame()
        if cells_raw is None and cells_expanded is None:
            # Mesh-only fallback: nearest centroid assignment. This is less faithful than the original.
            self._log("Warning: cells_raw/cells_expanded missing. Using nearest-cell centroid assignment for nuclei.")
            cell_ids = np.array(list(cell_meshes.keys()), dtype=int)
            centers = np.array([cell_meshes[cid].centroid for cid in cell_ids], dtype=float)
            tree = cKDTree(centers)
            for nid, nuc in nucleus_data.items():
                centroid = np.asarray(nuc["mesh_centroid_zyx"], dtype=float)
                dist, idx = tree.query(centroid)
                owner = int(cell_ids[idx])
                info = self._compact_nucleus_info(nid, nuc, "nearest_cell_centroid")
                cell_nuclei_data[owner]["nuclei"].append(info)
                assigned.append({"nucleus_id": int(nid), "owner_cell_id": owner, "assignment_source": "nearest_cell_centroid", "distance": float(dist), "type_label": int(nuc["type_label"]), "type_name": nuc["type_name"]})
            self._finalize_cell_nuclei_data(cell_nuclei_data)
            return cell_nuclei_data, pd.DataFrame(assigned), pd.DataFrame(unassigned)

        ref_shape = cells_raw.shape if cells_raw is not None else cells_expanded.shape  # type: ignore[union-attr]
        for nid, nuc in nucleus_data.items():
            centroid = np.asarray(nuc["mesh_centroid_zyx"], dtype=float)
            zi, yi, xi = np.round(centroid).astype(int)
            inside = (0 <= zi < ref_shape[0]) and (0 <= yi < ref_shape[1]) and (0 <= xi < ref_shape[2])
            if not inside:
                unassigned.append({"nucleus_id": int(nid), "type_label": int(nuc.get("type_label", -1)), "type_name": nuc.get("type_name", ""), "reason": "centroid_outside_crop", "centroid_z": centroid[0], "centroid_y": centroid[1], "centroid_x": centroid[2]})
                continue
            owner = 0
            source = "none"
            if cells_raw is not None:
                owner = int(cells_raw[zi, yi, xi])
                source = "cells_raw"
            if owner == 0 and cells_expanded is not None:
                owner = int(cells_expanded[zi, yi, xi])
                source = "cells_expanded"
            if owner == 0:
                unassigned.append({"nucleus_id": int(nid), "type_label": int(nuc.get("type_label", -1)), "type_name": nuc.get("type_name", ""), "reason": "centroid_on_background", "centroid_z": centroid[0], "centroid_y": centroid[1], "centroid_x": centroid[2]})
                continue
            if owner not in cell_nuclei_data:
                unassigned.append({"nucleus_id": int(nid), "type_label": int(nuc.get("type_label", -1)), "type_name": nuc.get("type_name", ""), "reason": "owner_cell_not_in_cell_meshes", "owner_cell_id": int(owner), "centroid_z": centroid[0], "centroid_y": centroid[1], "centroid_x": centroid[2]})
                continue
            info = self._compact_nucleus_info(nid, nuc, source)
            cell_nuclei_data[owner]["nuclei"].append(info)
            assigned.append({"nucleus_id": int(nid), "owner_cell_id": int(owner), "assignment_source": source, "type_label": int(nuc["type_label"]), "type_name": nuc["type_name"]})

        self._finalize_cell_nuclei_data(cell_nuclei_data)
        return cell_nuclei_data, pd.DataFrame(assigned), pd.DataFrame(unassigned)

    def map_surface_topology(
        self,
        *,
        label_volume: np.ndarray,
        cell_meshes: Dict[int, trimesh.Trimesh],
        cell_nuclei_data: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
        cfg = self.config
        max_z, max_y, max_x = label_volume.shape
        search_distances = np.arange(cfg.probe_start_dist, cfg.probe_end_dist + cfg.probe_step_dist, cfg.probe_step_dist)
        mapped: Dict[int, Dict[str, Any]] = {}
        void_rows: List[Dict[str, Any]] = []

        for cid, mesh in tqdm(cell_meshes.items(), desc="Mapping face topology"):
            cid = int(cid)
            face_centers = np.asarray(mesh.triangles_center, dtype=float)
            face_normals = np.asarray(mesh.face_normals, dtype=float)
            face_areas = np.asarray(mesh.area_faces, dtype=float)
            n_faces = len(face_centers)
            face_labels = np.zeros(n_faces, dtype=np.int32)
            unresolved = np.ones(n_faces, dtype=bool)

            for dist in search_distances:
                current = np.where(unresolved)[0]
                if len(current) == 0:
                    break
                probe_coords = face_centers[current] + face_normals[current] * float(dist)
                probe_idx = np.round(probe_coords).astype(int)
                is_border = (
                    (probe_idx[:, 0] < cfg.border_margin)
                    | (probe_idx[:, 0] >= max_z - cfg.border_margin)
                    | (probe_idx[:, 1] < cfg.border_margin)
                    | (probe_idx[:, 1] >= max_y - cfg.border_margin)
                    | (probe_idx[:, 2] < cfg.border_margin)
                    | (probe_idx[:, 2] >= max_x - cfg.border_margin)
                )
                if np.any(is_border):
                    border_idx = current[is_border]
                    face_labels[border_idx] = cfg.id_border
                    unresolved[border_idx] = False

                inside_current = current[~is_border]
                if len(inside_current) == 0:
                    continue
                inside_idx = probe_idx[~is_border]
                voxel_ids = label_volume[inside_idx[:, 0], inside_idx[:, 1], inside_idx[:, 2]]
                is_valid_contact = (voxel_ids != 0) & (voxel_ids != cid)
                if np.any(is_valid_contact):
                    found_idx = inside_current[is_valid_contact]
                    found_values = voxel_ids[is_valid_contact]
                    face_labels[found_idx] = found_values.astype(np.int32, copy=False)
                    unresolved[found_idx] = False

            masks = self._face_class_masks(face_labels)
            areas = {name: float(np.sum(face_areas[mask])) for name, mask in masks.items()}
            area_mapped = areas["apical"] + areas["basal"] + areas["lateral"] + areas["void"] + areas["cv"] + areas["pv"]

            if cfg.collect_void_face_diagnostics:
                void_idx = np.where(face_labels == 0)[0]
                if cfg.max_void_faces_per_cell is not None:
                    void_idx = void_idx[: int(cfg.max_void_faces_per_cell)]
                for fid in void_idx:
                    void_rows.append({
                        "cell_id": cid,
                        "face_id": int(fid),
                        "face_area": float(face_areas[fid]),
                        "face_center_z": float(face_centers[fid, 0]),
                        "face_center_y": float(face_centers[fid, 1]),
                        "face_center_x": float(face_centers[fid, 2]),
                        "face_normal_z": float(face_normals[fid, 0]),
                        "face_normal_y": float(face_normals[fid, 1]),
                        "face_normal_x": float(face_normals[fid, 2]),
                        "final_label": 0,
                        "probe_status": "void_after_max_distance",
                    })

            mapped[cid] = {
                "cell_id": cid,
                "centroid": np.asarray(mesh.centroid, dtype=float),
                "face_labels": face_labels,
                "face_areas": face_areas,
                "area_apical": areas["apical"],
                "area_basal": areas["basal"],
                "area_lateral": areas["lateral"],
                "area_border": areas["border"],
                "area_void": areas["void"],
                "area_cv": areas["cv"],
                "area_pv": areas["pv"],
                "mapped_surface_area": area_mapped,
                "nuclei": cell_nuclei_data.get(cid, None),
            }
        return mapped, pd.DataFrame(void_rows)

    def select_valid_cells(self, cell_data: Dict[int, Dict[str, Any]]) -> Tuple[List[int], pd.DataFrame, pd.DataFrame]:
        border_rows = []
        gt2_rows = []
        valid_ids = []
        for cid in sorted(cell_data.keys()):
            data = cell_data[cid]
            area_border = float(data.get("area_border", 0.0))
            is_border = area_border > 0
            nuclei = data.get("nuclei", {}) or {}
            n_nuc = int(nuclei.get("n_nuclei", 0))
            has_gt2 = n_nuc > 2
            data["is_border_cell"] = is_border
            data["has_more_than_two_nuclei"] = has_gt2
            use = (not is_border) if self.config.exclude_border_cells else True
            data["use_for_graph"] = use
            data["use_for_physics"] = use
            if use:
                valid_ids.append(int(cid))
            if is_border:
                border_rows.append({"cell_id": int(cid), "area_border": area_border, "n_nuclei": n_nuc})
            if has_gt2:
                gt2_rows.append({
                    "cell_id": int(cid),
                    "n_nuclei": n_nuc,
                    "assigned_nucleus_ids": ";".join(str(n["nucleus_id"]) for n in nuclei.get("nuclei", [])),
                    "n_normal": nuclei.get("n_normal", 0),
                    "n_metaphase": nuclei.get("n_metaphase", 0),
                    "n_anaphase_telophase": nuclei.get("n_anaphase_telophase", 0),
                    "n_cytokinesis": nuclei.get("n_cytokinesis", 0),
                })
        return valid_ids, pd.DataFrame(border_rows), pd.DataFrame(gt2_rows)

    def compute_raw_physics(self, cell_data: Dict[int, Dict[str, Any]], cell_meshes: Dict[int, trimesh.Trimesh], cell_ids: Sequence[int]) -> None:
        cfg = self.config
        for cid in tqdm(cell_ids, desc="Computing raw tensors"):
            mesh = cell_meshes[int(cid)]
            data = cell_data[int(cid)]
            face_labels = np.asarray(data["face_labels"])
            projection_center = compute_surface_center(mesh)
            apical_idx = np.where(face_labels == cfg.id_bile)[0]
            basal_idx = np.where(face_labels == cfg.id_sinu)[0]
            apical = compute_surface_nematic_tensor(mesh, apical_idx, projection_center)
            basal = compute_surface_nematic_tensor(mesh, basal_idx, projection_center)
            inertia = compute_inertia_axes(mesh)
            data["physics_raw"] = {
                "projection_center": projection_center,
                "apical": apical,
                "basal": basal,
                "inertia": inertia,
            }

    def compute_reference_frame(
        self,
        cell_data: Dict[int, Dict[str, Any]],
        cell_ids: Sequence[int],
        *,
        master_labels: np.ndarray,
        vcvp_img: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.config
        if vcvp_img is not None:
            cv_mask = vcvp_img == cfg.raw_id_cv
            pv_mask = vcvp_img == cfg.raw_id_pv
        else:
            cv_mask = master_labels == cfg.id_cv
            pv_mask = master_labels == cfg.id_pv
        if not np.any(cv_mask) or not np.any(pv_mask):
            for cid in cell_ids:
                cell_data[int(cid)]["reference_frame"] = build_local_reference_frame(np.full(3, np.nan))
                cell_data[int(cid)]["portal_central_position"] = self._empty_portal_central_position()
            return

        cv_mesh = mesh_binary_mask(cv_mask, pad=cfg.vessel_mesh_pad)
        pv_mesh = mesh_binary_mask(pv_mask, pad=cfg.vessel_mesh_pad)
        if cv_mesh is None or pv_mesh is None:
            for cid in cell_ids:
                cell_data[int(cid)]["reference_frame"] = build_local_reference_frame(np.full(3, np.nan))
                cell_data[int(cid)]["portal_central_position"] = self._empty_portal_central_position()
            return

        cv_centers, cv_weights = get_surface_charges(cv_mesh, max_faces=cfg.max_reference_faces)
        pv_centers, pv_weights = get_surface_charges(pv_mesh, max_faces=cfg.max_reference_faces)

        for cid in tqdm(cell_ids, desc="Computing reference field"):
            cid = int(cid)
            point = np.asarray(cell_data[cid].get("physics_raw", {}).get("projection_center", cell_data[cid].get("centroid", np.full(3, np.nan))), dtype=float)
            J_pv = field_from_surface(point, pv_centers, pv_weights, sign=+1.0, min_distance=cfg.reference_min_distance)
            J_cv = field_from_surface(point, cv_centers, cv_weights, sign=-1.0, min_distance=cfg.reference_min_distance)
            J = J_pv + J_cv
            cell_data[cid]["reference_frame"] = build_local_reference_frame(J)

        self._compute_nearest_vessel_distances(cell_data, cell_ids, cv_mesh, pv_mesh)

    def compute_coarse_grained_fields(self, cell_data: Dict[int, Dict[str, Any]], cell_ids: Sequence[int]) -> None:
        axis_getters: Dict[str, Callable[[Dict[str, Any]], np.ndarray]] = {
            "apical_bipolar": lambda d: d.get("physics_raw", {}).get("apical", {}).get("bipolar_axis", np.full(3, np.nan)),
            "apical_ring": lambda d: d.get("physics_raw", {}).get("apical", {}).get("ring_axis", np.full(3, np.nan)),
            "basal_bipolar": lambda d: d.get("physics_raw", {}).get("basal", {}).get("bipolar_axis", np.full(3, np.nan)),
            "basal_ring": lambda d: d.get("physics_raw", {}).get("basal", {}).get("ring_axis", np.full(3, np.nan)),
            "inertia_vec1": lambda d: d.get("physics_raw", {}).get("inertia", {}).get("vec1", np.full(3, np.nan)),
            "inertia_vec3": lambda d: d.get("physics_raw", {}).get("inertia", {}).get("vec3", np.full(3, np.nan)),
            "nucleus_pair": lambda d: self._get_nucleus_pair_axis(d),
            "nucleus_inertia_vec1": lambda d: d.get("nuclear_aggregate", {}).get("axis_nucleus_inertia_vec1", np.full(3, np.nan)),
            "nucleus_inertia_vec3": lambda d: d.get("nuclear_aggregate", {}).get("axis_nucleus_inertia_vec3", np.full(3, np.nan)),
        }
        # Prepare per-cell nuclear inertia aggregates before coarse-graining.
        for cid in cell_ids:
            self._compute_cell_nuclear_aggregate(cell_data[int(cid)])
        for output_key, getter in axis_getters.items():
            self._compute_coarse_grained_axis_field(cell_data, cell_ids, getter, output_key)

    def assemble_node_features(
        self,
        cell_data: Dict[int, Dict[str, Any]],
        cell_meshes: Dict[int, trimesh.Trimesh],
        cell_ids: Sequence[int],
    ) -> pd.DataFrame:
        self._register_base_features()
        rows: List[Dict[str, Any]] = []
        cfg = self.config
        for cid in cell_ids:
            cid = int(cid)
            mesh = cell_meshes[cid]
            data = cell_data[cid]
            nuclei = data.get("nuclei", {}) or {}
            self._compute_cell_nuclear_aggregate(data)
            nucagg = data.get("nuclear_aggregate", {})

            row: Dict[str, Any] = {
                "cell_id": cid,
                "has_more_than_two_nuclei": bool(data.get("has_more_than_two_nuclei", False)),
                "cell_volume": float(mesh.volume),
                "cell_surface_area": float(mesh.area),
                "cell_centroid_z": float(mesh.centroid[0]),
                "cell_centroid_y": float(mesh.centroid[1]),
                "cell_centroid_x": float(mesh.centroid[2]),
                "projection_center_z": float(data.get("physics_raw", {}).get("projection_center", np.full(3, np.nan))[0]),
                "projection_center_y": float(data.get("physics_raw", {}).get("projection_center", np.full(3, np.nan))[1]),
                "projection_center_x": float(data.get("physics_raw", {}).get("projection_center", np.full(3, np.nan))[2]),
            }

            for key in ["apical", "basal", "lateral", "void", "cv", "pv"]:
                row[f"area_{key}"] = float(data.get(f"area_{key}", np.nan))
            denom = sum(row[f"area_{k}"] for k in ["apical", "basal", "lateral", "void", "cv", "pv"] if np.isfinite(row[f"area_{k}"]))
            for key in ["apical", "basal", "lateral", "void", "cv", "pv"]:
                row[f"frac_{key}"] = row[f"area_{key}"] / denom if denom > 0 else np.nan

            self._add_nuclear_features_to_row(row, nuclei, nucagg)
            self._add_raw_physics_to_row(row, data)
            self._add_reference_to_row(row, data)
            self._add_cg_to_row(row, data)
            rows.append(row)
        return pd.DataFrame(rows)

    def add_pairwise_cosine_features(self, node_features: pd.DataFrame) -> pd.DataFrame:
        axis_names = self.retained_axis_names(node_features.columns)
        for a_i, a in enumerate(axis_names):
            cols_a = vector_columns(f"axis_{a}") if not a.startswith("axis_") else vector_columns(a)
            # retained_axis_names returns names without 'axis_' prefix by design.
            cols_a = vector_columns(f"axis_{a}")
            if not all(c in node_features.columns for c in cols_a):
                continue
            for b in axis_names[a_i + 1:]:
                cols_b = vector_columns(f"axis_{b}")
                if not all(c in node_features.columns for c in cols_b):
                    continue
                col = f"cos_{a}__{b}"
                vals = []
                A = node_features[cols_a].to_numpy(dtype=float)
                B = node_features[cols_b].to_numpy(dtype=float)
                for u, v in zip(A, B):
                    vals.append(cos_alignment(u, v))
                node_features[col] = vals
                self.registry.add_scalar(col, level="node", group="alignment", description=f"Absolute nematic cosine alignment between {a} and {b}.")
        return node_features

    def build_graph(
        self,
        cell_data: Dict[int, Dict[str, Any]],
        node_features: pd.DataFrame,
        cell_ids: Sequence[int],
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        cfg = self.config
        valid_set = {int(cid) for cid in cell_ids}
        special = {int(cfg.id_bile), int(cfg.id_sinu), int(cfg.id_border), int(cfg.id_cv), int(cfg.id_pv)}
        G = nx.Graph()
        feature_table = node_features.set_index("cell_id")
        for cid in cell_ids:
            cid = int(cid)
            if cid not in feature_table.index:
                continue
            attrs = feature_table.loc[cid].to_dict()
            attrs["cell_id"] = cid
            G.add_node(cid, **attrs)

        for cid in tqdm(cell_ids, desc="Building graph edges"):
            cid = int(cid)
            face_labels = np.asarray(cell_data[cid]["face_labels"])
            face_areas = np.asarray(cell_data[cid]["face_areas"], dtype=float)
            for nid in np.unique(face_labels):
                nid = int(nid)
                if nid == 0 or nid == cid or nid in special or nid not in valid_set:
                    continue
                mask = face_labels == nid
                area = float(np.sum(face_areas[mask]))
                if area <= 0:
                    continue
                u, v = sorted([cid, nid])
                if not G.has_edge(u, v):
                    G.add_edge(u, v, contact_area_total=0.0)
                G[u][v]["contact_area_total"] += area

        edge_rows: List[Dict[str, Any]] = []
        for u, v, attrs in G.edges(data=True):
            cu = np.array([G.nodes[u].get("projection_center_z", np.nan), G.nodes[u].get("projection_center_y", np.nan), G.nodes[u].get("projection_center_x", np.nan)], dtype=float)
            cv = np.array([G.nodes[v].get("projection_center_z", np.nan), G.nodes[v].get("projection_center_y", np.nan), G.nodes[v].get("projection_center_x", np.nan)], dtype=float)
            dist = float(norm(cu - cv)) if np.all(np.isfinite(cu)) and np.all(np.isfinite(cv)) else np.nan
            attrs["centroid_distance"] = dist
            edge_rows.append({"cell_u": int(u), "cell_v": int(v), "contact_area_total": float(attrs["contact_area_total"]), "centroid_distance": dist})

        self.registry.add_scalar("cell_u", level="edge", group="identity", description="First hepatocyte ID in undirected edge.")
        self.registry.add_scalar("cell_v", level="edge", group="identity", description="Second hepatocyte ID in undirected edge.")
        self.registry.add_scalar("contact_area_total", level="edge", group="topological", description="Accumulated mapped hepatocyte-hepatocyte contact area from both probing directions.")
        self.registry.add_scalar("centroid_distance", level="edge", group="geometric", description="Distance between hepatocyte projection centers.")
        return G, pd.DataFrame(edge_rows)

    def export_result(self, result: GraphGeneratorResult, output_dir: Union[str, Path]) -> None:
        out = Path(output_dir)
        graph_dir = out / "graph"
        diag_dir = out / "diagnostics"
        graph_dir.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(parents=True, exist_ok=True)

        safe_write_table(result.node_features, graph_dir / "node_features", prefer_parquet=self.config.prefer_parquet, index=False)
        safe_write_table(result.edge_features, graph_dir / "edge_features", prefer_parquet=self.config.prefer_parquet, index=False)

        with open(graph_dir / "feature_registry.json", "w", encoding="utf-8") as f:
            json.dump(result.feature_registry.to_dict(), f, indent=2, default=_json_default)
        with open(graph_dir / "graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result.metadata, f, indent=2, default=_json_default)
        if self.config.save_graph_pickle:
            with open(graph_dir / "hepatocyte_graph.pkl", "wb") as f:
                pickle.dump(result.graph, f)

        for name, obj in result.diagnostics.items():
            if isinstance(obj, pd.DataFrame):
                safe_write_table(obj, diag_dir / name, prefer_parquet=self.config.prefer_parquet, index=False)
            else:
                with open(diag_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _validate_shapes(self, *arrays: Optional[np.ndarray]) -> None:
        shapes = [arr.shape for arr in arrays if arr is not None]
        if len(shapes) == 0:
            return
        ref = shapes[0]
        for s in shapes[1:]:
            if s != ref:
                raise ValueError(f"All provided volumes must have the same shape. Got {shapes}")

    def _augment_nucleus_inertia(self, nucleus_meshes: Dict[int, trimesh.Trimesh], nucleus_data: Dict[int, Dict[str, Any]]) -> None:
        for nid, mesh in nucleus_meshes.items():
            if nid in nucleus_data and "inertia" not in nucleus_data[nid]:
                nucleus_data[nid]["inertia"] = compute_inertia_axes(mesh)

    def _compact_nucleus_info(self, nid: int, nuc: Dict[str, Any], source: str) -> Dict[str, Any]:
        inertia = nuc.get("inertia", {})
        return {
            "nucleus_id": int(nid),
            "type_label": int(nuc.get("type_label", -1)),
            "type_name": str(nuc.get("type_name", "")),
            "mesh_volume": float(nuc.get("mesh_volume", np.nan)),
            "mesh_area": float(nuc.get("mesh_area", np.nan)),
            "voxel_count": int(nuc.get("voxel_count", -1)),
            "bbox_max_size": float(nuc.get("bbox_max_size", np.nan)),
            "mesh_centroid_zyx": nuc.get("mesh_centroid_zyx", [np.nan, np.nan, np.nan]),
            "mesh_center_mass_zyx": nuc.get("mesh_center_mass_zyx", [np.nan, np.nan, np.nan]),
            "inertia_eigvals": np.asarray(inertia.get("eigvals", np.full(3, np.nan)), dtype=float).tolist(),
            "inertia_vec1": np.asarray(inertia.get("vec1", np.full(3, np.nan)), dtype=float).tolist(),
            "inertia_vec3": np.asarray(inertia.get("vec3", np.full(3, np.nan)), dtype=float).tolist(),
            "assignment_source": source,
        }

    def _finalize_cell_nuclei_data(self, cell_nuclei_data: Dict[int, Dict[str, Any]]) -> None:
        prefix_by_label = self.config.nucleus_feature_prefix
        for cid, cdata in cell_nuclei_data.items():
            nuclei = cdata["nuclei"]
            cdata["n_nuclei"] = len(nuclei)
            # Count all configured categories, and preserve original simple names.
            for label, prefix in prefix_by_label.items():
                cdata[f"n_{prefix}"] = int(sum(int(n["type_label"]) == int(label) for n in nuclei))
            # Backward-compatible expected names.
            cdata["n_normal"] = cdata.get("n_normal", 0)
            cdata["n_metaphase"] = cdata.get("n_metaphase", 0)
            cdata["n_anaphase_telophase"] = cdata.get("n_anaphase_telophase", 0)
            cdata["n_cytokinesis"] = cdata.get("n_cytokinesis", 0)
            if len(nuclei) == 2:
                p0 = np.asarray(nuclei[0]["mesh_centroid_zyx"], dtype=float)
                p1 = np.asarray(nuclei[1]["mesh_centroid_zyx"], dtype=float)
                vec = p1 - p0
                d = float(norm(vec))
                if d > 0:
                    cdata["has_nucleus_pair_vector"] = True
                    cdata["nucleus_pair_vector_zyx"] = vec.tolist()
                    cdata["nucleus_pair_unit_vector_zyx"] = (vec / d).tolist()
                    cdata["nucleus_pair_distance"] = d

    def _face_class_masks(self, face_labels: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        mask_apical = face_labels == cfg.id_bile
        mask_basal = face_labels == cfg.id_sinu
        mask_border = face_labels == cfg.id_border
        mask_void = face_labels == 0
        mask_cv = face_labels == cfg.id_cv
        mask_pv = face_labels == cfg.id_pv
        mask_lateral = (face_labels > 0) & (~mask_apical) & (~mask_basal) & (~mask_border) & (~mask_cv) & (~mask_pv)
        return {"apical": mask_apical, "basal": mask_basal, "lateral": mask_lateral, "border": mask_border, "void": mask_void, "cv": mask_cv, "pv": mask_pv}

    def _empty_portal_central_position(self) -> Dict[str, Any]:
        return {"valid": False, "d_cv": np.nan, "d_pv": np.nan, "chi_cv_to_pv": np.nan, "nearest_cv_triangle_index": -1, "nearest_pv_triangle_index": -1, "nearest_cv_point": np.full(3, np.nan), "nearest_pv_point": np.full(3, np.nan)}

    def _compute_nearest_vessel_distances(self, cell_data: Dict[int, Dict[str, Any]], cell_ids: Sequence[int], cv_mesh: trimesh.Trimesh, pv_mesh: trimesh.Trimesh) -> None:
        cv_centers = np.asarray(cv_mesh.triangles_center, dtype=float)
        pv_centers = np.asarray(pv_mesh.triangles_center, dtype=float)
        cv_tree = cKDTree(cv_centers)
        pv_tree = cKDTree(pv_centers)
        for cid in tqdm(cell_ids, desc="Computing nearest CV/PV distances"):
            cid = int(cid)
            point = np.asarray(cell_data[cid].get("physics_raw", {}).get("projection_center", cell_data[cid].get("centroid", np.full(3, np.nan))), dtype=float)
            if np.any(~np.isfinite(point)):
                cell_data[cid]["portal_central_position"] = self._empty_portal_central_position()
                continue
            d_cv, idx_cv = cv_tree.query(point)
            d_pv, idx_pv = pv_tree.query(point)
            denom = float(d_cv + d_pv)
            chi = float(d_cv / denom) if denom > 1e-12 else np.nan
            cell_data[cid]["portal_central_position"] = {
                "valid": bool(np.isfinite(chi)),
                "d_cv": float(d_cv),
                "d_pv": float(d_pv),
                "chi_cv_to_pv": chi,
                "nearest_cv_triangle_index": int(idx_cv),
                "nearest_pv_triangle_index": int(idx_pv),
                "nearest_cv_point": cv_centers[int(idx_cv)],
                "nearest_pv_point": pv_centers[int(idx_pv)],
            }

    def _get_nucleus_pair_axis(self, data: Dict[str, Any]) -> np.ndarray:
        nuclei = data.get("nuclei", {}) or {}
        if not nuclei.get("has_nucleus_pair_vector", False):
            return np.full(3, np.nan)
        return normalize_vector(nuclei.get("nucleus_pair_unit_vector_zyx", np.full(3, np.nan)))

    def _compute_cell_nuclear_aggregate(self, data: Dict[str, Any]) -> None:
        if "nuclear_aggregate" in data:
            return
        nuclei_info = data.get("nuclei", {}) or {}
        nuclei = nuclei_info.get("nuclei", [])
        agg: Dict[str, Any] = {
            "nucleus_inertia_eigvals_mean": np.full(3, np.nan),
            "nucleus_inertia_vec1_field": empty_axis_field_output(),
            "nucleus_inertia_vec3_field": empty_axis_field_output(),
            "axis_nucleus_inertia_vec1": np.full(3, np.nan),
            "axis_nucleus_inertia_vec3": np.full(3, np.nan),
        }
        if len(nuclei) > 0:
            eigvals = np.array([n.get("inertia_eigvals", np.full(3, np.nan)) for n in nuclei], dtype=float)
            if eigvals.ndim == 2 and eigvals.shape[1] == 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    agg["nucleus_inertia_eigvals_mean"] = np.nanmean(eigvals, axis=0)
            vec1_axes = [n.get("inertia_vec1", np.full(3, np.nan)) for n in nuclei]
            vec3_axes = [n.get("inertia_vec3", np.full(3, np.nan)) for n in nuclei]
            field1 = aggregate_nematic_axes(vec1_axes)
            field3 = aggregate_nematic_axes(vec3_axes)
            agg["nucleus_inertia_vec1_field"] = field1
            agg["nucleus_inertia_vec3_field"] = field3
            agg["axis_nucleus_inertia_vec1"] = field1.get("vec3", np.full(3, np.nan))
            agg["axis_nucleus_inertia_vec3"] = field3.get("vec3", np.full(3, np.nan))
        data["nuclear_aggregate"] = agg

    def _compute_coarse_grained_axis_field(
        self,
        cell_data: Dict[int, Dict[str, Any]],
        cell_ids: Sequence[int],
        axis_getter: Callable[[Dict[str, Any]], np.ndarray],
        output_key: str,
    ) -> None:
        cfg = self.config
        ids_used = []
        centers = []
        axes = []
        for cid in cell_ids:
            cid = int(cid)
            axis = normalize_vector(axis_getter(cell_data[cid]))
            center = np.asarray(cell_data[cid].get("physics_raw", {}).get("projection_center", cell_data[cid].get("centroid", np.full(3, np.nan))), dtype=float)
            if np.any(np.isnan(axis)) or np.any(~np.isfinite(center)):
                continue
            ids_used.append(cid)
            centers.append(center)
            axes.append(axis)
        for cid in cell_ids:
            cell_data[int(cid)].setdefault("physics_cg", {})[output_key] = empty_axis_field_output()
        if len(ids_used) < 2:
            return
        centers_arr = np.vstack(centers)
        axes_arr = np.vstack(axes)
        # Full distance matrix, consistent with original implementation. For very large tissues,
        # this can be replaced by a sparse radius-neighbor version later.
        d2 = np.sum((centers_arr[:, None, :] - centers_arr[None, :, :]) ** 2, axis=2)
        W = np.exp(-d2 / (2.0 * cfg.coarse_sigma_vox**2))
        if cfg.coarse_exclude_self:
            np.fill_diagonal(W, 0.0)
        I = np.eye(3)
        for i, cid in enumerate(ids_used):
            weights = W[i]
            total = float(np.sum(weights))
            if total <= 1e-12:
                continue
            Q = np.zeros((3, 3), dtype=float)
            for j, w in enumerate(weights):
                if w <= 0:
                    continue
                n = axes_arr[j]
                Q += float(w) * (np.outer(n, n) - (1.0 / 3.0) * I)
            Q = (3.0 / 2.0) * Q / total
            eigvals, eigvecs = sorted_eigh(Q)
            cell_data[int(cid)]["physics_cg"][output_key] = {
                "valid": True,
                "tensor": Q,
                "eigvals": eigvals,
                "vec1": normalize_vector(eigvecs[:, 0]),
                "vec3": normalize_vector(eigvecs[:, 2]),
                "total_weight": total,
                "n_neighbors_used": int(np.sum(weights > 0)),
            }

    def _add_nuclear_features_to_row(self, row: Dict[str, Any], nuclei_info: Dict[str, Any], nucagg: Dict[str, Any]) -> None:
        n_total = int(nuclei_info.get("n_nuclei", 0))
        row["n_nuclei"] = n_total
        for label, prefix in self.config.nucleus_feature_prefix.items():
            count = int(nuclei_info.get(f"n_{prefix}", 0))
            row[f"n_{prefix}"] = count
            row[f"frac_{prefix}"] = count / n_total if n_total > 0 else np.nan
            volume = 0.0
            area = 0.0
            for n in nuclei_info.get("nuclei", []):
                if int(n.get("type_label", -1)) == int(label):
                    volume += float(n.get("mesh_volume", 0.0))
                    area += float(n.get("mesh_area", 0.0))
            row[f"{prefix}_nucleus_volume"] = volume
            row[f"{prefix}_nucleus_area"] = area
        row["has_nucleus_pair_vector"] = bool(nuclei_info.get("has_nucleus_pair_vector", False))
        row["nucleus_pair_distance"] = float(nuclei_info.get("nucleus_pair_distance", np.nan)) if nuclei_info.get("nucleus_pair_distance", None) is not None else np.nan
        add_vector_columns(row, "axis_nucleus_pair", nuclei_info.get("nucleus_pair_unit_vector_zyx", np.full(3, np.nan)))

        eigmean = nucagg.get("nucleus_inertia_eigvals_mean", np.full(3, np.nan))
        add_eigval_columns(row, "nucleus_inertia_mean", eigmean)
        field1 = nucagg.get("nucleus_inertia_vec1_field", empty_axis_field_output())
        field3 = nucagg.get("nucleus_inertia_vec3_field", empty_axis_field_output())
        add_eigval_columns(row, "nucleus_inertia_vec1_order", field1.get("eigvals", np.full(3, np.nan)))
        add_eigval_columns(row, "nucleus_inertia_vec3_order", field3.get("eigvals", np.full(3, np.nan)))
        add_vector_columns(row, "axis_nucleus_inertia_vec1", nucagg.get("axis_nucleus_inertia_vec1", np.full(3, np.nan)))
        add_vector_columns(row, "axis_nucleus_inertia_vec3", nucagg.get("axis_nucleus_inertia_vec3", np.full(3, np.nan)))

    def _add_raw_physics_to_row(self, row: Dict[str, Any], data: Dict[str, Any]) -> None:
        raw = data.get("physics_raw", {})
        for key in ["apical", "basal"]:
            tensor = raw.get(key, empty_tensor_output())
            add_eigval_columns(row, key, tensor.get("eigvals", np.full(3, np.nan)))
            add_vector_columns(row, f"axis_{key}_bipolar", tensor.get("bipolar_axis", np.full(3, np.nan)))
            add_vector_columns(row, f"axis_{key}_ring", tensor.get("ring_axis", np.full(3, np.nan)))
        inertia = raw.get("inertia", {})
        add_eigval_columns(row, "inertia", inertia.get("eigvals", np.full(3, np.nan)))
        add_vector_columns(row, "axis_inertia_vec1", inertia.get("vec1", np.full(3, np.nan)))
        add_vector_columns(row, "axis_inertia_vec3", inertia.get("vec3", np.full(3, np.nan)))

    def _add_reference_to_row(self, row: Dict[str, Any], data: Dict[str, Any]) -> None:
        pc = data.get("portal_central_position", self._empty_portal_central_position())
        row["d_cv_nearest_surface"] = float(pc.get("d_cv", np.nan))
        row["d_pv_nearest_surface"] = float(pc.get("d_pv", np.nan))
        row["chi_cv_to_pv"] = float(pc.get("chi_cv_to_pv", np.nan))
        ref = data.get("reference_frame", build_local_reference_frame(np.full(3, np.nan)))
        add_vector_columns(row, "axis_ref_u", ref.get("u", np.full(3, np.nan)))
        add_vector_columns(row, "axis_ref_v", ref.get("v", np.full(3, np.nan)))
        add_vector_columns(row, "axis_ref_w", ref.get("w", np.full(3, np.nan)))

    def _add_cg_to_row(self, row: Dict[str, Any], data: Dict[str, Any]) -> None:
        cg = data.get("physics_cg", {})
        for key, axis_name in [
            ("apical_bipolar", "cg_apical_bipolar"),
            ("apical_ring", "cg_apical_ring"),
            ("basal_bipolar", "cg_basal_bipolar"),
            ("basal_ring", "cg_basal_ring"),
            ("inertia_vec1", "cg_inertia_vec1"),
            ("inertia_vec3", "cg_inertia_vec3"),
            ("nucleus_pair", "cg_nucleus_pair"),
            ("nucleus_inertia_vec1", "cg_nucleus_inertia_vec1"),
            ("nucleus_inertia_vec3", "cg_nucleus_inertia_vec3"),
        ]:
            out = cg.get(key, empty_axis_field_output())
            add_eigval_columns(row, axis_name, out.get("eigvals", np.full(3, np.nan)))
            add_vector_columns(row, f"axis_{axis_name}", out.get("vec3", np.full(3, np.nan)))

    def retained_axis_names(self, columns: Iterable[str]) -> List[str]:
        base = [
            "apical_bipolar",
            "apical_ring",
            "basal_bipolar",
            "basal_ring",
            "inertia_vec1",
            "inertia_vec3",
            "nucleus_inertia_vec1",
            "nucleus_inertia_vec3",
            "cg_apical_bipolar",
            "cg_apical_ring",
            "cg_basal_bipolar",
            "cg_basal_ring",
            "cg_inertia_vec1",
            "cg_inertia_vec3",
            "nucleus_pair",
            "cg_nucleus_pair",
            "cg_nucleus_inertia_vec1",
            "cg_nucleus_inertia_vec3",
            "ref_u",
            "ref_v",
            "ref_w",
        ]
        colset = set(columns)
        return [a for a in base if all(c in colset for c in vector_columns(f"axis_{a}"))]

    def _build_probing_summary(self, cell_data: Dict[int, Dict[str, Any]], valid_ids: Sequence[int]) -> Dict[str, Any]:
        total_faces = 0
        total_void_faces = 0
        total_void_area = 0.0
        total_border_area = 0.0
        for data in cell_data.values():
            labels = np.asarray(data.get("face_labels", []))
            areas = np.asarray(data.get("face_areas", []), dtype=float)
            total_faces += int(len(labels))
            if len(labels) > 0:
                total_void_faces += int(np.sum(labels == 0))
                total_void_area += float(np.sum(areas[labels == 0]))
                total_border_area += float(data.get("area_border", 0.0))
        return {
            "n_cells_processed": int(len(cell_data)),
            "n_valid_cells": int(len(valid_ids)),
            "total_faces": int(total_faces),
            "n_void_faces": int(total_void_faces),
            "void_area_total": float(total_void_area),
            "border_area_total": float(total_border_area),
            "fraction_void_faces": float(total_void_faces / total_faces) if total_faces > 0 else np.nan,
        }

    def _register_base_features(self) -> None:
        r = self.registry
        # Identity / flags
        r.add_scalar("cell_id", level="node", group="identity", description="Unique hepatocyte ID.")
        r.add_boolean("has_more_than_two_nuclei", level="node", group="nuclear", description="True if cell has more than two assigned nuclei.")
        # Cell geometry
        for name in ["cell_volume", "cell_surface_area"]:
            r.add_scalar(name, level="node", group="geometric")
        r.add_vector("cell_centroid", level="node", group="geometric", columns=vector_columns("cell_centroid"))
        r.add_vector("projection_center", level="node", group="geometric", columns=vector_columns("projection_center"))
        # Topology
        for name in ["area_apical", "area_basal", "area_lateral", "area_void", "area_cv", "area_pv", "frac_apical", "frac_basal", "frac_lateral", "frac_void", "frac_cv", "frac_pv"]:
            r.add_scalar(name, level="node", group="topological")
        # Nuclear composition
        r.add_scalar("n_nuclei", level="node", group="nuclear")
        for prefix in self.config.nucleus_feature_prefix.values():
            for n in [f"n_{prefix}", f"frac_{prefix}", f"{prefix}_nucleus_volume", f"{prefix}_nucleus_area"]:
                r.add_scalar(n, level="node", group="nuclear")
        r.add_boolean("has_nucleus_pair_vector", level="node", group="nuclear")
        r.add_scalar("nucleus_pair_distance", level="node", group="nuclear")
        r.add_vector("axis_nucleus_pair", level="node", group="nuclear", columns=vector_columns("axis_nucleus_pair"))
        # Nuclear inertia aggregate
        for p in ["nucleus_inertia_mean", "nucleus_inertia_vec1_order", "nucleus_inertia_vec3_order"]:
            for i in range(1, 4):
                r.add_scalar(f"{p}_eigval_{i}", level="node", group="nuclear")
        r.add_vector("axis_nucleus_inertia_vec1", level="node", group="nuclear", columns=vector_columns("axis_nucleus_inertia_vec1"))
        r.add_vector("axis_nucleus_inertia_vec3", level="node", group="nuclear", columns=vector_columns("axis_nucleus_inertia_vec3"))
        # Raw physics
        for p in ["apical", "basal", "inertia"]:
            for i in range(1, 4):
                r.add_scalar(f"{p}_eigval_{i}", level="node", group="polarity" if p in {"apical", "basal"} else "geometric")
        for vname, group in [
            ("axis_apical_bipolar", "polarity"),
            ("axis_apical_ring", "polarity"),
            ("axis_basal_bipolar", "polarity"),
            ("axis_basal_ring", "polarity"),
            ("axis_inertia_vec1", "geometric"),
            ("axis_inertia_vec3", "geometric"),
        ]:
            r.add_vector(vname, level="node", group=group, columns=vector_columns(vname))
        # Reference
        for name in ["d_cv_nearest_surface", "d_pv_nearest_surface", "chi_cv_to_pv"]:
            r.add_scalar(name, level="node", group="reference")
        for vname in ["axis_ref_u", "axis_ref_v", "axis_ref_w"]:
            r.add_vector(vname, level="node", group="reference", columns=vector_columns(vname))
        # CG
        for p in ["cg_apical_bipolar", "cg_apical_ring", "cg_basal_bipolar", "cg_basal_ring", "cg_inertia_vec1", "cg_inertia_vec3", "cg_nucleus_pair", "cg_nucleus_inertia_vec1", "cg_nucleus_inertia_vec3"]:
            for i in range(1, 4):
                r.add_scalar(f"{p}_eigval_{i}", level="node", group="coarse_grained")
            r.add_vector(f"axis_{p}", level="node", group="coarse_grained", columns=vector_columns(f"axis_{p}"))


__all__ = [
    "GraphGeneratorConfig",
    "GraphGeneratorResult",
    "FeatureRegistry",
    "FeatureSpec",
    "LiverGraphGenerator",
    "get_tiff_metadata",
    "read_crop_zyx",
    "sanitize_crop_bounds",
    "compute_surface_nematic_tensor",
    "compute_inertia_axes",
    "normalize_vector",
    "cos_alignment",
]
