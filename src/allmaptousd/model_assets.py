"""3D equipment models (USD / STEP / STL) that are placed on the map and merged into the exported USD.

An imported model becomes a :class:`ModelAsset`: a self-contained USD layer whose root prim
``/Model`` is in metres, Z-up, centred on its top-view outline and standing on z = 0, so placing it
only takes (x, y, yaw). Exports copy its prims into the stage (no references to the source file), and
its top-view outline is an obstacle for the 2D collision check. STEP files are converted to a USD
stage first (:mod:`allmaptousd.step_import`), then go the same way as USD files.

A USD model is uploaded as one file, a folder or a .zip. Every file its parts use (textures, MDL
materials, ...) is kept in ``ModelAsset.files`` and its asset paths are rewritten to
``./allmaptousd_assets/<model id>/<path in the upload>``; exports write those files next to the USD.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from allmaptousd.model_preview import PreviewMesh, build_preview, simplify_mesh, world_triangles
from allmaptousd.step_import import STEP_EXTENSIONS, step_to_stage, unique_name

USD_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")
MODEL_EXTENSIONS = (*USD_EXTENSIONS, *STEP_EXTENSIONS, ".stl")
ASSET_DIR = "allmaptousd_assets"  # exports put a model's files in <USD folder>/allmaptousd_assets/<model id>/
MAX_UPLOAD_FILES = 20_000
MAX_UNZIPPED_BYTES = 4 << 30
UNIT_SCALES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
MAX_TREE_NODES = 4000
_FOOTPRINT_TRIANGLES = 400_000  # rasterized for the outline
_FOOTPRINT_PIXELS = 512         # outline raster size along the longer side
_SHADING_TYPES = {"Material", "Shader", "NodeGraph", "GeomSubset"}
_MODEL = Sdf.Path("/Model")


# ---------------------------------------------------------------------------
# Uploaded source: the file the user is choosing parts from
# ---------------------------------------------------------------------------

@dataclass
class ModelSource:
    """Uploaded files (one model file, a folder or a zip), kept until the user picks what to import.

    ``filename`` is the model file in use (its path inside the upload); ``candidates`` lists every
    model file of the upload, the likeliest main file first."""

    root: Path                                                  # the uploaded files on disk
    candidates: list[str]
    files_digest: str                                           # of every uploaded file
    filename: str = ""
    stage: Usd.Stage | None = None                              # USD and STEP
    mesh: tuple[np.ndarray, np.ndarray] | None = None           # STL: vertices, triangles
    missing: list[str] = field(default_factory=list)            # files the model uses that are not uploaded
    _tempdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return Path(self.filename).stem

    @property
    def kind(self) -> str:
        """"usd", "step" or "stl"."""
        ext = Path(self.filename).suffix.lower()
        return "stl" if ext == ".stl" else "step" if ext in STEP_EXTENSIONS else "usd"

    @property
    def digest(self) -> str:
        """Identifies the upload and the model file in use (for model ids)."""
        if len(self.candidates) == 1:
            return self.files_digest
        return hashlib.sha1(f"{self.files_digest}|{self.filename}".encode()).hexdigest()

    def describe(self) -> dict:
        """The part tree and the file's own units, for the import dialog."""
        upload = {"main": self.filename, "candidates": self.candidates, "missing": self.missing}
        if self.stage is None:
            return {
                "name": self.name, "format": "stl", "truncated": False,
                "tree": [{"path": f"/{_prim_name(self.name)}", "name": Path(self.filename).name, "type": "Mesh",
                          "children": []}],
                "units": "mm", "units_note": "STL files have no units; most CAD tools write millimetres",
                "up_axis": "z", **upload,
            }
        tree, truncated = _usd_tree(self.stage)
        authored = self.stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit)
        unit = _unit_name(UsdGeom.GetStageMetersPerUnit(self.stage))
        note = "from the file" if authored else "not set in the file (USD default)"
        if self.kind == "step":
            note = "STEP lengths are read in millimetres whatever unit the file uses"
        return {
            "name": self.name, "format": self.kind, "tree": tree, "truncated": truncated,
            "units": unit, "units_note": note,
            "up_axis": UsdGeom.GetStageUpAxis(self.stage).lower(), **upload,
        }

    def select(self, main: str) -> None:
        """Open ``main`` (one of ``candidates``) as the model file."""
        if main not in self.candidates:
            raise ValueError(f"'{main}' is not a model file of this upload.")
        path = self.root / main
        self.stage, self.mesh, self.missing, self.filename = None, None, [], main
        if self.kind == "stl":
            self.mesh = _read_stl(path.read_bytes())
        elif self.kind == "step":
            self.stage = step_to_stage(str(path))
        else:
            try:
                stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
            except Exception as exc:  # pxr raises Tf.ErrorException
                raise ValueError(f"'{Path(main).name}' could not be opened as USD: {exc}") from exc
            if stage is None:
                raise ValueError(f"'{Path(main).name}' could not be opened as USD.")
            self.stage = stage
            self.missing = _unresolved_files(path, self.root)

    def close(self) -> None:
        self.stage = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None


def open_model_source(filename: str, data: bytes) -> ModelSource:
    """One uploaded file: a USD, STEP or STL model, or a .zip holding a model and the files it uses."""
    name = Path(filename.replace("\\", "/")).name
    if Path(name).suffix.lower() == ".zip":
        return open_model_files(_unzip(name, data))
    if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model file '{name}'. Use {', '.join(MODEL_EXTENSIONS)} or .zip.")
    return open_model_files({name: data})


def open_model_files(files: dict[str, bytes], main: str | None = None) -> ModelSource:
    """Uploaded files by their path in the upload (e.g. a folder: ``Crate/crate.usd``,
    ``Crate/textures/wood.png``). ``main`` picks the model file; by default the likeliest one."""
    clean: dict[str, bytes] = {}
    for raw, data in files.items():
        clean[_clean_relpath(raw)] = data
    if len(clean) > MAX_UPLOAD_FILES:
        raise ValueError(f"The upload has more than {MAX_UPLOAD_FILES} files.")
    candidates = [n for n in clean if Path(n).suffix.lower() in MODEL_EXTENSIONS]
    if not candidates:
        raise ValueError(f"No model file was found. Include a {', '.join(MODEL_EXTENSIONS)} file.")
    if len(clean) == 1:
        files_digest = hashlib.sha1(next(iter(clean.values()))).hexdigest()
    else:
        h = hashlib.sha1()
        for n in sorted(clean):
            h.update(n.encode("utf-8") + b"\0" + hashlib.sha1(clean[n]).digest())
        files_digest = h.hexdigest()

    tempdir = tempfile.TemporaryDirectory(prefix="allmaptousd_model_")
    root = Path(tempdir.name)
    try:
        write_tree(root, clean)
        source = ModelSource(root, _rank_candidates(root, candidates), files_digest, _tempdir=tempdir)
        source.select(main or source.candidates[0])
    except Exception:
        tempdir.cleanup()
        raise
    return source


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    """Write ``files`` (paths relative to ``root``) below ``root``; a path leading outside it is refused."""
    base = root.resolve()
    for rel, data in files.items():
        path = (base / rel).resolve()
        if base not in path.parents:
            raise ValueError(f"Refusing to write outside {base}: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _clean_relpath(name: str) -> str:
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        raise ValueError(f"'{name}' is not a valid file name in the upload.")
    return "/".join(parts)


def _unzip(name: str, data: bytes) -> dict[str, bytes]:
    invalid = (zipfile.BadZipFile, EOFError, OSError, NotImplementedError, ValueError)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except invalid as exc:
        raise ValueError(f"'{name}' is not a valid zip file.") from exc
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")]
        if len(infos) > MAX_UPLOAD_FILES:
            raise ValueError(f"'{name}' holds more than {MAX_UPLOAD_FILES} files.")
        if sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
            raise ValueError(f"'{name}' unpacks to more than 4 GiB.")
        try:
            return {i.filename: archive.read(i) for i in infos}
        except invalid as exc:
            raise ValueError(f"'{name}' is not a valid zip file.") from exc


def _rank_candidates(root: Path, candidates: list[str]) -> list[str]:
    """USD files no other uploaded USD file uses first, then STEP, then STL, shallowest first."""
    referenced: set[str] = set()
    base = root.resolve()
    for rel in candidates:
        if Path(rel).suffix.lower() not in USD_EXTENSIONS:
            continue
        try:
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(root / rel))
        except Exception:  # pxr raises Tf.ErrorException for files it cannot read
            continue
        for ref in (*sublayers, *references, *payloads):
            if not ref or "://" in ref:
                continue
            outer = Ar.SplitPackageRelativePathOuter(ref)[0] if Ar.IsPackageRelativePath(ref) else ref
            target = Path(outer) if Path(outer).is_absolute() else (root / rel).parent / outer
            try:
                referenced.add(target.resolve().relative_to(base).as_posix())
            except (ValueError, OSError):
                pass

    def rank(rel: str) -> tuple:
        ext = Path(rel).suffix.lower()
        kind = 0 if ext in USD_EXTENSIONS else 1 if ext in STEP_EXTENSIONS else 2
        return rel in referenced, kind, rel.count("/"), rel.lower()

    return sorted(candidates, key=rank)


def _is_library_path(path: str) -> bool:
    """A bare name such as ``OmniPBR.mdl``, found on the renderer's search paths, or a URL."""
    return "://" in path or ("/" not in path and "\\" not in path)


def _upload_label(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path


def _unresolved_files(path: Path, root: Path) -> list[str]:
    """Files the USD file uses (other USD files, textures, materials) that were not uploaded."""
    try:
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception:  # pxr raises Tf.ErrorException
        return []
    return sorted({_upload_label(p, root) for p in unresolved if p and not _is_library_path(p)})


def _usd_tree(stage: Usd.Stage) -> tuple[list[dict], bool]:
    """Prims that hold visible geometry somewhere below them (materials, lights and cameras left out)."""
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    count = 0
    truncated = False

    def visit(prim: Usd.Prim) -> dict | None:
        nonlocal count, truncated
        if prim.GetTypeName() in _SHADING_TYPES:
            return None
        children = []
        for child in prim.GetFilteredChildren(predicate):
            if count >= MAX_TREE_NODES:
                truncated = True
                break
            node = visit(child)
            if node is not None:
                children.append(node)
        if not children and not prim.IsA(UsdGeom.Gprim):
            return None
        count += 1
        return {"path": str(prim.GetPath()), "name": prim.GetName(), "type": prim.GetTypeName(), "children": children}

    roots = [visit(prim) for prim in stage.GetPseudoRoot().GetFilteredChildren(predicate)]
    return [r for r in roots if r is not None], truncated


def _read_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (N, 3), triangles (M, 3)) from a binary or ASCII STL file."""
    tri: np.ndarray | None = None
    if len(data) >= 84:
        count = int.from_bytes(data[80:84], "little")
        if 84 + 50 * count == len(data):
            record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            tri = np.frombuffer(data, dtype=record, count=count, offset=84)["v"].reshape(-1, 3)
    if tri is None:
        text = data.decode("ascii", errors="ignore")
        values = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
        try:
            tri = np.array(values, dtype=np.float64).reshape(-1, 3)
        except ValueError as exc:
            raise ValueError("This STL file is damaged: a vertex is not a number.") from exc
        tri = tri[: len(tri) // 3 * 3]
    if not len(tri) or not np.isfinite(tri).all():
        raise ValueError("This STL file has no triangles.")
    vertices, inverse = np.unique(tri.astype(np.float64), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3).astype(np.int64)


def _unit_name(meters_per_unit: float) -> str:
    return min(UNIT_SCALES, key=lambda k: abs(math.log(UNIT_SCALES[k] / max(meters_per_unit, 1e-12))))


def _prim_name(name: str, fallback: str = "Part") -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = f"{fallback}_{clean}"
    return clean


# ---------------------------------------------------------------------------
# Imported asset
# ---------------------------------------------------------------------------

@dataclass
class ModelAsset:
    """A normalized model: ``layer`` holds ``/Model`` (metres, Z-up, centred, base on z = 0)."""

    id: str
    name: str
    layer: Sdf.Layer
    size: tuple[float, float, float]
    footprint: list[np.ndarray]                 # top-view outlines (K, 2), model frame
    preview_mesh: PreviewMesh                   # coloured and simplified for the browser
    triangle_count: int
    files: dict[str, bytes] = field(default_factory=dict)  # path in the upload -> file the model uses
    missing: list[str] = field(default_factory=list)       # files it uses that were not uploaded

    def summary(self) -> dict:
        dx, dy, dz = self.size
        warning = None
        if max(dx, dy, dz) > 200.0 or max(dx, dy, dz) < 0.02:
            warning = f"This model is {dx:.3g} × {dy:.3g} × {dz:.3g} m. Check the units."
        return {
            "id": self.id,
            "name": self.name,
            "size": [round(v, 4) for v in self.size],
            "footprint": [np.round(poly, 4).tolist() for poly in self.footprint],
            "triangles": self.triangle_count,
            "files": len(self.files),
            "missing_files": self.missing,
            "warning": warning,
        }

    def preview(self) -> dict:
        def pack(array: np.ndarray, dtype: str) -> str:
            return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")

        mesh = self.preview_mesh
        return {
            "positions": pack(mesh.positions, "<f4"),
            "indices": pack(mesh.indices, "<u4"),
            "colors": pack(mesh.colors, "u1"),
            "uvs": pack(mesh.uvs, "<f4"),
            "groups": [list(g) for g in mesh.groups],
            "textures": len(mesh.textures),
        }

    def package_files(self) -> dict[str, bytes]:
        """The model's files by their path next to a USD file that uses it."""
        return {f"{ASSET_DIR}/{self.id}/{rel}": data for rel, data in self.files.items()}

    def to_usdc(self) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.usdc"
            self.layer.Export(str(path))
            return path.read_bytes()

    @classmethod
    def from_usdc(cls, model_id: str, name: str, data: bytes, files: dict[str, bytes] | None = None) -> ModelAsset:
        """Reload a model saved with :meth:`to_usdc` and its ``files`` (e.g. from a saved map)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.usdc"
            path.write_bytes(data)
            opened = Sdf.Layer.OpenAsAnonymous(str(path))
            if opened is None or not opened.GetPrimAtPath(_MODEL):
                raise ValueError(f"The saved model '{name}' is damaged.")
            layer = Sdf.Layer.CreateAnonymous("model.usda")
            layer.TransferContent(opened)
            del opened
        return _analyze(model_id, name, layer, recentre=False, files=dict(files or {}))


def import_model(source: ModelSource, paths: list[str], unit: str = "auto", up_axis: str = "auto") -> ModelAsset:
    """Normalize the chosen parts of ``source`` (all of it when ``paths`` is empty) into a ModelAsset."""
    if unit != "auto" and unit not in UNIT_SCALES:
        raise ValueError(f"Unknown unit '{unit}'.")
    if up_axis not in ("auto", "z", "y"):
        raise ValueError(f"Unknown up axis '{up_axis}'.")
    key = "|".join([source.digest, *sorted(paths), unit, up_axis])
    model_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    layer = Sdf.Layer.CreateAnonymous("model.usda")
    model = Sdf.CreatePrimInLayer(layer, _MODEL)
    model.specifier, model.typeName = Sdf.SpecifierDef, "Xform"
    files: dict[str, bytes] = {}
    missing: list[str] = []
    if source.mesh is not None:
        meters_per_unit = UNIT_SCALES["mm" if unit == "auto" else unit]
        up = "z" if up_axis == "auto" else up_axis
        _write_stl_mesh(layer, source)
        name = source.name
    else:
        stage = source.stage
        meters_per_unit = UNIT_SCALES[unit] if unit != "auto" else UsdGeom.GetStageMetersPerUnit(stage)
        up = UsdGeom.GetStageUpAxis(stage).lower() if up_axis == "auto" else up_axis
        chosen = _copy_usd_parts(layer, stage, paths)
        name = source.name if not paths else (chosen[0] if len(chosen) == 1 else f"{source.name}_parts")
        if source.kind == "usd":
            files, missing = _bundle_files(layer, source.root, asset_prefix(model_id))

    layer.pseudoRoot.SetInfo(UsdGeom.Tokens.upAxis, UsdGeom.Tokens.z)
    layer.pseudoRoot.SetInfo(UsdGeom.Tokens.metersPerUnit, 1.0)
    layer.defaultPrim = _MODEL.name
    to_z_up = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)) if up == "y" else Gf.Matrix4d(1.0)
    _set_model_matrix(layer, Gf.Matrix4d().SetScale(Gf.Vec3d(meters_per_unit)) * to_z_up)
    _add_collision(layer)
    asset = _analyze(model_id, name, layer, recentre=True, files=files)
    asset.missing = missing
    return asset


def _write_stl_mesh(layer: Sdf.Layer, source: ModelSource) -> None:
    vertices, triangles = source.mesh
    stage = Usd.Stage.Open(layer)
    mesh = UsdGeom.Mesh.Define(stage, _MODEL.AppendChild(_prim_name(source.name)))
    mesh.CreatePointsAttr(vertices.astype(np.float32))
    mesh.CreateFaceVertexCountsAttr(np.full(len(triangles), 3, np.int32))
    mesh.CreateFaceVertexIndicesAttr(triangles.astype(np.int32).ravel())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.66, 0.72)])
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(vertices.astype(np.float32)))


def _copy_usd_parts(layer: Sdf.Layer, stage: Usd.Stage, paths: list[str]) -> list[str]:
    """Copy the chosen prims (flattened, instancing expanded) under /Model and return their names."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):  # plain copies instead of instance prototypes
        while instances := [p for p in stage.Traverse(Usd.TraverseInstanceProxies()) if p.IsInstance()]:
            for prim in instances:
                prim.SetInstanceable(False)
    flat = stage.Flatten(False)

    if paths:
        chosen = [Sdf.Path(p) for p in paths]
        for path in chosen:
            if not path.IsAbsolutePath() or not path.IsPrimPath() or not stage.GetPrimAtPath(path):
                raise ValueError(f"'{path}' is not a part of this model.")
        chosen = [p for p in chosen if not any(p != q and p.HasPrefix(q) for q in chosen)]
    else:
        tree, _ = _usd_tree(stage)
        chosen = [Sdf.Path(node["path"]) for node in tree]
    if not chosen:
        raise ValueError("This model has no visible geometry.")

    used: set[str] = set()
    mapping: dict[Sdf.Path, Sdf.Path] = {}
    cache = UsdGeom.XformCache()
    for path in chosen:
        name = unique_name(_prim_name(path.name), used)
        parent_to_world = cache.GetParentToWorldTransform(stage.GetPrimAtPath(path))
        if Gf.IsClose(parent_to_world, Gf.Matrix4d(1.0), 1e-12):
            dest = _MODEL.AppendChild(name)
        else:  # keep where the part sits in the whole model
            frame = Sdf.CreatePrimInLayer(layer, _MODEL.AppendChild(name))
            frame.specifier, frame.typeName = Sdf.SpecifierDef, "Xform"
            _author_matrix(layer, frame.path, parent_to_world)
            dest = frame.path.AppendChild(path.name)
        if not Sdf.CopySpec(flat, path, layer, dest):
            raise ValueError(f"'{path}' could not be copied.")
        mapping[path] = dest
    _remap_outside_targets(layer, flat, mapping, used)
    return [p.name for p in chosen]


def _remap_outside_targets(
    layer: Sdf.Layer, flat: Sdf.Layer, mapping: dict[Sdf.Path, Sdf.Path], used: set[str],
) -> None:
    """Point relationships and connections at the copies. Materials (or other prims) the copied
    parts use from elsewhere in the file are copied too, under /Model/Looks."""
    looks = _MODEL.AppendChild("Looks")

    def remap(target: Sdf.Path) -> Sdf.Path | None:
        if target.HasPrefix(_MODEL):
            return target
        for src in target.GetPrimPath().GetPrefixes()[::-1]:  # the deepest copied prim first
            if src in mapping:
                return target.ReplacePrefix(src, mapping[src])
        prim_path = target.GetPrimPath()
        if not flat.GetPrimAtPath(prim_path):
            return None  # dangling in the source as well: drop it
        source = prim_path
        while source.pathElementCount > 1 and flat.GetPrimAtPath(source).typeName != "Material":
            source = source.GetParentPath()
        if flat.GetPrimAtPath(source).typeName != "Material":
            source = prim_path
        if not layer.GetPrimAtPath(looks):
            scope = Sdf.CreatePrimInLayer(layer, looks)
            scope.specifier, scope.typeName = Sdf.SpecifierDef, "Scope"
        dest = looks.AppendChild(unique_name(_prim_name(source.name), used))
        Sdf.CopySpec(flat, source, layer, dest)
        mapping[source] = dest
        return target.ReplacePrefix(source, dest)

    def properties() -> list[Sdf.Path]:
        found: list[Sdf.Path] = []
        layer.Traverse(_MODEL, lambda p: found.append(p) if p.IsPropertyPath() else None)
        return found

    changed = True
    while changed:  # a copied material can itself point further out
        changed = False
        for prop_path in properties():
            spec = layer.GetPropertyAtPath(prop_path)
            if isinstance(spec, Sdf.RelationshipSpec):
                listop = spec.targetPathList
            elif isinstance(spec, Sdf.AttributeSpec):
                listop = spec.connectionPathList
            else:
                continue
            for kind in ("explicitItems", "prependedItems", "appendedItems", "addedItems", "deletedItems"):
                items = list(getattr(listop, kind))
                if not items or all(p.HasPrefix(_MODEL) for p in items):
                    continue
                count = len(mapping)
                setattr(listop, kind, [q for q in (remap(p) for p in items) if q is not None])
                changed |= len(mapping) != count


def _author_matrix(layer: Sdf.Layer, path: Sdf.Path, matrix: Gf.Matrix4d) -> None:
    stage = Usd.Stage.Open(layer)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xformable.ClearXformOpOrder()
    xformable.MakeMatrixXform().Set(matrix)


def _set_model_matrix(layer: Sdf.Layer, matrix: Gf.Matrix4d) -> None:
    _author_matrix(layer, _MODEL, matrix)


def _add_collision(layer: Sdf.Layer) -> None:
    """Static colliders on every mesh and shape, unless the model brings its own physics setup."""
    stage = Usd.Stage.Open(layer)
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(_MODEL)))
    if any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims):
        return
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
        elif any(prim.IsA(t) for t in (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Capsule)):
            UsdPhysics.CollisionAPI.Apply(prim)


# ---------------------------------------------------------------------------
# Geometry: size, outline and preview
# ---------------------------------------------------------------------------

def asset_prefix(model_id: str) -> str:
    """How a model's layer names its files: relative to the USD file that uses the model."""
    return f"./{ASSET_DIR}/{model_id}/"


def _analyze(model_id: str, name: str, layer: Sdf.Layer, *, recentre: bool, files: dict[str, bytes]) -> ModelAsset:
    """Size, top-view outline and preview of the visible geometry under /Model."""
    stage = Usd.Stage.Open(layer)  # kept open while the parts' prims are in use
    parts = world_triangles(stage, _MODEL)
    if not parts:
        raise ValueError("The selected parts have no visible geometry.")
    offsets = np.cumsum([0] + [len(p.points) for p in parts[:-1]])
    vertices = np.concatenate([p.points for p in parts])
    triangles = np.concatenate([p.triangles + o for p, o in zip(parts, offsets, strict=True)])
    prefix = asset_prefix(model_id)
    preview = build_preview(parts, lambda p: files.get(p[len(prefix):]) if p.startswith(prefix) else None)

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if recentre:  # centre on the top view, base on the floor
        shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(_MODEL)).GetLocalTransformation()
        _set_model_matrix(layer, matrix * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*shift.tolist())))
        vertices = vertices + shift
        lo, hi = lo + shift, hi + shift
        preview.positions = preview.positions + shift.astype(np.float32)
    size = tuple(float(v) for v in hi - lo)
    if len(triangles) > _FOOTPRINT_TRIANGLES:
        outline_v, outline_t, _ = simplify_mesh(vertices, triangles, _FOOTPRINT_TRIANGLES)
    else:
        outline_v, outline_t = vertices, triangles
    return ModelAsset(
        id=model_id, name=name, layer=layer, size=size,
        footprint=_footprint(outline_v, outline_t, lo, hi),
        preview_mesh=preview, triangle_count=len(triangles), files=files,
    )


def _footprint(vertices: np.ndarray, triangles: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    """Top-view outlines of the model (every part of it, at any height), slightly on the safe side."""
    px = max(float(max(hi[0] - lo[0], hi[1] - lo[1])) / _FOOTPRINT_PIXELS, 0.005)
    origin = lo[:2] - 2 * px
    cols, rows = (np.ceil((hi[:2] - lo[:2]) / px).astype(int) + 5).tolist()
    image = np.zeros((rows, cols), np.uint8)
    pixels = np.rint(((vertices[:, :2] - origin) / px - 0.5) * 16).astype(np.int32)
    for tri in pixels[triangles]:  # one at a time: a polygon list would cancel overlaps out
        cv2.fillConvexPoly(image, tri, 1, cv2.LINE_8, 4)
    image = cv2.dilate(image, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in contours:
        simple = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(simple) >= 3:
            outlines.append((simple + 0.5) * px + origin)
    return outlines


# ---------------------------------------------------------------------------
# Files the model uses
# ---------------------------------------------------------------------------

def _bundle_files(layer: Sdf.Layer, root: Path, prefix: str) -> tuple[dict[str, bytes], list[str]]:
    """Read every uploaded file that an asset path under /Model points at (after flattening they are
    absolute paths into ``root``) and point the path at ``prefix + <path in the upload>``. MDL modules
    bring the files they name along. Returns (files, paths that were not uploaded)."""
    base = root.resolve()
    files: dict[str, bytes] = {}
    missing: set[str] = set()
    rewritten: dict[str, str] = {}

    def read_file(path: Path) -> str | None:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(base).as_posix()
        except (ValueError, OSError):
            return None
        if rel not in files:
            if not resolved.is_file():
                return None
            files[rel] = resolved.read_bytes()
            if resolved.suffix.lower() == ".mdl":
                for dep in _mdl_dependencies(resolved):
                    read_file(dep)
        return rel

    def read_packaged(package: str, inner: str) -> str | None:
        try:
            outer = Path(package).resolve().relative_to(base).as_posix()
            with zipfile.ZipFile(base / outer) as archive:
                data = archive.read(inner.replace("\\", "/"))
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            return None
        if Ar.IsPackageRelativePath(inner):
            return None
        rel = f"{PurePosixPath(outer).with_suffix('')}/{_clean_relpath(inner)}"
        files[rel] = data
        return rel

    def read_udim(path: Path) -> str | None:
        pattern = re.compile(re.escape(path.name).replace(re.escape("<UDIM>"), r"\d{4}") + "$")
        tiles = [p for p in path.parent.glob("*") if pattern.match(p.name)] if path.parent.is_dir() else []
        rels = [read_file(p) for p in tiles]
        if not any(rels):
            return None
        return (PurePosixPath(next(r for r in rels if r)).parent / path.name).as_posix()

    def localize(authored: str) -> str:
        if authored in rewritten:
            return rewritten[authored]
        new = authored
        if authored and "://" not in authored:
            if Ar.IsPackageRelativePath(authored):
                rel = read_packaged(*Ar.SplitPackageRelativePathOuter(authored))
            else:
                path = Path(authored) if Path(authored).is_absolute() else root / authored
                rel = read_udim(path) if "<UDIM>" in authored else read_file(path)
            if rel is not None:
                new = prefix + rel
            elif not _is_library_path(authored):  # where the file would go, so it can be added later
                label = _upload_label(authored, root)
                missing.add(label)
                if not Path(label).is_absolute() and not Ar.IsPackageRelativePath(label):
                    new = prefix + label
        rewritten[authored] = new
        return new

    UsdUtils.ModifyAssetPaths(layer, localize)
    return files, sorted(missing)


_MDL_STRING = re.compile(r'"([^"\n]{1,512})"')
_MDL_IMPORT = re.compile(r"\b(?:import|using)\s+((?:\.\.?::)+[\w:]+)")


def _mdl_dependencies(path: Path) -> list[Path]:
    """Files an MDL module names relative to itself: texture strings and relative module imports."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    found = []
    for value in _MDL_STRING.findall(text):
        if "://" not in value and not value.startswith("/") and (path.parent / value).is_file():
            found.append(path.parent / value)
    for module in _MDL_IMPORT.findall(text):
        parts = [p for p in module.split("::") if p and p != "*"]
        for end in range(len(parts), 0, -1):  # the module is a prefix: "import .::a::b::name;"
            candidate = path.parent.joinpath(*parts[:end]).with_suffix(".mdl")
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def add_model_copy(stage: Usd.Stage, path: str, asset: ModelAsset) -> None:
    """Copy the model's prims to ``path`` in the stage's root layer (its parent prim must exist)."""
    if not Sdf.CopySpec(asset.layer, _MODEL, stage.GetRootLayer(), Sdf.Path(path)):
        raise ValueError(f"The model '{asset.name}' could not be added to the USD file.")


__all__ = [
    "ASSET_DIR", "MODEL_EXTENSIONS", "ModelAsset", "ModelSource", "UNIT_SCALES",
    "add_model_copy", "asset_prefix", "import_model", "open_model_files", "open_model_source", "write_tree",
]
