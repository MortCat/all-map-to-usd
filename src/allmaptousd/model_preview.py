"""Coloured, textured preview meshes of imported models for the browser's 3D thumbnails and top views.

Each visible Gprim, or each of its faces bound to another material through a GeomSubset, takes the
colour of its material: UsdPreviewSurface ``diffuseColor``, the base colour inputs of the MDL materials
Isaac Sim uses (OmniPBR, OmniSurface), or else ``primvars:displayColor``. A diffuse texture that came
with the model is sent as an image along with the UVs. A model above the triangle budget is simplified
and its textures are baked into vertex colours.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

PREVIEW_TRIANGLES = 60_000
TEXTURE_PIXELS = 512  # longer side of a preview texture
DEFAULT_COLOR = (0.62, 0.66, 0.72)  # linear, like an STL import
# Shader inputs holding the base colour (a value or a texture node), then MDL texture inputs.
_COLOR_INPUTS = (
    "diffuseColor", "diffuse_color_constant", "diffuse_reflection_color", "base_color", "baseColor",
    "albedo", "diffuse_color", "color",
)
_TEXTURE_INPUTS = (
    "diffuse_texture", "diffuse_reflection_color_image", "base_color_texture", "basecolor_texture",
    "albedo_texture", "BaseColor_Tex", "diffuse_color_texture",
)
_TINT_INPUTS = ("diffuse_tint", "base_color_tint")
_MDL_DEFAULTS = {"OmniPBR": (0.2, 0.2, 0.2), "OmniSurface": (0.8, 0.8, 0.8), "OmniGlass": (0.9, 0.95, 1.0)}


@dataclass
class Look:
    color: tuple[float, float, float] | None = None  # linear RGB; None: use displayColor
    texture: str | None = None                       # asset path of the diffuse texture
    uv_name: str = "st"


@dataclass
class PreviewMesh:
    positions: np.ndarray                  # (N, 3) float32
    colors: np.ndarray                     # (N, 3) uint8, sRGB
    uvs: np.ndarray                        # (N, 2) float32
    indices: np.ndarray                    # (M, 3) uint32, ordered by group
    groups: list[tuple[int, int, int]]     # (first triangle, triangle count, texture index or -1)
    textures: list[bytes] = field(default_factory=list)  # PNG images

    @property
    def triangle_count(self) -> int:
        return len(self.indices)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.clip(linear, 0.0, 1.0) ** (1 / 2.2)


def _value(attr: Usd.Attribute):
    value = attr.Get()
    return attr.Get(Usd.TimeCode.EarliestTime()) if value is None else value


def _rgb(value) -> tuple[float, float, float] | None:
    try:
        rgb = tuple(float(v) for v in value)
    except TypeError:
        return None
    return rgb[:3] if len(rgb) >= 3 and np.isfinite(rgb[:3]).all() else None


def _texture_node(shader: UsdShade.Shader) -> tuple[str | None, str]:
    """(texture file, UV primvar name) of a texture node such as UsdUVTexture."""
    file_input = shader.GetInput("file") or shader.GetInput("texture")
    value = file_input.Get() if file_input else None
    path = value.path if isinstance(value, Sdf.AssetPath) and value.path else None
    uv_name = "st"
    st = shader.GetInput("st")
    for attr in (st.GetValueProducingAttributes() if st else []):
        if UsdShade.Output.IsOutput(attr):
            reader = UsdShade.Shader(attr.GetPrim()).GetInput("varname")
            name = reader.Get() if reader else None
            if name:
                uv_name = str(name)
    return path, uv_name


def material_look(material: UsdShade.Material) -> Look | None:
    """What a material looks like in the preview, or None when it names no colour."""
    if not material:
        return None
    for context in (UsdShade.Tokens.universalRenderContext, "mdl"):
        shader = material.ComputeSurfaceSource(context)[0]
        if not shader:
            continue
        look = Look()
        for name in _COLOR_INPUTS:
            shader_input = shader.GetInput(name)
            if not shader_input:
                continue
            for attr in shader_input.GetValueProducingAttributes():
                if UsdShade.Output.IsOutput(attr):
                    look.texture, look.uv_name = _texture_node(UsdShade.Shader(attr.GetPrim()))
                elif (rgb := _rgb(attr.Get())) is not None:
                    look.color = rgb
            if look.color is not None or look.texture:
                break
        for name in _TEXTURE_INPUTS:
            shader_input = shader.GetInput(name)
            value = shader_input.Get() if shader_input else None
            if isinstance(value, Sdf.AssetPath) and value.path:
                look.texture = value.path
                break
        if look.color is None:
            sub_id = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
            look.color = _MDL_DEFAULTS.get(str(sub_id or ""))
        if look.texture:  # the texture replaces the colour; a tint multiplies it
            tint = next((shader.GetInput(n).Get() for n in _TINT_INPUTS if shader.GetInput(n)), None)
            look.color = _rgb(tint) if tint is not None else (1.0, 1.0, 1.0)
        if look.color is not None or look.texture:
            return look
    return None


class _Textures:
    """Decoded diffuse textures (RGB uint8, at most TEXTURE_PIXELS), indexed in first-use order."""

    def __init__(self, read_file: Callable[[str], bytes | None]) -> None:
        self.read_file = read_file
        self.index: dict[str, int] = {}
        self.images: list[np.ndarray] = []

    def get(self, path: str | None) -> int:
        if not path:
            return -1
        if path not in self.index:
            image = _decode(self.read_file(path))
            self.index[path] = len(self.images) if image is not None else -1
            if image is not None:
                self.images.append(image)
        return self.index[path]


def _decode(data: bytes | None) -> np.ndarray | None:
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    scale = TEXTURE_PIXELS / max(image.shape[:2])
    if scale < 1:  # smaller first: the conversions below then touch fewer pixels
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    elif image.dtype != np.uint8:  # float (e.g. EXR): linear values
        image = (_to_srgb(np.nan_to_num(image.astype(np.float32))) * 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return image


def _per_corner(values: np.ndarray, interpolation: str, faces: np.ndarray, points: np.ndarray,
                corners: np.ndarray) -> np.ndarray | None:
    """Primvar values for each triangle corner (T, 3, k), or None when they do not fit the mesh."""
    try:
        if interpolation == UsdGeom.Tokens.constant:
            return np.broadcast_to(values[0], (*corners.shape, values.shape[1]))
        if interpolation == UsdGeom.Tokens.uniform:
            return np.repeat(values[faces][:, None, :], 3, axis=1)
        if interpolation == UsdGeom.Tokens.faceVarying:
            return values[corners]
        return values[points]  # vertex / varying
    except IndexError:
        return None


def _primvar(prim: Usd.Prim, name: str) -> tuple[np.ndarray, str] | None:
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not primvar or not primvar.HasValue():
        return None
    values = primvar.ComputeFlattened()
    if values is None:
        values = primvar.ComputeFlattened(Usd.TimeCode.EarliestTime())
    if values is None or not len(values):
        return None
    return np.asarray(values, np.float64).reshape(len(values), -1), primvar.GetInterpolation()


@dataclass
class GprimTriangles:
    """One visible Gprim as world-space triangles (shapes other than meshes: their bounding boxes)."""

    prim: Usd.Prim
    points: np.ndarray     # (P, 3) world positions
    triangles: np.ndarray  # (T, 3) indices into ``points``
    corners: np.ndarray    # (T, 3) face-vertex corner of each triangle point (for faceVarying primvars)
    faces: np.ndarray      # (T,) face of each triangle
    face_count: int


def world_triangles(stage: Usd.Stage, root: Sdf.Path) -> list[GprimTriangles]:
    """Every visible render Gprim under ``root``."""
    cache = UsdGeom.XformCache()
    found: list[GprimTriangles] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        shape = _mesh_triangles(prim) if prim.IsA(UsdGeom.Mesh) else _extent_box(prim)
        if shape is None:
            continue
        points, triangles, corners, faces, face_count = shape
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)
        found.append(GprimTriangles(prim, points @ matrix[:3, :3] + matrix[3, :3], triangles, corners, faces,
                                    face_count))
    return found


def _mesh_triangles(prim: Usd.Prim) -> tuple | None:
    mesh = UsdGeom.Mesh(prim)
    points, counts, indices = (_value(a) for a in (
        mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()))
    if points is None or counts is None or indices is None or not len(points):
        return None
    counts, indices = np.asarray(counts, np.int64), np.asarray(indices, np.int64)
    fan = _fan_corners(counts) if counts.sum() == len(indices) else None
    if fan is None or not len(fan[0]):
        return None
    corners, faces = fan
    triangles = indices[corners]
    if triangles.min() < 0 or triangles.max() >= len(points):
        return None
    return np.asarray(points, np.float64), triangles, corners, faces, len(counts)


def _extent_box(prim: Usd.Prim) -> tuple | None:
    boundable = UsdGeom.Boundable(prim)
    extent_attr = boundable.GetExtentAttr()  # unauthored, it reads the schema's fallback size
    extent = _value(extent_attr) if extent_attr.HasAuthoredValue() else None
    if extent is None:
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, Usd.TimeCode.Default())
    if extent is None:
        return None
    points, triangles = _box(np.asarray(extent, np.float64))
    return points, triangles, np.zeros_like(triangles), np.zeros(len(triangles), np.int64), 1


def _fan_corners(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(corner indices (T, 3) into faceVertexIndices, face of each triangle (T,))."""
    if (counts < 0).any():
        return None
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    per_face = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), per_face)
    k = np.arange(per_face.sum()) - np.repeat(np.cumsum(per_face) - per_face, per_face)
    s = starts[face]
    return np.stack([s, s + k + 1, s + k + 2], axis=1), face


def _box(extent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = extent
    points = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return points, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


class _Collector:
    def __init__(self, textures: _Textures) -> None:
        self.textures = textures
        self.looks: dict[str, Look | None] = {}
        self.pos: list[np.ndarray] = []   # (T, 3, 3)
        self.col: list[np.ndarray] = []   # (T, 3, 3) linear
        self.uv: list[np.ndarray] = []    # (T, 3, 2)
        self.tex: list[np.ndarray] = []   # (T,)

    def look(self, prim: Usd.Prim) -> Look | None:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        key = str(material.GetPath()) if material else ""
        if key not in self.looks:
            self.looks[key] = material_look(material) if material else None
        return self.looks[key]

    def add(self, part: GprimTriangles) -> None:
        prim, points, corners, faces, face_count = part.prim, part.triangles, part.corners, part.faces, part.face_count
        n = len(points)
        face_look = np.zeros(face_count, np.int64)
        looks = [self.look(prim)]
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            ids = subset.GetIndicesAttr().Get()
            if ids is None:
                continue
            ids = np.asarray(ids, np.int64)
            looks.append(self.look(subset.GetPrim()))
            face_look[ids[(ids >= 0) & (ids < face_count)]] = len(looks) - 1
        tri_look = face_look[faces]

        display = _primvar(prim, "displayColor")
        fallback = None
        if display is not None and display[0].shape[1] >= 3:
            fallback = _per_corner(display[0][:, :3], display[1], faces, points, corners)
        if fallback is None:
            fallback = np.broadcast_to(np.array(DEFAULT_COLOR), (n, 3, 3))
        colors = np.array(fallback, np.float32)
        uvs = np.zeros((n, 3, 2), np.float32)
        tex = np.full(n, -1, np.int64)
        for i, look in enumerate(looks):
            mask = tri_look == i
            if look is None or not mask.any():
                continue
            texture = self.textures.get(look.texture)
            if look.color is not None and (texture >= 0 or not look.texture):  # a lost texture: displayColor
                colors[mask] = look.color
            st = _primvar(prim, look.uv_name) if texture >= 0 else None
            corner_uv = _per_corner(st[0][:, :2], st[1], faces, points, corners) if st and st[0].shape[1] >= 2 else None
            if corner_uv is not None:
                uvs[mask] = corner_uv[mask]
                tex[mask] = texture
        self.pos.append(part.points[points].astype(np.float32))
        self.col.append(colors)
        self.uv.append(uvs)
        self.tex.append(tex)


def build_preview(parts: list[GprimTriangles], read_file: Callable[[str], bytes | None],
                  budget: int = PREVIEW_TRIANGLES) -> PreviewMesh:
    """``parts`` (from :func:`world_triangles`) with their colours; ``read_file`` reads a texture."""
    textures = _Textures(read_file)
    out = _Collector(textures)
    for part in parts:
        out.add(part)

    if not out.pos:
        empty = np.zeros((0, 3))
        return PreviewMesh(empty.astype(np.float32), empty.astype(np.uint8), np.zeros((0, 2), np.float32),
                           empty.astype(np.uint32), [])
    pos, col = np.concatenate(out.pos), _to_srgb(np.concatenate(out.col))
    uv, tex = np.concatenate(out.uv), np.concatenate(out.tex)
    if len(pos) > budget:
        for t, image in enumerate(textures.images):  # bake: the colour under each corner
            mask = tex == t
            if mask.any():
                h, w = image.shape[:2]
                u, v = uv[mask][..., 0], uv[mask][..., 1]
                cols = (np.floor(np.mod(u, 1.0) * w).astype(np.int64)).clip(0, w - 1)
                rows = (np.floor((1.0 - np.mod(v, 1.0)) * h).astype(np.int64)).clip(0, h - 1)
                col[mask] = col[mask] * (image[rows, cols] / 255.0)
        return _simplified(pos, col, budget)

    order = np.argsort(tex, kind="stable")
    pos, col, uv, tex = pos[order], col[order], uv[order], tex[order]
    rows = np.concatenate([
        pos.reshape(-1, 3).astype(np.float32).view(np.uint32),
        np.rint(col.reshape(-1, 3) * 255).astype(np.uint32),
        uv.reshape(-1, 2).astype(np.float32).view(np.uint32),
        np.repeat(tex, 3)[:, None].astype(np.int64).astype(np.uint32),
    ], axis=1)
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    groups, first = [], 0
    for t, count in zip(*np.unique(tex, return_counts=True)):
        groups.append((first, int(count), int(t)))
        first += int(count)
    used = sorted({g[2] for g in groups if g[2] >= 0})
    remap = {t: i for i, t in enumerate(used)}
    return PreviewMesh(
        positions=np.ascontiguousarray(unique[:, :3]).view(np.float32),
        colors=unique[:, 3:6].astype(np.uint8),
        uvs=np.ascontiguousarray(unique[:, 6:8]).view(np.float32),
        indices=inverse.reshape(-1, 3).astype(np.uint32),
        groups=[(f, c, remap.get(t, -1)) for f, c, t in groups],
        textures=[_png(textures.images[t]) for t in used],
    )


def _png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return data.tobytes() if ok else b""


def _simplified(pos: np.ndarray, col: np.ndarray, limit: int) -> PreviewMesh:
    """At most ``limit`` triangles; a merged vertex takes the mean colour of the corners it replaces."""
    vertices = pos.reshape(-1, 3)
    centres, tris, (mean,) = simplify_mesh(vertices, np.arange(len(vertices)).reshape(-1, 3), limit,
                                           col.reshape(-1, 3))
    return PreviewMesh(
        positions=centres.astype(np.float32),
        colors=np.rint(mean * 255).astype(np.uint8),
        uvs=np.zeros((len(centres), 2), np.float32),
        indices=tris.astype(np.uint32),
        groups=[(0, len(tris), -1)],
    )


def simplify_mesh(vertices: np.ndarray, triangles: np.ndarray, limit: int,
                  *attributes: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Vertex clustering: merge vertices on ever coarser grids until at most ``limit`` triangles remain.
    A merged vertex sits at the mean of the vertices it replaces; per-vertex ``attributes`` are
    averaged the same way. Returns (vertices, triangles, attributes)."""
    lo = vertices.min(axis=0)
    extent = max(float((vertices.max(axis=0) - lo).max()), 1e-9)
    for cells in (1024, 512, 256, 128, 64, 32, 16):
        key = np.floor((vertices - lo) / (extent / cells)).astype(np.int64)
        flat_key = (key[:, 0] * (cells + 1) + key[:, 1]) * (cells + 1) + key[:, 2]
        _, cluster = np.unique(flat_key, return_inverse=True)
        cluster = cluster.ravel()
        tris = cluster[triangles]
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
        _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
        tris = tris[np.sort(first)]
        if len(tris) <= limit or cells == 16:
            break
    n = int(cluster.max()) + 1
    weight = np.maximum(np.bincount(cluster, minlength=n), 1)[:, None]
    used, compact = np.unique(tris, return_inverse=True)

    def mean(values: np.ndarray) -> np.ndarray:
        sums = np.stack([np.bincount(cluster, values[:, i], n) for i in range(values.shape[1])], axis=1)
        return (sums / weight)[used]

    return mean(vertices), compact.reshape(-1, 3), [mean(a) for a in attributes]
