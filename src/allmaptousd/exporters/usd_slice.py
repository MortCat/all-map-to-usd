"""Read-only stage projection. This module also runs inside Isaac Sim."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_EPS = 1e-8
_MAX_FACES = 500_000
_MAX_PROJECTED_VERTICES = 500_000


def _stage_path(value: str, allow_root: bool) -> Sdf.Path:
    valid, _ = Sdf.Path.IsValidPathString(value)
    if valid:
        path = Sdf.Path(value)
        if path.IsAbsolutePath() and (path.IsPrimPath() or (allow_root and path == Sdf.Path.absoluteRootPath)):
            return path
    raise ValueError(f"Invalid absolute prim path: {value}")


def _mesh_geometry(prim: Usd.Prim, time: Usd.TimeCode):
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if not points or not counts or not indices:
        return [], [], False
    if len(counts) > _MAX_FACES:
        raise ValueError(f"{prim.GetPath()} has too many mesh faces. Select a smaller or simpler stage scope.")
    holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
    faces = []
    offset = 0
    for face_index, count in enumerate(counts):
        end = offset + count
        if count >= 3 and face_index not in holes:
            face = list(indices[offset:end])
            if len(face) == count and min(face) >= 0 and max(face) < len(points):
                faces.append(face)
        offset = end
    return points, faces, None


def _primitive_geometry(prim: Usd.Prim, time: Usd.TimeCode):
    if prim.IsA(UsdGeom.Cube):
        half = UsdGeom.Cube(prim).GetSizeAttr().Get(time) / 2
        points = [(x * half, y * half, z * half) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
        return points, [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6)], True

    shape_type = prim.GetTypeName()
    schema = getattr(UsdGeom, shape_type)(prim)
    radius = schema.GetRadiusAttr().Get(time)
    axis = "Z" if shape_type == "Sphere" else schema.GetAxisAttr().Get(time)
    half = 0 if shape_type == "Sphere" else schema.GetHeightAttr().Get(time) / 2
    if shape_type == "Cylinder":
        rings = [(-half, radius), (half, radius)]
    elif shape_type == "Cone":
        rings = [(-half, radius), (half, 0)]
    else:
        latitudes = [math.pi * n / 16 for n in range(-8, 9)]
        rings = [
            (radius * math.sin(angle) + (-half if angle < 0 else half), radius * math.cos(angle))
            for angle in latitudes
        ]
        if shape_type == "Capsule":
            rings.insert(8, (-half, radius))

    sides = 32
    points = []
    for z, ring_radius in rings:
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            x, y = ring_radius * math.cos(angle), ring_radius * math.sin(angle)
            points.append({"X": (z, x, y), "Y": (y, z, x), "Z": (x, y, z)}[axis])
    faces = [list(range(sides)), list(range(len(points) - sides, len(points)))]
    for ring in range(len(rings) - 1):
        for i in range(sides):
            a, b = ring * sides + i, ring * sides + (i + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return points, faces, True


def _world_points(points, transform: Gf.Matrix4d, meters: float, y_up: bool):
    world = []
    for point in points:
        x, y, z = transform.Transform(Gf.Vec3d(*point)) * meters
        world.append((x, -z, y) if y_up else (x, y, z))
    return world


def _crossing(a, b, height: float):
    t = (height - a[2]) / (b[2] - a[2])
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, height


def _clip_face(vertices, height: float, above: bool):
    if not vertices:
        return []
    result = []
    a = vertices[-1]
    for b in vertices:
        a_inside = a[2] >= height if above else a[2] <= height
        b_inside = b[2] >= height if above else b[2] <= height
        if a_inside != b_inside:
            result.append(_crossing(a, b, height))
        if b_inside:
            result.append(b)
        a = b
    return result


def _closed_mesh(faces) -> bool:
    edges = Counter(tuple(sorted((a, b))) for face in faces for a, b in zip(face, face[1:] + face[:1]))
    return bool(edges) and all(count == 2 for count in edges.values())


def _section_loops(points, faces, height: float):
    neighbors = defaultdict(set)
    for face in faces:
        vertices = [points[i] for i in face]
        crossings = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            if (a[2] < height) != (b[2] < height):
                x, y, _ = _crossing(a, b, height)
                crossings.append((round(x, 8), round(y, 8)))
        if len(crossings) > 2:
            axis = max(range(2), key=lambda i: max(p[i] for p in crossings) - min(p[i] for p in crossings))
            crossings.sort(key=lambda p: p[axis])
        for a, b in zip(crossings[::2], crossings[1::2]):
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)
    loops = []
    remaining = set(neighbors)
    for start in sorted(neighbors):
        if start not in remaining:
            continue
        loop = [start]
        previous, current = None, start
        while len(neighbors[current]) == 2:
            end = min(neighbors[current] - {previous})
            if end == start:
                if len(loop) >= 3:
                    loops.append(loop)
                break
            if end not in remaining:
                break
            remaining.remove(end)
            loop.append(end)
            previous, current = current, end
        remaining.remove(start)
    return loops


def _is_map_helper(prim: Usd.Prim) -> bool:
    if prim.GetName() == "Navigation":
        return prim.GetParent().HasCustomDataKey("grid_resolution_m")
    if prim.GetName() == "TestVehicle":
        sibling = prim.GetParent().GetChild("Map")
        return bool(sibling and sibling.HasCustomDataKey("grid_resolution_m"))
    return False


def collect_stage_slice(
    stage: Usd.Stage,
    time: Usd.TimeCode,
    root_path: str,
    exclude_path: str,
    low: float,
    high: float,
    collision_only: bool,
) -> dict:
    """Project geometry intersecting the vertical interval onto the stage's horizontal plane."""
    root = stage.GetPrimAtPath(_stage_path(root_path, allow_root=True))
    if not root:
        raise ValueError(f"Stage prim not found: {root_path}")
    excluded = _stage_path(exclude_path, allow_root=False) if exclude_path else None
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    transforms = UsdGeom.XformCache(time)
    projected, sections = [], []
    prim_count = unsupported = total_faces = vertex_count = 0
    supported = (UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Sphere, UsdGeom.Capsule)
    prim_range = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    for prim in prim_range:
        if (excluded is not None and prim.GetPath().HasPrefix(excluded)) or (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI) or _is_map_helper(prim)
        ):
            prim_range.PruneChildren()
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        has_collision = bool(collision and collision.GetCollisionEnabledAttr().Get(time))
        visible = (
            imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
        )
        if not has_collision and (collision_only or not visible):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points, faces, closed = _mesh_geometry(prim, time)
        elif any(prim.IsA(shape) for shape in supported):
            points, faces, closed = _primitive_geometry(prim, time)
        else:
            unsupported += bool(prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer))
            continue
        if not faces:
            continue
        transform = transforms.GetLocalToWorldTransform(prim)
        world = _world_points(points, transform, meters, y_up)
        min_z, max_z = min(p[2] for p in world), max(p[2] for p in world)
        if max_z <= low + _EPS or min_z > high:
            continue
        total_faces += len(faces)
        if total_faces > _MAX_FACES:
            raise ValueError("The height slice intersects too many faces. Select a smaller stage scope.")
        before = len(projected)
        for face in faces:
            vertices = [world[i] for i in face]
            if max(p[2] for p in vertices) <= low + _EPS or min(p[2] for p in vertices) > high:
                continue
            clipped = _clip_face(_clip_face(vertices, low, above=True), high, above=False)
            if clipped:
                projected.append([(round(x, 7), round(y, 7)) for x, y, _ in clipped])
                vertex_count += len(clipped)
        if min_z < low and (closed or (closed is None and _closed_mesh(faces))):
            # Surfaces plus the solid's cross-section at the bottom cover its whole swept footprint.
            loops = _section_loops(world, faces, low + _EPS)
            if loops:
                sections.append(loops)
                vertex_count += sum(map(len, loops))
        if vertex_count > _MAX_PROJECTED_VERTICES:
            raise ValueError("The projected slice is too detailed. Select a smaller stage scope.")
        prim_count += len(projected) != before

    return {
        "stage": stage.GetRootLayer().GetDisplayName() or "Isaac Sim",
        "up_axis": "Y" if y_up else "Z",
        "root_prim_path": root_path,
        "prim_count": prim_count,
        "unsupported": unsupported,
        "projected": projected,
        "sections": sections,
    }
