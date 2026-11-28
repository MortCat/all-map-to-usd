# AllMapToUSD

Bring AGV/AMR maps from ROS, SEER, CAD, and image floor plans into NVIDIA Isaac Sim for testing. AllMapToUSD reconstructs obstacles as OpenUSD geometry with colliders, so you can reuse existing maps, draw custom layouts, and check vehicle clearance before running a simulation. Import and position 3D equipment on the same map, then deploy the full layout or just the equipment to an existing scene.

![AllMapToUSD showing a reconstructed ROS map with placed storage racks, AGV clearance, and the Export & Save and Isaac Sim Sync panels](docs/images/ui_overview.png)

## Quick start

1. On Windows, install Python 3.11+ and run `start_app.cmd`. The first launch installs the dependencies. Use `start_app.cmd --browser` to open the app in your browser.
2. In **Map Source**, drop in a map, try **Sample: ROS.yaml**, or choose **Empty Map** to draw a layout from scratch.
3. Use **Ruler** to measure clearances and **AGV Footprint Collision Check** to test a vehicle pose. Then save an editable project, export OpenUSD, or send the layout directly to Isaac Sim.

## Load or draw a test map

| Source | Input | Settings |
| --- | --- | --- |
| ROS / ROS 2 Nav2 | YAML with PGM/PNG, or ZIP | Optionally seal unknown boundaries |
| SEER Robotics | SMAP or JSON | Obstacles, forbidden zones, stations, and routes |
| CAD / SICK SMET | DXF | Drawing units, layer selection, and obstacle heights |
| Raster floor plans / SICK exports | PNG, PGM, BMP, JPEG, TIFF | Meters per pixel |

- In **Edit**, use **Brush**, **Line**, or **Rectangle**. Brush over imported or drawn obstacles with **Eraser**, or click it again for **Box Eraser**. Completed strokes take effect immediately. Use the toolbar's undo and redo icons. Press **R** to use the ruler during editing, and press it again to resume drawing.
- **Contour Polygon** follows obstacle outlines. **Axis-Aligned Grid Merge (AABB)** produces rectangular blocks. **Fill Solid Loops** can fill small closed regions formed by imported and drawn obstacles together. Adjust **Max Fill Area** to keep larger enclosed spaces open. Click **Rebuild Obstacles** after changing reconstruction settings.
- Enter the AGV dimensions and safety margin, then drag the footprint and rotate it with **Q** or **E** to check clearance at a specific pose.

## Place 3D equipment

1. In **Edit → Model**, import STL, STEP (`.step` or `.stp`), or OpenUSD (`.usd`, `.usda`, `.usdc`, `.usdz`). Import the **Whole model** or select parts from the USD hierarchy or STEP assembly. Check the units and up axis. STL defaults to millimeters.
2. For USDs with external textures, materials, or parts, use **Choose Folder** or upload a ZIP that preserves their relative paths. If the package contains several USDs, select the **Main file**.
3. Right-drag a thumbnail or Shift-drag to orbit. Left-drag it onto the map to place it in top view. Drag a placed model to move it, or right-click to set **Yaw** or delete it. Its top-view footprint contributes to the clearance check.

The Windows launcher attempts to install OpenCASCADE for STEP. If that step fails, see `setup.log`. Other formats remain available. In a source checkout, install STEP support with `python -m pip install -e ".[step]"`.

## Save a project or export USD

| Action | Result |
| --- | --- |
| **Save Map (ROS .zip)** | A ROS / Nav2 occupancy map (YAML + PGM) bundled with project settings, zones, stations, routes, the AGV pose, and the models with their placement and asset files. Drop the ZIP into **Map Source** to resume editing. |
| **Export .usd** / **Export .usda** | A simulation layout with obstacle and equipment colliders. Keep the `allmaptousd_assets/` folder beside the USD when moving an export with textures or materials. |

Export works offline. Set the output origin and heading in **Export & Save** when the target scene uses a different frame. Use Isaac Sim to configure articulation, control, and sensor behavior for simulated robots.

## Sync with Isaac Sim

From the Isaac Sim installation directory, launch Isaac Sim with its Python Server enabled.

```powershell
.\isaac-sim.bat --enable isaacsim.code_editor.python_server
```

On Linux, use `./isaac-sim.sh --enable isaacsim.code_editor.python_server`. Under **Isaac Sim Sync**, use `127.0.0.1:8226` for a local instance and click **Test Connection**. For a remote Linux host with a loopback-only server, use its host address and configure key-based SSH for the tunnel.

| Action | Use it to |
| --- | --- |
| **Push to Isaac Sim** | Select **Reference in Current USD** to mount or update the full map at `/World/AGV_Map` by default. Select **Open as New Stage** to open the exported layout. |
| **Pull from Isaac Sim** | Replace the canvas with an editable occupancy map from the active stage. Use **Pull Source Prim** and **Exclude Prim** to select the area, and **Colliders Only** to restrict the geometry. |
| **Push Models Only** | Update the equipment in an existing scene. Pull the scene first to align coordinates, place the models on the canvas, then push them to **Models Prim** (default `/World/AGV_Equipment`). Each push replaces the previous set at that prim. |

**Pull height range** runs from floor elevation plus **Floor Clearance** to floor elevation plus **AGV Height**. At floor elevation 0 m, clearance 0.02 m, and AGV height 0.60 m, the app projects geometry between 0.02 and 0.60 m onto the map. Pull uses a geometry snapshot at the current timeline position.

**Model Floor** controls equipment placement. **Auto** searches for a surface under each model and falls back to **Floor Elevation**. The default directory for transferred files on the Isaac host is `~/allmaptousd_maps/`. Keep its `allmaptousd_assets/` folder with the deployment. A later pull restores equipment as movable models when their source models are still in the library. If a source model is missing, its geometry in the selected height range is projected as obstacles.

## CLI and development

Install from a source checkout with `python -m pip install -e .`.

```powershell
allmaptousd formats
allmaptousd convert src/allmaptousd/samples/sample_factory.yaml -o factory.usd
```

For development, install `requirements.txt` in a virtual environment and run `python -m ruff check .` and `python -m pytest`. See the [conversion pipeline](src/allmaptousd/pipeline.py) and [parser interface](src/allmaptousd/parsers/base.py) to add another format.

For Windows setup issues, check `setup.log` and retry with `start_app.cmd --force-setup`. Application logs are in `app.log`.

[MIT License](LICENSE)
