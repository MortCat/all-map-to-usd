# AllMapToUSD

Convert AGV/AMR navigation maps into simulation-ready OpenUSD obstacle layouts. Check aisle widths and vehicle clearance in 2D, then export a USD file or push the layout into a live NVIDIA Isaac Sim stage.

![AllMapToUSD showing a reconstructed ROS map, AGV clearance check, and aisle measurement](docs/images/ui_overview.png)

[Collision and export preview](docs/images/ui_collision_export.png) · [Isaac Sim stage preview](docs/images/isaac_sim_sample.jpg)

## Quick start

1. On Windows, install Python 3.11+ and double-click **start_app.cmd**. The first launch creates a virtual environment and installs dependencies. Use `start_app.cmd --browser` to open the UI in your default browser.
2. Drop a map into **Map Source**, or click **Sample: ROS.yaml** to try the bundled example.
3. Use **Ruler** to measure an aisle. Under **AGV Footprint Collision Check**, enter the vehicle dimensions, drag the footprint, and rotate it with **Q** or **E** to check clearance.
4. Export **.usd** (binary) or **.usda** (text). A **Save As** window asks where to save the file and what to call it. It suggests the map name and opens in the folder of the last export, or **exports/** the first time. You can set the output origin and heading and include stations, routes, and an AGV proxy. Export works offline.
5. Click **Save Map (ROS .zip)** to keep your work. The zip is a standard ROS / Nav2 map (YAML + PGM) of the obstacles. It also stores the forbidden zones, stations, routes, build settings, and AGV pose. Drop it into **Map Source** to reopen the map where you left off.

If setup fails, check **setup.log** and rerun with `start_app.cmd --force-setup`. The app writes to **app.log**.

## Edit a map

Load a map or choose **Empty Map** in **Map Source** and set its width, height, and grid resolution. Open **Edit** next to **Ruler** to draw obstacle points, lines, or outlined and filled rectangles. Use **Eraser** to remove drawn or imported obstacles. Click it again to switch to **Box Eraser**, then drag a box to erase the obstacles inside it. Drag with the middle or right mouse button to pan.

Each completed stroke updates the collision preview and the geometry used for USD export and Isaac Sim. With **Fill Solid Loops** enabled, imported obstacles, brush and line strokes, and rectangle outlines can form a closed boundary together. Small edited pockets with entrances too narrow for the AGV can also be filled. This check uses a disk based on the smaller AGV dimension and respects **Max Fill Area (m²)** and eraser openings. Use **Filled** for a rectangle that is solid regardless of the loop settings. Click **Rebuild Obstacles** after changing these settings or the AGV dimensions.

Use the **Undo** and **Redo** icons next to **Width**, or press **Ctrl+Z** and **Ctrl+Shift+Z**. Press **R** to measure during editing and again to resume drawing. Edits live in the current app session. Use **Save Map (ROS .zip)** to keep them after closing the app or loading another map.

## Place 3D models

Choose **Model** in the **Edit** tools to import equipment from **.usd**, **.usda**, **.usdc**, **.usdz**, **.step**, **.stp**, or **.stl** files. In the import dialog, keep **Whole model** or pick parts from the USD hierarchy or the STEP assembly tree. Check the units and up axis. STL files have no units and default to millimetres. STEP files are read in millimetres. Each model is converted to meters, Z up, and stands on the floor.

STEP import uses OpenCASCADE, a large download. `start_app.cmd` installs it during setup. If that step fails, for example because no build exists for your Python version yet, setup notes it in **setup.log** and the app still runs without STEP. A pip install needs the extra: `python -m pip install "allmaptousd[step]"`. STEP parts keep their colours but not their textures.

A USD model often keeps its textures, materials (`.mdl`), or parts in separate files. To import the whole model, click **Choose Folder** and select the model's folder, or choose a **.zip** of it. A **.usdz** already holds its files. If the folder has more than one model file, pick the right one in **Main file**. The dialog lists any file the model uses that was not uploaded. Materials that come with Isaac Sim, such as `OmniPBR.mdl`, need no file.

Imported models appear as 3D pictures in the tray, in the colours and textures of their materials. Drag a picture onto the map to place the model. Right-drag a picture, or Shift-drag it on a trackpad, to turn it. Drag a placed model to move it. Right-click it to set its yaw or delete it, or press **Delete**. A placed model's top-view outline always counts in the AGV collision check.

USD export and Isaac Sim push copy each placed model's geometry and materials into the file under `/World/Map/Equipment`, with static colliders, so the result does not depend on the source file. The files the models use go in an `allmaptousd_assets` folder next to the USD. **Export USD** saves them side by side in the folder you chose. **Save Map** stores the models, their files, and their positions in the zip.

## Supported maps

| Source | Input | Options |
| --- | --- | --- |
| ROS / ROS 2 Nav2 | YAML + PGM/PNG, or ZIP | Seal unknown boundaries in occupancy grids |
| SEER Robotics | SMAP or JSON | Import obstacles, no-go zones, stations, and routes |
| CAD / SICK SMET | DXF | Detect or override drawing units and configure layers and extrusion heights |
| Raster floor plans / SICK exports | PNG, PGM, BMP, JPEG, TIFF | Set the image resolution in meters per pixel |

Use **Auto Detect** or select the format in **Map Source**. For reconstruction, **Contour Polygon** follows the obstacle outlines and **Axis-Aligned Grid Merge (AABB)** creates rectangular blocks. **Fill Solid Loops** can also fill small enclosed rooms. Lower the area limit or disable the option to keep those rooms open.

## Isaac Sim Sync

1. On the Isaac Sim host, enable `isaacsim.code_editor.python_server` when launching Isaac Sim. On Windows, run `.isaac-sim.bat --enable isaacsim.code_editor.python_server` from the Isaac Sim installation directory. On Linux, use the following command.

   ```bash
   ./isaac-sim.sh --enable isaacsim.code_editor.python_server
   ```

2. In **Isaac Sim Sync**, enter `127.0.0.1` for a local instance or the address of your remote Linux host. The default port is `8226`. For a remote server bound to loopback, configure key-based SSH so the app can open a tunnel. Click **Test Connection** to check the active stage.
3. To send the canvas, select **Reference in Current USD** to mount or update the map at `/World/AGV_Map`, or select **Open as New Stage**. Click **Push to Isaac Sim** to frame the map in the active viewport. Reference mode uses the existing stage lights when available.
4. To read the scene, set **Pull Source Prim** to `/` for the whole stage or enter a subtree such as `/World/Warehouse`. Set **Exclude Prim** to the robot's subtree if needed. Select **Colliders Only** to use enabled USD collision geometries, or keep **Visible or Collidable** to also use visible meshes and shapes.
5. Set **Floor Elevation** and **Floor Clearance** in meters. Pull projects geometry between floor elevation plus clearance and floor elevation plus the **AGV Height** into editable occupancy cells. For example, with a floor elevation of 0 m, a clearance of 0.02 m, and an AGV height of 0.60 m, the range is 0.02–0.60 m. Click **Pull from Isaac Sim** to replace the canvas after a successful read.

Pull reads the active USD stage at its current timeline position, accounts for stage units and up axis, and leaves the stage untouched. It uses polygon meshes, analytic shapes such as cubes and cylinders, and geometry inside USD references and native instances. It samples their geometry rather than running a physics or LiDAR simulation. Point instancers and other unsupported geometry are reported in the sync status. The initial build uses exact grid rectangles with noise cleanup disabled to preserve the projected cells.

## Command line

Install the package in your active Python environment with `python -m pip install -e .`, then convert the bundled sample from the project root.

```powershell
allmaptousd formats
allmaptousd convert sample_maps/sample_factory.yaml -o factory.usd
allmaptousd convert --help
```

## Development

Install the desktop and development dependencies from `requirements.txt` in a virtual environment, then run the checks.

```powershell
python -m pip install -r requirements.txt
python -m ruff check .
python -m pytest
```

The shared [conversion pipeline](src/allmaptousd/pipeline.py) handles parsing, reconstruction, and export. To support another format, implement [MapParser](src/allmaptousd/parsers/base.py) and register it in [create_default_registry()](src/allmaptousd/parsers/__init__.py).

## License

[MIT](LICENSE)
