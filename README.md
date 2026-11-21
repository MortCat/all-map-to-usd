# AllMapToUSD

Convert AGV/AMR navigation maps into simulation-ready OpenUSD obstacle layouts. Check aisle widths and vehicle clearance in 2D, then export a USD file or push the layout into a live NVIDIA Isaac Sim stage.

![AllMapToUSD showing a reconstructed ROS map, AGV clearance check, and aisle measurement](docs/images/ui_overview.png)

[Collision and export preview](docs/images/ui_collision_export.png) · [Isaac Sim stage preview](docs/images/isaac_sim_sample.jpg)

## Quick start

1. On Windows, install Python 3.11+ and double-click **start_app.cmd**. The first launch creates a virtual environment and installs dependencies. Use `start_app.cmd --browser` to open the UI in your default browser.
2. Drop a map into **Map Source**, or click **Sample: ROS.yaml** to try the bundled example.
3. Use **Ruler** to measure an aisle. Under **AGV Footprint Collision Check**, enter the vehicle dimensions, drag the footprint, and rotate it with **Q** or **E** to check clearance.
4. Export **.usd** (binary) or **.usda** (text) to **exports/**. You can set the output origin and heading and include stations, routes, and an AGV proxy. Export works offline.

If setup fails, check **setup.log** and rerun with `start_app.cmd --force-setup`. The app writes to **app.log**.

## Supported maps

| Source | Input | Options |
| --- | --- | --- |
| ROS / ROS 2 Nav2 | YAML + PGM/PNG, or ZIP | Seal unknown boundaries in occupancy grids |
| SEER Robotics | SMAP or JSON | Import obstacles, no-go zones, stations, and routes |
| CAD / SICK SMET | DXF | Detect or override drawing units and configure layers and extrusion heights |
| Raster floor plans / SICK exports | PNG, PGM, BMP, JPEG, TIFF | Set the image resolution in meters per pixel |

Use **Auto Detect** or select the format in **Map Source**. For reconstruction, **Contour Polygon** follows the obstacle outlines and **Axis-Aligned Grid Merge (AABB)** creates rectangular blocks. **Fill Solid Loops** can also fill small enclosed rooms. Lower the area limit or disable the option to keep those rooms open.

## Push to Isaac Sim

1. On the Isaac Sim host, enable `isaacsim.code_editor.python_server` when launching Isaac Sim. Use this command on Linux.

   ```bash
   ./isaac-sim.sh --enable isaacsim.code_editor.python_server
   ```

2. In **Push to Isaac Sim (Live Stage)**, enter `127.0.0.1` for a local instance or the IP of your remote Linux host. The default port is `8226`. For a remote server bound to loopback, configure key-based SSH so the app can open a tunnel.
3. Choose **Reference in Current USD** to mount or update the map at `/World/AGV_Map`, or choose **Open as New Stage**. Click **Test Connection**, then **Push to Isaac Sim**.

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
