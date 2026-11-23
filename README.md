# Orchid

Orchid is a ROS2 mobile-manipulation orchestration stack. It loads a robot
manifest, starts per-robot dispatch/session nodes, uses an LLM to produce
behavior-tree task plans, and executes those plans through action and controller
plugins. The stack includes navigation/manipulation actions, scene observation,
GPD-based grasp estimation, replanning/user-input hooks, and UI feedback topics
for the Orchid HRI dashboard.

## Configuration

The main configuration locations are:

- `.env.manifest`: local secrets and runtime env vars. Copy it from
  `.env.manifest.example`; never commit real keys.
- `manifests/`: runtime robot manifests. The default is
  `manifests/example_manifest_hsrb.json`.
- Manifest module configs: use `api_key_env` to refer to env vars such as
  `DEEPSEEK_API_KEY` or `OPENROUTER_API_KEY`.

Create the local env file:

```bash
cp .env.manifest.example .env.manifest
```

Fill in the API keys and any endpoint overrides required by the manifest.

## Pull / Build The Image

Log in to the registry to be able to pull the image:

```bash
docker login registry.gitlab.uni-bonn.de:5050
```

If not accessible, build from the repository root:
Run the build from the repo root because the Docker build context includes
`pyproject.toml`, `uv.lock`, and `submodules/`.

```bash
uv sync
uv lock --check
./docker_build.sh
```

The Docker image intentionally uses `uv pip install --system` because it installs
dependencies into the image's system Python environment.

## Start The Main Server

Start the Orchid server container:

```bash
docker compose -f humble/docker-compose.yml up -d
```

The compose file loads `.env.manifest`, mounts this repo at
`/home/orchid/orchid_server`, and keeps the container interactive so the dispatch
server can be launched inside it.

The default compose workflow is Docker-first for development: Orchid is mounted
read-write, editable ROS packages from `submodules/` are mounted into
`/home/orchid/ros2_ws/src`, and the ROS workspace `build`, `install`, and `log`
directories are persistent Docker volumes. Recreate the container after compose
mount changes:

```bash
docker compose -f humble/docker-compose.yml up -d --force-recreate
```

## Start The Robot Stack

Start the dispatch server:

```bash
docker exec -it orchid_server bash -lc \
  'cd /home/orchid/orchid_server && scripts/launch_dispatch_server.sh'
```

Launch MoveIt:

```bash
docker exec -it orchid_server bash -lc \
  'source /opt/ros/humble/setup.bash && source /home/orchid/ros2_ws/install/setup.bash && ros2 launch orchid_moveit_config hsrc_moveit.launch.py'
```

Start the HSR Nav2 stack from the container with a map file:

```bash
docker exec -it orchid_server bash -lc \
    'source /opt/ros/humble/setup.bash && source /home/orchid/ros2_ws/install/setup.bash && ros2 launch orchid_nav2_config navigation.launch.py localization_map:=/absolute/path/to/map_loc.yaml navigation_map:=/absolute/path/to/map_nav.yaml'
```

## Orchid UI

The UI lives in the `submodules/orchid_ui` repository. See
`submodules/orchid_ui/README.md` for its full setup and interface details.

Typical startup inside the server container:

```bash
docker exec -it orchid_server /home/orchid/orchid_ui/start_dashboard.sh
```

Open:

```text
http://localhost:8000
```

## Model API Server

TODO: add the model API server repository link.

## LLM Configuration

The current manifests support OpenAI-compatible chat endpoints. The default HSRB
manifest includes examples for DeepSeek, OpenRouter, Ollama/local LLMs, and a
local SAM-style model endpoint. Configure URLs/models in `manifests/` and keep
keys in `.env.manifest`.