# HiFi: photoreal Godot twins of the GymBox envs

GymBox keeps the physics; Godot only renders it. Every OrthoWorlds env in `ortho/game_envs` has
a high-fidelity twin that renders the same four views from the same state:

- **Top**, **Rear** and **Side**: orthographic
- **3D FPV**: perspective, orbited by the Center/Left/Right/Up/Down camera commands

Dynamics, rewards, seeds, observations and the camera state stay identical. Trained agents,
prompts and logged camera dicts therefore carry over, and every hi-fi frame can be paired with
the GymBox frame from the same step.

```
 GymBox env + wrapper (ortho/, unchanged)          Godot 4 (hifi/godot, one project for all envs)
 ┌──────────────────────────────────┐  hello/scene  ┌───────────────────────────────────────┐
 │ physics, rewards, camera orbit   │──────────────►│ envs/<env>/world.gd builds the scene  │
 │ exporters/<env>.py               │  state+cams   │ 4 SubViewports: Top Rear Side FPV     │
 │ wrappers.HiFiOrthographicWrapper │──────────────►│ + flat object-ID mask viewports       │
 │   render() -> (top,rear,side,fpv)│◄──────────────│ frames (lockstep) / window (live)     │
 └──────────────────────────────────┘ frames, keys  └───────────────────────────────────────┘
```

## Quick start

From the repo root, in the `GAME` conda env:

```bash
python hifi/tools/fetch_assets.py          # ~250 MB of CC0 Poly Haven HDRIs, PBR textures, models (once)
python hifi/tools/get_godot.py             # Windows Godot 4.3 into hifi/tools/bin (GPU from WSL; optional)

python hifi/run_hifi.py --env tank         # play: game keys + arrows orbit the FPV; 1-5 views, 0 grid
python hifi/test_hifi.py --all             # automated checks + GymBox-vs-Godot sheets in hifi/test_out/
```

### Godot binaries

- Godot 4.3 or newer is required.
- **WSL2.** The Linux build only gets software Vulkan (llvmpipe). It is fine for tests, which run
  off-screen under xvfb with no window. Expect about 0.5 s per lockstep frame for the four views
  plus masks.
- **Windows build** (`hifi/tools/bin`, or `--godot path/to/Godot_*_win64_console.exe`). It uses the
  GPU (Forward+, Vulkan) and connects to the WSL sim over localhost. On an RTX 3070, a lockstep
  frame takes about 60 ms.

## Use it like the GymBox envs

```python
import sys; sys.path.insert(0, "hifi")
from wrappers import make_hifi_env, build_hifi_registry, build_hifi_heldout_registry

env = make_hifi_env("snowball", aov=("rgb", "mask"))   # starts Godot
obs, _ = env.reset(seed=0)
top, rear, side, fpv = env.render(camera_pose="Left")  # same shapes as the GymBox wrapper
obs, r, term, trunc, info = env.step(env.action_space.sample())
masks = env.render_masks("cover")                       # per-view binary masks (aov must include "mask")
env.cam_x, env.cam_yaw                                  # same camera state as GymBox
```

- `build_hifi_registry()` and `build_hifi_heldout_registry(obstacles)` mirror `ortho/utils.py`.
  Their factories share one Godot process, which switches worlds per env. They drop into
  `collect_rollouts`-style loops.
- `pair=True` also runs the GymBox `render()` and keeps it in `env.last_lowfi`, for paired data.
- `replay.py` records rollouts as renderer messages without needing Godot, then renders them
  offline. It can shard across several Godot instances.

```bash
python hifi/replay.py record --env tank --episodes 8 --steps 200 --out data/tank.jsonl.gz
python hifi/replay.py render --log data/tank.jsonl.gz --out data/tank_hifi --workers 2 --aov rgb mask
```

## Cameras

Python computes every camera pose in `exporters/<env>.py` from the wrapper's own state, and Godot
applies it.

Each view keeps the GymBox framing:

- what it follows and where that sits in the frame
- the viewing axis
- the px-per-metre scale, so ortho `size` = frame height ÷ px/m
- the FPV orbit: wrapper `cam_pos`, pivot, focal length → vertical FOV, near clip

Because every view is now a real camera, GymBox's painter artefacts are gone:

- mirrored side views
- fake fixed horizons
- the agent always drawn on top
- 2×-stretched or floating elevations
- views pointing the wrong way (Tank's "Rear", Goat's "Rear")
- MultiCar's FPV forward-vector bug (declared in `camera_fix`)

All four views render one consistent world. A tree, building or prop appears in every view whose
frame contains it. The only exception is a cutaway, where an object standing between an ortho
camera and the subject is removed (see `ENV_AUTHORING.md`).

The orthographic views also get a world-anchored backdrop: the HDRI sky above the horizon and a
receding ground below it. Without it, Godot draws an HDRI sky through an ortho camera as streaks.

## Object-ID masks

With `aov=("rgb", "mask")` each view also has a flat-colour mask:

- R + 256·G = instance index
- B = class code
- background and unregistered geometry = 0, so the masks are occlusion-correct

The id and class tables come from the world, available as `renderer.mask_table`. `render_masks(target)`
accepts ids, classes, exporter `mask_groups`, or int obstacle indices, matching the GymBox
`render_masks()` targets for Parkour, Snowball and Rescue. The masks can replace the luminance
threshold in voxel carving.

## Determinism

A lockstep frame depends only on the state it renders:

- no TAA, SDFGI, auto-exposure or particles
- effects are functions of the phase Python sends
- shaders use a global `sim_time`

The test harness checks this: re-rendering the same state gives identical pixels.

## Layout

```
hifi/
├── run_hifi.py      live play / watch          ├── wrappers.py   HiFiOrthographicWrapper + registries
├── test_hifi.py     checks + comparison sheets ├── replay.py     record / offline render
├── renderer.py      launches Godot, lockstep   ├── bridge.py     TCP protocol v2
├── controls.py      key bindings               ├── ENV_AUTHORING.md  how to add or extend an env
├── exporters/       one per env (+ base.py)    ├── assets_manifest.json  CC0 asset list
├── tools/           fetch_assets.py, get_godot.py, check_views.py (cross-view consistency)
└── godot/           project.godot, core/ (session, view rig, masks, look, assets, meshes),
                     shared/ (terrain, vfx), envs/<env>/world.gd, assets/ (fetched, gitignored)
```

## Envs

Training registry (`build_hifi_registry`) and held-out obstacle-transfer envs (`build_hifi_heldout_registry`):

| Env | Hi-fi scene | GymBox camera quirks fixed |
|---|---|---|
| Bipedal Walker | Four-link mech walker on a grassland strip; flag with GymBox colour states | Mirrored side view; rear view cuts terrain behind the walker; FPV up-vector sign |
| Lunar Lander | Apollo-style lander, regolith strip, crater-rim far field, space sky with Earth, engine plumes, dust | (views were already correct) |
| MultiCar Racing | Open-wheel cars, PBR track with kerbs, grandstand, trees, skid marks; `--opponent` adds the autopilot car | FPV forward vector (`camera_fix`) |
| Mario Escape | Original adventurer character, brick platforms, real pits, flagpole, meadow | Mirrored FPV and side view; invisible pits in FPV; west wall drawn over the player |
| Drone Dogfight | Racing quadcopters, turf arena with netting and floodlights, bolt tracers, explosions | Inverted elevation views; FPV without floor and walls |
| Excavator | Tracked hydraulic excavator, posed rams, boulder payload, steel hopper, quarry site | Mirrored side view and FPV |
| Tank Duel | Two MBTs with scrolling tracks, desert arena, concrete cover, tracers, fireballs | "Rear" was a mirrored side profile; 2× vertical stretch; floating slabs grounded at the same top heights |
| Catapult War | Timber mangonel with winch, castle-brick tower, wind banners, meadow | Bounding-box-only top and rear views; fake FPV ground and crosshair |
| Robot Sumo | Wheeled sumo robots with plows, clay dohyo with straw-bale ring, indoor hall | 2× vertical stretch |
| Mountain Goat | Furred mountain goat, alpine ridge with exact snow line, clouds | "Rear" showed the face; mirrored side view |
| Wrecking Ball | Crawler crane with lattice boom, sagging cable from `cable_points()`, concrete pillars, dust | Mirrored side view and FPV |
| Toxic Gas Escape | Concrete maze facility, utility rover, exit gate, hard-edged green gas front | Car always drawn on top; depth fading; gas drawn differently per view |
| Rescue Ambulance *(held out)* | City blocks with facades, ambulance, bus, roadblocks, debris, casualty site | Fake horizon; footprint clipping becomes a real section cut |
| Parkour Runner *(held out)* | Runner-palette mech on an urban course: hurdles, crates, walls, gaps, crests (crest occlusion matches GymBox) | Same as Bipedal |
| Snowball Fight *(held out)* | Kids in winter clothes, snow forts, ice pillars, fence, snowfall | Fake horizon; GymBox cutaway reproduced with real cameras |

`python hifi/test_hifi.py --all` checks every env and writes `hifi/test_out/report.md`. The 8 checks are
shapes, not_blank, masks, deterministic, camera_state, probe, views and godot_errors. `views` checks
cross-view consistency: every object inside a view's frame is drawn there, except cutaways in front
of the subject. `python hifi/tools/check_views.py --env <name>` explains a `views` failure.

Render time per lockstep frame:

| Renderer | Lockstep frame (4 RGB + 4 mask views) |
|---|---|
| Software (llvmpipe, xvfb) | ~0.5-1.8 s |
| RTX 3070 (Windows Godot 4.3) | 45-120 ms |
