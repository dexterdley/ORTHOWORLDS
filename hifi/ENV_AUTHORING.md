# Writing a hi-fi env

A hi-fi env is two files plus an optional asset list. Each one is a twin of a GymBox env in
`ortho/game_envs`. Python keeps the physics, Godot only renders.

| File | What it does |
|---|---|
| `hifi/exporters/<env>.py` | Turns the GymBox wrapper into messages: `scene` (once per reset), `state` (every step), and the four cameras. |
| `hifi/godot/envs/<env>/world.gd` | `extends EnvWorld`. Builds the photoreal scene and moves it. |
| `hifi/godot/envs/<env>/assets.json` | Optional extra CC0 Poly Haven assets, in the same layout as `hifi/assets_manifest.json`. |

The reference implementation is `lunar_lander`: `hifi/exporters/lunar_lander.py` and
`hifi/godot/envs/lunar_lander/world.gd`.

## Contract

### Exporter (`hifi/exporters/base.py`)

```python
class XExporter(Exporter):
    name = "x"                 # folder in hifi/godot/envs/
    title = "X"                # display name from ortho/utils.py
    view_size = (800, 600)     # GymBox frame (w, h); render() returns exactly this shape
    render_size = None         # set only if GymBox resizes anisotropically (Godot renders this, then resizes)
    heldout = False            # True for Rescue Ambulance, Parkour Runner, Snowball Fight; the class
                               # then accepts obstacles=True/False

    def make_env(self): ...            # build the GymBox orthographic wrapper exactly as ortho/utils.py does
    def scene(self, env): ...          # dict: static or procedural content (terrain, obstacles, ...)
    def state(self, env): ...          # {"xf": {id: xf}, "dyn": [...], "p": {...}, "vis": {...}}
    def cameras(self, env): ...        # [cam_ortho("top"...), cam_ortho("rear"...), cam_ortho("side"...), cam_persp("fpv"...)]
    def advance_camera(self, env, cmd) # default: env.update_camera_pose(cmd). Override if the wrapper's
                                       # render() changes more camera state before drawing (e.g. Mario's follow_y)
    def on_step(self, env, action)     # optional: keep what state() needs but the env does not store
```

**Coordinates.** Godot uses metres with +Y up. Every conversion happens in Python.

- **Side-view envs.** Box2D `(x, y)` maps to Godot `(x, y)`, and the depth is `z`. Use `xf_side(x, y, angle, z)`.
- **Top-down envs.** Box2D `(x, y)` at height `h` maps to Godot `(x, h, -y)`, and the yaw is the Box2D angle.
  Use `xf_top(x, y, angle, h)`.
- A world builds each object in its body-local frame:
  - side-view: local `(lx, ly)` becomes `(lx, ly, z)`
  - top-down: local `(lx, ly)` becomes `(lx, h, -ly)`

  `M.prism()` and `M.prism_up()` take Box2D polygons directly.
- Wrapper camera values are sometimes in pixels (SCALE px/m). Divide them out.

**Messages.**

- `state["xf"]` moves the nodes the world registered under the same ids.
- `state["dyn"]` holds transient objects: `{"id", "k": kind, "xf": [...], "mask": class (optional), ...}`.
  The world spawns one in `spawn_dyn(kind, d)` when its id first appears and frees it when the id
  disappears. Give effects their phase (`age / life`) so they render deterministically.
- `state["vis"]` is `{view: [ids hidden in that view]}`. Use it for GymBox section cuts.
- `state["p"]` carries anything else: flags, flashes, the gas front, and so on.

### Cameras (decided policy)

Each view keeps the GymBox wrapper's framing:

- what it follows and where that sits in the frame
- the viewing axis
- the scale: ortho `size` = frame height in px ÷ GymBox px per metre

It is a real camera, so GymBox artefacts disappear:

- mirrored painter order
- fake fixed horizons
- the agent always drawn on top
- floating or 2×-stretched elevation views
- views that were obviously meant the other way round, such as a "Rear" that shows the face

Every GymBox frame row must land on the same world point as the Godot frame row. Check this in the
test sheets.

**FPV.**

- Position: the wrapper's own `cam_pos`, unit-converted. Its yaw and pitch come from
  `env.cam_yaw` / `env.cam_pitch`, which `advance_camera` updates.
- Look-at: the wrapper's orbit pivot (where its optical axis points).
- `vfov_from_focal(focal_px, frame_h_px)`.
- `near`: the wrapper's near clip.
- Never re-implement the orbit in GDScript.

**Ortho views.** Keep `near`/`far` as tight as the scenery allows, so the directional shadow
cascades stay sharp. But `far` must reach every object inside the frame, and the ortho sky card
sits at 0.97·`far`. Keep the camera outside all geometry.

**Scenery: one consistent world.** All four views render the same world. An object (tree, prop,
building, fence, ridge, skyline, ...) appears in every view whose frame contains it, even though
ortho views show distant scenery at full size. Its shadows stay consistent too.

The only exception is a **cutaway**: an object that lies wholly between a camera and the subject
(the camera's `look` point) may be removed from that one view, so the subject stays visible.
There are two ways to make one:

- Hide the front group in that view with `set_views` or `state["vis"]`. To do this, split
  scattered scenery into front and back groups.
- For cameras that follow the subject, use a section plane: set `near` to the camera-to-subject
  distance minus the subject's half-extent and a small clearance.

Geometry that exists only to draw a section's cut face in one view is marked
`set_meta("section_cap", true)`. `hifi/tools/check_views.py` and the `views` test check all of
this.

**Ortho backdrop.** Ortho views get a world-anchored backdrop automatically: the HDRI sky above the
horizon, and a receding ground below it. Tune it with these `Look.configure` keys:

| Key | What it sets |
|---|---|
| `horizon_y` | World height of the ground |
| `card_ground` | Ground texture name |
| `card_ground_tint` | Ground tint |
| `card_ground_scale` | Ground texture scale |
| `card_eye` | Virtual eye height |
| `ortho_bg` | Flat background colour, sRGB (e.g. indoor scenes or space) |
| `ortho_fog` | Depth fog in ortho views; off by default, because ortho camera distances are arbitrary |

### World (`hifi/godot/core/env_world.gd`)

```gdscript
extends EnvWorld
func setup_look(lk: Look) -> void: lk.configure({...})      # sky (Poly Haven HDRI name), sun, fog, exposure: see core/look.gd
func build_scene(scene: Dictionary) -> void: ...            # build under `content`; register(id, node, mask_class)
func on_state(state, prev, alpha) -> void: ...              # flags, effects, wheel spin, ...
func spawn_dyn(kind, d) -> Node3D: ...                      # transient objects
func update_dyn(node, a, b, alpha) -> void: ...             # e.g. effect phase
func chase_pose() -> Transform3D: ...                       # live-mode hi-fi chase camera
```

Helpers you will use:

| Helper | What it gives you |
|---|---|
| `Assets.pbr(name, {"uv", "tint", "color", "metallic", "roughness", "world"})` | PBR texture sets |
| `Assets.fitted_model(name, size)` | Poly Haven models scaled to a box, base on y = 0 |
| `M.*` (`core/meshes.gd`) | Bevelled boxes, prisms, lathes, tubes, wheels, materials, emissive, glass, paint; `M.tri` / `M.quad` / `M.quad_uv` for custom SurfaceTool meshes |
| `Terrain.*` (`shared/terrain.gd`) | Side-view terrain strips (`get_meta("profile").height(x, z)`), ground slabs, scatter, procedural trees |
| `Vfx.*` (`shared/vfx.gd`) | Explosions, puffs, flames, glowing capsules, all driven by phase |

`hifi/godot/assets/polyhaven/index.json` lists the assets that are available locally, including
`model_tris`. `Assets.model_tris(name)` returns a model's triangle count.

### Rules

- **Determinism.** No unseeded randomness. No wall-clock time: shaders use `global uniform float sim_time`,
  scripts use `sim_t`. No particle systems.
- **Masks.** Register each moving object with a semantic class:
  - Bipedal and Parkour: `"walker"`
  - Rescue: the obstacle kind
  - Snowball: `"player"` / `"opponent"` / `"cover"`

  Mask twins are made when you call `register()`. Geometry added to a node afterwards becomes a black
  occluder unless you call `refresh_mask(id)`. Mask twins of alpha-cut materials keep their cut-out
  shape. Label3D text is masked too. Mask classes of transient objects appear in the table that
  comes with each frame. You can also declare them in `build_scene` with `class_code("name")`.
  Use classes that match what the env's GymBox `render_masks()` accepts. Unions like `"obstacle"` go in
  the exporter's `mask_groups = {"obstacle": ["hurdle", "crate", ...]}`. Integer targets map to the id
  `"obs<i>"`, so register obstacle i as `"obs<i>"`. Unregistered geometry
  becomes a black occluder automatically. Effects and transparent fog carry `set_meta("no_mask", true)`.
- **Occlusion semantics.** Where the env's experiments depend on what hides what, keep the GymBox
  heights and camera geometry exact. This covers Parkour crests and walls, and Snowball fort and pillar heights.
- **Budget.** Keep a scene under roughly 300k triangles, and remember the 4 RGB views plus 4 mask
  views each draw it.
  - Many Poly Haven models are 30-70k triangles: for example `street_lamp_01`, `concrete_road_barrier`
    and `boulder_01`. Use them for a few hero props only. Repeated props should be procedural
    (`M.*`) or one MultiMesh.
  - Software rendering (xvfb) is the bottleneck while you iterate. Aim for under ~1.5 s per
    lockstep frame there; the GPU build is about 10× faster.
- **Scope.**
  - No `class_name` in env scripts. Load helpers with `preload("res://envs/<env>/x.gd")`.
  - Do not edit `core/`, `shared/` or other envs. Report a core issue rather than patching it.

## Testing

```bash
# from the repo root, GAME conda env; software Vulkan under xvfb, no window
/home/dexterneo/anaconda3/envs/GAME/bin/python hifi/test_hifi.py --env <name>
```

Every check must pass: shapes, not_blank, masks, deterministic, camera_state, probe, views, godot_errors.

When `views` fails, run `python hifi/tools/check_views.py --env <name>`:

- `hifi/test_out/<env>/views_report.md` lists each offending geometry, the view, and the reason:
  - `layer`: set_views / state vis
  - `near` / `far`: clip planes
  - `card`: behind the ortho sky card
  - `range`: visibility_range_end
- `views_<step>.png` shows what per-view hiding removes. Its rows are GymBox, Godot, Godot with
  everything forced visible, and the forced-visible pixels in magenta. Only cutaways in front of
  the subject may remain magenta.

Then read `hifi/test_out/<env>/sheet_*.png` (rows: GymBox, Godot, mask) and `hifi_*.png`
(native-resolution Godot views). Iterate until the views frame the same content as GymBox and
look photoreal: PBR materials, real sky, soft shadows, no floating or interpenetrating parts,
readable silhouettes at 96×96.

Godot logs are in `hifi/test_out/logs/`.
