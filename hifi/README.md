# HiFi — high-fidelity twins of the GymBox envs

GymBox keeps the physics; a game engine (Godot) only renders it. Python runs the env at its
native rate and streams state over TCP; the Godot project draws it and sends player input back.

```
hifi/play_hifi.py ── track + state (JSON lines, :9876) ──► Godot (hifi/multicar_racing)
       ▲                                                       │
       └──────────── input {steer, gas, brake} / reset ◄───────┘
```

## Layout

```
hifi/
├── play_hifi.py        # entry point: runs the sim, streams to Godot
├── godot_bridge.py     # TCP server + state export (protocol documented at the top)
├── multicar_env.py     # two-car CarRacing-v3 with an autopilot opponent
└── multicar_racing/    # Godot 4.3+ project (renderer)
    ├── main.gd         # environment, cameras, input, HUD, interpolation, --screenshot mode
    ├── net_client.gd   # TCP client (auto-reconnect)
    ├── track_builder.gd# road/kerb meshes from the track message, seeded scenery
    ├── car_visual.gd   # procedural car; cosmetic roll/pitch, wheel spin/steer, smoke, brake light
    └── skid_marks.gd   # shared skid-mark ring buffer
```

## Run

1. Install Godot 4.3+ (standard build, not .NET).
2. From the repo root: `python hifi/play_hifi.py`
   - `--lowfi top` / `--lowfi grid`: also show the GymBox render(s) side by side
   - `--ego autopilot`: watch both cars race
   - `--seed N`: fixed track
   - `--host 0.0.0.0`: accept Godot from another machine or VM (e.g. Windows Godot -> WSL sim)
3. Launch the game: `godot --path hifi/multicar_racing` (or open the folder in Godot and press F5).
   Under WSL add `--rendering-driver opengl3`. To connect to a remote sim: `... -- --host=<ip>`.

Controls: W/S or Up/Down gas/brake · A/D or Left/Right steer · gamepad (left stick, RT, LT) ·
1-5 or C camera · R new episode · F1 toggle SDFGI · Esc quit.

## Cameras

| Key | Camera | Matches |
|---|---|---|
| 1 | Chase | (hi-fi only) |
| 2 | GymBox 3D FPV | `MultiCarOrthographicWrapper` FPV: 12 behind, 8 up, 90° horizontal FOV |
| 3 | GymBox Top (ortho) | CarRacing top-down: 16.2 px/unit, car-up |
| 4 | GymBox Rear (ortho) | wrapper rear view: 25 px/unit, culls > 9.5 units behind the car |
| 5 | GymBox Side (ortho) | wrapper side view: 25 px/unit, from the car's right |

The ortho rear/side cameras hide the scenery layer (trees, stands, gantry) because the GymBox
views contain only ground, road and cars.

Screenshot every camera after N seconds, then quit:
`godot --path hifi/multicar_racing -- --screenshot=/tmp/shot --shot-delay=4`
