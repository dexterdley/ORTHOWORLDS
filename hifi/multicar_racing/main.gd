extends Node3D
## OrthoWorlds hi-fi renderer for MultiCar Racing.
## GymBox (hifi/play_hifi.py) owns the physics at 50 Hz; this scene renders the streamed
## state, interpolating between the last two snapshots, and sends player input back.
##
## Keys: W/S or Up/Down gas/brake, A/D or Left/Right steer (gamepad: left stick, RT, LT)
##       1-5 or C camera, R new episode, F1 toggle SDFGI, Esc quit
##
## Screenshot mode (user args): godot --path . -- --screenshot=C:/tmp/shot --shot-delay=4
## saves C:/tmp/shot_<camera>.png for every camera after the delay, then quits.

const NetClient := preload("res://net_client.gd")
const TrackBuilder := preload("res://track_builder.gd")
const CarVisual := preload("res://car_visual.gd")
const SkidMarks := preload("res://skid_marks.gd")

const STEP_DT := 1.0 / 50.0
const KEY_STEER_RATE := 5.0  # keyboard steering ramps instead of snapping to +-1

enum Cam { CHASE, FPV, TOP, REAR, SIDE }
const CAM_NAMES := ["Chase", "GymBox 3D FPV", "GymBox Top (ortho)", "GymBox Rear (ortho)", "GymBox Side (ortho)"]

var net: Node
var track: Node3D
var skids: MultiMeshInstance3D
var camera: Camera3D
var env: Environment
var hud: Label
var cars := {}  # name -> CarVisual
var prev_state := {}
var cur_state := {}
var since_state := 0.0
var cam_mode: int = Cam.CHASE
var chase_yaw := 0.0
var key_steer := 0.0
var shot_prefix := ""
var shot_delay := 4.0
var shot_taken := false


func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--screenshot="):
			shot_prefix = arg.get_slice("=", 1)
		elif arg.begins_with("--shot-delay="):
			shot_delay = float(arg.get_slice("=", 1))
	_setup_environment()
	track = TrackBuilder.new()
	add_child(track)
	skids = SkidMarks.new()
	add_child(skids)
	camera = Camera3D.new()
	add_child(camera)
	camera.make_current()
	_apply_cam_mode()
	_setup_hud()
	net = NetClient.new()
	add_child(net)
	net.track_received.connect(_on_track)
	net.state_received.connect(_on_state)


func _setup_environment() -> void:
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.2, 0.4, 0.75)
	sky_mat.sky_horizon_color = Color(0.66, 0.74, 0.84)
	# Green "ground" half of the sky: the ortho side/rear views see the ground plane edge-on,
	# and GymBox fills that half with the grass colour too
	sky_mat.ground_horizon_color = Color(0.45, 0.56, 0.38)
	sky_mat.ground_bottom_color = Color(0.3, 0.42, 0.24)
	var sky := Sky.new()
	sky.sky_material = sky_mat

	env = Environment.new()
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_white = 6.0
	env.glow_enabled = true
	env.glow_intensity = 0.5
	env.glow_bloom = 0.05
	env.ssao_enabled = true
	env.ssr_enabled = true
	env.fog_enabled = true
	env.fog_light_color = Color(0.72, 0.79, 0.88)
	env.fog_density = 0.0012
	env.fog_aerial_perspective = 0.6
	env.fog_sky_affect = 0.3
	env.adjustment_enabled = true
	env.adjustment_saturation = 1.1
	env.adjustment_contrast = 1.05
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	add_child(world_env)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-40, -35, 0)
	sun.light_energy = 1.4
	sun.light_color = Color(1.0, 0.96, 0.9)
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 250.0
	add_child(sun)


func _setup_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	hud.position = Vector2(18, 14)
	hud.add_theme_font_size_override("font_size", 20)
	hud.add_theme_color_override("font_outline_color", Color.BLACK)
	hud.add_theme_constant_override("outline_size", 6)
	layer.add_child(hud)


# ---------------------------------------------------------------- network

func _on_track(msg: Dictionary) -> void:
	track.build(msg)
	skids.clear()
	for c in cars.values():
		c.queue_free()
	cars.clear()
	prev_state = {}
	cur_state = {}


func _on_state(msg: Dictionary) -> void:
	prev_state = msg if cur_state.is_empty() else cur_state
	cur_state = msg
	since_state = 0.0
	for c in msg["cars"]:
		if not cars.has(c["name"]):
			var v = CarVisual.new()
			add_child(v)
			var rgb: Array = c["color"]
			v.setup(Color(rgb[0], rgb[1], rgb[2]), skids)
			cars[c["name"]] = v
			if c["name"] == "ego":
				chase_yaw = c["a"]


func _send_input(delta: float) -> void:
	var target := 0.0
	if Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT):
		target -= 1.0
	if Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT):
		target += 1.0
	key_steer = move_toward(key_steer, target, KEY_STEER_RATE * delta)
	var steer := key_steer
	var gas := 1.0 if Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP) else 0.0
	var brake := 0.8 if Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN) else 0.0

	var jx := Input.get_joy_axis(0, JOY_AXIS_LEFT_X)
	if absf(jx) > 0.12:
		steer = jx
	var rt := Input.get_joy_axis(0, JOY_AXIS_TRIGGER_RIGHT)
	if rt > 0.05:
		gas = rt
	var lt := Input.get_joy_axis(0, JOY_AXIS_TRIGGER_LEFT)
	if lt > 0.05:
		brake = lt * 0.8
	net.send({"type": "input", "steer": steer, "gas": gas, "brake": brake})


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	var key: Key = event.physical_keycode
	if key == KEY_ESCAPE:
		get_tree().quit()
	elif key == KEY_R:
		net.send({"type": "reset"})
	elif key == KEY_C:
		cam_mode = (cam_mode + 1) % CAM_NAMES.size()
		_apply_cam_mode()
	elif key >= KEY_1 and key <= KEY_5:
		cam_mode = key - KEY_1
		_apply_cam_mode()
	elif key == KEY_F1:
		env.sdfgi_enabled = not env.sdfgi_enabled


# ---------------------------------------------------------------- frame

func _process(delta: float) -> void:
	_send_input(delta)
	if cur_state.is_empty():
		hud.text = "Waiting for GymBox sim on %s:%d ...\nRun:  python hifi/play_hifi.py" % [net.host, net.port]
		return
	since_state += delta
	var t := clampf(since_state / STEP_DT, 0.0, 1.0)
	var prev_cars := {}
	for c in prev_state["cars"]:
		prev_cars[c["name"]] = c
	for c in cur_state["cars"]:
		if cars.has(c["name"]):
			cars[c["name"]].apply_state(prev_cars.get(c["name"], c), c, t, delta)
	_update_camera(delta)
	_update_hud()
	if shot_prefix != "" and not shot_taken:
		shot_delay -= delta
		if shot_delay <= 0.0:
			shot_taken = true
			_save_screenshots()


func _save_screenshots() -> void:
	## Renders every camera mode for the current frame and saves <prefix>_<mode>.png.
	for mode in CAM_NAMES.size():
		cam_mode = mode
		_apply_cam_mode()
		_update_camera(0.0)
		await RenderingServer.frame_post_draw
		var path := "%s_%d_%s.png" % [shot_prefix, mode + 1, CAM_NAMES[mode].get_slice(" (", 0).replace(" ", "_")]
		get_viewport().get_texture().get_image().save_png(path)
		print("saved ", path)
	get_tree().quit()


func _apply_cam_mode() -> void:
	camera.near = 0.05
	camera.far = 3000.0
	camera.cull_mask = 0xFFFFF
	if cam_mode == Cam.CHASE or cam_mode == Cam.FPV:
		camera.projection = Camera3D.PROJECTION_PERSPECTIVE
		camera.keep_aspect = Camera3D.KEEP_HEIGHT if cam_mode == Cam.CHASE else Camera3D.KEEP_WIDTH
		camera.fov = 65.0 if cam_mode == Cam.CHASE else 90.0  # GymBox FPV: focal 500 px on a 1000 px width
		return
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	if cam_mode == Cam.TOP:
		camera.size = 49.4  # CarRacing zoom 16.2 px/unit on an 800 px tall frame
	else:
		camera.size = 32.0  # GymBox ortho scale 25 px/unit on an 800 px tall frame
		camera.cull_mask = 1  # GymBox side/rear views contain only ground, road and cars
	if cam_mode == Cam.REAR:
		camera.near = 20.5  # GymBox drops polygons more than 9.5 units behind the car


func _update_camera(delta: float) -> void:
	var ego = cars.get("ego")
	if ego == null:
		return
	var pos: Vector3 = ego.position
	var yaw: float = ego.yaw
	var fwd := Vector3(-sin(yaw), 0.0, -cos(yaw))
	var right := Vector3(cos(yaw), 0.0, -sin(yaw))
	match cam_mode:
		Cam.CHASE:
			chase_yaw = lerp_angle(chase_yaw, yaw, 1.0 - exp(-delta * 5.0))
			var cfwd := Vector3(-sin(chase_yaw), 0.0, -cos(chase_yaw))
			camera.fov = 65.0 + clampf(ego.speed * 0.12, 0.0, 15.0)
			camera.global_position = pos - cfwd * 10.0 + Vector3.UP * 3.6
			camera.look_at(pos + cfwd * 6.0 + Vector3.UP * 1.0)
		Cam.FPV:  # MultiCarOrthographicWrapper.cam_pos with zero orbit: 12 behind, 8 up, level
			camera.global_position = pos - fwd * 12.0 + Vector3.UP * 8.0
			camera.look_at(camera.global_position + fwd)
		Cam.TOP:  # car-up, car placed a quarter of the frame above the bottom edge
			var center := pos + fwd * 12.35
			camera.global_position = center + Vector3.UP * 150.0
			camera.look_at(center, fwd)
		Cam.REAR:
			camera.global_position = pos - fwd * 30.0 + Vector3.UP * 8.0
			camera.look_at(camera.global_position + fwd)
		Cam.SIDE:
			camera.global_position = pos + right * 60.0 + Vector3.UP * 8.0
			camera.look_at(camera.global_position - right)


func _update_hud() -> void:
	var ego := {}
	var opp := {}
	for c in cur_state["cars"]:
		if c["name"] == "ego":
			ego = c
		else:
			opp = c
	if ego.is_empty():
		return
	var kmh := Vector2(ego["vx"], ego["vy"]).length() * 3.6
	var place := 1
	if not opp.is_empty() and float(opp["progress"]) > float(ego["progress"]):
		place = 2
	hud.text = "P%d / %d     %3d km/h\nLap %3d%%     return %+.1f     step %d\nCamera [1-5/C]: %s\n[R] reset   [F1] GI %s" % [
		place, cur_state["cars"].size(), int(kmh),
		int(float(cur_state["lap"]) * 100.0), float(cur_state["ret"]), int(cur_state["step"]),
		CAM_NAMES[cam_mode], "on" if env.sdfgi_enabled else "off",
	]
