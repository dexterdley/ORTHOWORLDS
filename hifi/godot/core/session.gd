extends Node
## OrthoWorlds hi-fi renderer session (main scene).
## Python (hifi/renderer.py + hifi/bridge.py) owns the GymBox physics and every camera pose; this
## renders what it is sent. "hello" selects the env world (res://envs/<env>/world.gd) and the
## views, "scene" rebuilds the world, "state" moves it, "render" captures the four views
## (Top, Rear, Side, 3D FPV) plus optional object-ID masks and sends them back.
##
## Live mode (python hifi/run_hifi.py): states arrive at the env rate and are interpolated;
## keys go back to Python.
##   game keys      as printed by run_hifi.py      arrows  orbit the 3D FPV camera
##   1-4            Top / Rear / Side / FPV         5      hi-fi chase camera
##   0, G or Tab    2x2 grid of all four views      M      show object-ID masks
##   F5/Backspace   new episode                     F1     toggle SDFGI      Esc  quit

const NetClient := preload("res://core/net_client.gd")
const ViewRig := preload("res://core/view_rig.gd")
const VIEW_TITLES := ["Top", "Rear", "Side", "3D FPV", "Chase (hi-fi)"]

var net: Node
var rig: Node
var world: EnvWorld
var look: Look
var env_name := ""
var mode := ""            # "live" | "lockstep", set by hello
var fps := 50.0
var aovs: Array = ["rgb"]
var view_size := Vector2i(600, 400)
var prev_state := {}
var cur_state := {}
var since := 0.0
var has_scene := false
var layout := -1          # -1 grid, 0..3 single view, 4 chase
var show_mask := false
var _chase_xf := Transform3D()
var _chase_init := false

var _ui: Control
var _grid: GridContainer
var _cells: Array[TextureRect] = []
var _labels: Array[Label] = []
var _single: TextureRect
var _hud: Label


func _ready() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	DisplayServer.window_set_title("OrthoWorlds HiFi")
	net = NetClient.new()
	add_child(net)
	net.message.connect(_on_message)
	rig = ViewRig.new()
	rig.name = "Views"
	add_child(rig)
	_build_ui()


# ================================================================ protocol

func _on_message(msg: Dictionary) -> void:
	match String(msg.get("type", "")):
		"hello":
			_on_hello(msg)
		"scene":
			_on_scene(msg)
		"state":
			_on_state(msg)
		"render":
			_on_render(msg)
		"probe":
			net.send({"type": "probe", "id": msg.get("id", 0), "pixels": rig.probe(msg.get("points", []))})
		"inspect":
			net.send({"type": "inspect", "id": msg.get("id", 0),
				"nodes": world.inspect(int(msg.get("max_instances", 4000))) if world != null else []})
		"reveal":
			if world != null:
				world.reveal(bool(msg.get("on", false)))
			net.send({"type": "revealed", "id": msg.get("id", 0)})
		"quit":
			_quit()


func _on_hello(msg: Dictionary) -> void:
	mode = String(msg.get("mode", "live"))
	fps = float(msg.get("fps", 50.0))
	aovs = msg.get("aov", ["rgb"])
	var sz: Array = msg.get("size", [600, 400])
	view_size = Vector2i(int(sz[0]), int(sz[1]))
	var new_env := String(msg.get("env", ""))
	if new_env != env_name or world == null:
		if world != null:
			world.free()
			world = null
		var path := "res://envs/%s/world.gd" % new_env
		var scr: Script = load(path) if ResourceLoader.exists(path) else null
		if scr == null:
			net.send({"type": "error", "message": "no Godot world for env '%s' (%s)" % [new_env, path]})
			env_name = ""
			return
		world = scr.new()
		world.name = new_env
		add_child(world)
		look = Look.new()
		look.name = "Look"
		world.add_child(look)
		world.look = look
		world.setup_look(look)
		env_name = new_env
		DisplayServer.window_set_title("OrthoWorlds HiFi - " + env_name)
	rig.setup(view_size, aovs.has("mask"), mode == "lockstep")
	rig.set_environments(look.env, look.env_ortho, look.env_mask)
	rig.set_sky_card(look.card_material())
	rig.set_live_updates(mode == "live", layout == 4)
	_layout_ui()
	prev_state = {}
	cur_state = {}
	has_scene = false


func _on_scene(msg: Dictionary) -> void:
	if world == null:
		net.send({"type": "error", "message": "scene before a valid hello"})
		return
	world.begin_scene(msg)
	has_scene = true
	prev_state = {}
	cur_state = {}
	_chase_init = false
	net.send({"type": "ready", "env": env_name,
		"renderer": str(ProjectSettings.get_setting("rendering/renderer/rendering_method")),
		"device": RenderingServer.get_video_adapter_name(), "mask": world.mask_table()})


func _on_state(msg: Dictionary) -> void:
	if world == null or not has_scene:
		return
	prev_state = msg if cur_state.is_empty() else cur_state
	cur_state = msg
	since = 0.0
	if mode == "lockstep":
		_apply(cur_state, cur_state, 1.0)


func _on_render(msg: Dictionary) -> void:
	if world == null or cur_state.is_empty():
		net.send({"type": "error", "message": "render requested before scene/state"})
		return
	var res: Dictionary = await rig.capture(aovs)
	var header := {"type": "frames", "id": msg.get("id", 0), "w": view_size.x, "h": view_size.y, "items": res["items"]}
	if aovs.has("mask"):
		header["mask"] = world.mask_table()   # classes first used by transient objects show up here
	net.send_with_payload(header, res["chunks"])


func _quit() -> void:
	for c in _cells:
		c.texture = null
	_single.texture = null
	if world != null:
		world.free()
		world = null
	for c in rig.get_children():
		c.free()
	Assets.clear_cache()
	Vfx.clear_cache()
	get_tree().quit()


func _apply(a: Dictionary, b: Dictionary, alpha: float) -> void:
	world.apply_state(a, b, alpha)
	rig.apply_cameras(a.get("cams", []), b.get("cams", []), alpha)


# ================================================================ live mode

func _process(delta: float) -> void:
	if mode == "live" and world != null and not cur_state.is_empty():
		since += delta
		_apply(prev_state, cur_state, clampf(since * fps, 0.0, 1.0))
		if layout == 4:
			_update_chase(delta)
	if mode == "live":
		_send_input()
	_update_hud()


func _update_chase(delta: float) -> void:
	var target: Transform3D = world.chase_pose()
	if not _chase_init:
		_chase_xf = target
		_chase_init = true
	else:
		var k := 1.0 - exp(-delta * 4.0)
		_chase_xf.origin = _chase_xf.origin.lerp(target.origin, k)
		_chase_xf.basis = Basis(Quaternion(_chase_xf.basis).slerp(Quaternion(target.basis), k))
	rig.chase_cam.global_transform = _chase_xf


func _send_input() -> void:
	if not net.is_connected_to_sim():
		return
	var keys: Array = []
	for k in range(KEY_A, KEY_Z + 1):
		if Input.is_physical_key_pressed(k):
			keys.append(char(k).to_lower())
	if Input.is_physical_key_pressed(KEY_SPACE):
		keys.append("space")
	var cam := "Center"
	if Input.is_physical_key_pressed(KEY_LEFT):
		cam = "Left"
	elif Input.is_physical_key_pressed(KEY_RIGHT):
		cam = "Right"
	elif Input.is_physical_key_pressed(KEY_UP):
		cam = "Up"
	elif Input.is_physical_key_pressed(KEY_DOWN):
		cam = "Down"
	net.send({"type": "input", "keys": keys, "cam": cam})


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	if mode != "live":
		return  # lockstep (data rendering) ignores the keyboard, e.g. stray keys into a new window
	var key: Key = event.physical_keycode
	match key:
		KEY_ESCAPE:
			net.send({"type": "bye"})
			_quit()
		KEY_F5, KEY_BACKSPACE:
			net.send({"type": "reset"})
		KEY_0, KEY_G, KEY_TAB:
			layout = -1
		KEY_1, KEY_2, KEY_3, KEY_4, KEY_5:
			layout = key - KEY_1
		KEY_M:
			show_mask = not show_mask and aovs.has("mask")
		KEY_F1:
			if look != null and look.env != null:
				look.env.sdfgi_enabled = not look.env.sdfgi_enabled
	if mode == "live":
		rig.set_live_updates(true, layout == 4)
	_layout_ui()


# ================================================================ UI

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	_ui = Control.new()
	_ui.set_anchors_preset(Control.PRESET_FULL_RECT)
	_ui.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(_ui)
	var bg := ColorRect.new()
	bg.color = Color(0.05, 0.05, 0.06)
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	_ui.add_child(bg)
	_grid = GridContainer.new()
	_grid.columns = 2
	_grid.set_anchors_preset(Control.PRESET_FULL_RECT)
	_grid.add_theme_constant_override("h_separation", 2)
	_grid.add_theme_constant_override("v_separation", 2)
	_ui.add_child(_grid)
	for i in 4:
		var cell := TextureRect.new()
		cell.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		cell.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		cell.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		cell.size_flags_vertical = Control.SIZE_EXPAND_FILL
		_grid.add_child(cell)
		_cells.append(cell)
		var lbl := _label(14)
		lbl.text = VIEW_TITLES[i]
		cell.add_child(lbl)
		lbl.position = Vector2(6, 4)
		_labels.append(lbl)
	_single = TextureRect.new()
	_single.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_single.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_single.set_anchors_preset(Control.PRESET_FULL_RECT)
	_single.visible = false
	_ui.add_child(_single)
	_hud = _label(15)
	_hud.position = Vector2(10, 8)
	_ui.add_child(_hud)


func _label(size: int) -> Label:
	var l := Label.new()
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_outline_color", Color.BLACK)
	l.add_theme_constant_override("outline_size", 5)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return l


func _layout_ui() -> void:
	if rig.rgb.is_empty():
		return
	for i in 4:
		_cells[i].texture = rig.mask_texture(i) if show_mask else rig.texture(i)
	_grid.visible = layout < 0
	_single.visible = layout >= 0
	if layout == 4:
		_single.texture = rig.chase_vp.get_texture()
	elif layout >= 0:
		_single.texture = rig.mask_texture(layout) if show_mask else rig.texture(layout)


func _update_hud() -> void:
	if not net.is_connected_to_sim():
		_hud.text = "Waiting for the GymBox sim on %s:%d ...\nRun:  python hifi/run_hifi.py --env <name>" % [net.host, net.port]
		return
	if world == null or cur_state.is_empty():
		_hud.text = "Connected (%s), waiting for a scene ..." % mode
		return
	if mode != "live":
		_hud.text = "%s  [lockstep]  step %d" % [env_name, int(cur_state.get("step", 0))]
		return
	var h: Dictionary = cur_state.get("hud", {})
	var view: String = "Grid" if layout < 0 else VIEW_TITLES[layout]
	_hud.text = "%s   step %d   reward %+.2f   return %+.2f\nview [1-5, 0]: %s   camera [arrows]: %s%s\n%s" % [
		env_name, int(cur_state.get("step", 0)), float(h.get("reward", 0.0)), float(h.get("ret", 0.0)), view,
		String(h.get("cam", "Center")), "   masks" if show_mask else "", String(h.get("text", ""))]
