extends Node
## The four OrthoWorlds views (Top, Rear, Side, 3D FPV), each a SubViewport + Camera3D sharing the
## main World3D, plus optional flat object-ID mask viewports and a hi-fi-only chase view (live).
##
## Render layers (VisualInstance3D.layers bits, 0-based):
##   0      everything by default: seen by all four RGB views and the chase view
##   1..4   per-view RGB visibility for objects hidden from some views (EnvWorld.set_hidden)
##   5      chase camera (live only)
##   11..14 flat-colour mask duplicates, seen only by the mask cameras (EnvWorld.register)
## Python poses every camera ("cams" in each state message); nothing here computes a pose.

const VIEWS := ["top", "rear", "side", "fpv"]
const CHASE_BIT := 5

var size := Vector2i(600, 400)
var want_mask := false
var rgb: Array[SubViewport] = []
var cams: Array[Camera3D] = []
var mask_vps: Array[SubViewport] = []
var mask_cams: Array[Camera3D] = []
var chase_vp: SubViewport
var chase_cam: Camera3D
var _env_persp: Environment
var _env_ortho: Environment
var _env_mask: Environment
var _lockstep := true
var _cards: Array[MeshInstance3D] = []


func setup(view_size: Vector2i, with_mask: bool, lockstep: bool) -> void:
	for c in get_children():
		c.queue_free()
	rgb.clear()
	cams.clear()
	_cards.clear()
	mask_vps.clear()
	mask_cams.clear()
	size = view_size
	want_mask = with_mask
	_lockstep = lockstep
	for i in VIEWS.size():
		var vp := _make_vp(VIEWS[i], true)
		var cam := Camera3D.new()
		cam.cull_mask = 1 | (1 << (i + 1))
		vp.add_child(cam)
		cam.current = true
		rgb.append(vp)
		cams.append(cam)
		if with_mask:
			var mvp := _make_vp(VIEWS[i] + "_mask", false)
			var mcam := Camera3D.new()
			mcam.cull_mask = 1 << (11 + i)
			mvp.add_child(mcam)
			mcam.current = true
			mask_vps.append(mvp)
			mask_cams.append(mcam)
	chase_vp = _make_vp("chase", true)
	chase_vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	chase_cam = Camera3D.new()
	chase_cam.cull_mask = 1 | (1 << CHASE_BIT)
	chase_cam.fov = 62.0
	chase_cam.far = 4000.0
	chase_vp.add_child(chase_cam)
	chase_cam.current = true
	_apply_envs()


func _make_vp(vp_name: String, shaded: bool) -> SubViewport:
	var vp := SubViewport.new()
	vp.name = vp_name
	vp.size = size
	vp.own_world_3d = false
	vp.transparent_bg = false
	vp.handle_input_locally = false
	vp.gui_disable_input = true
	if shaded:
		vp.msaa_3d = Viewport.MSAA_4X
		vp.use_debanding = true
		vp.positional_shadow_atlas_size = 2048
	else:
		# Exact flat colours: no MSAA blending, no dithering, no AA
		vp.msaa_3d = Viewport.MSAA_DISABLED
		vp.use_debanding = false
		vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED if _lockstep else SubViewport.UPDATE_ALWAYS
	add_child(vp)
	return vp


func set_sky_card(mat: Material) -> void:
	## Far backdrop quad per RGB camera, shown only while that camera is orthographic and only to
	## that camera (its own view layer), standing in for the sky in ortho views (Look.card_material).
	for c in _cards:
		c.free()
	_cards.clear()
	for i in cams.size():
		var q := MeshInstance3D.new()
		var qm := QuadMesh.new()
		qm.size = Vector2.ONE
		q.mesh = qm
		q.material_override = mat
		q.layers = 1 << (i + 1)
		q.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		q.gi_mode = GeometryInstance3D.GI_MODE_DISABLED
		q.visible = false
		cams[i].add_child(q)
		_cards.append(q)


func set_environments(persp: Environment, ortho: Environment, mask_env: Environment) -> void:
	_env_persp = persp
	_env_ortho = ortho
	_env_mask = mask_env
	_apply_envs()


func _apply_envs() -> void:
	for i in cams.size():
		cams[i].environment = _env_ortho if cams[i].projection == Camera3D.PROJECTION_ORTHOGONAL else _env_persp
	for mc in mask_cams:
		mc.environment = _env_mask
	if chase_cam:
		chase_cam.environment = _env_persp


func set_live_updates(on: bool, chase: bool) -> void:
	if chase_vp == null:
		return  # not set up yet (no hello received)
	var mode := SubViewport.UPDATE_ALWAYS if on else SubViewport.UPDATE_DISABLED
	for vp in rgb:
		vp.render_target_update_mode = mode
	for vp in mask_vps:
		vp.render_target_update_mode = mode
	chase_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS if (on and chase) else SubViewport.UPDATE_DISABLED


# ---------------------------------------------------------------- cameras

func apply_cameras(prev: Array, cur: Array, alpha: float) -> void:
	for c in cur:
		var i := VIEWS.find(String(c.get("view", "")))
		if i < 0:
			continue
		var p: Dictionary = c
		for q in prev:
			if q.get("view", "") == c["view"]:
				p = q
				break
		_pose(cams[i], p, c, alpha)
		if i < _cards.size():
			var ortho := cams[i].projection == Camera3D.PROJECTION_ORTHOGONAL
			_cards[i].visible = ortho
			if ortho:
				var d := cams[i].far * 0.97
				var aspect := float(size.x) / float(size.y)
				_cards[i].position = Vector3(0, 0, -d)
				_cards[i].scale = Vector3(cams[i].size * aspect * 1.05, cams[i].size * 1.05, 1.0)
		if i < mask_cams.size():
			_pose(mask_cams[i], p, c, alpha)
	_apply_envs()


static func _v(a) -> Vector3:
	return Vector3(float(a[0]), float(a[1]), float(a[2]))


func _pose(cam: Camera3D, a: Dictionary, b: Dictionary, t: float) -> void:
	var pos := _v(a["pos"]).lerp(_v(b["pos"]), t)
	var look := _v(a["look"]).lerp(_v(b["look"]), t)
	var up := _v(a["up"]).lerp(_v(b["up"]), t).normalized()
	cam.keep_aspect = Camera3D.KEEP_HEIGHT
	cam.near = float(b.get("near", 0.05))
	cam.far = float(b.get("far", 4000.0))
	if b.get("proj", "p") == "o":
		cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		cam.size = lerpf(float(a.get("size", 10.0)), float(b.get("size", 10.0)), t)
	else:
		cam.projection = Camera3D.PROJECTION_PERSPECTIVE
		cam.fov = lerpf(float(a.get("fov", 60.0)), float(b.get("fov", 60.0)), t)
	var fwd := (look - pos).normalized()
	if absf(fwd.dot(up)) > 0.9999:
		up = Vector3(0, 0, -1) if absf(fwd.y) > 0.9 else Vector3.UP
	cam.look_at_from_position(pos, look, up)


# ---------------------------------------------------------------- capture

func capture(aovs: Array) -> Dictionary:
	## Renders the current state once and returns {"items": [...], "chunks": [PackedByteArray]}.
	var targets: Array = []
	for i in VIEWS.size():
		if aovs.has("rgb"):
			targets.append([VIEWS[i], "rgb", rgb[i]])
		if aovs.has("mask") and i < mask_vps.size():
			targets.append([VIEWS[i], "mask", mask_vps[i]])
	for t in targets:
		t[2].render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	var items: Array = []
	var chunks: Array = []
	for t in targets:
		var img: Image = t[2].get_texture().get_image()
		if img.get_format() != Image.FORMAT_RGB8:
			img.convert(Image.FORMAT_RGB8)
		var data := img.get_data()
		items.append({"view": t[0], "aov": t[1], "bytes": data.size()})
		chunks.append(data)
	return {"items": items, "chunks": chunks}


func probe(points: Array) -> Dictionary:
	var out := {}
	for i in VIEWS.size():
		var px: Array = []
		for p in points:
			var v := _v(p)
			var uv := cams[i].unproject_position(v)
			px.append([uv.x, uv.y, cams[i].is_position_behind(v)])
		out[VIEWS[i]] = px
	return out


func texture(i: int) -> Texture2D:
	return rgb[i].get_texture()


func mask_texture(i: int) -> Texture2D:
	return mask_vps[i].get_texture() if i < mask_vps.size() else null
