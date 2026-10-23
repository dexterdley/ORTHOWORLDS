class_name EnvWorld
extends Node3D
## Base class of every hi-fi env world (res://envs/<env>/world.gd).
##
## Python sends "scene" once per reset and "state" every step (hifi/bridge.py). A world
##   - configures sky/sun/fog in setup_look(look)
##   - builds the static and moving content in build_scene(scene), under `content`, and calls
##     register(id, node, mask_class) for every node Python moves through state["xf"][id]
##   - optionally reacts to per-step extras in on_state(state, prev, alpha)
##   - spawns transient objects ("dyn": projectiles, effects) in spawn_dyn(kind, d)
## The base class streams transforms (interpolated in live mode), spawns/frees transient objects,
## applies per-view visibility, and builds the flat-colour object-ID mask duplicates.
##
## Mask encoding (mask AOV): R + 256*G = instance index (1-based, per registered id; 65535 = any
## transient object), B = class code. Everything unregistered is a black occluder, so masks are
## occlusion-correct. Python gets the id/class tables in the "ready" message.

const VIEWS := ["top", "rear", "side", "fpv"]
const RGB_HIDDEN_BASE := 0b111110          # layers bits 1..5 (4 views + chase), used once hidden somewhere
const MASK_ALL := (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14)
const DYN_INDEX := 65535

var look: Look
var content: Node3D
var objects := {}          # id -> Node3D moved by state["xf"]
var dyn_nodes := {}        # id -> Node3D transient
var dyn_kind := {}         # id -> kind
var mask_objects := {}     # id -> instance index
var mask_classes := {}     # class -> code
var _mask_code := {}       # id -> class code
var scene_msg := {}
var sim_t := 0.0
var _hidden := {}          # id -> bitmask of hidden views
var _mask_mats := {}
var _next_index := 1
var _occluder_mat: ShaderMaterial


# ================================================================ override points

func setup_look(lk: Look) -> void:
	lk.configure({})


func build_scene(_scene: Dictionary) -> void:
	pass


func on_state(_state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	pass


func spawn_dyn(_kind: String, _d: Dictionary) -> Node3D:
	return null


func update_dyn(_node: Node3D, _a: Dictionary, _b: Dictionary, _alpha: float) -> void:
	pass


func chase_pose() -> Transform3D:
	## Live-mode chase camera (hi-fi only). Default: behind and above the first registered object.
	for id in objects:
		var n: Node3D = objects[id]
		var p := n.global_position
		return Transform3D(Basis(), p + Vector3(0, 4, 9)).looking_at(p, Vector3.UP)
	return Transform3D(Basis(), Vector3(0, 5, 10))


# ================================================================ session hooks

func begin_scene(scene: Dictionary) -> void:
	if content:
		content.free()
	content = Node3D.new()
	content.name = "Content"
	add_child(content)
	objects.clear()
	dyn_nodes.clear()
	dyn_kind.clear()
	mask_objects.clear()
	_mask_code.clear()
	_hidden.clear()
	_cut_mats.clear()
	_next_index = 1
	scene_msg = scene
	build_scene(scene)
	add_occluders(content)


func apply_state(prev: Dictionary, cur: Dictionary, alpha: float) -> void:
	sim_t = lerpf(float(prev.get("t", 0.0)), float(cur.get("t", 0.0)), alpha)
	RenderingServer.global_shader_parameter_set("sim_time", sim_t)
	var pxf: Dictionary = prev.get("xf", {})
	var cxf: Dictionary = cur.get("xf", {})
	for id in cxf:
		var n = objects.get(id)
		if n == null:
			continue
		_set_xf(n, pxf.get(id, cxf[id]), cxf[id], alpha)
	_apply_dyn(prev.get("dyn", []), cur.get("dyn", []), alpha)
	if cur.has("vis"):
		_apply_vis(cur["vis"])
	on_state(cur, prev, alpha)


func mask_table() -> Dictionary:
	return {"objects": mask_objects, "classes": mask_classes}


# ================================================================ registration / masks

func register(id: String, node: Node3D, mask_class := "") -> Node3D:
	## Makes node movable by state["xf"][id]; mask_class != "" gives it an object-ID mask.
	objects[id] = node
	if mask_class != "":
		var idx := _next_index
		_next_index += 1
		mask_objects[id] = idx
		_mask_code[id] = class_code(mask_class)
		attach_mask(node, _mask_color(idx, int(_mask_code[id])))
	return node


func set_views(node: Node3D, views: Array) -> void:
	## Shows node (and its subtree) only in the listed views: any of "top", "rear", "side", "fpv",
	## "chase". Composes with per-step hiding (state["vis"]). Call during build_scene, before the
	## mask twins. Cross-view consistency (hifi/tools/check_views.py): scenery must stay visible in
	## every view whose frame contains it; hide a node only from a view where it stands between the
	## camera and the subject (a cutaway), or when it is a section cap (set_meta("section_cap", true)).
	var bits := 0
	for v in views:
		var i := VIEWS.find(String(v))
		if i >= 0:
			bits |= 1 << i
	node.set_meta("view_bits", bits)
	node.set_meta("chase", views.has("chase"))
	for gi in _geometry(node):
		_apply_layers(gi, 0)


func _apply_layers(gi: GeometryInstance3D, hidden_views: int) -> void:
	var base := _view_bits(gi)
	var vis := base & ~hidden_views & 0b1111
	if gi.has_meta("is_mask"):
		gi.layers = vis << 11
		return
	var chase := _chase_ok(gi)
	if vis == 0b1111 and chase:
		gi.layers = 1
	else:
		gi.layers = (vis << 1) | ((1 << 5) if chase else 0)


func _chase_ok(n: Node) -> bool:
	var cur := n
	while cur != null and cur != self:
		if cur.has_meta("chase"):
			return bool(cur.get_meta("chase"))
		cur = cur.get_parent()
	return true


func _view_bits(n: Node) -> int:
	var cur := n
	while cur != null and cur != self:
		if cur.has_meta("view_bits"):
			return int(cur.get_meta("view_bits"))
		cur = cur.get_parent()
	return 0b1111


func class_code(cls: String) -> int:
	if not mask_classes.has(cls):
		mask_classes[cls] = mask_classes.size() + 1
	return int(mask_classes[cls])


func _mask_color(idx: int, code: int) -> Color:
	return Color8(idx & 255, (idx >> 8) & 255, code & 255)


func attach_mask(node: Node, color: Color) -> void:
	## Adds an unshaded flat-colour twin of every mesh under node, seen only by the mask cameras.
	var key := color.to_html(false)
	var mat: ShaderMaterial = _mask_mats.get(key)
	if mat == null:
		mat = ShaderMaterial.new()
		mat.shader = _flat_shader()
		mat.set_shader_parameter("flat_color", color)
		_mask_mats[key] = mat
	for gi in _geometry(node):
		_twin(gi, mat)


func add_occluders(node: Node) -> void:
	## Black mask twins for every geometry that has none yet (terrain, scenery...), so the mask
	## views are occlusion-correct. Called after build_scene and for new transient objects.
	if _occluder_mat == null:
		_occluder_mat = ShaderMaterial.new()
		_occluder_mat.shader = _flat_shader()
		_occluder_mat.set_shader_parameter("flat_color", Color(0, 0, 0))
	for gi in _geometry(node):
		if not gi.has_meta("mask_twin"):
			_twin(gi, _occluder_mat)


func refresh_mask(id: String) -> void:
	## Re-creates the mask twins of a registered object after its meshes changed.
	var n: Node3D = objects.get(id)
	if n == null:
		return
	for c in n.find_children("*", "GeometryInstance3D", true, false):
		if c.has_meta("is_mask"):
			c.free()
	for gi in _geometry(n):
		gi.remove_meta("mask_twin")
	if mask_objects.has(id):
		attach_mask(n, _mask_color(int(mask_objects[id]), int(_mask_code[id])))
	add_occluders(n)


func _geometry(node: Node) -> Array:
	var out: Array = []
	if node is GeometryInstance3D and not node.has_meta("is_mask"):
		out.append(node)
	for c in node.find_children("*", "GeometryInstance3D", true, false):
		if not c.has_meta("is_mask"):
			out.append(c)
	return out


func _twin(gi: GeometryInstance3D, mat: Material) -> void:
	if gi.has_meta("mask_twin") or gi.has_meta("no_mask"):
		return
	if not gi.visible:
		return
	var twin: GeometryInstance3D
	if gi is MeshInstance3D:
		if gi.mesh == null:
			return
		var m := MeshInstance3D.new()
		m.mesh = gi.mesh
		m.skeleton = NodePath()
		twin = m
	elif gi is MultiMeshInstance3D:
		var mm := MultiMeshInstance3D.new()
		mm.multimesh = gi.multimesh
		twin = mm
	elif gi is Label3D:
		# Text keeps its glyph shape (alpha-cut, no blending) in the flat mask colour
		var lb: Label3D = gi.duplicate(0)
		for c in lb.get_children():
			lb.remove_child(c)
			c.free()
		var col: Color = mat.get_shader_parameter("flat_color") if mat is ShaderMaterial else Color.BLACK
		lb.modulate = col
		lb.outline_modulate = col
		lb.shaded = false
		lb.double_sided = true
		lb.alpha_cut = Label3D.ALPHA_CUT_DISCARD
		lb.transform = Transform3D()
		lb.set_meta("is_mask", true)
		lb.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		lb.layers = _view_bits(gi) << 11
		gi.add_child(lb)
		gi.set_meta("mask_twin", lb)
		return
	else:
		return
	twin.set_meta("is_mask", true)
	twin.material_override = _alpha_cut_variant(gi, mat)
	twin.visibility_range_begin = gi.visibility_range_begin
	twin.visibility_range_end = gi.visibility_range_end
	twin.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	twin.gi_mode = GeometryInstance3D.GI_MODE_DISABLED
	twin.layers = _view_bits(gi) << 11
	gi.add_child(twin)
	gi.set_meta("mask_twin", twin)


static var _flat: Shader
static var _flat_cut: Shader
var _cut_mats := {}


func _alpha_cut_variant(gi: GeometryInstance3D, mat: Material) -> Material:
	## Alpha-tested or transparent textured materials (foliage cards, fences) keep their cut-out
	## shape in the mask: same flat colour, discarding where the albedo alpha is low.
	var src: Material = gi.material_override
	if src == null and gi is MeshInstance3D and gi.mesh != null and gi.mesh.get_surface_count() > 0:
		src = gi.get_active_material(0)
	if not (src is BaseMaterial3D) or not (mat is ShaderMaterial):
		return mat
	var bm: BaseMaterial3D = src
	if bm.transparency == BaseMaterial3D.TRANSPARENCY_DISABLED or bm.albedo_texture == null or bm.uv1_triplanar:
		return mat
	var col: Color = mat.get_shader_parameter("flat_color")
	var key := "%s:%d:%s" % [col.to_html(false), bm.albedo_texture.get_instance_id(), str(bm.uv1_scale)]
	if _cut_mats.has(key):
		return _cut_mats[key]
	if _flat_cut == null:
		_flat_cut = Shader.new()
		_flat_cut.code = """
shader_type spatial;
render_mode unshaded, fog_disabled, cull_disabled, shadows_disabled, ambient_light_disabled;
uniform vec4 flat_color : source_color = vec4(0.0, 0.0, 0.0, 1.0);
uniform sampler2D alpha_tex : filter_linear_mipmap, repeat_enable;
uniform vec3 uv_scale = vec3(1.0);
uniform vec3 uv_offset = vec3(0.0);
uniform float cut = 0.5;
uniform float base_alpha = 1.0;
void fragment() {
	if (texture(alpha_tex, UV * uv_scale.xy + uv_offset.xy).a * base_alpha < cut) { discard; }
	ALBEDO = flat_color.rgb;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _flat_cut
	m.set_shader_parameter("flat_color", col)
	m.set_shader_parameter("alpha_tex", bm.albedo_texture)
	m.set_shader_parameter("uv_scale", bm.uv1_scale)
	m.set_shader_parameter("uv_offset", bm.uv1_offset)
	m.set_shader_parameter("base_alpha", bm.albedo_color.a)
	m.set_shader_parameter("cut", bm.alpha_scissor_threshold if bm.transparency == BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR else 0.5)
	_cut_mats[key] = m
	return m


static func _flat_shader() -> Shader:
	if _flat == null:
		_flat = Shader.new()
		_flat.code = """
shader_type spatial;
render_mode unshaded, fog_disabled, cull_disabled, shadows_disabled, ambient_light_disabled;
uniform vec4 flat_color : source_color = vec4(0.0, 0.0, 0.0, 1.0);
void fragment() { ALBEDO = flat_color.rgb; }
"""
	return _flat


# ================================================================ transforms / transient objects

static func xf_pos(a: Array) -> Vector3:
	return Vector3(float(a[0]), float(a[1]), float(a[2]))


static func xf_quat(a: Array) -> Quaternion:
	if a.size() < 7:
		return Quaternion.IDENTITY
	return Quaternion(float(a[3]), float(a[4]), float(a[5]), float(a[6])).normalized()


static func _set_xf(n: Node3D, a: Array, b: Array, t: float) -> void:
	var p := xf_pos(a).lerp(xf_pos(b), t)
	var q := xf_quat(a).slerp(xf_quat(b), t) if t < 1.0 else xf_quat(b)
	n.transform = Transform3D(Basis(q), p)


func _apply_dyn(prev: Array, cur: Array, alpha: float) -> void:
	var seen := {}
	var prev_by_id := {}
	for d in prev:
		prev_by_id[str(d.get("id", ""))] = d
	for d in cur:
		var id := str(d.get("id", ""))
		var kind := str(d.get("k", ""))
		seen[id] = true
		var n: Node3D = dyn_nodes.get(id)
		if n != null and dyn_kind.get(id, "") != kind:
			n.queue_free()
			n = null
		if n == null:
			n = spawn_dyn(kind, d)
			if n == null:
				continue
			if n.get_parent() == null:
				content.add_child(n)
			dyn_nodes[id] = n
			dyn_kind[id] = kind
			if d.has("mask"):
				attach_mask(n, _mask_color(DYN_INDEX, class_code(str(d["mask"]))))
			else:
				add_occluders(n)
		var a: Dictionary = prev_by_id.get(id, d)
		if d.has("xf"):
			_set_xf(n, a.get("xf", d["xf"]), d["xf"], alpha)
		update_dyn(n, a, d, alpha)
	for id in dyn_nodes.keys():
		if not seen.has(id):
			dyn_nodes[id].queue_free()
			dyn_nodes.erase(id)
			dyn_kind.erase(id)


func _apply_vis(vis: Dictionary) -> void:
	var want := {}
	for v in vis:
		var vi := VIEWS.find(String(v))
		if vi < 0:
			continue
		for id in vis[v]:
			want[id] = int(want.get(id, 0)) | (1 << vi)
	for id in _hidden.keys():
		if not want.has(id):
			want[id] = 0
	for id in want:
		if int(_hidden.get(id, 0)) != int(want[id]):
			set_hidden_mask(id, int(want[id]))


func set_hidden_mask(id: String, hidden_views: int) -> void:
	## Hides object id in the views whose bit is set (bit 0 = top ... bit 3 = fpv).
	var n = objects.get(id, dyn_nodes.get(id))
	if n == null:
		return
	_hidden[id] = hidden_views
	for gi in n.find_children("*", "GeometryInstance3D", true, false) + ([n] if n is GeometryInstance3D else []):
		_apply_layers(gi, hidden_views)
	if hidden_views == 0:
		_hidden.erase(id)


# ================================================================ debug: scene inspection

func inspect(max_instances := 4000) -> Array:
	## Every rendered geometry under `content` (mask twins excluded) with its world-space AABBs
	## (one per MultiMesh instance), render layers and visibility range. Used by
	## hifi/tools/check_views.py to compare what each view's frustum contains with what it shows.
	var owners := {}
	for id in objects:
		owners[objects[id]] = id
	for id in dyn_nodes:
		owners[dyn_nodes[id]] = id
	var out: Array = []
	if content == null:
		return out
	for gi in _geometry(content):
		if not gi.is_visible_in_tree():
			continue
		var owner_id := ""
		var cur: Node = gi
		while cur != null and cur != content:
			if owners.has(cur):
				owner_id = String(owners[cur])
				break
			cur = cur.get_parent()
		var boxes: Array = []
		var count := 1
		if gi is MultiMeshInstance3D:
			var mm: MultiMesh = gi.multimesh
			if mm == null or mm.mesh == null:
				continue
			count = mm.instance_count if mm.visible_instance_count < 0 else mm.visible_instance_count
			var local := mm.mesh.get_aabb()
			var step := maxi(1, int(ceil(float(count) / float(max_instances))))
			for i in range(0, count, step):
				boxes.append(_aabb_arr(gi.global_transform * mm.get_instance_transform(i) * local))
		else:
			boxes.append(_aabb_arr(gi.global_transform * gi.get_aabb()))
		var cap := false
		cur = gi
		while cur != null and cur != content:
			if cur.has_meta("section_cap"):
				cap = true
				break
			cur = cur.get_parent()
		out.append({"path": String(content.get_path_to(gi)), "cls": gi.get_class(), "layers": gi.layers, "cap": cap,
			"owner": owner_id, "vr": [gi.visibility_range_begin, gi.visibility_range_end],
			"n": count, "aabbs": boxes, "no_mask": gi.has_meta("no_mask")})
	return out


static func _aabb_arr(b: AABB) -> Array:
	return [b.position.x, b.position.y, b.position.z, b.end.x, b.end.y, b.end.z]


func reveal(on: bool) -> void:
	## Debug: on = every RGB geometry visible in all four views (ignores set_views and state["vis"]);
	## off = restores the layers. Lets hifi/tools/check_views.py render what per-view hiding removes.
	if content == null:
		return
	for gi in _geometry(content):
		if on:
			if not gi.has_meta("reveal_layers"):
				gi.set_meta("reveal_layers", gi.layers)
			gi.layers = 1
		elif gi.has_meta("reveal_layers"):
			gi.layers = int(gi.get_meta("reveal_layers"))
			gi.remove_meta("reveal_layers")
