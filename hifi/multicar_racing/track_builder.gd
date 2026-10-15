extends Node3D
## Builds the race track scene from the GymBox "track" message: asphalt strip, edge lines,
## kerbs, start line, plus seeded scenery (trees, grandstand, gantry).
## Box2D (x, y) maps to Godot (x, h, -y); a Box2D angle is a Godot yaw (rotation.y).

const ROAD_Y := 0.02
const LINE_Y := 0.028
const CURB_Y := 0.045
const SCENERY_LAYER := 2  # render-layer bitmask; the GymBox-matched ortho cameras hide it

var _asphalt: StandardMaterial3D
var _grass: StandardMaterial3D
var _line: StandardMaterial3D
var _curb_red: StandardMaterial3D
var _curb_white: StandardMaterial3D
var _center: Array = []
var _width := 6.67


func _init() -> void:
	_asphalt = _mat(Color(0.34, 0.34, 0.36), 0.0, 0.85)
	_asphalt.albedo_texture = _noise_tex(0.08, false, 1)
	_asphalt.normal_enabled = true
	_asphalt.normal_texture = _noise_tex(0.2, true, 2)
	_asphalt.normal_scale = 0.4
	_asphalt.uv1_scale = Vector3(3.0, 3.0, 1.0)

	_grass = _mat(Color(0.42, 0.62, 0.28), 0.0, 0.95)
	_grass.albedo_texture = _noise_tex(0.03, false, 3)
	_grass.normal_enabled = true
	_grass.normal_texture = _noise_tex(0.12, true, 4)
	_grass.uv1_scale = Vector3(500.0, 500.0, 1.0)

	_line = _mat(Color(0.92, 0.92, 0.9), 0.0, 0.6)
	_curb_red = _mat(Color(0.8, 0.06, 0.05), 0.0, 0.55)
	_curb_white = _mat(Color(0.93, 0.93, 0.93), 0.0, 0.55)


static func b2g(x: float, y: float, h: float = 0.0) -> Vector3:
	return Vector3(x, h, -y)


func build(msg: Dictionary) -> void:
	for c in get_children():
		c.queue_free()
	_center = msg["center"]
	_width = msg["width"]
	var track_seed: int = int(msg.get("seed", -1))
	if track_seed < 0:
		track_seed = int(absf(_center[0][0]) * 1000.0 + absf(_center[0][1]))

	_build_ground()
	_add_mesh(_strip(-_width, _width, ROAD_Y, _width * 2.0), _asphalt)
	_add_mesh(_strip(-_width + 0.25, -_width + 0.55, LINE_Y, 1.0), _line)
	_add_mesh(_strip(_width - 0.55, _width - 0.25, LINE_Y, 1.0), _line)
	_build_curbs(msg["curbs"])
	_build_start_line()

	var rng := RandomNumberGenerator.new()
	rng.seed = track_seed
	_build_gantry()
	_build_grandstand(rng)
	_build_trees(rng, float(msg["playfield"]))


# ---------------------------------------------------------------- road

func _strip(off_a: float, off_b: float, h: float, uv_len: float) -> ArrayMesh:
	## Closed strip between two lateral offsets of the centreline (same geometry as the tiles).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := _center.size()
	var dist := 0.0
	var pa := Vector3.ZERO
	var pb := Vector3.ZERO
	var pv := 0.0
	for k in range(n + 1):
		var p: Array = _center[k % n]
		var c := cos(float(p[2]))
		var s := sin(float(p[2]))
		var a := b2g(p[0] + off_a * c, p[1] + off_a * s, h)
		var b := b2g(p[0] + off_b * c, p[1] + off_b * s, h)
		if k > 0:
			dist += ((a + b) * 0.5).distance_to((pa + pb) * 0.5)
			var v := dist / uv_len
			# Wound so the face points up (back faces get a flipped normal and render unlit)
			_vert(st, pa, Vector2(0, pv))
			_vert(st, b, Vector2(1, v))
			_vert(st, pb, Vector2(1, pv))
			_vert(st, pa, Vector2(0, pv))
			_vert(st, a, Vector2(0, v))
			_vert(st, b, Vector2(1, v))
			pv = v
		pa = a
		pb = b
	st.generate_tangents()
	return st.commit()


func _vert(st: SurfaceTool, p: Vector3, uv: Vector2) -> void:
	st.set_normal(Vector3.UP)
	st.set_uv(uv)
	st.add_vertex(p)


func _build_curbs(curbs: Array) -> void:
	var red := SurfaceTool.new()
	var white := SurfaceTool.new()
	red.begin(Mesh.PRIMITIVE_TRIANGLES)
	white.begin(Mesh.PRIMITIVE_TRIANGLES)
	for q in curbs:
		var st := red if q[8] > 0.5 else white
		var v: Array = []
		for i in 4:
			v.append(b2g(q[2 * i], q[2 * i + 1], CURB_Y))
		for idx in [0, 1, 2, 0, 2, 3]:
			_vert(st, v[idx], Vector2.ZERO)
	if not curbs.is_empty():
		_add_mesh(red.commit(), _curb_red)
		_add_mesh(white.commit(), _curb_white)


func _build_start_line() -> void:
	var img := Image.create(16, 2, false, Image.FORMAT_RGB8)
	for x in 16:
		for y in 2:
			img.set_pixel(x, y, Color.WHITE if (x + y) % 2 == 0 else Color(0.05, 0.05, 0.05))
	var mat := _mat(Color.WHITE, 0.0, 0.6)
	mat.albedo_texture = ImageTexture.create_from_image(img)
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	var plane := PlaneMesh.new()
	plane.size = Vector2(_width * 2.0, 1.6)
	var node := _local_frame(0, -3.0, 0.0)
	var mi := MeshInstance3D.new()
	mi.mesh = plane
	mi.material_override = mat
	mi.position.y = LINE_Y + 0.004
	node.add_child(mi)


# ---------------------------------------------------------------- scenery

func _build_ground() -> void:
	var plane := PlaneMesh.new()
	plane.size = Vector2(2400, 2400)
	var mi := MeshInstance3D.new()
	mi.mesh = plane
	mi.material_override = _grass
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)


func _build_gantry() -> void:
	var node := _local_frame(0, -3.0, 0.0)
	var frame := _mat(Color(0.18, 0.18, 0.2), 0.6, 0.4)
	var banner := _mat(Color(0.05, 0.05, 0.06), 0.0, 0.5)
	banner.emission_enabled = true
	banner.emission = Color(0.1, 0.6, 1.0)
	banner.emission_energy_multiplier = 1.5
	var span := _width * 2.0 + 4.0
	for side in [-1.0, 1.0]:
		_box(node, Vector3(0.7, 8.0, 0.7), Vector3(side * span * 0.5, 4.0, 0), frame)
	_box(node, Vector3(span + 0.7, 1.4, 0.8), Vector3(0, 8.0, 0), frame)
	_box(node, Vector3(span * 0.8, 0.9, 0.85), Vector3(0, 8.0, 0), banner)
	_set_layer(node)


func _build_grandstand(rng: RandomNumberGenerator) -> void:
	## Stepped stand beside the start straight, on whichever side has room.
	for side in [1.0, -1.0]:
		var off: float = side * (_width + 9.0)
		var p: Array = _center[0]
		var c := cos(float(p[2]))
		var s := sin(float(p[2]))
		if _dist_to_track(p[0] + off * c, p[1] + off * s, 8) < _width + 18.0:
			continue
		var node := _local_frame(0, 0.0, off)
		var concrete := _mat(Color(0.62, 0.62, 0.6), 0.0, 0.9)
		var roof := _mat(Color(0.85, 0.87, 0.9), 0.3, 0.4)
		var steps := 6
		var length := 44.0
		var crowd := MultiMesh.new()
		crowd.transform_format = MultiMesh.TRANSFORM_3D
		crowd.use_colors = true
		var body := BoxMesh.new()
		body.size = Vector3(0.45, 0.9, 0.45)
		var cm := _mat(Color.WHITE, 0.0, 0.8)
		cm.vertex_color_use_as_albedo = true
		body.material = cm
		crowd.mesh = body
		var spots: Array = []
		for k in steps:
			var h := 0.9 * (k + 1)
			var x: float = side * (k * 1.6 + 0.8)
			_box(node, Vector3(1.6, h, length), Vector3(x, h * 0.5, 0), concrete)
			var z := -length * 0.5 + 0.6
			while z < length * 0.5 - 0.6:
				if rng.randf() < 0.8:
					spots.append(Vector3(x + rng.randf_range(-0.2, 0.2), h + 0.45, z))
				z += 0.75
		crowd.instance_count = spots.size()
		for i in spots.size():
			crowd.set_instance_transform(i, Transform3D(Basis(), spots[i]))
			crowd.set_instance_color(i, Color.from_hsv(rng.randf(), rng.randf_range(0.4, 0.9), rng.randf_range(0.5, 1.0)))
		var cmi := MultiMeshInstance3D.new()
		cmi.multimesh = crowd
		node.add_child(cmi)
		var depth := steps * 1.6
		_box(node, Vector3(depth + 2.0, 0.3, length + 2.0), Vector3(side * depth * 0.5, 9.5, 0), roof)
		for z in [-length * 0.5, 0.0, length * 0.5]:
			_box(node, Vector3(0.4, 9.5, 0.4), Vector3(side * (depth + 0.5), 4.75, z), concrete)
		_set_layer(node)
		return


func _build_trees(rng: RandomNumberGenerator, playfield: float) -> void:
	var trunk_mesh := CylinderMesh.new()
	trunk_mesh.top_radius = 0.22
	trunk_mesh.bottom_radius = 0.35
	trunk_mesh.height = 3.0
	trunk_mesh.radial_segments = 8
	var trunk_mat := _mat(Color(0.33, 0.22, 0.13), 0.0, 0.95)
	trunk_mesh.material = trunk_mat

	var leaf_mat := _mat(Color.WHITE, 0.0, 0.9)
	leaf_mat.vertex_color_use_as_albedo = true
	var cone_low := CylinderMesh.new()
	cone_low.top_radius = 0.0
	cone_low.bottom_radius = 2.4
	cone_low.height = 4.5
	cone_low.radial_segments = 10
	cone_low.material = leaf_mat
	var cone_high := CylinderMesh.new()
	cone_high.top_radius = 0.0
	cone_high.bottom_radius = 1.7
	cone_high.height = 3.5
	cone_high.radial_segments = 10
	cone_high.material = leaf_mat

	var xforms: Array = []
	var colors: Array = []
	var extent := playfield + 60.0
	for _i in 3500:
		var x := rng.randf_range(-extent, extent)
		var y := rng.randf_range(-extent, extent)
		var d := _dist_to_track(x, y, 2)
		if d < _width + 7.0 or d > 140.0:
			continue
		var sc := rng.randf_range(0.7, 1.6)
		var tree_basis := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(sc, sc * rng.randf_range(0.85, 1.25), sc))
		xforms.append(Transform3D(tree_basis, b2g(x, y)))
		colors.append(Color.from_hsv(rng.randf_range(0.24, 0.36), rng.randf_range(0.45, 0.75), rng.randf_range(0.22, 0.42)))

	# (mesh, local offset, colored) per tree part
	for part in [[trunk_mesh, Vector3(0, 1.5, 0), false], [cone_low, Vector3(0, 4.6, 0), true], [cone_high, Vector3(0, 6.8, 0), true]]:
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = part[2]
		mm.mesh = part[0]
		mm.instance_count = xforms.size()
		for i in xforms.size():
			var xf: Transform3D = xforms[i]
			mm.set_instance_transform(i, xf.translated_local(part[1]))
			if part[2]:
				mm.set_instance_color(i, colors[i])
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.layers = SCENERY_LAYER
		add_child(mmi)


# ---------------------------------------------------------------- helpers

func _dist_to_track(x: float, y: float, stride: int) -> float:
	var best := INF
	for i in range(0, _center.size(), stride):
		var p: Array = _center[i]
		var dx: float = x - p[0]
		var dy: float = y - p[1]
		best = minf(best, dx * dx + dy * dy)
	return sqrt(best)


func _local_frame(i: int, forward: float, across: float) -> Node3D:
	## Node at centreline point i, rotated so local +X is across the road and -Z is the driving direction.
	var p: Array = _center[i]
	var b := float(p[2])
	var node := Node3D.new()
	node.rotation.y = b
	node.position = b2g(p[0], p[1]) + node.basis * Vector3(across, 0, -forward)
	add_child(node)
	return node


func _box(parent: Node3D, size: Vector3, pos: Vector3, mat: Material) -> void:
	var m := BoxMesh.new()
	m.size = size
	var mi := MeshInstance3D.new()
	mi.mesh = m
	mi.material_override = mat
	mi.position = pos
	parent.add_child(mi)


func _add_mesh(mesh: Mesh, mat: Material) -> void:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = mat
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mi)


func _set_layer(node: Node) -> void:
	if node is VisualInstance3D:
		node.layers = SCENERY_LAYER
	for c in node.get_children():
		_set_layer(c)


static func _mat(color: Color, metallic: float, roughness: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.metallic = metallic
	m.roughness = roughness
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m


static func _noise_tex(freq: float, normal: bool, noise_seed: int) -> NoiseTexture2D:
	var noise := FastNoiseLite.new()
	noise.frequency = freq
	noise.seed = noise_seed
	noise.fractal_octaves = 4
	var t := NoiseTexture2D.new()
	t.width = 512
	t.height = 512
	t.seamless = true
	t.as_normal_map = normal
	t.bump_strength = 4.0
	t.noise = noise
	return t
