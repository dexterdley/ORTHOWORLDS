extends Node3D
## Race track from the CarRacing "scene" message: asphalt strip, edge lines, kerbs, start line,
## grass, plus seeded scenery (gantry, grandstand, trees). Box2D (x, y) -> Godot (x, h, -y).
## Scenery nodes are listed in `scenery`. They are drawn in every view whose frame contains them
## (one consistent world); the elevations cut away only what lies between their camera and the car.

const ROAD_Y := 0.02
const LINE_Y := 0.028
const CURB_Y := 0.04

var scenery: Array[Node3D] = []
var center: Array = []
var width := 6.67


static func b2g(x: float, y: float, h := 0.0) -> Vector3:
	return Vector3(x, h, -y)


func build(msg: Dictionary) -> void:
	center = msg.get("center", [])
	width = float(msg.get("width", 6.67))
	if center.is_empty():
		return
	var asphalt := Assets.pbr("asphalt_track", {"uv": 0.22, "world": true, "tint": Color(0.82, 0.82, 0.84), "color": Color(0.3, 0.3, 0.32)})
	var line := M.mat(Color(0.93, 0.93, 0.9), 0.55, 0.0, {"normal_noise": 0.3})
	var grass := Assets.pbr("aerial_grass_rock", {"uv": 0.12, "world": true, "tint": Color(0.82, 0.98, 0.72), "color": Color(0.3, 0.5, 0.2)})
	var soil := Assets.pbr("brown_mud_02", {"uv": 0.25, "world": true, "color": Color(0.35, 0.25, 0.18)})
	var pf := float(msg.get("playfield", 333.3))
	# Thick slab: the horizontal ortho elevations see its grassy flank as the ground band below the horizon
	add_child(Terrain.ground_slab(Vector2.ZERO, Vector2(pf * 2.6, pf * 2.6), 40.0, grass, grass, 0.0))
	_mesh(_strip(-width, width, ROAD_Y), asphalt)
	_mesh(_strip(-width + 0.25, -width + 0.55, LINE_Y), line)
	_mesh(_strip(width - 0.55, width - 0.25, LINE_Y), line)
	# Road shoulders: a thin darker band of gravel/dirt where the asphalt meets the grass
	_mesh(_strip(-width - 1.2, -width, ROAD_Y * 0.5), soil)
	_mesh(_strip(width, width + 1.2, ROAD_Y * 0.5), soil)
	_build_curbs(msg.get("curbs", []))
	_build_start_line()
	var rng := RandomNumberGenerator.new()
	rng.seed = int(absf(float(center[0][0])) * 1000.0 + absf(float(center[0][1])))
	_build_gantry()
	_build_grandstand(rng)
	_build_trees(rng, pf)


func _strip(off_a: float, off_b: float, h: float) -> ArrayMesh:
	## Closed strip between two lateral offsets of the centreline (same geometry as the road tiles).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := center.size()
	var pa := Vector3.ZERO
	var pb := Vector3.ZERO
	for k in range(n + 1):
		var p: Array = center[k % n]
		var c := cos(float(p[2]))
		var s := sin(float(p[2]))
		var a := b2g(float(p[0]) + off_a * c, float(p[1]) + off_a * s, h)
		var b := b2g(float(p[0]) + off_b * c, float(p[1]) + off_b * s, h)
		if k > 0:
			M._quad(st, [pa, pb, b, a], Vector3.UP)
		pa = a
		pb = b
	st.generate_tangents()
	return st.commit()


func _build_curbs(curbs: Array) -> void:
	var red := SurfaceTool.new()
	var white := SurfaceTool.new()
	red.begin(Mesh.PRIMITIVE_TRIANGLES)
	white.begin(Mesh.PRIMITIVE_TRIANGLES)
	for q in curbs:
		var st := red if float(q[8]) > 0.5 else white
		var v: Array = []
		for i in 4:
			v.append(b2g(float(q[2 * i]), float(q[2 * i + 1]), CURB_Y))
		M._quad(st, v, Vector3.UP)
	if not curbs.is_empty():
		_mesh(red.commit(), M.mat(Color(0.78, 0.07, 0.05), 0.45, 0.0, {"normal_noise": 0.4}))
		_mesh(white.commit(), M.mat(Color(0.92, 0.92, 0.9), 0.45, 0.0, {"normal_noise": 0.4}))


func _build_start_line() -> void:
	var img := Image.create(16, 2, false, Image.FORMAT_RGB8)
	for x in 16:
		for y in 2:
			img.set_pixel(x, y, Color.WHITE if (x + y) % 2 == 0 else Color(0.05, 0.05, 0.05))
	var mat := M.mat(Color.WHITE, 0.6)
	mat.albedo_texture = ImageTexture.create_from_image(img)
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	var node := _local_frame(0, -3.0, 0.0)
	M.mi(node, M.plane(Vector2(width * 2.0, 1.6)), mat, Vector3(0, LINE_Y + 0.004, 0))


func _build_gantry() -> void:
	var node := _local_frame(0, -3.0, 0.0)
	var frame := Assets.pbr("metal_plate", {"uv": 1.0, "tint": Color(0.35, 0.36, 0.4), "color": Color(0.2, 0.2, 0.22)})
	var banner := M.emissive(Color(0.1, 0.55, 1.0), 1.4)
	var span := width * 2.0 + 4.0
	for side in [-1.0, 1.0]:
		M.mi(node, M.box(Vector3(0.7, 8.0, 0.7), 0.05), frame, Vector3(side * span * 0.5, 4.0, 0))
	M.mi(node, M.box(Vector3(span + 0.7, 1.4, 0.8), 0.06), frame, Vector3(0, 8.0, 0))
	M.mi(node, M.box(Vector3(span * 0.8, 0.9, 0.85), 0.02), banner, Vector3(0, 8.0, 0))
	scenery.append(node)


func _build_grandstand(rng: RandomNumberGenerator) -> void:
	## Stepped stand beside the start straight, on whichever side has room.
	for side in [1.0, -1.0]:
		var off: float = side * (width + 9.0)
		var p: Array = center[0]
		var c := cos(float(p[2]))
		var s := sin(float(p[2]))
		if _dist_to_track(float(p[0]) + off * c, float(p[1]) + off * s, 8) < width + 18.0:
			continue
		var node := _local_frame(0, 0.0, off)
		var concrete := Assets.pbr("concrete_floor_worn_001", {"uv": 0.5, "color": Color(0.6, 0.6, 0.58)})
		var roof := Assets.pbr("corrugated_iron", {"uv": 0.5, "tint": Color(0.85, 0.87, 0.9), "color": Color(0.8, 0.8, 0.82)})
		var steps := 6
		var length := 44.0
		var crowd := MultiMesh.new()
		crowd.transform_format = MultiMesh.TRANSFORM_3D
		crowd.use_colors = true
		var body := M.capsule(0.22, 0.9, 8)
		var cm := M.mat(Color.WHITE, 0.8)
		cm.vertex_color_use_as_albedo = true
		crowd.mesh = body
		var spots: Array = []
		for k in steps:
			var h := 0.9 * (k + 1)
			var x: float = side * (k * 1.6 + 0.8)
			M.mi(node, M.box(Vector3(1.6, h, length), 0.03), concrete, Vector3(x, h * 0.5, 0))
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
		cmi.material_override = cm
		node.add_child(cmi)
		var depth := steps * 1.6
		M.mi(node, M.box(Vector3(depth + 2.0, 0.3, length + 2.0), 0.05), roof, Vector3(side * depth * 0.5, 9.5, 0))
		for z in [-length * 0.5, 0.0, length * 0.5]:
			M.mi(node, M.box(Vector3(0.4, 9.5, 0.4), 0.04), concrete, Vector3(side * (depth + 0.5), 4.75, z))
		scenery.append(node)
		return


func _build_trees(rng: RandomNumberGenerator, playfield: float) -> void:
	var pines: Array = []
	var broad: Array = []
	var bushes: Array = []
	var extent := playfield + 60.0
	for _i in 4200:
		var x := rng.randf_range(-extent, extent)
		var y := rng.randf_range(-extent, extent)
		var d := _dist_to_track(x, y, 2)
		if d < width + 6.0 or d > 150.0:
			continue
		var sc := rng.randf_range(0.75, 1.5)
		var xf := Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(sc, sc * rng.randf_range(0.85, 1.25), sc)), b2g(x, y))
		var r := rng.randf()
		if r < 0.45:
			pines.append(xf)
		elif r < 0.85:
			broad.append(xf)
		elif d > width + 9.0:
			bushes.append(xf)
	for part in [Terrain.tree_multimesh(pines, "pine", 3), Terrain.tree_multimesh(broad, "broadleaf", 5),
			Terrain.multimesh_model("shrub_02", bushes, 2.0)]:
		add_child(part)
		scenery.append(part)


func _dist_to_track(x: float, y: float, stride: int) -> float:
	var best := INF
	for i in range(0, center.size(), stride):
		var p: Array = center[i]
		var dx: float = x - float(p[0])
		var dy: float = y - float(p[1])
		best = minf(best, dx * dx + dy * dy)
	return sqrt(best)


func _local_frame(i: int, forward: float, across: float) -> Node3D:
	## Node at centreline point i: local +X across the road, -Z the driving direction.
	var p: Array = center[i]
	var node := Node3D.new()
	node.rotation.y = float(p[2])
	node.position = b2g(float(p[0]), float(p[1])) + node.basis * Vector3(across, 0, -forward)
	add_child(node)
	return node


func _mesh(mesh: Mesh, mat: Material) -> void:
	var mi := M.mi(self, mesh, mat)
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
