extends EnvWorld
## Excavator: tracked hydraulic excavator at a quarry floor; steel hopper bin, blasted rock ledge.
## Python: hifi/exporters/excavator.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Moving ids: chassis, boom, dipper, bucket (class "excavator"), payload ("payload").
## Static: bin ("bin"), hydraulics (rams + bucket linkage, "excavator", posed from the link poses).
## p: tread (chassis x = distance driven, drives the track shoes), vx.

const Parts := preload("res://envs/excavator/parts.gd")

const GROUND := 2.0
const LANE := 9.0           # flat worksite |z| < LANE; berms beyond
const PAD := 4.5            # compacted gravel pad half-width (material blend)
const X0 := -60.0           # terrain strip start (the rear camera sits just behind it)
const X1 := 260.0
const Z_BACK := -90.0
const Z_FRONT := 60.0

# Arm attachment points (body-local)
const BOOM_RAM_BASE := Vector3(1.5, 0.05, 0.0)      # chassis
const BOOM_RAM_END := Vector3(0.5, -0.56, 0.0)      # boom
const BOOM_RAM_Z := 0.6
const DIPPER_RAM_BASE := Vector3(-0.7, 0.6, 0.0)    # boom
const DIPPER_RAM_END := Vector3(-2.25, 0.5, 0.0)    # dipper tail lug
const BUCKET_RAM_BASE := Vector3(-1.0, 0.52, 0.0)   # dipper
const ROCKER_PIVOT := Vector2(1.25, 0.12)           # dipper
const ROCKER_LEN := 0.55
const LINK_LEN := 0.6
const BUCKET_LUG := Vector3(-0.5, 0.32, 0.0)        # bucket

var _chassis: Node3D
var _boom: Node3D
var _dipper: Node3D
var _bucket: Node3D
var _tracks: Array = []
var _boom_rams: Array = []
var _dipper_ram: Node3D
var _bucket_ram: Node3D
var _rockers: Array = []
var _links: Array = []
var _smoke: Array = []
var _noise := FastNoiseLite.new()

var _yellow: Material
var _yellow_dirty: Material
var _dark: Material
var _steel: Material
var _worn: Material
var _chrome: Material
var _pin: Material
var _shoe: Material
var _glass: Material
var _rubber: Material


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "quarry_01_puresky", "sun_from_hdri": false, "sun_elev": 42.0, "sun_azim": 215.0,
		"sun_energy": 2.4, "sun_color": Color(1.0, 0.94, 0.84), "ambient_energy": 0.9,
		"exposure": 0.95, "white": 7.0, "ssao": true, "ssil": true, "glow": true,
		"saturation": 1.06, "contrast": 1.06, "shadow_distance": 180.0,
		"horizon_y": GROUND, "card_ground": "brown_mud_02", "card_ground_tint": Color(0.85, 0.8, 0.75),
	})


func build_scene(scene: Dictionary) -> void:
	_make_materials()
	_build_terrain()
	_build_backdrop()
	_build_props()
	_build_bin(scene.get("bin", {}))
	_build_rock_face(scene.get("face", {}))
	_chassis = _build_chassis()
	register("chassis", _chassis, "excavator")
	_boom = _build_boom()
	register("boom", _boom, "excavator")
	_dipper = _build_dipper()
	register("dipper", _dipper, "excavator")
	_bucket = _build_bucket()
	register("bucket", _bucket, "excavator")
	register("hydraulics", _build_hydraulics(), "excavator")
	register("payload", _build_payload(), "payload")


func _make_materials() -> void:
	_yellow = Parts.grime_paint(Color(0.92, 0.6, 0.02), {"grime": 0.4, "bottom": 1.0, "low": 2.0, "high": 2.9, "coat": 0.35, "seed": 3})
	_yellow_dirty = Parts.grime_paint(Color(0.92, 0.6, 0.02), {"grime": 0.85, "rough": 0.5, "coat": 0.2, "seed": 8, "scale": 1.1})
	_dark = Parts.grime_paint(Color(0.07, 0.07, 0.075), {"grime": 0.3, "bottom": 1.0, "low": 2.0, "high": 2.9, "rough": 0.55, "metal": 0.3, "coat": 0.1, "seed": 5})
	_steel = Parts.grime_paint(Color(0.2, 0.2, 0.21), {"grime": 0.7, "bottom": 1.0, "low": 2.0, "high": 2.8, "rough": 0.45, "metal": 0.8, "coat": 0.0, "seed": 6})
	_worn = Parts.grime_paint(Color(0.42, 0.41, 0.4), {"grime": 0.75, "rough": 0.35, "metal": 0.9, "coat": 0.0, "seed": 9, "scale": 1.4})
	_shoe = Parts.grime_paint(Color(0.1, 0.095, 0.09), {"grime": 0.9, "bottom": 1.0, "low": 2.0, "high": 2.7, "rough": 0.6, "metal": 0.6, "coat": 0.0, "seed": 12, "scale": 1.6})
	_chrome = M.mat(Color(0.86, 0.87, 0.9), 0.08, 1.0)
	_pin = M.mat(Color(0.3, 0.3, 0.31), 0.35, 0.9)
	_glass = M.glass(Color(0.08, 0.11, 0.13), 0.62)
	_rubber = M.mat(Color(0.03, 0.03, 0.03), 0.75, 0.0)


# ---------------------------------------------------------------- terrain / scenery

func _build_terrain() -> void:
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	var n := int((X1 - X0) / 1.0)
	for i in n + 1:
		xs.append(X0 + i * 1.0)
		ys.append(GROUND)
	var top := Parts.ground_blend("gravel", "brown_mud_02", {"a_uv": 0.32, "b_uv": 0.13, "lane": PAD, "soft": 3.5,
		"a_tint": Color(1.05, 1.0, 0.94), "b_tint": Color(1.0, 0.95, 0.9), "mask_freq": 0.025, "ruts": 0.85, "macro": 0.32,
		"rut_z": Vector2(1.15, 0.4), "seed": 31})
	var sec := Assets.pbr("excavated_soil_wall", {"uv": 0.3, "tint": Color(0.9, 0.85, 0.8)})
	_noise.seed = 4
	_noise.frequency = 0.05
	_noise.fractal_octaves = 3
	var t := Parts.heightfield(xs, Z_BACK, Z_FRONT, 0.75, _height, -3.0, top, sec)
	content.add_child(t)
	# Ground beyond the strip: a ring of quads at the lane height, in every view (one consistent
	# world; the ortho elevations see it edge-on). It casts no shadow: it would only shade the
	# strip's soil cross-sections below it.
	var ring := _ground_ring(Rect2(X0, Z_BACK, X1 - X0, Z_FRONT - Z_BACK), 2500.0, GROUND - 0.04,
		Parts.matte(Assets.pbr("brown_mud_02", {"uv": 0.2, "world": true, "tint": Color(0.9, 0.86, 0.8)})))
	ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(ring)


func _ground_ring(inner: Rect2, half: float, y: float, mat: Material) -> MeshInstance3D:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var x0 := inner.position.x
	var x1 := inner.end.x
	var z0 := inner.position.y
	var z1 := inner.end.y
	var quads := [[-half, -half, x0, half], [x1, -half, half, half], [x0, -half, x1, z0], [x0, z1, x1, half]]
	for q in quads:
		var a := Vector3(q[0], y, q[1])
		var b := Vector3(q[2], y, q[1])
		var c := Vector3(q[2], y, q[3])
		var d := Vector3(q[0], y, q[3])
		M._quad(st, [a, b, c, d], Vector3.UP)
	st.generate_tangents()
	var mi := MeshInstance3D.new()
	mi.name = "GroundRing"
	mi.mesh = st.commit()
	mi.material_override = mat
	return mi


func _build_backdrop() -> void:
	## Quarry walls around the site. Cross-view consistency: drawn in every view whose frame
	## contains them; the ortho rear / side elevations show them full size behind the site (their
	## far planes in exporters/excavator.py reach past them), so the rims stay low (~5-10 m) and the
	## elevations keep sky in their upper half.
	var rock := Assets.pbr("rock_face", {"uv": 0.06, "world": true, "tint": Color(0.78, 0.72, 0.66)})
	var bench := Assets.pbr("dry_ground_01", {"uv": 0.04, "world": true, "tint": Color(0.72, 0.62, 0.52)})
	var noise := FastNoiseLite.new()
	noise.seed = 19
	noise.frequency = 0.008
	var walls := Node3D.new()
	walls.name = "QuarryWalls"
	content.add_child(walls)
	# Side walls (behind and in front of the lane), long in x
	for side in [-1.0, 1.0]:
		var xs := PackedFloat32Array()
		var ys := PackedFloat32Array()
		for i in 121:
			var x := -700.0 + i * 12.0
			xs.append(x)
			ys.append(GROUND + 6.0 + 3.5 * noise.get_noise_1d(x + side * 300.0) + 1.5 * noise.get_noise_1d(x * 3.0))
		var z0 := -520.0 if side < 0.0 else 210.0
		var z1 := -240.0 if side < 0.0 else 490.0
		var s := Terrain.side_strip(xs, ys, z0, z1, -5.0, bench, rock, {"lane": -1.0, "rough": 2.5, "dz": 14.0, "freq": 0.02, "seed": 7 + int(side), "front_down": false})
		walls.add_child(s)
	# End walls across the x axis (ahead of and behind the machine): blasted benches (4 m faces,
	# 25 m berms)
	for end in [-1.0, 1.0]:
		var xs := PackedFloat32Array()
		var ys := PackedFloat32Array()
		for i in 161:
			var d := i * 2.5                     # distance from the toe of the wall
			var x := 300.0 + d if end > 0.0 else -300.0 - d
			var lvl := 0.0
			for k in 2:
				lvl += smoothstep(k * 28.0, k * 28.0 + 2.0, d) * (4.0 + 1.2 * noise.get_noise_1d(x * 0.5 + k * 70.0))
			xs.append(x)
			ys.append(GROUND - 1.0 + lvl)
		if end < 0.0:
			xs.reverse()
			ys.reverse()
		var s := Terrain.side_strip(xs, ys, -520.0, 490.0, -5.0, rock, rock, {"lane": -1.0, "rough": 2.5, "dz": 10.0, "freq": 0.04, "seed": 13 + int(end), "front_down": false})
		walls.add_child(s)


func _build_props() -> void:
	## Worksite clutter behind the lane (z < 0) or low in the dip in front; nothing inside the
	## gameplay lane. Cross-view consistency: every prop is drawn in every view whose frame contains
	## it. Nothing stands between the machine and a camera: the rear camera looks along the lane
	## (props are beside it), and the front rocks sit in the dip with their tops at most ~0.3 m
	## above the lane, so the side view still sees the tracks.
	var rng := RandomNumberGenerator.new()
	rng.seed = 77
	var gravel := Assets.pbr("gravel", {"uv": 0.6, "tint": Color(0.85, 0.8, 0.74)})
	var soil := Assets.pbr("brown_mud_02", {"uv": 0.4})
	# Stockpiles (noisy cones)
	for sp in [[4.0, -15.0, 3.2, 1.9, gravel], [13.0, -17.5, 4.0, 2.6, soil], [30.0, -14.0, 3.5, 2.2, gravel], [-12.0, -16.0, 3.0, 1.6, soil]]:
		var base := INF
		for k in 12:
			var a := TAU * k / 12.0
			base = minf(base, _h(sp[0] + cos(a) * sp[2], sp[1] + sin(a) * sp[2]))
		var pile := M.mi(content, _pile_mesh(sp[2], sp[3], rng.randi()), sp[4], Vector3(sp[0], base - 0.3, sp[1]))
		pile.name = "Pile"
	# Blasted rocks on the berms (low-poly procedural: the scanned boulder is 66k triangles)
	var rock_mat := Assets.pbr("rock_face", {"uv": 0.9, "tint": Color(0.8, 0.74, 0.68)})
	var back_rocks := M.node(content, Vector3.ZERO, Vector3.ZERO, "BermRocks")
	var front_rocks := M.node(content, Vector3.ZERO, Vector3.ZERO, "FrontRocks")
	for i in 34:
		var x := rng.randf_range(-30.0, 60.0)
		var z := rng.randf_range(-34.0, -LANE - 0.5) if i % 3 != 0 else rng.randf_range(LANE + 3.0, 30.0)
		var s := rng.randf_range(0.5, 1.7)
		var yaw := rng.randf() * 360.0
		var sy := rng.randf_range(0.7, 1.1)
		if z > 0.0:
			# front rocks: down in the dip (z 22..42), low enough not to hide the tracks from the side
			z = remap(z, LANE + 3.0, 30.0, 22.0, 42.0)
			s = clampf((GROUND + 0.3 - _h(x, z)) / (0.8 * sy * 1.15 - 0.12), 0.35, s)
		var mesh := Parts.rock_mesh(Vector3(1.3, 0.8, 1.0), i % 7, 0.6)
		var r := M.mi(back_rocks if z < 0.0 else front_rocks, mesh, rock_mat, Vector3(x, _h(x, z) - 0.12 * s, z),
			Vector3(0, yaw, 0), Vector3(s, s * sy, s))
		r.name = "Rock"
	# Site kit behind the lane: barrels, tyres, a site container and a light mast
	var kit := M.node(content, Vector3.ZERO, Vector3.ZERO, "SiteKit")
	for i in 5:
		var br := Assets.fitted_model("Barrel_01", Vector3(0.6, 0.9, 0.6))
		if br:
			br.position = Vector3(-3.0 + (i % 3) * 0.66, GROUND, -7.0 - (i / 3) * 0.66)
			br.rotation.y = rng.randf() * TAU
			kit.add_child(br)
	for i in 4:
		var ty := Assets.fitted_model("old_tyre", Vector3(1.3, 1.3, 0.38))
		if ty:
			ty.rotation = Vector3(PI * 0.5, i * 0.7, 0)
			ty.position = Vector3(0.8, GROUND + 0.19 + i * 0.36, -7.4)
			kit.add_child(ty)
	_build_container(kit, Vector3(-9.0, GROUND, -7.7), 0.04)
	_build_light_mast(Vector3(27.0, GROUND, -7.6))


func _pile_mesh(r: float, h: float, seed: int) -> ArrayMesh:
	var noise := FastNoiseLite.new()
	noise.seed = seed
	noise.frequency = 0.9
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var rings := 8
	var segs := 28
	var pts: Array = []
	# smooth shading: positions only, indexed, normals generated
	for i in rings + 1:
		var t := float(i) / rings
		var row: Array = []
		for j in segs:
			var a := TAU * j / segs
			var rr := r * (1.0 - t) * (1.0 + 0.18 * noise.get_noise_2d(cos(a) * 3.0, sin(a) * 3.0 + t * 2.0))
			var y := h * (1.0 - pow(1.0 - t, 1.6)) * (1.0 + 0.1 * noise.get_noise_2d(a * 2.0, t * 5.0))
			row.append(Vector3(cos(a) * rr, y, sin(a) * rr))
		pts.append(row)
	for i in rings:
		for j in segs:
			var j1 := (j + 1) % segs
			var a: Vector3 = pts[i][j]
			var b: Vector3 = pts[i][j1]
			var c: Vector3 = pts[i + 1][j1]
			var d: Vector3 = pts[i + 1][j]
			var nrm := ((a + b + c + d) * 0.25 - Vector3(0, h * 0.3, 0)).normalized()
			Parts._tri_auto(st, a, b, c, nrm)
			Parts._tri_auto(st, a, c, d, nrm)
	st.index()
	st.generate_normals()
	return st.commit()


func _h(x: float, z: float) -> float:
	return _height(x, z)


func _height(x: float, z: float) -> float:
	## Flat worksite (|z| < LANE, the Box2D ground), spoil berms behind it (the side view's
	## background), a shallow dip in front (never above the lane, so nothing hides the machine
	## from the side camera) that comes back to the lane height at the strip edge.
	if z < -LANE:
		var t := clampf((-z - LANE) / 7.0, 0.0, 1.0)
		var fade := clampf((z - Z_BACK) / 25.0, 0.0, 1.0)
		var n1 := _noise.get_noise_2d(x, z)
		var n2 := _noise.get_noise_2d(x * 5.0 + 100.0, z * 5.0)
		return GROUND + t * t * (3.0 - 2.0 * t) * fade * (1.7 + 1.5 * n1 + 0.15 * n2)
	if z > LANE:
		var t := clampf((z - LANE) / (Z_FRONT - LANE), 0.0, 1.0)
		var n1 := _noise.get_noise_2d(x + 50.0, z)
		return GROUND - sin(PI * t) * (0.75 + 0.45 * n1)
	return GROUND


func _build_container(parent: Node3D, pos: Vector3, yaw: float) -> void:
	var root := M.node(parent, pos, Vector3(0, rad_to_deg(yaw), 0), "Container")
	var iron := Assets.pbr("corrugated_iron", {"uv": 0.8, "tint": Color(0.35, 0.5, 0.42)})
	var frame := M.mat(Color(0.18, 0.26, 0.22), 0.6, 0.5)
	M.mi(root, M.box(Vector3(6.06, 2.59, 2.44), 0.03), iron, Vector3(0, 1.295, 0))
	for x in [-3.0, 3.0]:
		for z in [-1.19, 1.19]:
			M.mi(root, M.box(Vector3(0.14, 2.6, 0.14), 0.01), frame, Vector3(x, 1.3, z))
	M.mi(root, M.box(Vector3(0.9, 1.1, 0.03), 0.01), M.glass(Color(0.1, 0.12, 0.14), 0.9), Vector3(1.2, 1.5, 1.23))
	M.mi(root, M.box(Vector3(0.9, 2.0, 0.03), 0.01), frame, Vector3(-1.5, 1.0, 1.23))


func _build_light_mast(pos: Vector3) -> Node3D:
	var root := M.node(content, pos, Vector3.ZERO, "LightMast")
	var paint := Parts.grime_paint(Color(0.85, 0.55, 0.05), {"grime": 0.5, "coat": 0.3})
	M.mi(root, M.box(Vector3(2.2, 0.9, 1.3), 0.06), paint, Vector3(0, 0.75, 0))
	for x in [-0.8, 0.8]:
		var w := M.node(root, Vector3(x, 0.32, 0.72))
		M.wheel(w, 0.32, 0.2, _rubber, _pin).rotation = Vector3(0, PI * 0.5, 0)
	M.mi(root, M.cyl(0.07, 0.1, 7.5, 10), _pin, Vector3(0, 4.9, 0))
	var head := M.node(root, Vector3(0, 8.6, 0))
	M.mi(head, M.box(Vector3(1.6, 0.08, 0.08)), _pin)
	for k in 4:
		var lamp := M.node(head, Vector3(-0.6 + k * 0.4, 0.2, 0.0), Vector3(-15, 0, 0))
		M.mi(lamp, M.box(Vector3(0.34, 0.3, 0.14), 0.02), M.mat(Color(0.15, 0.15, 0.16), 0.5, 0.6))
		M.mi(lamp, M.box(Vector3(0.28, 0.24, 0.02)), M.mat(Color(0.85, 0.85, 0.8), 0.1, 0.0), Vector3(0, 0, 0.075))
	return root


# ---------------------------------------------------------------- bin and rock face

func _build_bin(d: Dictionary) -> void:
	var pos: Array = d.get("pos", [17.0, 2.0])
	var root := M.node(content, Vector3(float(pos[0]), float(pos[1]), 0), Vector3.ZERO, "Bin")
	var blue := Assets.pbr("blue_metal_plate", {"uv": 0.45, "tint": Color(0.85, 0.9, 1.0), "metallic": 0.6})
	var blue_frame := Parts.grime_paint(Color(0.06, 0.16, 0.42), {"grime": 0.55, "bottom": 1.0, "low": 2.0, "high": 2.8, "rough": 0.5, "metal": 0.4, "coat": 0.3, "seed": 17})
	var bar := Parts.grime_paint(Color(0.12, 0.13, 0.15), {"grime": 0.7, "rough": 0.5, "metal": 0.7, "coat": 0.0, "seed": 18})
	# Floor (y 0..0.2) and end walls (x = +-1.7..2.0, y 0.2..1.8) = the Box2D fixtures
	M.mi(root, M.box(Vector3(4.0, 0.2, 4.0), 0.02), blue, Vector3(0, 0.1, 0))
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.3, 1.6, 4.0), 0.03), blue, Vector3(sx * 1.85, 1.0, 0))
		# Outside stiffeners and top rim
		for y in [0.55, 1.15]:
			M.mi(root, M.box(Vector3(0.05, 0.12, 3.9), 0.01), blue_frame, Vector3(sx * 2.0, y, 0))
		M.mi(root, M.box(Vector3(0.32, 0.08, 4.0), 0.015), blue_frame, Vector3(sx * 1.85, 1.76, 0))
		for z in [-1.7, 1.7]:
			var lug := M.mi(root, M.cyl(0.07, 0.07, 0.05, 12), blue_frame, Vector3(sx * 1.85, 1.86, z), Vector3(0, 0, 90))
			lug.name = "Lug"
	# Side cages at z = +-2: frame + vertical bars (the payload stays visible from the side)
	for sz in [-1.0, 1.0]:
		var z: float = sz * 1.94
		M.mi(root, M.box(Vector3(3.4, 0.1, 0.1), 0.01), blue_frame, Vector3(0, 1.72, z))
		M.mi(root, M.box(Vector3(3.4, 0.12, 0.12), 0.01), blue_frame, Vector3(0, 0.28, z))
		for x in [-1.0, 0.0, 1.0]:
			M.mi(root, M.box(Vector3(0.1, 1.4, 0.1), 0.01), blue_frame, Vector3(x, 1.0, z))
		for k in 16:
			var x := -1.6 + k * (3.2 / 15.0)
			if absf(x - roundf(x)) < 0.06:
				continue
			M.mi(root, M.box(Vector3(0.035, 1.38, 0.035)), bar, Vector3(x, 1.0, z))
		# diagonal braces
		for k in 2:
			var br := M.mi(root, M.box(Vector3(0.05, 1.7, 0.05)), blue_frame, Vector3(-1.5 + k * 3.0, 1.0, z * 0.985))
			br.rotation = Vector3(0, 0, (0.38 if k == 0 else -0.38))
	register("bin", root, "bin")


func _build_rock_face(d: Dictionary) -> void:
	var pos: Array = d.get("pos", [25.5, 3.5])
	var root := M.node(content, Vector3(float(pos[0]), float(pos[1]), 0), Vector3.ZERO, "RockFace")
	var rock := Assets.pbr("rock_face", {"uv": 0.32, "tint": Color(0.82, 0.76, 0.7)})
	# The near face (-x, where the payload rests) and the base stay inside the Box2D box
	var mesh := Parts.rock_mass(Vector3(1.5, 1.5, 4.0), {"amp": 0.32, "freq": 0.55, "seed": 23, "n": 4.0, "round": 0.45,
		"keep": [[0, -1.0, 0.1], [1, -1.0], [1, 1.0, 0.18]]})
	M.mi(root, mesh, rock)
	# Rubble and boulders around the back and the flanks (never in front of the face)
	var rng := RandomNumberGenerator.new()
	rng.seed = 41
	var rubble := Assets.pbr("rock_face", {"uv": 0.7, "tint": Color(0.78, 0.72, 0.66)})
	for i in 12:
		var back := i < 6
		var x := rng.randf_range(1.4, 3.4) if back else rng.randf_range(-0.8, 1.4)
		var z := rng.randf_range(-4.5, 4.5) if back else (rng.randf_range(4.1, 5.8) * (1.0 if i % 2 == 0 else -1.0))
		var s := rng.randf_range(0.6, 1.5)
		if i < 2:
			var b := Assets.fitted_model("boulder_01", Vector3(s * 1.2, s * 0.8, s), "stretch")
			if b != null:
				b.position = Vector3(x, -1.5 - 0.12 * s, z)
				b.rotation = Vector3(0, rng.randf() * TAU, 0)
				root.add_child(b)
				continue
		M.mi(root, Parts.rock_mesh(Vector3(1.2, 0.85, 1.0), 20 + i % 5, 0.7), rubble, Vector3(x, -1.5 - 0.1 * s, z),
			Vector3(0, rng.randf() * 360.0, 0), Vector3(s, s, s))


# ---------------------------------------------------------------- excavator

func _build_chassis() -> Node3D:
	## Box2D chassis: 4 x 2 m box (local y -1..1), extruded z +-1.5. Boom pin at (1.0, 1.0).
	var root := M.node(content, Vector3(8, 3, 0), Vector3.ZERO, "Chassis")
	# Undercarriage: two crawler tracks (z +-1.15, 0.6 wide), car body, swing bearing
	for zc in [-1.15, 1.15]:
		_tracks.append(Parts.track(root, {"half_len": 1.66, "cy": -0.70, "r": 0.27, "z": zc, "w": 0.6, "pitch": 0.19,
			"shoe": _shoe, "frame": _dark, "steel": _steel, "rollers": 5, "carriers": 2}))
	M.mi(root, M.box(Vector3(2.2, 0.32, 1.66), 0.05), _dark, Vector3(0, -0.66, 0))
	M.mi(root, M.cyl(0.8, 0.85, 0.12, 32), _dark, Vector3(0, -0.36, 0))
	# Upper structure: deck plate, counterweight (rounded rear), engine hood, tool box
	M.mi(root, M.box(Vector3(3.45, 0.2, 2.84), 0.04), _yellow, Vector3(-0.22, -0.2, 0))
	var cw := M.poly([[-1.22, -1.42], [-1.22, 1.42], [-1.55, 1.42], [-1.82, 1.32], [-1.95, 1.0], [-1.99, 0.0],
		[-1.95, -1.0], [-1.82, -1.32], [-1.55, -1.42]])
	M.mi(root, M.prism_up(cw, -0.3, 0.5), _yellow)
	M.mi(root, M.prism_up(_inset(cw, 0.04), 0.5, 0.54), _dark)
	# Hazard plate and tail lights on the counterweight back
	M.mi(root, M.box(Vector3(0.02, 0.22, 1.5)), Parts.stripes(Color(0.86, 0.56, 0.04), Color(0.03, 0.03, 0.03), 0.1), Vector3(-1.995, 0.2, 0))
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.03, 0.08, 0.16), 0.01), M.emissive(Color(0.8, 0.05, 0.03), 0.6), Vector3(-1.97, 0.38, sz * 0.95))
	M.mi(root, M.box(Vector3(0.9, 0.56, 2.76), 0.07), _yellow, Vector3(-0.8, 0.17, 0))
	for sz in [-1.0, 1.0]:
		for k in 3:
			M.mi(root, M.box(Vector3(0.5, 0.035, 0.02)), _dark, Vector3(-0.85, 0.05 + k * 0.09, sz * 1.385))
	M.mi(root, M.box(Vector3(0.3, 0.12, 0.5), 0.03), _dark, Vector3(-0.9, 0.5, -0.6))
	M.mi(root, M.cyl(0.05, 0.05, 0.42, 10), _pin, Vector3(-0.7, 0.66, -0.95))
	M.mi(root, M.cyl(0.062, 0.05, 0.06, 10), _dark, Vector3(-0.7, 0.87, -0.95))
	_smoke = Parts.exhaust(root, Vector3(-0.7, 1.0, -0.95))
	M.mi(root, M.box(Vector3(1.9, 0.42, 0.78), 0.05), _yellow, Vector3(0.53, 0.11, -1.02))
	M.mi(root, M.box(Vector3(1.5, 0.02, 0.6), 0.0), _dark, Vector3(0.53, 0.33, -1.02))
	# Boom foot: twin plates around the boom root, pin at (1.0, 1.0)
	var foot := M.poly([[0.3, -0.12], [1.62, -0.12], [1.62, 0.2], [1.28, 0.75], [1.2, 1.12], [0.8, 1.12], [0.62, 0.62]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(foot, sz * 0.42 - 0.06, sz * 0.42 + 0.06), _yellow)
	M.mi(root, M.box(Vector3(1.1, 0.5, 0.78), 0.04), _yellow, Vector3(0.95, 0.12, 0))
	Parts.pin(root, Vector3(1.0, 1.0, 0), 0.075, 1.02, _pin)
	# Boom ram base pins
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.22, 0.26, 0.1), 0.02), _yellow, Vector3(1.5, 0.05, sz * (BOOM_RAM_Z + 0.12)))
	# Cab on the +z side (the side camera's side), roof at the chassis top (y = 1.0)
	var cab := Parts.cab(root, Vector3(1.7, 1.12, 0.7), _yellow, _glass, M.mat(Color(0.05, 0.05, 0.05), 0.7))
	cab.position = Vector3(0.38, -0.12, 1.07)
	M.mi(cab, M.box(Vector3(0.14, 0.06, 0.1), 0.01), M.emissive(Color(1.0, 0.45, 0.05), 1.5), Vector3(-0.6, 1.14, 0.2))
	for k in 2:
		M.mi(cab, M.box(Vector3(0.08, 0.07, 0.14), 0.01), M.mat(Color(0.9, 0.9, 0.85), 0.15, 0.2), Vector3(0.62, 1.12, -0.22 + k * 0.44))
	# Handrails along the hood and the tool box
	var rail := M.mat(Color(0.86, 0.56, 0.04), 0.5, 0.3)
	M.mi(root, M.tube(PackedVector3Array([Vector3(-1.3, 0.45, -1.3), Vector3(-1.3, 0.7, -1.3), Vector3(1.2, 0.7, -1.3), Vector3(1.2, 0.32, -1.3)]), 0.022, 6), rail)
	# Steps on the track frame below the cab
	M.mi(root, M.box(Vector3(0.36, 0.03, 0.22), 0.0), _dark, Vector3(0.9, -0.28, 1.36))
	return root


func _inset(p: PackedVector2Array, d: float) -> PackedVector2Array:
	var res := Geometry2D.offset_polygon(p, -d)
	return res[0] if res.size() > 0 else p


func _build_boom() -> Node3D:
	## Box2D boom: 6 x 0.8 m (local x -3..3), extruded z +-0.6; pins at x = -2.8 (root) and 2.8 (tip).
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Boom")
	var body := M.poly([[-2.55, -0.36], [0.8, -0.4], [2.35, -0.3], [2.45, -0.27], [2.45, 0.27], [2.35, 0.3], [0.8, 0.4], [-2.55, 0.34]])
	M.mi(root, M.prism(body, -0.38, 0.38), _yellow)
	# Root (full width, rounded) and tip tongue (narrow, between the dipper fork plates)
	M.mi(root, M.prism(_round_end(Vector2(-2.8, 0), 0.26, -2.5, true), -0.38, 0.38), _yellow)
	M.mi(root, M.prism(_round_end(Vector2(2.8, 0), 0.24, 2.4, false), -0.16, 0.16), _yellow)
	Parts.pin(root, Vector3(-2.8, 0, 0), 0.17, 0.84, _yellow)
	Parts.pin(root, Vector3(2.8, 0, 0), 0.075, 0.98, _pin)
	# Weld seams / top plate lines
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(4.6, 0.025, 0.01)), _dark, Vector3(-0.1, 0.2, sz * 0.385))
	# Boom ram lug (underneath) with a cross tube, dipper ram bracket (on top)
	M.mi(root, M.prism(M.poly([[0.1, -0.38], [0.9, -0.38], [0.62, -0.62], [0.38, -0.62]]), -0.3, 0.3), _yellow)
	Parts.pin(root, BOOM_RAM_END, 0.07, BOOM_RAM_Z * 2.0 + 0.16, _pin)
	M.mi(root, M.prism(M.poly([[-1.2, 0.32], [-0.2, 0.38], [-0.55, 0.7], [-0.85, 0.7]]), -0.14, 0.14), _yellow)
	Parts.pin(root, DIPPER_RAM_BASE, 0.06, 0.34, _pin)
	# Hydraulic lines along the top
	for sz in [-0.12, 0.12]:
		Parts.hose(root, PackedVector3Array([Vector3(-2.6, 0.36, sz), Vector3(0.6, 0.44, sz), Vector3(2.3, 0.34, sz), Vector3(2.62, 0.18, sz)]), 0.025, _rubber)
	return root


func _round_end(c: Vector2, r: float, x_join: float, left: bool) -> PackedVector2Array:
	## Polygon: half-disc of radius r around c towards the open end, joined by a rectangle to x_join.
	var pts := PackedVector2Array()
	var n := 10
	if left:
		pts.append(Vector2(x_join, -r))
		for i in n + 1:
			var a := -PI * 0.5 - PI * i / n
			pts.append(c + Vector2(cos(a), sin(a)) * r)
		pts.append(Vector2(x_join, r))
	else:
		pts.append(Vector2(x_join, r))
		for i in n + 1:
			var a := PI * 0.5 - PI * i / n
			pts.append(c + Vector2(cos(a), sin(a)) * r)
		pts.append(Vector2(x_join, -r))
	if M.signed_area(pts) < 0.0:
		pts.reverse()
	return pts


func _build_dipper() -> Node3D:
	## Box2D dipper (stick): 4 x 0.6 m (local x -2..2), extruded z +-0.5; pins at -1.8 / 1.8.
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Dipper")
	var body := M.poly([[-1.45, -0.29], [1.2, -0.27], [2.0, -0.15], [2.0, 0.15], [1.45, 0.24], [-1.45, 0.3]])
	M.mi(root, M.prism(body, -0.3, 0.3), _yellow)
	# Root fork (around the boom tongue) with the tail lug for the dipper ram
	var fork := M.poly([[-1.35, -0.29], [-1.8, -0.27], [-2.06, -0.08], [-2.08, 0.2], [-2.38, 0.42], [-2.36, 0.6], [-2.1, 0.68], [-1.35, 0.3]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(fork, sz * 0.24 - 0.065, sz * 0.24 + 0.065), _yellow)
	Parts.pin(root, Vector3(-1.8, 0, 0), 0.14, 0.62, _yellow)
	Parts.pin(root, DIPPER_RAM_END, 0.055, 0.6, _pin)
	# Bucket ram bracket on top, rocker pivot and the bucket pin bosses at the tip
	M.mi(root, M.prism(M.poly([[-1.35, 0.28], [-0.65, 0.3], [-0.85, 0.62], [-1.12, 0.62]]), -0.12, 0.12), _yellow)
	Parts.pin(root, BUCKET_RAM_BASE, 0.05, 0.3, _pin)
	Parts.pin(root, Vector3(ROCKER_PIVOT.x, ROCKER_PIVOT.y, 0), 0.07, 0.86, _pin)
	Parts.pin(root, Vector3(1.8, 0, 0), 0.11, 0.66, _yellow)
	Parts.pin(root, Vector3(1.8, 0, 0), 0.06, 0.92, _pin)
	for sz in [-0.1, 0.1]:
		Parts.hose(root, PackedVector3Array([Vector3(-1.5, 0.33, sz), Vector3(0.6, 0.3, sz), Vector3(1.3, 0.26, sz)]), 0.022, _rubber)
	return root


func _build_bucket() -> Node3D:
	## Box2D bucket: back wall box (0.16 x 1.1 at (-0.8, -0.35)) + floor polygon with a cutting edge,
	## extruded z +-0.8; pivot at (-0.9, 0). Visual side plates keep the payload contained.
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Bucket")
	var wall := M.poly([[-0.88, -0.9], [-0.72, -0.9], [-0.72, 0.2], [-0.88, 0.2]])
	var floor := M.poly([[-0.88, -0.92], [1.05, -0.92], [0.55, -0.8], [-0.88, -0.8]])
	M.mi(root, M.prism(wall, -0.76, 0.76), _yellow_dirty)
	M.mi(root, M.prism(floor, -0.76, 0.76), _worn)
	# Rounded heel (outer back-bottom corner) and a wear strip under the floor
	M.mi(root, M.prism(M.poly([[-0.88, -0.92], [-0.88, -0.6], [-0.98, -0.72], [-0.98, -0.84], [-0.94, -0.9]]), -0.76, 0.76), _yellow_dirty)
	M.mi(root, M.box(Vector3(1.6, 0.03, 1.5)), _steel, Vector3(0.05, -0.935, 0))
	# Side plates (scooped profile) with a reinforcing rim
	var side := M.poly([[-0.88, 0.2], [-0.88, -0.92], [1.05, -0.92], [0.98, -0.84], [0.6, -0.72], [0.2, -0.58],
		[-0.2, -0.42], [-0.5, -0.22], [-0.66, 0.0], [-0.72, 0.2]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(side, sz * 0.78 - 0.025, sz * 0.78 + 0.025), _yellow_dirty)
		M.mi(root, M.tube(PackedVector3Array([Vector3(0.98, -0.84, sz * 0.78), Vector3(0.6, -0.72, sz * 0.78), Vector3(0.2, -0.58, sz * 0.78),
			Vector3(-0.2, -0.42, sz * 0.78), Vector3(-0.5, -0.22, sz * 0.78), Vector3(-0.66, 0.0, sz * 0.78), Vector3(-0.72, 0.2, sz * 0.78)]), 0.03, 6), _steel)
	# Teeth along the cutting edge
	var tooth := M.poly([[0.82, -0.925], [1.21, -0.905], [1.23, -0.88], [0.82, -0.81]])
	for k in 5:
		var z := -0.6 + k * 0.3
		M.mi(root, M.prism(tooth, z - 0.055, z + 0.055), _steel)
	# Ears (pivot and linkage lug) on top of the back wall
	var ear := M.poly([[-1.06, -0.04], [-0.88, -0.24], [-0.72, 0.2], [-0.44, 0.22], [-0.38, 0.42], [-0.58, 0.48], [-1.0, 0.18]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(ear, sz * 0.38 - 0.045, sz * 0.38 + 0.045), _yellow_dirty)
	Parts.pin(root, Vector3(-0.9, 0, 0), 0.08, 0.9, _pin)
	Parts.pin(root, BUCKET_LUG, 0.06, 0.9, _pin)
	return root


func _build_hydraulics() -> Node3D:
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Hydraulics")
	var body := Parts.grime_paint(Color(0.1, 0.1, 0.1), {"grime": 0.45, "rough": 0.4, "metal": 0.5, "coat": 0.3, "seed": 14})
	for sz in [-1.0, 1.0]:
		_boom_rams.append(Parts.ram(root, {"r": 0.1, "rod": 0.055, "lb": 2.15, "lr": 2.15, "body": _yellow, "chrome": _chrome, "pin": _pin}))
	_dipper_ram = Parts.ram(root, {"r": 0.095, "rod": 0.05, "lb": 2.25, "lr": 2.1, "body": _yellow, "chrome": _chrome, "pin": _pin})
	_bucket_ram = Parts.ram(root, {"r": 0.08, "rod": 0.045, "lb": 1.55, "lr": 1.45, "body": _yellow, "chrome": _chrome, "pin": _pin})
	# Bucket linkage: rocker (dipper -> joint) and connecting link (joint -> bucket lug), twin plates
	var rk := M.node(root, Vector3.ZERO, Vector3.ZERO, "Rocker")
	for sz in [-1.0, 1.0]:
		M.mi(rk, M.box(Vector3(0.12, ROCKER_LEN + 0.12, 0.05), 0.02), body, Vector3(0, ROCKER_LEN * 0.5, sz * 0.4))
	Parts.pin(rk, Vector3(0, ROCKER_LEN, 0), 0.055, 0.88, _pin)
	_rockers.append(rk)
	var lk := M.node(root, Vector3.ZERO, Vector3.ZERO, "Link")
	for sz in [-1.0, 1.0]:
		M.mi(lk, M.box(Vector3(0.13, LINK_LEN + 0.12, 0.05), 0.02), _yellow, Vector3(0, LINK_LEN * 0.5, sz * 0.47))
	_links.append(lk)
	_update_hydraulics()
	return root


func _build_payload() -> Node3D:
	## Octagonal rock (circumradius 0.35, flats 0.323 from the centre), extruded z +-0.4.
	var root := M.node(content, Vector3(23.5, 2.32, 0), Vector3.ZERO, "Payload")
	var rock := Assets.fitted_model("boulder_01", Vector3(0.7, 0.66, 0.8), "stretch")
	if rock != null:
		rock.position = Vector3(0, -0.33, 0)
		root.add_child(rock)
	else:
		M.mi(root, M.sphere(0.34), Assets.pbr("rock_face", {"uv": 1.0}))
	return root


# ---------------------------------------------------------------- per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var tread := float(p.get("tread", _chassis.position.x))
	for t in _tracks:
		Parts.set_track(t, tread)
	_update_hydraulics()
	Parts.set_exhaust(_smoke, sim_t, float(p.get("load", 0.0)))


func _update_hydraulics() -> void:
	if _chassis == null or _boom == null or _dipper == null or _bucket == null:
		return
	var ct := _chassis.transform
	var bt := _boom.transform
	var dt := _dipper.transform
	var kt := _bucket.transform
	for i in _boom_rams.size():
		var z := BOOM_RAM_Z * (1.0 if i == 0 else -1.0)
		Parts.set_ram(_boom_rams[i], ct * (BOOM_RAM_BASE + Vector3(0, 0, z)), bt * (BOOM_RAM_END + Vector3(0, 0, z)))
	Parts.set_ram(_dipper_ram, bt * DIPPER_RAM_BASE, dt * DIPPER_RAM_END)
	# Four-bar bucket linkage, solved in the dipper frame
	var lug_w := kt * BUCKET_LUG
	var lug := dt.affine_inverse() * lug_w
	var j := _circle_joint(ROCKER_PIVOT, ROCKER_LEN, Vector2(lug.x, lug.y), LINK_LEN)
	var j_w := dt * Vector3(j.x, j.y, 0)
	Parts.set_ram(_bucket_ram, dt * BUCKET_RAM_BASE, j_w)
	_orient(_rockers[0], dt * Vector3(ROCKER_PIVOT.x, ROCKER_PIVOT.y, 0), j_w)
	_orient(_links[0], j_w, lug_w)


static func _circle_joint(c0: Vector2, r0: float, c1: Vector2, r1: float) -> Vector2:
	## Intersection of two circles on the left of c0 -> c1 (continuous over the bucket's range).
	var dv := c1 - c0
	var d := dv.length()
	if d < 1e-5:
		return c0 + Vector2(0, r0)
	var u := dv / d
	if d >= r0 + r1 or d <= absf(r0 - r1):
		return c0 + u * r0
	var a := (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
	var h := sqrt(maxf(r0 * r0 - a * a, 0.0))
	return c0 + u * a + Vector2(-u.y, u.x) * h


static func _orient(n: Node3D, a: Vector3, b: Vector3) -> void:
	var d := b - a
	if d.length() < 1e-5:
		return
	var y := d.normalized()
	var x := y.cross(Vector3(0, 0, 1)).normalized()
	n.transform = Transform3D(Basis(x, y, x.cross(y)), a)


func chase_pose() -> Transform3D:
	var p := _chassis.global_position if _chassis else Vector3(8, 3, 0)
	return Transform3D(Basis(), p + Vector3(-6.0, 4.0, 11.0)).looking_at(p + Vector3(3.5, 1.0, 0), Vector3.UP)
