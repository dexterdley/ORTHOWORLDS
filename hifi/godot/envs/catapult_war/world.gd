extends EnvWorld
## Catapult War: timber siege catapult (mangonel) with iron fittings and a rope winch on a meadow,
## lobbing stone boulders at a brick tower flanked by two sandstone pillars; banners show the wind.
## Python: hifi/exporters/catapult_war.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Ids: base (static, "catapult"), arm ("catapult"), rigging (winch rope, "catapult"), block0..4
## ("block": 0-2 tower, 3-4 pillars). dyn: boulder<n> (class "boulder", recreated every shot),
## dust<i> (toppled block). p: tension, wind, aim, loaded, shots_left, toppled[5].

const Parts := preload("res://envs/excavator/parts.gd")
const Nature := preload("res://envs/catapult_war/nature.gd")

const GROUND := 1.0
const LANE := 5.0
const X0 := -60.0           # terrain strip start (the rear camera sits just behind it)
const X1 := 260.0
const Z_BACK := -90.0
const Z_FRONT := 45.0
const PIVOT := Vector2(9.0, 3.5)
const ROPE_ARM := Vector3(0.3, -0.2, 0.0)      # arm-local rope anchor (under the arm)
const DRUM := Vector3(11.0, 3.28, 0.0)         # winch drum axis (world)

var _base: Node3D
var _arm: Node3D
var _drum: Node3D
var _rope_mesh := ArrayMesh.new()
var _banners: Array = []
var _block_mats: Array = []
var _noise := FastNoiseLite.new()
var _wind := 0.0

var _wood: Material
var _wood_dark: Material
var _iron: Material
var _rope_mat: Material


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "qwantani_late_afternoon_puresky", "sun_from_hdri": false, "sun_elev": 34.0, "sun_azim": 215.0,
		"sun_energy": 1.8, "sun_color": Color(1.0, 0.93, 0.82), "ambient_energy": 0.8,
		"exposure": 0.9, "white": 7.0, "ssao": true, "ssil": true, "glow": true,
		"saturation": 1.05, "contrast": 1.05, "shadow_distance": 220.0,
		"horizon_y": GROUND, "card_ground": "leafy_grass", "card_ground_tint": Color(0.8, 0.85, 0.7),
	})


func build_scene(scene: Dictionary) -> void:
	_wind = float(scene.get("wind", 0.0))
	class_code("boulder")   # the boulder is transient: put its class in the mask table sent with "ready"
	_wood = Assets.pbr("rough_wood", {"uv": 0.9, "tint": Color(0.78, 0.66, 0.52)})
	_wood_dark = Assets.pbr("weathered_planks", {"uv": 0.8, "tint": Color(0.8, 0.7, 0.6)})
	_iron = Parts.grime_paint(Color(0.1, 0.095, 0.09), {"grime": 0.5, "rough": 0.55, "metal": 0.85, "coat": 0.0, "seed": 61,
		"dirt": Color(0.42, 0.25, 0.12), "dirt_tex": "rusty_metal_02", "scale": 1.5})
	_rope_mat = M.mat(Color(0.55, 0.45, 0.3), 0.9, 0.0, {"normal_noise": 2.0, "noise_scale": 8.0})
	_build_terrain()
	_build_backdrop()
	_build_dressing(scene)
	var base: Array = scene.get("base", [10.0, 2.0])
	_base = _build_base(Vector3(float(base[0]), float(base[1]), 0))
	register("base", _base, "catapult")
	_arm = _build_arm()
	register("arm", _arm, "catapult")
	register("rigging", _build_rigging(), "catapult")
	var blocks: Array = scene.get("blocks", [])
	var tower_x := float(scene.get("tower_x", 35.0))
	for i in 5:
		register("block%d" % i, _build_block(i, blocks[i] if i < blocks.size() else []), "block")
	_banners.append(Nature.banner(content, Vector3(1.5, _height(1.5, -9.0), -9.0), {"wind": _wind, "wood": _wood, "iron": _iron, "height": 6.5}))
	_banners.append(Nature.banner(content, Vector3(tower_x + 4.8, _height(tower_x + 4.8, -3.2), -3.2),
		{"wind": _wind, "wood": _wood, "iron": _iron, "col_a": Color(0.08, 0.16, 0.42), "col_b": Color(0.86, 0.82, 0.7), "height": 8.0}))


# ---------------------------------------------------------------- terrain / scenery

func _height(x: float, z: float) -> float:
	## Flat field around the lane (the Box2D ground), gentle meadow swells behind it, a shallow
	## dip in front (never above the lane: nothing hides the scene from the side camera).
	if z < -LANE:
		var t := clampf((-z - LANE) / 10.0, 0.0, 1.0)
		var fade := clampf((z - Z_BACK) / 25.0, 0.0, 1.0)
		return GROUND + t * t * (3.0 - 2.0 * t) * fade * (0.7 + 0.6 * _noise.get_noise_2d(x, z))
	if z > LANE:
		var t := clampf((z - LANE) / (Z_FRONT - LANE), 0.0, 1.0)
		return GROUND - sin(PI * t) * (0.45 + 0.25 * _noise.get_noise_2d(x + 30.0, z))
	return GROUND


func _build_terrain() -> void:
	_noise.seed = 15
	_noise.frequency = 0.04
	_noise.fractal_octaves = 3
	var xs := PackedFloat32Array()
	for i in int((X1 - X0) / 1.0) + 1:
		xs.append(X0 + i)
	var top := Parts.ground_blend("dirt", "leafy_grass", {"a_uv": 0.3, "b_uv": 0.22, "lane": 2.2, "soft": 3.0,
		"a_tint": Color(0.82, 0.76, 0.66), "b_tint": Color(0.72, 0.84, 0.56), "mask_freq": 0.03, "lane_bias": 1.15, "seed": 71,
		"ruts": 0.5, "rut_z": Vector2(0.95, 0.3)})
	var sec := Assets.pbr("excavated_soil_wall", {"uv": 0.3, "tint": Color(0.75, 0.68, 0.6)})
	content.add_child(Parts.heightfield(xs, Z_BACK, Z_FRONT, 0.75, _height, -4.0, top, sec))
	# Ground beyond the strip, in every view (one consistent world; the ortho elevations see it
	# edge-on). No shadow: it would only shade the strip's soil cross-sections below it.
	var ring := _ground_ring(Rect2(X0, Z_BACK, X1 - X0, Z_FRONT - Z_BACK), 2500.0, GROUND - 0.04,
		Parts.matte(Assets.pbr("leafy_grass", {"uv": 0.1, "world": true, "tint": Color(0.72, 0.84, 0.56)})))
	ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(ring)
	# Grass clumps and wildflowers on both sides of the lane. Cross-view consistency: drawn in every
	# view whose frame contains them. They stand beside the lane (|z| > 4.5), so the rear camera
	# looks past them; in front of the lane they are shorter near its edge (_front_trim), so the
	# side view still sees the catapult wheels and the tower base.
	var rng := RandomNumberGenerator.new()
	rng.seed = 33
	var mesh := Nature.clump_mesh(14, 0.55, 3)
	var mat := Nature.grass_material(_wind, {"height": 0.55, "base": Color(0.1, 0.17, 0.04), "tip": Color(0.34, 0.45, 0.14), "dry": Color(0.48, 0.42, 0.22)})
	var fmesh := Nature.flower_mesh(6, 0.45, 9)
	var fmats := [Nature.grass_material(_wind, {"height": 0.45, "base": Color(0.1, 0.17, 0.04), "tip": Color(0.25, 0.35, 0.1), "flower": Color(0.95, 0.82, 0.18)}),
		Nature.grass_material(_wind, {"height": 0.45, "base": Color(0.1, 0.17, 0.04), "tip": Color(0.25, 0.35, 0.1), "flower": Color(0.92, 0.92, 0.88)})]
	for side in [-1.0, 1.0]:
		for k in 2:
			var fx: Array = []
			for i in 500:
				var x := rng.randf_range(-24.0, 50.0)
				var z: float = side * rng.randf_range(LANE + 1.0, 24.0)
				fx.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(1.0, _front_trim(z), 1.0)), Vector3(x, _height(x, z) - 0.02, z)))
			var fl := Nature.grass(fx, fmats[k], fmesh)
			content.add_child(fl)
		var xf: Array = []
		for i in 4200:
			var x := rng.randf_range(-30.0, 75.0)
			var z: float = side * rng.randf_range(LANE - 0.5, 26.0 if side > 0.0 else 34.0)
			if (x > 50.0 or absf(z) > 22.0) and rng.randf() < 0.5:
				continue
			if absf(z) < LANE + 1.5 and rng.randf() < 0.6:
				continue
			var s := rng.randf_range(0.6, 1.4)
			xf.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s * rng.randf_range(0.7, 1.3) * _front_trim(z), s)), Vector3(x, _height(x, z) - 0.02, z)))
		var g := Nature.grass(xf, mat, mesh)
		content.add_child(g)


func _front_trim(z: float) -> float:
	## Height factor of the field dressing in front of the lane (z > 0): half height at the lane edge,
	## full height 6 m out, where the ground has started to dip.
	return 1.0 if z < 0.0 else lerpf(0.5, 1.0, clampf((z - LANE) / 6.0, 0.0, 1.0))


func _ground_ring(inner: Rect2, half: float, y: float, mat: Material) -> MeshInstance3D:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var x0 := inner.position.x
	var x1 := inner.end.x
	var z0 := inner.position.y
	var z1 := inner.end.y
	for q in [[-half, -half, x0, half], [x1, -half, half, half], [x0, -half, x1, z0], [x0, z1, x1, half]]:
		M._quad(st, [Vector3(q[0], y, q[1]), Vector3(q[2], y, q[1]), Vector3(q[2], y, q[3]), Vector3(q[0], y, q[3])], Vector3.UP)
	st.generate_tangents()
	var mi := MeshInstance3D.new()
	mi.name = "GroundRing"
	mi.mesh = st.commit()
	mi.material_override = mat
	return mi


func _build_backdrop() -> void:
	## Hills, forests and a far castle. Cross-view consistency: drawn in every view whose frame
	## contains them (the ortho rear / side elevations show them full size behind the field; their
	## far planes in exporters/catapult_war.py reach past them). So the hills rise gently from the
	## meadow to low ridges (~10-20 m, the elevations keep sky above them), and no tree or building
	## stands in the fixed rear / side cameras' sight lines to the lane.
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Backdrop")
	var hill := Assets.pbr("aerial_grass_rock", {"uv": 0.03, "world": true, "tint": Color(0.75, 0.8, 0.62)})
	var cliff := Assets.pbr("rock_wall_08", {"uv": 0.04, "world": true, "tint": Color(0.7, 0.68, 0.62)})
	var n := FastNoiseLite.new()
	n.seed = 8
	n.frequency = 0.006
	# Hill ranges on both sides of the field (long in x): slopes from the meadow at |z| = 260 up to
	# rolling ridges ~100 m further out
	var hx := PackedFloat32Array()
	for i in 121:
		hx.append(-700.0 + i * 12.0)
	for side in [-1.0, 1.0]:
		var hz := func(x: float, z: float) -> float:
			var ramp := smoothstep(0.0, 1.0, (absf(z) - 260.0) / 110.0)
			var ridge := 8.0 + 8.0 * (0.5 + 0.5 * n.get_noise_1d(x + side * 200.0)) + 2.5 * n.get_noise_1d(x * 3.0)
			return GROUND - 0.5 + ramp * (ridge + 4.0 * n.get_noise_2d(x * 1.5, z * 1.5))
		var z0 := -650.0 if side < 0.0 else 260.0
		var z1 := -260.0 if side < 0.0 else 600.0
		var hills := Parts.heightfield(hx, z0, z1, 13.0, hz, -5.0, hill, cliff)
		hills.name = "HillRange"
		root.add_child(hills)
	# Rolling hills across the far end (behind the tower) and behind the catapult. The toe starts
	# `rough` below the meadow, so the strip's end cap stays under the ground (no cliff in the rear view).
	for end in [-1.0, 1.0]:
		var xs := PackedFloat32Array()
		var ys := PackedFloat32Array()
		for i in 61:
			var t := float(i) / 60.0
			var x := lerpf(320.0, 800.0, t) if end > 0.0 else lerpf(-800.0, -260.0, t)
			var rise := smoothstep(0.0, 0.25, t) if end > 0.0 else smoothstep(1.0, 0.75, t)
			xs.append(x)
			ys.append(GROUND - 5.5 + rise * (15.0 + 5.0 * n.get_noise_1d(x)))
		root.add_child(Terrain.side_strip(xs, ys, -650.0, 600.0, -5.0, hill, cliff, {"lane": -1.0, "rough": 5.0, "dz": 15.0, "freq": 0.015, "seed": 11 + int(end), "front_down": false}))
	# Forest belts
	var rng := RandomNumberGenerator.new()
	rng.seed = 21
	var xf_b: Array = []
	var xf_p: Array = []
	for i in 260:
		var a := rng.randf() * TAU
		var d := rng.randf_range(95.0, 230.0)
		var p := Vector3(15.0 + cos(a) * d * 1.4, 0.0, sin(a) * d)
		p.y = _height(p.x, p.z) - 0.2
		var t := Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * rng.randf_range(1.0, 2.0)), p)
		if i % 3 == 0:
			xf_p.append(t)
		else:
			xf_b.append(t)
	root.add_child(Terrain.tree_multimesh(xf_b, "broadleaf", 4))
	root.add_child(Terrain.tree_multimesh(xf_p, "pine", 6))
	_build_castle(root, Vector3(340.0, GROUND, 62.0))   # just outside the rear view's frame (|z| < 27)


func _build_castle(parent: Node3D, pos: Vector3) -> void:
	var root := M.node(parent, pos, Vector3(0, 25, 0), "Castle")
	var stone := Assets.pbr("castle_brick_01", {"uv": 0.25, "world": true, "tint": Color(0.82, 0.8, 0.75)})
	var roof := M.mat(Color(0.25, 0.12, 0.1), 0.8)
	for i in 4:
		var c := Vector3(-24.0 + i * 16.0, 0, 0)
		M.mi(root, M.cyl(4.0, 4.4, 20.0, 16), stone, c + Vector3(0, 10.0, 0))
		M.mi(root, M.cyl(0.0, 4.8, 6.0, 16), roof, c + Vector3(0, 23.0, 0))
		if i < 3:
			M.mi(root, M.box(Vector3(12.0, 13.0, 3.0)), stone, c + Vector3(8.0, 6.5, 0))
			for k in 6:
				M.mi(root, M.box(Vector3(1.0, 1.2, 3.2)), stone, c + Vector3(3.0 + k * 2.0, 13.6, 0))
	M.mi(root, M.box(Vector3(10.0, 30.0, 10.0)), stone, Vector3(0, 15.0, -14.0))
	M.mi(root, M.cyl(0.0, 7.5, 9.0, 4), roof, Vector3(0, 34.5, -14.0), Vector3(0, 45, 0))


func _build_dressing(scene: Dictionary) -> void:
	## Near props behind the lane: spare boulders, hay, a supply cart; stakes and a gabion by the tower.
	## Cross-view consistency: drawn in every view whose frame contains them (beside the lane in the
	## rear view, behind it in the side view).
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Dressing")
	var rng := RandomNumberGenerator.new()
	rng.seed = 44
	var stone := Assets.pbr("rock_wall_08", {"uv": 1.2, "tint": Color(0.75, 0.72, 0.68)})
	for i in 6:
		var p := Vector3(6.6 + (i % 3) * 0.95, 0, -2.6 - (i / 3) * 0.9)
		p.y = _height(p.x, p.z) + (0.62 if i == 5 else 0.0)
		M.mi(root, Parts.rock_mesh(Vector3(1.0, 0.9, 1.0), 80 + i, 0.2), stone, p - Vector3(0, 0.05, 0), Vector3(0, rng.randf() * 360.0, 0), Vector3.ONE * 0.92)
	# Hay bales and crates
	var hay := M.mat(Color(0.72, 0.6, 0.32), 0.95, 0.0, {"noise": 1.0, "normal_noise": 3.0, "noise_scale": 4.0})
	for i in 3:
		var p := Vector3(2.0 + i * 1.25, 0, -4.2)
		M.mi(root, M.box(Vector3(1.1, 0.5, 0.6), 0.06), hay, Vector3(p.x, _height(p.x, p.z) + 0.25, p.z), Vector3(0, rng.randf_range(-10, 10), 0))
	M.mi(root, M.box(Vector3(1.1, 0.5, 0.6), 0.06), hay, Vector3(3.25, _height(3.25, -4.2) + 0.75, -4.2), Vector3(0, 6, 0))
	var crate := Assets.fitted_model("wooden_crate_02", Vector3(0.6, 0.55, 1.3))
	if crate:
		crate.position = Vector3(12.8, _height(12.8, -3.4), -3.4)
		crate.rotation.y = 1.4
		root.add_child(crate)
	# Wooden stakes (palisade) behind the tower
	var tower_x := float(scene.get("tower_x", 35.0))
	for k in 12:
		var x := tower_x - 6.0 + k * 1.1
		var z := -5.5 - 0.3 * sin(k * 1.3)
		var h := rng.randf_range(1.5, 2.1)
		M.mi(root, M.cyl(0.0, 0.13, 0.35, 7), _wood, Vector3(x, _height(x, z) + h + 0.17, z))
		M.mi(root, M.cyl(0.13, 0.14, h, 7), _wood, Vector3(x, _height(x, z) + h * 0.5, z), Vector3(rng.randf_range(-4, 4), 0, rng.randf_range(-4, 4)))
	M.mi(root, M.box(Vector3(13.2, 0.08, 0.06)), _rope_mat, Vector3(tower_x + 0.05, _height(tower_x, -5.5) + 1.2, -5.55))


# ---------------------------------------------------------------- catapult

func _beam(parent: Node3D, a: Vector3, b: Vector3, w: float, t: float, mat: Material) -> MeshInstance3D:
	## Timber beam of cross-section w x t (t along z) from a to b (a, b in the XY plane of parent).
	var d := b - a
	var mi := M.mi(parent, M.box(Vector3(d.length(), w, t), 0.02), mat, (a + b) * 0.5)
	mi.rotation = Vector3(0, 0, atan2(d.y, d.x))
	return mi


func _build_base(pos: Vector3) -> Node3D:
	## Box2D base: 3 x 2 m (local x -1.5..1.5, y -1..1), extruded z +-1. Pivot at world (9, 3.5).
	var root := M.node(content, pos, Vector3.ZERO, "Base")
	var pv := Vector3(PIVOT.x - pos.x, PIVOT.y - pos.y, 0)
	# Wheels (solid planked discs with iron tyres) at the corners, axles along z
	for wx in [-1.05, 1.05]:
		M.mi(root, M.cyl(0.05, 0.05, 2.0, 10), _iron, Vector3(wx, -0.58, 0), Vector3(90, 0, 0))
		for sz in [-1.0, 1.0]:
			var w := M.node(root, Vector3(wx, -0.58, sz * 0.91))
			M.mi(w, M.cyl(0.4, 0.4, 0.14, 24), _wood_dark, Vector3.ZERO, Vector3(90, 0, 0))
			M.mi(w, M.cyl(0.42, 0.42, 0.12, 24), _iron, Vector3.ZERO, Vector3(90, 0, 0))
			M.mi(w, M.cyl(0.39, 0.39, 0.16, 24), _wood_dark, Vector3.ZERO, Vector3(90, 0, 0))
			M.mi(w, M.cyl(0.1, 0.12, 0.2, 12), _iron, Vector3(0, 0, sz * 0.03), Vector3(90, 0, 0))
			for k in 2:
				M.mi(w, M.box(Vector3(0.74, 0.07, 0.02)), _iron, Vector3(0, 0, sz * 0.082), Vector3(0, 0, 45 + k * 90))
	# Side frames (z = +-0.66): bottom and top rails, corner posts, cross bracing
	for sz in [-1.0, 1.0]:
		var z: float = sz * 0.66
		var f := M.node(root, Vector3(0, 0, z))
		_beam(f, Vector3(-1.5, -0.42, 0), Vector3(1.5, -0.42, 0), 0.26, 0.24, _wood)
		_beam(f, Vector3(-1.5, 0.88, 0), Vector3(1.5, 0.88, 0), 0.24, 0.22, _wood)
		for x in [-1.36, 1.36]:
			_beam(f, Vector3(x, -0.3, 0), Vector3(x, 0.77, 0), 0.22, 0.2, _wood)
		_beam(f, Vector3(-1.25, -0.3, sz * 0.07), Vector3(1.25, 0.77, sz * 0.07), 0.16, 0.12, _wood_dark)
		_beam(f, Vector3(-1.25, 0.77, sz * 0.07), Vector3(1.25, -0.3, sz * 0.07), 0.16, 0.12, _wood_dark)
		for x in [-1.36, 0.0, 1.36]:
			M.mi(f, M.box(Vector3(0.26, 0.05, 0.26)), _iron, Vector3(x, 0.88, 0))
			M.mi(f, M.box(Vector3(0.28, 0.06, 0.28)), _iron, Vector3(x, -0.42, 0))
	# Cross beams along z
	for x in [-1.36, 1.36]:
		M.mi(root, M.box(Vector3(0.24, 0.24, 1.56), 0.02), _wood, Vector3(x, -0.42, 0))
	for x in [pv.x, 1.0, 1.36]:
		M.mi(root, M.box(Vector3(0.24, 0.22, 1.56), 0.02), _wood, Vector3(x, 0.88, 0))
	# Plank deck between the bottom rails
	for k in 7:
		M.mi(root, M.box(Vector3(0.36, 0.05, 1.1), 0.01), _wood_dark, Vector3(-1.1 + k * 0.37, -0.27, 0), Vector3(0, rng_yaw(k), 0))
	# Uprights carrying the arm axle at the pivot, braced forward
	for sz in [-1.0, 1.0]:
		var z: float = sz * 0.46
		var u := M.node(root, Vector3(0, 0, z))
		_beam(u, Vector3(pv.x, 0.99, 0), Vector3(pv.x, pv.y + 0.3, 0), 0.2, 0.17, _wood)
		_beam(u, Vector3(pv.x + 0.05, pv.y + 0.1, 0), Vector3(pv.x + 1.15, 0.99, 0), 0.14, 0.14, _wood)
		_beam(u, Vector3(pv.x - 0.05, pv.y + 0.0, 0), Vector3(pv.x - 0.42, 0.99, 0), 0.12, 0.14, _wood)
		M.mi(u, M.box(Vector3(0.24, 0.3, 0.2), 0.01), _iron, Vector3(pv.x, pv.y, 0))
	M.mi(root, M.cyl(0.07, 0.07, 1.22, 14), _iron, pv, Vector3(90, 0, 0))
	for sz in [-1.0, 1.0]:
		M.mi(root, M.cyl(0.11, 0.11, 0.04, 8), _iron, pv + Vector3(0, 0, sz * 0.6), Vector3(90, 0, 0))
	# Padded stop beam at the front (the arm's follow-through) and the winch brackets
	M.mi(root, M.box(Vector3(0.3, 0.16, 0.9), 0.04), M.mat(Color(0.32, 0.22, 0.14), 0.95), Vector3(1.36, 1.07, 0))
	var drum_local := DRUM - pos
	for sz in [-1.0, 1.0]:
		var br := M.node(root, Vector3(0, 0, sz * 0.47))
		_beam(br, Vector3(drum_local.x - 0.25, 0.99, 0), Vector3(drum_local.x, drum_local.y + 0.05, 0), 0.1, 0.1, _wood)
		_beam(br, Vector3(drum_local.x + 0.25, 0.99, 0), Vector3(drum_local.x, drum_local.y + 0.05, 0), 0.1, 0.1, _wood)
	# pennant on the rear corner post (part of the catapult: shows the wind)
	_banners.append(Nature.banner(root, Vector3(-1.36, 0.88, -0.66), {"wind": _wind, "wood": _wood, "iron": _iron,
		"height": 2.4, "length": 1.0, "width": 0.38, "col_a": Color(0.55, 0.06, 0.05), "col_b": Color(0.85, 0.68, 0.22)}))
	return root


static func rng_yaw(k: int) -> float:
	return sin(k * 12.9898) * 2.0


func _build_arm() -> Node3D:
	## Box2D arm: 5 x 0.4 m (local x -2.5..2.5), extruded z +-0.3; pivot at local (-2, 0),
	## boulder seat at local (2.0, 0.7) (on top, near the tip).
	var root := M.node(content, Vector3(10.4, 4.9, 0), Vector3(0, 0, 45), "Arm")
	var prof := M.poly([[-2.5, -0.19], [2.5, -0.13], [2.5, 0.13], [-2.5, 0.19]])
	M.mi(root, M.prism(prof, -0.25, 0.25), _wood)
	# Iron bands and the axle bearing
	for x in [-2.3, -1.6, -0.6, 0.6, 1.45, 2.4]:
		var hy: float = 0.19 - (x + 2.5) * 0.012
		M.mi(root, M.box(Vector3(0.08, hy * 2.0 + 0.03, 0.53)), _iron, Vector3(x, 0, 0))
	M.mi(root, M.cyl(0.17, 0.17, 0.54, 16), _iron, Vector3(-2.0, 0, 0), Vector3(90, 0, 0))
	# Rope lashing below the cup
	for k in 5:
		var lash := M.mi(root, M.cyl(0.155, 0.155, 0.035, 10), _rope_mat, Vector3(1.0 + k * 0.05, 0, 0), Vector3(0, 0, 90))
		lash.scale = Vector3(1, 1, 1.75)
	# Cup (wooden bowl with an iron rim) holding the boulder; bowl base at local y 0.15
	var bowl := PackedVector2Array([Vector2(0.05, 0.0), Vector2(0.4, 0.0), Vector2(0.57, 0.1), Vector2(0.62, 0.27), Vector2(0.57, 0.31),
		Vector2(0.51, 0.24), Vector2(0.38, 0.11), Vector2(0.05, 0.08)])
	M.mi(root, M.lathe(bowl, 20), _wood_dark, Vector3(2.0, 0.13, 0))
	M.mi(root, M.lathe(PackedVector2Array([Vector2(0.6, 0.255), Vector2(0.63, 0.27), Vector2(0.6, 0.305)]), 20), _iron, Vector3(2.0, 0.13, 0))
	# Rope anchor eye under the arm
	M.mi(root, M.box(Vector3(0.12, 0.08, 0.2), 0.01), _iron, ROPE_ARM + Vector3(0, -0.03, 0))
	return root


func _build_rigging() -> Node3D:
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Rigging")
	_drum = M.node(root, DRUM, Vector3.ZERO, "Drum")
	M.mi(_drum, M.cyl(0.17, 0.17, 0.72, 16), _wood, Vector3.ZERO, Vector3(90, 0, 0))
	M.mi(_drum, M.cyl(0.2, 0.2, 0.42, 16), _rope_mat, Vector3.ZERO, Vector3(90, 0, 0))
	for sz in [-1.0, 1.0]:
		M.mi(_drum, M.cyl(0.24, 0.24, 0.05, 16), _iron, Vector3(0, 0, sz * 0.38), Vector3(90, 0, 0))
		for k in 4:
			var spoke := M.mi(_drum, M.cyl(0.025, 0.025, 0.85, 6), _wood_dark, Vector3(0, 0, sz * 0.42), Vector3(0, 0, 45.0 * (k % 2) + 90.0 * (k / 2)))
			spoke.name = "Spoke"
	M.tube(PackedVector3Array([DRUM + Vector3(0, 0.2, 0), DRUM + Vector3(-0.5, 1.2, 0)]), 0.025, 6, _rope_mesh)
	M.mi(root, _rope_mesh, _rope_mat).name = "Rope"
	return root


func _build_block(i: int, shapes: Array) -> Node3D:
	## Tower blocks 0-2: 1.6 m brick cubes; pillars 3-4: 1 x 2 m sandstone; all extruded z +-0.8.
	var root := M.node(content, Vector3(35.0, 1.8 + 1.6 * i, 0), Vector3.ZERO, "Block%d" % i)
	var tower := i < 3
	var hx := 0.8 if tower else 0.5
	var hy := 0.8 if tower else 1.0
	if shapes.size() > 0 and shapes[0].has("poly"):
		var xs: Array = []
		var ys: Array = []
		for v in shapes[0]["poly"]:
			xs.append(float(v[0]))
			ys.append(float(v[1]))
		hx = (xs.max() - xs.min()) * 0.5
		hy = (ys.max() - ys.min()) * 0.5
	var mat: BaseMaterial3D
	if tower:
		mat = Assets.pbr("castle_brick_01", {"uv": 0.8, "tint": Color(0.9, 0.86, 0.8)}).duplicate()
	else:
		mat = Assets.pbr("large_sandstone_blocks", {"uv": 0.5, "tint": Color(0.95, 0.9, 0.8)}).duplicate()
		mat.uv1_offset = Vector3(0.0, -0.45, 0.0)   # the texture's weathered band sits at the pillar foot
	_block_mats.append(mat)
	M.mi(root, M.box(Vector3(hx * 2.0, hy * 2.0, 1.6), 0.035), mat)
	var dark := M.mat(Color(0.04, 0.035, 0.03), 0.9)
	if i == 0:
		# timber door on the face towards the catapult (-x) and an iron-strapped frame
		M.mi(root, M.box(Vector3(0.02, 1.0, 0.62), 0.005), _wood_dark, Vector3(-hx - 0.006, -hy + 0.52, 0))
		for k in 2:
			M.mi(root, M.box(Vector3(0.025, 0.05, 0.64)), _iron, Vector3(-hx - 0.012, -hy + 0.25 + k * 0.5, 0))
	elif i == 1:
		for face in [[Vector3(-hx - 0.004, 0.1, 0), Vector3(0.02, 0.6, 0.12)], [Vector3(0, 0.1, 0.804), Vector3(0.12, 0.6, 0.02)], [Vector3(0, 0.1, -0.804), Vector3(0.12, 0.6, 0.02)]]:
			M.mi(root, M.box(face[1]), dark, face[0])
	elif i == 2:
		for face in [[Vector3(-hx - 0.004, 0.05, 0), Vector3(0.02, 0.45, 0.1)], [Vector3(0, 0.05, 0.804), Vector3(0.1, 0.45, 0.02)]]:
			M.mi(root, M.box(face[1]), dark, face[0])
	return root


# ---------------------------------------------------------------- per step

func spawn_dyn(kind: String, d: Dictionary) -> Node3D:
	if kind == "boulder":
		var holder := Node3D.new()
		holder.name = "Boulder"
		var rock := Assets.fitted_model("boulder_01", Vector3(1.0, 0.96, 1.0), "stretch")
		if rock != null:
			rock.position = Vector3(0, -0.48, 0)
			holder.add_child(rock)
		else:
			M.mi(holder, M.sphere(0.5), Assets.pbr("rock_wall_08", {"uv": 1.0}))
		return holder
	if kind == "dust":
		var holder := Node3D.new()
		holder.name = "Dust"
		Vfx.puff(holder, Color(0.45, 0.41, 0.35), abs(str(d.get("id", "")).hash()) % 1000, 7)
		return holder
	return null


func update_dyn(node: Node3D, a: Dictionary, b: Dictionary, alpha: float) -> void:
	if node.name.begins_with("Dust") and node.get_child_count() > 0:
		var t := lerpf(float(a.get("t", 0.0)), float(b.get("t", 0.0)), alpha)
		Vfx.set_puff(node.get_child(0), t, 1.8, 0.7)


func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var tension := float(p.get("tension", 0.0))
	# Winch: drum turns with the wound-in tension, the rope tightens (less sag) as it winds
	_drum.rotation = Vector3(0, 0, -tension * 7.0)
	var a := DRUM + Vector3(0, 0.2, 0)
	var b := _arm.transform * ROPE_ARM
	var sag := (1.0 - tension) * 0.22 + 0.02
	var pts := PackedVector3Array()
	for k in 9:
		var t := float(k) / 8.0
		pts.append(a.lerp(b, t) - Vector3(0, sag * 4.0 * t * (1.0 - t), 0))
	M.tube(pts, 0.025, 6, _rope_mesh)
	var top: Array = p.get("toppled", [])
	for i in mini(top.size(), _block_mats.size()):
		var base := Color(0.9, 0.86, 0.8) if i < 3 else Color(0.95, 0.9, 0.8)
		_block_mats[i].albedo_color = base * Color(0.82, 0.8, 0.78) if int(top[i]) == 1 else base


func chase_pose() -> Transform3D:
	var p := Vector3(10, 3, 0)
	return Transform3D(Basis(), p + Vector3(-8.0, 5.0, 10.0)).looking_at(p + Vector3(8.0, 1.0, 0), Vector3.UP)
