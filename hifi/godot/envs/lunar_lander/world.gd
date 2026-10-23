extends EnvWorld
## Lunar Lander: Apollo-style lander over a regolith terrain strip, black sky with Earth, hard sun.
## Python: hifi/exporters/lunar_lander.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Moving ids: lander, leg0, leg1. p: main / rcs_px / rcs_nx engine power, contact[2], landed.

const LANE := 2.5          # |z| where the terrain follows the Box2D profile exactly
const Z_BACK := -70.0
const Z_FRONT := 12.0

var _lander: Node3D
var _main_flame: Node3D
var _rcs_px: Node3D
var _rcs_nx: Node3D
var _dust: Node3D
var _flags: Array = []
var _flag_mats: Array = []
var _profile  # Terrain.SideProfile of the main strip


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "space", "stars": 1.0, "planet": [0.82, 0.38, -0.42, 1.9], "ortho_bg": Color(0.004, 0.004, 0.006),
		"sun_elev": 50.0, "sun_azim": 228.0, "sun_energy": 2.0, "sun_color": Color(1.0, 0.98, 0.95),
		"ambient_color": Color(0.55, 0.6, 0.7), "ambient_energy": 0.22,
		"exposure": 1.12, "white": 8.0, "glow": true, "ssao": true, "ssil": true,
		"saturation": 0.95, "contrast": 1.08, "shadow_distance": 160.0,
	})


func build_scene(scene: Dictionary) -> void:
	_build_terrain(scene)
	_build_pad(scene)
	_build_backdrop()
	_lander = _build_lander()
	register("lander", _lander, "lander")
	for i in 2:
		register("leg%d" % i, _build_leg(), "lander")
	_dust = Vfx.puff(content, Color(0.55, 0.53, 0.5), 11, 7)


# ---------------------------------------------------------------- terrain

func _build_terrain(scene: Dictionary) -> void:
	var pts: Array = scene.get("terrain", [])
	var W: float = scene.get("W", 20.0)
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	# Low continuation beyond the playfield so the ortho views never show a void
	var y0: float = pts[0][1] if pts.size() > 0 else 3.0
	var y1: float = pts[-1][1] if pts.size() > 0 else 3.0
	for k in range(24, 0, -1):
		xs.append(-k * 2.5)
		ys.append(y0 - 0.06 * k + 0.25 * sin(k * 1.7))
	# Densify the playfield profile (0.25 m) so the shoulders get smooth noise
	for i in pts.size():
		var a: Array = pts[i]
		if i > 0:
			var p: Array = pts[i - 1]
			var n := int(ceil((float(a[0]) - float(p[0])) / 0.25))
			for s in range(1, n):
				var t := float(s) / n
				xs.append(lerpf(p[0], a[0], t))
				ys.append(lerpf(p[1], a[1], t))
		xs.append(a[0])
		ys.append(a[1])
	for k in range(1, 25):
		xs.append(W + k * 2.5)
		ys.append(y1 - 0.06 * k + 0.25 * sin(k * 2.3))
	var top := Assets.pbr("moon_dusted_02", {"uv": 0.3, "tint": Color(0.6, 0.59, 0.57), "color": Color(0.38, 0.38, 0.38)})
	var sec := Assets.pbr("moon_01", {"uv": 0.35, "tint": Color(0.42, 0.41, 0.4), "color": Color(0.3, 0.3, 0.3)})
	var t := Terrain.side_strip(xs, ys, Z_BACK, Z_FRONT, -2.0, top, sec,
		{"lane": LANE, "rough": 1.5, "dz": 0.6, "freq": 0.14, "seed": 3})
	content.add_child(t)
	_profile = t.get_meta("profile")
	# Rocks: behind the lane and small ones in front, never inside it
	var rng := RandomNumberGenerator.new()
	rng.seed = 1234
	var names := ["moon_rock_01", "moon_rock_03", "moon_rock_05"]
	for i in 70:
		var x := rng.randf_range(xs[0], xs[-1])
		var z := rng.randf_range(Z_BACK * 0.6, -LANE - 0.6) if i % 4 != 0 else rng.randf_range(LANE + 0.8, Z_FRONT - 1.0)
		var s := rng.randf_range(0.3, 1.6) if z < 0.0 else rng.randf_range(0.15, 0.4)
		var rock := Assets.fitted_model(names[i % 3], Vector3(s, s * 0.6, s))
		if rock == null:
			break
		rock.position = Vector3(x, _profile.height(x, z) - 0.12 * s, z)
		rock.rotation = Vector3(0, rng.randf() * TAU, 0)
		content.add_child(rock)


func _height_at(x: float) -> float:
	return _profile.base(x) if _profile != null else 0.0


func _build_pad(scene: Dictionary) -> void:
	var pad: Array = scene.get("pad", [8.0, 12.0, 3.33])
	var y: float = pad[2]
	var metal := M.mat(Color(0.75, 0.76, 0.78), 0.35, 0.9)
	_flags.clear()
	_flag_mats.clear()
	for x in [float(pad[0]), float(pad[1])]:
		var root := M.node(content, Vector3(x, y, 0))
		M.mi(root, M.cyl(0.025, 0.03, 1.7, 10), metal, Vector3(0, 0.85, 0))
		var cloth := M.mat(Color(0.85, 0.12, 0.05), 0.8, 0.0, {"cull": false})
		var flag := M.mi(root, M.box(Vector3(0.8, 0.34, 0.01), 0.0), cloth, Vector3(0.42, 1.48, 0))
		_flags.append(flag)
		_flag_mats.append(cloth)
		# Beacon at the pole foot, plus a lit ground marker
		M.mi(root, M.cyl(0.1, 0.14, 0.12, 12), M.mat(Color(0.2, 0.2, 0.22), 0.5, 0.6), Vector3(0, 0.06, 0))
		M.mi(root, M.sphere(0.05), M.emissive(Color(1.0, 0.35, 0.1), 6.0), Vector3(0, 1.74, 0))
	# Landing target: a faint ring of reflectors on the pad
	for k in 12:
		var x := lerpf(float(pad[0]) + 0.3, float(pad[1]) - 0.3, k / 11.0)
		M.mi(content, M.box(Vector3(0.18, 0.03, 0.18), 0.01), M.mat(Color(0.9, 0.9, 0.85), 0.3, 0.2),
			Vector3(x, y + 0.01, LANE * (1.0 if k % 2 == 0 else -1.0) * 0.5))


func _build_backdrop() -> void:
	## Far field: a heightfield all around the terrain strip (below it where they overlap) that
	## rises into crater rims and hills with distance from the strip. One consistent world: it is
	## drawn in every view whose frame contains it (the ortho elevations see the rims above the
	## strip; their far planes reach its edge, see exporters/lunar_lander.py). The part between an
	## elevation camera and the strip's section face lies in front of that camera's near plane.
	var noise := FastNoiseLite.new()
	noise.seed = 77
	noise.frequency = 0.008
	var detail := FastNoiseLite.new()
	detail.seed = 78
	detail.frequency = 0.05
	var base := _height_at(10.0) - 0.6
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var x0 := -320.0
	var z0 := -460.0
	var z1 := 320.0
	var step := 5.0
	var sx0: float = _profile.xs[0]
	var sx1: float = _profile.xs[-1]
	var nx := int((340.0 - x0) / step) + 1
	var nz := int((z1 - z0) / step) + 1
	for i in nx:
		for j in nz:
			var x := x0 + i * step
			var z := z0 + j * step
			var dx := maxf(maxf(sx0 - x, x - sx1), 0.0)
			var dz := maxf(maxf(Z_BACK - z, z - Z_FRONT), 0.0)
			var far := clampf(sqrt(dx * dx + dz * dz) / 300.0, 0.0, 1.0)
			# rims stay low (<= ~6 m above the pad): the true-scale elevations (frame 13.3 m) see
			# them at full height, and taller rims would replace their whole sky with regolith
			var h := base + far * far * 6.0 * (0.45 + 0.55 * noise.get_noise_2d(x, z)) + 0.8 * detail.get_noise_2d(x, z)
			h -= (1.0 - minf(far * 8.0, 1.0)) * 2.5   # tucked below the strip where they meet
			if dx <= 0.0 and dz <= 0.0:
				# under the strip: stay below its lowest nearby top (front shoulder noise dips by `rough`)
				var lo := minf(minf(_profile.base(x - step), _profile.base(x)), _profile.base(x + step))
				h = minf(h, lo - _profile.rough - 0.6)
			st.set_uv(Vector2(x, z))
			st.add_vertex(Vector3(x, h, z))
	for i in nx - 1:
		for j in nz - 1:
			var a := i * nz + j
			var b := a + nz
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)
	st.generate_normals()
	st.generate_tangents()
	var m := Assets.pbr("moon_01", {"uv": 0.04, "world": true, "tint": Color(0.5, 0.49, 0.47), "color": Color(0.35, 0.35, 0.35)})
	var far_field := M.mi(content, st.commit(), m)
	far_field.name = "FarField"


# ---------------------------------------------------------------- lander

func _build_lander() -> Node3D:
	var root := Node3D.new()
	root.name = "Lander"
	content.add_child(root)
	var foil := M.mat(Color(0.86, 0.64, 0.22), 0.32, 1.0, {"normal_noise": 3.0, "noise_scale": 3.5})
	var foil_dark := M.mat(Color(0.25, 0.22, 0.2), 0.45, 0.8, {"normal_noise": 2.0, "noise_scale": 4.0})
	var hull := Assets.pbr("metal_plate", {"uv": 2.5, "tint": Color(0.82, 0.83, 0.85), "color": Color(0.75, 0.75, 0.78)})
	var black := M.mat(Color(0.03, 0.03, 0.035), 0.25, 0.5)
	var bell := M.mat(Color(0.18, 0.17, 0.17), 0.35, 0.9)
	var glass := M.glass(Color(0.05, 0.07, 0.1), 0.85)
	# Descent stage: octagonal, gold foil, y -0.33..0.0 (Box2D polygon bottom)
	var ds := M.mi(root, M.cyl(0.58, 0.58, 0.34, 8), foil, Vector3(0, -0.165, 0), Vector3(0, 22.5, 0))
	ds.name = "Descent"
	M.mi(root, M.cyl(0.6, 0.6, 0.03, 8), foil_dark, Vector3(0, 0.0, 0), Vector3(0, 22.5, 0))
	# Ascent stage (y 0..0.57), faceted cabin narrowing towards the top
	var prof := PackedVector2Array([Vector2(-0.5, 0.0), Vector2(0.5, 0.0), Vector2(0.45, 0.38), Vector2(0.3, 0.55), Vector2(-0.3, 0.55), Vector2(-0.45, 0.38)])
	M.mi(root, M.prism(prof, -0.42, 0.42), hull)
	M.mi(root, M.box(Vector3(0.22, 0.2, 0.3), 0.02), foil_dark, Vector3(-0.42, 0.28, 0.0))
	# Windows (front: +z, the face the x-y elevation camera sees) and docking hatch on top
	for sx in [-1.0, 1.0]:
		var w := M.mi(root, M.prism(PackedVector2Array([Vector2(0, 0), Vector2(0.18, 0), Vector2(0.06, 0.16)]), 0.0, 0.01), glass,
			Vector3(sx * 0.1 - (0.18 if sx > 0 else 0.0) * 0.0, 0.3, 0.425))
		w.scale = Vector3(sx, 1, 1)
	M.mi(root, M.cyl(0.13, 0.15, 0.06, 16), hull, Vector3(0, 0.58, 0))
	# Antenna mast + dish
	M.mi(root, M.cyl(0.012, 0.012, 0.32, 6), hull, Vector3(0.22, 0.7, -0.15))
	var dish := M.mi(root, M.lathe(PackedVector2Array([Vector2(0.0, 0.0), Vector2(0.08, 0.01), Vector2(0.15, 0.04)]), 16), M.mat(Color(0.92, 0.92, 0.9), 0.4, 0.3, {"cull": false}),
		Vector3(0.22, 0.86, -0.15), Vector3(35, 0, 0))
	dish.name = "Dish"
	# RCS quads at the four upper corners
	for sx in [-1.0, 1.0]:
		for sz in [-1.0, 1.0]:
			var q := M.node(root, Vector3(sx * 0.46, 0.47, sz * 0.36))
			M.mi(q, M.box(Vector3(0.09, 0.09, 0.09), 0.01), hull)
			for d in [Vector3(sx, 0, 0), Vector3(0, 1, 0), Vector3(0, -1, 0)]:
				M.mi(q, M.cyl(0.012, 0.022, 0.05, 8), black, d * 0.065, Vector3(0, 0, -90 * d.x) if d.x != 0.0 else (Vector3(180, 0, 0) if d.y < 0.0 else Vector3.ZERO))
	# Main engine bell
	M.mi(root, M.lathe(PackedVector2Array([Vector2(0.08, 0.0), Vector2(0.12, -0.08), Vector2(0.2, -0.22)]), 20), bell, Vector3(0, -0.34, 0))
	_main_flame = Vfx.flame(root, 1.6, 0.17, Color(0.55, 0.7, 1.0))
	_main_flame.position = Vector3(0, -0.56, 0)
	# Side nozzles: action 1 fires the +x one (exhaust +x), action 3 the -x one
	_rcs_px = Vfx.flame(root, 0.6, 0.05, Color(0.8, 0.85, 1.0))
	_rcs_px.position = Vector3(0.52, 0.47, 0)
	_rcs_px.rotation_degrees = Vector3(0, 0, 90)
	_rcs_nx = Vfx.flame(root, 0.6, 0.05, Color(0.8, 0.85, 1.0))
	_rcs_nx.position = Vector3(-0.52, 0.47, 0)
	_rcs_nx.rotation_degrees = Vector3(0, 0, -90)
	return root


func _build_leg() -> Node3D:
	## One Box2D leg (0.13 x 0.53 m box) drives a front and a back strut, like the LM's four legs.
	var root := Node3D.new()
	root.name = "Leg"
	content.add_child(root)
	var foil := M.mat(Color(0.86, 0.64, 0.22), 0.3, 1.0, {"normal_noise": 2.5, "noise_scale": 5.0})
	var steel := M.mat(Color(0.7, 0.7, 0.72), 0.3, 0.9)
	for z in [-0.42, 0.42]:
		M.mi(root, M.cyl(0.035, 0.035, 0.56, 10), foil, Vector3(0, 0.0, z))
		M.mi(root, M.cyl(0.012, 0.012, 0.6, 6), steel, Vector3(0, 0.02, z * 0.55), Vector3(0, 0, 0)).rotation = Vector3(atan2(z * 0.45, 0.6), 0, 0)
		M.mi(root, M.cyl(0.12, 0.14, 0.035, 18), steel, Vector3(0, -0.28, z))
	return root


# ---------------------------------------------------------------- per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	Vfx.set_flame(_main_flame, float(p.get("main", 0.0)))
	Vfx.set_flame(_rcs_px, float(p.get("rcs_px", 0.0)))
	Vfx.set_flame(_rcs_nx, float(p.get("rcs_nx", 0.0)))
	# Flags turn green once landed (as in GymBox)
	var landed := int(p.get("landed", 0)) == 1
	for m in _flag_mats:
		m.albedo_color = Color(0.15, 0.75, 0.2) if landed else Color(0.85, 0.12, 0.05)
	# Gentle cloth ripple from sim time (deterministic)
	for i in _flags.size():
		_flags[i].rotation = Vector3(0, 0.25 * sin(sim_t * 1.3 + i), 0.05 * sin(sim_t * 2.1 + i))
	# Dust kicked up by the main engine near the ground
	var lx := _lander.position.x
	var ground := _height_at(lx)
	var alt := _lander.position.y - ground
	var main := float(p.get("main", 0.0))
	if main > 0.0 and alt < 4.0:
		_dust.position = Vector3(lx, ground + 0.05, 0)
		Vfx.set_puff(_dust, 0.35 + 0.25 * sin(sim_t * 3.0), 1.6 * (1.0 - alt / 4.0) + 0.4, 0.7 * (1.0 - alt / 4.0))
	else:
		_dust.visible = false


func chase_pose() -> Transform3D:
	var p := _lander.global_position if _lander else Vector3.ZERO
	return Transform3D(Basis(), p + Vector3(-4.5, 1.8, 6.5)).looking_at(p + Vector3(0, -0.2, 0), Vector3.UP)
