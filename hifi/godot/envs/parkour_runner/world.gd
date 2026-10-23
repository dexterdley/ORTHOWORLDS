extends "res://envs/bipedal_walker/walker_world.gd"
## Parkour Runner (held-out): the Bipedal Walker mech in the runner palette (teal hull, navy thighs,
## white shins, skin-tone sensor head) on an urban parkour yard at sunset.
## Python: hifi/exporters/parkour_runner.py. Terrain / robot / flag / rear-cut logic is shared with
## Bipedal Walker (envs/bipedal_walker/walker_world.gd).
##
## Obstacles keep their exact Box2D x/y extents and GymBox z extents (the crest-occlusion and wall
## experiments depend on them): hurdle = orange steel vault barrier (solid, z +-4), crate = row of
## wooden crates (z +-2.5), wall = brick wall with concrete coping (z +-9.9), gap = pit with concrete
## lips (z +-9.9) and retaining end walls at z = +-10, crest = sand mound (lane piece, tapering out
## to |z| = 18 beside it). Obstacle i is registered "obs<i>" with its kind as mask class, including
## the crest / pit terrain pieces GymBox tags with it.
##
## Scenery follows the walker_world.gd consistency rule: drawn in every view whose frame contains
## it; the front group (z > LANE) is the Side view's cutaway.

const Flora := preload("res://envs/bipedal_walker/flora.gd")

const SKY := "industrial_sunset_02_puresky"
const SINK := 0.23333            # GymBox buries box bases half a terrain step below the flat pad

var nc_prof                      # Strip.Profile: terrain without crests and pits (outside the lane)
var obs := {}                    # index -> scene obstacle dict
var _m := {}


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": SKY, "sun_from_hdri": false, "sun_elev": 27.0, "sun_azim": 215.0,
		"sun_energy": 2.0, "sun_color": Color(1.0, 0.83, 0.64), "ambient_energy": 0.95,
		"exposure": 1.0, "white": 6.0, "ssao": true, "ssil": true, "glow": true,
		"fog_density": 0.0022, "fog_color": Color(0.86, 0.77, 0.68), "fog_aerial": 0.5, "fog_sky_affect": 0.0,
		"shadow_distance": 70.0, "saturation": 1.04, "contrast": 1.06,
		"horizon_y": SKY_HORIZON, "card_ground": "asphalt_02", "card_ground_tint": Color(0.72, 0.68, 0.64),
	})
	lk.sun.directional_shadow_split_1 = 0.12
	lk.sun.directional_shadow_split_2 = 0.28
	lk.sun.directional_shadow_split_3 = 0.55


func robot_kind() -> String:
	return "runner"


func with_head() -> bool:
	return true


# ================================================================ materials

func mat(key: String) -> Material:
	if _m.has(key):
		return _m[key]
	var m: Material
	match key:
		"concrete":
			m = Strip.ground_material("concrete_floor_worn_001", "concrete_floor_02", {"scale_a": 0.3, "scale_b": 0.22,
				"tint_a": Color(1.2, 1.2, 1.25), "tint_b": Color(0.9, 0.9, 0.94), "mix": 0.3, "macro": 0.025, "macro_amp": 0.3, "normal": 0.8})
		"lot":
			m = Strip.ground_material("asphalt_02", "gravel", {"scale_a": 0.22, "scale_b": 0.35,
				"tint_a": Color(0.95, 0.93, 0.92), "tint_b": Color(0.85, 0.82, 0.78), "mix": 0.45, "macro": 0.02, "macro_amp": 0.3})
		"sand":
			m = Strip.ground_material("sand_01", "sand_01", {"scale_a": 0.55, "scale_b": 0.21,
				"tint_a": Color(1.0, 0.93, 0.8), "tint_b": Color(0.96, 0.88, 0.74), "mix": 0.45, "macro": 0.04, "macro_amp": 0.18, "normal": 0.7})
		"pit":
			m = Strip.ground_material("brown_mud_02", "gravel", {"scale_a": 0.5, "scale_b": 0.6,
				"tint_a": Color(0.55, 0.48, 0.42), "tint_b": Color(0.5, 0.45, 0.4), "mix": 0.4, "macro": 0.05})
		"sand_section":
			m = Assets.pbr("sand_01", {"uv": 0.6, "tint": Color(0.85, 0.76, 0.62), "color": Color(0.6, 0.5, 0.35)})
		"slab":
			m = Assets.pbr("concrete_wall_008", {"uv": 0.6, "tint": Color(0.72, 0.72, 0.74), "color": Color(0.5, 0.5, 0.5)})
		"soil":
			m = Assets.pbr("gravel", {"uv": 0.5, "tint": Color(0.7, 0.62, 0.55), "color": Color(0.3, 0.25, 0.2)})
		"lip":
			m = Assets.pbr("concrete_wall_008", {"uv": 0.7, "tint": Color(0.8, 0.8, 0.82), "color": Color(0.55, 0.55, 0.55)})
		"brick":
			m = Assets.pbr("red_brick", {"uv": 0.85, "tint": Color(0.92, 0.82, 0.8), "color": Color(0.6, 0.25, 0.18)})
		"coping":
			m = Assets.pbr("concrete_floor_02", {"uv": 0.8, "tint": Color(0.9, 0.9, 0.9), "color": Color(0.6, 0.6, 0.6)})
		"orange":
			m = Robot.painted(Color8(232, 128, 30), 0.5, 0.3, 0.6, 0.9)
		"orange_top":
			m = Robot.painted(Color8(246, 168, 74), 0.45, 0.3, 0.6, 0.9)
		"steel":
			m = M.mat(Color(0.5, 0.51, 0.53), 0.38, 0.95, {"normal_noise": 0.4, "noise_scale": 3.0})
		"dark_steel":
			m = M.mat(Color(0.09, 0.09, 0.1), 0.45, 0.8)
		"planks":
			m = Assets.pbr("wood_floor", {"uv": 1.1, "tint": Color(0.95, 0.78, 0.6), "color": Color(0.55, 0.38, 0.22), "roughness": 0.9})
		"battens":
			m = Assets.pbr("rough_wood", {"uv": 1.5, "tint": Color(0.75, 0.6, 0.45), "color": Color(0.4, 0.3, 0.2)})
		"hazard":
			m = _hazard_material()
		"yellow":
			m = Robot.painted(Color8(236, 196, 30), 0.55, 0.2, 0.5, 1.5)
	_m[key] = m
	return m


static func _hazard_material() -> StandardMaterial3D:
	## Diagonal black / yellow hazard stripes (seeded image, object-space triplanar).
	var n := 64
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	for y in n:
		for x in n:
			var band := int(floor(float(x + y) / (n * 0.25))) % 2
			img.set_pixel(x, y, Color(0.9, 0.72, 0.08) if band == 0 else Color(0.04, 0.04, 0.04))
	img.generate_mipmaps()
	var m := StandardMaterial3D.new()
	m.albedo_texture = ImageTexture.create_from_image(img)
	m.uv1_triplanar = true
	m.uv1_scale = Vector3.ONE * 2.5
	m.roughness = 0.6
	m.metallic = 0.1
	return m


func lane_material(tag_kind: String) -> Material:
	match tag_kind:
		"crest":
			return mat("sand")
		"gap":
			return mat("pit")
	return mat("concrete")


func band_material() -> Material:
	return mat("lot")


func near_band_material(tag_kind: String) -> Material:
	return mat("sand") if tag_kind == "crest" else mat("lot")


func section_layers() -> Array:
	return [[0.32, mat("slab")], [1.0, mat("soil")]]


func section_layers_for(tag_kind: String) -> Array:
	if tag_kind == "crest":
		return [[0.6, mat("sand_section")], [1.0, mat("soil")]]
	return section_layers()


# ================================================================ profiles

func prepare(scene: Dictionary) -> void:
	obs.clear()
	for ob in scene.get("obstacles", []):
		obs[int(ob["i"])] = ob
	# Crest heights (raised cosine, as GymBox adds them to terrain_y) and the plain terrain beside
	# the lane: crests taper out over LANE..NEAR, pits stay inside the lane (retaining walls at +-10)
	var ny := PackedFloat32Array()
	for i in xs.size():
		var x := xs[i]
		var add := 0.0
		var flat := NAN
		for k in obs:
			var ob: Dictionary = obs[k]
			var x0 := float(ob["x0"])
			var x1 := float(ob["x1"])
			if ob["kind"] == "crest":
				var t := (x - x0) / maxf(x1 - x0, 1e-6)
				if t > 0.0 and t < 1.0:
					add += float(ob.get("h", float(ob["yt"]) - float(ob["yb"]))) * 0.5 * (1.0 - cos(TAU * t))
			elif ob["kind"] == "gap" and x >= x0 - 1e-4 and x <= x1 + 1e-4:
				flat = float(ob["yt"])
		ny.append(flat if not is_nan(flat) else ys[i] - add)
	nc_prof = Strip.Profile.new(xs, ny)


func band_tag(xm: float) -> int:
	for k in obs:
		var ob: Dictionary = obs[k]
		if ob["kind"] == "crest" and xm > float(ob["x0"]) and xm < float(ob["x1"]):
			return int(k)
	return -1


func band_height(x: float, z: float) -> float:
	var dist := absf(z) - LANE
	if dist <= 0.0:
		return prof.h(x)
	var taper := 1.0 - smoothstep(0.0, NEAR - LANE, dist)
	# the lane's exact height at its edge (crests) fading to the plain terrain; pits stay in the lane
	var plain: float = nc_prof.h(x)
	var y: float = plain if _in_gap(x) else lerpf(plain, prof.h(x), taper)
	var bumps := 0.05 * noise.get_noise_2d(x, z) * smoothstep(0.0, 3.0, dist)
	var hills := 6.0 * smoothstep(150.0, 320.0, absf(z)) * (0.5 + 0.5 * noise_low.get_noise_2d(x, z))
	return y + bumps + hills


func _in_gap(x: float) -> bool:
	for k in obs:
		var ob: Dictionary = obs[k]
		if ob["kind"] == "gap" and x >= float(ob["x0"]) - 0.47 and x <= float(ob["x1"]) + 0.47:
			return true
	return false


# ================================================================ obstacles

func lane_piece_extra(parent: Node3D, px: PackedFloat32Array, tag: int) -> void:
	## Pit pieces get concrete retaining walls at z = +-LANE (the pit does not continue outside the
	## lane), from the pit floor up to the pad.
	if tag < 0 or not obs.has(tag) or obs[tag]["kind"] != "gap":
		return
	var ob: Dictionary = obs[tag]
	var x0 := float(ob["x0"])
	var x1 := float(ob["x1"])
	var top := float(ob["yt"])
	var wx := PackedFloat32Array()
	for x in px:
		if x >= x0 - 1e-4 and x <= x1 + 1e-4:
			wx.append(x)
	if wx.size() < 2:
		return
	var topf := func(_x: float, _z: float) -> float: return top
	var floorf := func(x: float, _z: float) -> float: return prof.h(x)
	for side in [-1.0, 1.0]:
		var w := M.mi(parent, Strip.wall(wx, LANE * side, topf, floorf, -side), mat("lip"))
		w.name = "PitWall"


func build_obstacles(scene: Dictionary) -> void:
	for b in scene.get("boxes", []):
		var i := int(b["i"])
		var kind := String(b["kind"])
		var v: Array = b["v"]
		var x0 := float(v[0][0])
		var x1 := float(v[1][0])
		var y0 := float(v[0][1])
		var y1 := float(v[2][1])
		var z0 := float(b["z"][0])
		var z1 := float(b["z"][1])
		var parent := obs_node(i)
		var ground_y := y0 + SINK
		if obs.has(i) and kind != "gap":
			ground_y = float(obs[i]["yb"])
		var cap: Material = mat("lip")
		match kind:
			"hurdle":
				_hurdle(parent, x0, x1, y0, y1, z0, z1)
				cap = mat("orange")
			"crate":
				_crates(parent, x0, x1, ground_y, y1, z0, z1)
				cap = mat("planks")
			"wall":
				_wall(parent, x0, x1, y0, y1, z0, z1)
				cap = mat("brick")
			"gap":
				_lip(parent, x0, x1, y0, y1, z0, z1)
		add_box_cap(parent, x0, x1, y0, y1, z0, z1, cap)


func _hurdle(p: Node3D, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> void:
	## Solid orange steel vault barrier filling the GymBox box (x0..x1, up to y1, z0..z1).
	var w := x1 - x0
	var cx := 0.5 * (x0 + x1)
	var cz := 0.5 * (z0 + z1)
	var L := z1 - z0
	var cap_h := 0.05
	M.mi(p, M.box(Vector3(w - 0.03, y1 - cap_h - y0, L - 0.03), 0.02), mat("orange"), Vector3(cx, 0.5 * (y0 + y1 - cap_h), cz))
	# painted top cap (GymBox draws the hurdle's top face light orange) and a dark kick plate
	M.mi(p, M.box(Vector3(w, cap_h, L), 0.018), mat("orange_top"), Vector3(cx, y1 - cap_h * 0.5, cz))
	M.mi(p, M.box(Vector3(w, 0.06, L), 0.01), mat("dark_steel"), Vector3(cx, y0 + SINK + 0.03, cz))
	# hazard bands on both ends and along both long faces
	for s in [-1.0, 1.0]:
		M.mi(p, M.box(Vector3(w - 0.04, 0.1, 0.012), 0.003), mat("hazard"), Vector3(cx, y1 - cap_h - 0.08, cz + s * (L * 0.5 - 0.004)))
		M.mi(p, M.box(Vector3(0.012, 0.09, L - 0.3), 0.003), mat("hazard"), Vector3(cx + s * (w * 0.5 - 0.004), y1 - cap_h - 0.08, cz))
	# vertical stiffeners every ~1.3 m (flush, darker)
	var n := int(round(L / 1.33))
	for k in range(1, n):
		var z := z0 + L * k / n
		for s in [-1.0, 1.0]:
			M.mi(p, M.box(Vector3(0.014, y1 - y0 - SINK - 0.2, 0.07), 0.004), mat("steel"), Vector3(cx + s * (w * 0.5 - 0.005), 0.5 * (y0 + SINK + y1) - 0.06, z))


func _crates(p: Node3D, x0: float, x1: float, gy: float, y1: float, z0: float, z1: float) -> void:
	## Row of wooden crates filling x0..x1, ground..y1, z0..z1 (GymBox crate: z +-2.5).
	var w := x1 - x0
	var h := y1 - gy
	var L := z1 - z0
	var n := maxi(1, int(round(L / maxf(w * 1.05, 0.5))))
	var cl := L / n
	var cx := 0.5 * (x0 + x1)
	var bt := 0.055
	for k in n:
		var cz := z0 + cl * (k + 0.5)
		var c := M.node(p, Vector3(cx, gy, cz), Vector3(0, 180.0 * (k % 2), 0))
		# plank body, slightly inset so the battens frame it
		M.mi(c, M.box(Vector3(w - 0.03, h + SINK - 0.03, cl - 0.03), 0.01), mat("planks"), Vector3(0, (h - SINK) * 0.5, 0))
		# edge battens (12 edges), all inside the crate's box
		var hx := w * 0.5 - bt * 0.5
		var hz := cl * 0.5 - bt * 0.5
		for sx in [-1.0, 1.0]:
			for sz in [-1.0, 1.0]:
				M.mi(c, M.box(Vector3(bt, h, bt), 0.008), mat("battens"), Vector3(sx * hx, h * 0.5, sz * hz))
		for sy in [bt * 0.5, h - bt * 0.5]:
			for sx in [-1.0, 1.0]:
				M.mi(c, M.box(Vector3(bt, bt, cl), 0.008), mat("battens"), Vector3(sx * hx, sy, 0))
			for sz in [-1.0, 1.0]:
				M.mi(c, M.box(Vector3(w, bt, bt), 0.008), mat("battens"), Vector3(0, sy, sz * hz))
		# diagonal braces on the long (x) faces
		var diag := Vector2(cl - 2.0 * bt, h - 2.0 * bt)
		var ang := rad_to_deg(atan2(diag.y, diag.x))
		for sx in [-1.0, 1.0]:
			M.mi(c, M.box(Vector3(0.03, bt, diag.length()), 0.006), mat("battens"), Vector3(sx * (w * 0.5 - 0.015), h * 0.5, 0), Vector3(ang * sx, 0, 0))


func _wall(p: Node3D, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> void:
	## Brick wall with a concrete coping, exactly the GymBox box (full occluder, z +-9.9).
	var w := x1 - x0
	var cx := 0.5 * (x0 + x1)
	var cz := 0.5 * (z0 + z1)
	var L := z1 - z0
	var cop := 0.09
	M.mi(p, M.box(Vector3(w - 0.03, y1 - cop - y0, L - 0.03), 0.01), mat("brick"), Vector3(cx, 0.5 * (y0 + y1 - cop), cz))
	M.mi(p, M.box(Vector3(w, cop, L), 0.02), mat("coping"), Vector3(cx, y1 - cop * 0.5, cz))
	# brick piers every ~3.3 m (flush with the faces) and a dark damp course at the foot
	var n := int(round(L / 3.3))
	for k in n + 1:
		var z := z0 + 0.2 + (L - 0.4) * k / maxf(n, 1)
		M.mi(p, M.box(Vector3(w, y1 - cop - y0, 0.36), 0.01), mat("brick"), Vector3(cx, 0.5 * (y0 + y1 - cop), z))
	M.mi(p, M.box(Vector3(w - 0.02, 0.12, L - 0.02), 0.005), mat("slab"), Vector3(cx, y0 + SINK + 0.06, cz))


func _lip(p: Node3D, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> void:
	## Pit lip: a concrete block from the pit floor to the pad, with a painted safety edge.
	var w := x1 - x0
	var cx := 0.5 * (x0 + x1)
	var cz := 0.5 * (z0 + z1)
	var L := z1 - z0
	M.mi(p, M.box(Vector3(w, y1 - y0, L), 0.012), mat("lip"), Vector3(cx, 0.5 * (y0 + y1), cz))
	M.mi(p, M.box(Vector3(w * 0.42, 0.012, L - 0.1), 0.003), mat("hazard"), Vector3(cx, y1 - 0.004, cz))


# ================================================================ scenery

func build_scenery(_scene: Dictionary) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 9091
	var x0 := course.x - 40.0
	var x1 := course.y + 60.0
	var props := Node3D.new()
	props.name = "Props"
	content.add_child(props)
	# Expansion joints across the concrete lane (seen from above: motion cues for the Top view)
	var joint := M.mat(Color(0.06, 0.06, 0.065), 0.8, 0.0)
	var jx := course.x + 1.0
	while jx < course.y + 30.0:
		if not _near_obstacle(jx, 0.2):
			var a: float = prof.h(jx - 0.2)
			var b: float = prof.h(jx + 0.2)
			var jm := M.mi(props, M.box(Vector3(0.03, 0.006, 2.0 * LANE)), joint, Vector3(jx, prof.h(jx) + 0.002, 0), Vector3(0, 0, rad_to_deg(atan2(b - a, 0.4))))
			jm.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			jm.set_meta("no_mask", true)
		jx += 3.0
	# One consistent world: every prop is drawn in every view whose frame contains it; only the
	# front group (z > LANE, between the Side camera and the walker) is hidden from the Side view
	# (FRONT_VIEWS, a cutaway). Band edges stay |z| >= 10.45 so no tuft enters the Rear frame.
	# Behind the lane: barriers, street lamps, utility boxes, barrels
	var x := x0 + rng.randf_range(0.0, 5.0)
	var k := 0
	while x < x1:
		var z := rng.randf_range(-13.5, -11.2)
		var pick := k % 6
		var node: Node3D = null
		match pick:
			0, 3:
				node = Assets.fitted_model("concrete_road_barrier_02", Vector3(1.6, 0.0, 0.0), "fit")
			1:
				node = Assets.fitted_model("utility_box_01", Vector3(0.0, 1.15, 0.0), "height")
			2:
				node = Assets.fitted_model("Barrel_01", Vector3(0.0, 0.9, 0.0), "height")
			4:
				node = Assets.fitted_model("old_tyre", Vector3(0.0, 0.6, 0.0), "height")
			5:
				node = Assets.fitted_model("metal_trash_can", Vector3(1.85, 0.0, 0.0), "fit")
		if node != null:
			node.position = Vector3(x, ground(x, z) - 0.02, z)
			node.rotation = Vector3(0, (rng.randf() - 0.5) * 0.6 + (PI * 0.5 if pick in [1, 2] else 0.0), 0)
			props.add_child(node)
		x += rng.randf_range(3.0, 7.0)
		k += 1
	var lx := x0 + 6.0
	while lx < x1:
		var lamp := Assets.fitted_model("street_lamp_01", Vector3(0.0, 5.5, 0.0), "height")
		if lamp == null:
			break
		lamp.position = Vector3(lx, ground(lx, -15.5), -15.5)
		lamp.rotation = Vector3(0, PI * 0.5, 0)
		props.add_child(lamp)
		lx += 22.0
	_fence(props, x0, x1, -19.0)
	# Weeds along the kerb and in the lot
	var gh := func(xx: float, zz: float) -> float: return ground(xx, zz)
	for gb in [[-30.0, -10.45, 900], [10.45, 30.0, 700]]:
		var xf := Flora.scatter_xforms(rng, int(gb[2]), x0, x1, float(gb[0]), float(gb[1]), gh, 0.5, 1.1)
		for mmi in Flora.chunked(props, Flora.tuft_mesh(17, 14, 0.28, Color(0.09, 0.1, 0.03), Color(0.3, 0.27, 0.1)), xf, 15.0, 60.0, Flora.blade_material()):
			if float(gb[0]) > LANE:
				set_views(mmi, FRONT_VIEWS)
	# A few scrubby trees and bushes in the lot, on both sides
	var bush: Array = []
	for i in 50:
		var fx := rng.randf_range(x0, x1 + 40.0)
		var fz := rng.randf_range(-45.0, -24.0) if i % 2 == 0 else rng.randf_range(20.0, 50.0)
		var s := rng.randf_range(0.7, 1.3)
		bush.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s), Vector3(fx, ground(fx, fz) - 0.08, fz)))
	plants_split(props, "bush", bush, 401, 1.5, Color(0.36, 0.42, 0.18), 150.0)
	var trees: Array = []
	for i in 40:
		var fx := rng.randf_range(x0, x1 + 60.0)
		var fz := rng.randf_range(-58.0, -30.0) if i % 2 == 0 else rng.randf_range(28.0, 60.0)
		var s := rng.randf_range(0.8, 1.2)
		trees.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s), Vector3(fx, ground(fx, fz) - 0.15, fz)))
	plants_split(props, "tree", trees, 431, 8.0, Color(0.33, 0.42, 0.18))
	_skyline(props, rng)



func _near_obstacle(x: float, pad: float) -> bool:
	for k in obs:
		var ob: Dictionary = obs[k]
		var e := 0.47 if ob["kind"] == "gap" else 0.0
		if x > float(ob["x0"]) - e - pad and x < float(ob["x1"]) + e + pad:
			return true
	return false


func _fence(parent: Node3D, x0: float, x1: float, z: float) -> void:
	## Chain-link fence: steel posts + rails, alpha-cut diamond mesh (seeded image).
	var n := 64
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.6, 0.62, 0.64, 0.0))
	for y in n:
		for x in n:
			var u := absf(fposmod(float(x + y), 32.0) - 16.0)
			var v := absf(fposmod(float(x - y), 32.0) - 16.0)
			if u < 1.6 or v < 1.6:
				img.set_pixel(x, y, Color(0.62, 0.64, 0.66, 1.0))
	img.generate_mipmaps()
	var mesh_mat := StandardMaterial3D.new()
	mesh_mat.albedo_texture = ImageTexture.create_from_image(img)
	mesh_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	mesh_mat.alpha_scissor_threshold = 0.4
	mesh_mat.alpha_antialiasing_mode = BaseMaterial3D.ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE
	mesh_mat.metallic = 0.8
	mesh_mat.roughness = 0.45
	mesh_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mesh_mat.uv1_scale = Vector3(8.0, 8.0, 1.0)
	var steel := mat("steel")
	var fence := Node3D.new()
	fence.name = "Fence"
	parent.add_child(fence)
	var h := 2.1
	var x := x0
	while x < x1:
		var xe := minf(x + 3.0, x1)
		var ya := ground(x, z)
		var yb := ground(xe, z)
		M.mi(fence, M.cyl(0.035, 0.035, h + 0.3, 8), steel, Vector3(x, ya + h * 0.5 - 0.15, z))
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var q := [Vector3(x, ya + 0.04, z), Vector3(xe, yb + 0.04, z), Vector3(xe, yb + h, z), Vector3(x, ya + h, z)]
		var uv := [Vector2(0, h / 1.0), Vector2((xe - x) / 1.0, h / 1.0), Vector2((xe - x) / 1.0, 0), Vector2(0, 0)]
		M._quad_uv(st, q, Vector3(0, 0, 1), uv)
		var panel := M.mi(fence, st.commit(), mesh_mat)
		panel.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		panel.set_meta("no_mask", true)
		var rail := PackedVector3Array([Vector3(x, ya + h, z), Vector3(xe, yb + h, z)])
		M.mi(fence, M.tube(rail, 0.022, 6), steel)
		x = xe


func _skyline(parent: Node3D, rng: RandomNumberGenerator) -> void:
	## Distant blocks of flats / warehouses, all in front of the course (z = 75..170). An orthographic
	## Side view shows any building behind the lane at full size, a wall across the whole frame, so
	## the skyline stands on the near side, where it is the Side view's cutaway (behind that camera);
	## Top and Rear frames do not reach it.
	var facades := ["exterior_wall_cladding", "concrete_block_wall", "brick_wall_02", "rectangular_facade_tiles"]
	var glass := M.mat(Color(0.05, 0.06, 0.08), 0.12, 0.6, {"clearcoat": 0.6})
	var city := Node3D.new()
	city.name = "Skyline"
	parent.add_child(city)
	for i in 46:
		var side := 1.0
		var bx := rng.randf_range(course.x - 120.0, course.y + 220.0)
		var bz := side * rng.randf_range(75.0, 170.0)
		var w := rng.randf_range(12.0, 30.0)
		var d := rng.randf_range(10.0, 22.0)
		var h := rng.randf_range(8.0, 34.0)
		var y := ground(bx, bz) - 0.5
		var fm := Assets.pbr(facades[i % facades.size()], {"uv": 0.25, "world": true, "color": Color(0.55, 0.52, 0.5)})
		M.mi(city, M.box(Vector3(w, h, d)), fm, Vector3(bx, y + h * 0.5, bz))
		M.mi(city, M.box(Vector3(w + 0.3, 0.6, d + 0.3)), mat("slab"), Vector3(bx, y + h + 0.3, bz))
		var floors := int(h / 3.2)
		for f in range(1, floors):
			M.mi(city, M.box(Vector3(w * 0.9, 1.3, d + 0.12)), glass, Vector3(bx, y + f * 3.2 + 0.4, bz))
	set_views(city, FRONT_VIEWS)
