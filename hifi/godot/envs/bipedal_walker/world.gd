extends "res://envs/bipedal_walker/walker_world.gd"
## Bipedal Walker: a purple four-link mech walker on a rolling grassland strip under a partly
## cloudy sky. Python: hifi/exporters/bipedal_walker.py. Shared terrain / robot / flag / rear-cut
## logic: envs/bipedal_walker/walker_world.gd (also used by Parkour Runner).

const Flora := preload("res://envs/bipedal_walker/flora.gd")

const SKY := "kloofendal_48d_partly_cloudy_puresky"

var _grass: Material
var _turf: Material
var _soil: Material


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": SKY, "sun_energy": 1.7, "sun_color": Color(1.0, 0.96, 0.9),
		"ambient_energy": 1.0, "exposure": 1.0, "white": 6.0,
		"ssao": true, "ssil": true, "glow": true,
		"fog_density": 0.0014, "fog_color": Color(0.74, 0.8, 0.88), "fog_aerial": 0.5, "fog_sky_affect": 0.0,
		"shadow_distance": 70.0, "saturation": 1.06, "contrast": 1.05,
		"horizon_y": SKY_HORIZON, "card_ground": "aerial_grass_rock", "card_ground_tint": Color(0.75, 0.85, 0.65),
	})
	lk.sun.directional_shadow_split_1 = 0.12
	lk.sun.directional_shadow_split_2 = 0.28
	lk.sun.directional_shadow_split_3 = 0.55


func _materials() -> void:
	if _grass == null:
		_grass = Strip.ground_material("aerial_grass_rock", "leafy_grass", {"scale_a": 0.17, "scale_b": 0.45,
			"tint_a": Color(0.6, 0.84, 0.48), "tint_b": Color(0.52, 0.72, 0.38), "mix": 0.25, "macro": 0.018, "macro_amp": 0.25})
		_turf = Assets.pbr("brown_mud_02", {"uv": 0.7, "tint": Color(0.55, 0.5, 0.42), "color": Color(0.2, 0.17, 0.12)})
		_soil = Assets.pbr("excavated_soil_wall", {"uv": 0.3, "tint": Color(0.6, 0.5, 0.44), "color": Color(0.3, 0.22, 0.15)})


func lane_material(_tag_kind: String) -> Material:
	_materials()
	return _grass


func band_material() -> Material:
	_materials()
	return _grass


func section_layers() -> Array:
	_materials()
	return [[0.3, _turf], [1.0, _soil]]


func build_scenery(_scene: Dictionary) -> void:
	## One consistent world: every plant / rock is drawn in every view whose frame contains it. The
	## only exception is the Side view's cutaway: the front group (z > LANE, between the Side camera
	## and the walker) is hidden there (FRONT_VIEWS); it lies behind that camera anyway, and hiding
	## it also keeps its shadows off the soil cross-section. Band edges stay |z| >= 10.45 so no
	## tuft enters the Rear frame (|z| <= 10) unless it is drawn there; the lane-shoulder tufts
	## (inside the Rear frame) have no visibility range, so the Rear and FPV views agree on them.
	var rng := RandomNumberGenerator.new()
	rng.seed = 4711
	var gh := func(x: float, z: float) -> float: return ground(x, z)
	var x0 := course.x - 40.0
	var x1 := course.y + 60.0
	var veg := Node3D.new()
	veg.name = "Vegetation"
	content.add_child(veg)
	# Grass tufts: dense beside the lane, small on the lane's shoulders (z0, z1, count, s0, s1,
	# visibility range, seed)
	var grass_bands := [[-24.0, -10.45, 2600, 0.8, 1.5, 70.0, 3],
		[10.45, 26.0, 2200, 0.8, 1.6, 50.0, 5],
		[2.6, 10.0, 700, 0.45, 0.9, 0.0, 7],
		[-10.0, -2.6, 700, 0.45, 0.9, 0.0, 9],
		[-50.0, -24.0, 1400, 1.0, 1.7, 90.0, 11]]
	for gb in grass_bands:
		var xf := Flora.scatter_xforms(rng, int(gb[2]), x0, x1, float(gb[0]), float(gb[1]), gh, float(gb[3]), float(gb[4]))
		for mmi in Flora.chunked(veg, Flora.tuft_mesh(int(gb[6])), xf, 15.0, float(gb[5]), Flora.blade_material()):
			if float(gb[0]) > LANE:
				set_views(mmi, FRONT_VIEWS)
	# Rocks, mostly behind the lane
	var rocks := ["rock_07", "rock_09"]
	for i in 18:
		var z := rng.randf_range(-40.0, -11.5) if i % 5 != 0 else rng.randf_range(11.5, 30.0)
		var x := rng.randf_range(x0, x1)
		var s := rng.randf_range(0.35, 1.3)
		var rock := Assets.fitted_model(rocks[i % 2], Vector3(s, s * 0.6, s))
		if rock == null:
			break
		rock.position = Vector3(x, ground(x, z) - 0.12 * s, z)
		rock.rotation = Vector3(0, rng.randf() * TAU, 0)
		veg.add_child(rock)
		if z > LANE:
			set_views(rock, FRONT_VIEWS)
	# Shrubs behind the lane (a hedge row seen by the Side view above the ground line)
	var bushes: Array = []
	var x := x0 + rng.randf_range(0.0, 4.0)
	while x < x1:
		var z := rng.randf_range(-30.0, -12.0)
		var s := rng.randf_range(0.6, 1.25)
		bushes.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s * rng.randf_range(0.8, 1.1), s)), Vector3(x, ground(x, z) - 0.08, z)))
		x += rng.randf_range(3.0, 8.0)
	Flora.plants(veg, "bush", bushes, 101, 1.6, Color(0.36, 0.5, 0.2), 120.0)
	# Scattered bushes on both sides
	var fb: Array = []
	for i in 70:
		var fx := rng.randf_range(x0, x1 + 40.0)
		var fz := rng.randf_range(16.0, 60.0) if i % 2 == 0 else rng.randf_range(-70.0, -30.0)
		var s := rng.randf_range(0.8, 1.6)
		fb.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s), Vector3(fx, ground(fx, fz) - 0.08, fz)))
	plants_split(veg, "bush", fb, 131, 1.8, Color(0.34, 0.48, 0.19), 200.0)
	# Trees: a loose row on both sides (the back ones stand behind the walker in the Side view, at
	# full size as an orthographic camera sees them) and a far belt. The Side camera's far plane
	# (sky card at z ~ -166) reaches everything behind the lane up to the back band's end
	# (z = BACK_Z), so far-belt trees behind the lane stand only on ridge ground above the Side frame
	# (base above the frame top y = H), where every ortho view's frame excludes them.
	var trees: Array = []
	var tx := x0 + rng.randf_range(0.0, 8.0)
	while tx < x1 + 120.0:
		var z := rng.randf_range(-130.0, -34.0) if rng.randf() < 0.6 else rng.randf_range(34.0, 140.0)
		var s := rng.randf_range(0.8, 1.3)
		trees.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s * rng.randf_range(0.9, 1.15), s)), Vector3(tx, ground(tx, z) - 0.15, z)))
		tx += rng.randf_range(4.0, 11.0)
	plants_split(veg, "tree", trees, 211, 9.0, Color(0.32, 0.45, 0.18))
	var belt: Array = []
	var tries := 0
	while belt.size() < 140 and tries < 2000:
		tries += 1
		var fx := rng.randf_range(course.x - 150.0, course.y + 250.0)
		var front := belt.size() % 2 == 0
		var fz := rng.randf_range(140.0, 260.0) if front else rng.randf_range(-300.0, -175.0)
		var s := rng.randf_range(0.9, 1.5)
		var gy := ground(fx, fz)
		if not front and gy - 0.2 - 0.3 * s < view_h + 0.2:
			continue
		belt.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s), Vector3(fx, gy - 0.2, fz)))
	plants_split(veg, "tree", belt, 211, 9.0, Color(0.32, 0.45, 0.18))

