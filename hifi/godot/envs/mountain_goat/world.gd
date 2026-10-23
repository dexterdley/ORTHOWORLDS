extends EnvWorld
## Mountain Goat: a white mountain goat trotting up an alpine ridge trail (alpine grass low, snow
## above the GymBox snow line), rocky shoulders falling away on both sides, cliff cross-sections,
## soft cumulus puffs, distant snowy ranges and pine valleys.
## Cross-view consistency: every scenery object is drawn in every view whose frame contains it; the
## only exceptions are cutaways of what stands between a camera and the goat (the clouds above it in
## the top view; the rear view's near plane east of it, see the exporter).
## Python: hifi/exporters/mountain_goat.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Moving ids: torso, head, horn0, horn1, leg0..leg7 (mask class "goat").

const Goat := preload("res://envs/mountain_goat/goat.gd")
const Alpine := preload("res://envs/mountain_goat/alpine.gd")

const LANE := 1.2          # |z| of the GymBox climbing lane (grass / snow): exact Box2D profile inside
const Z_BACK := -16.0
const Z_FRONT := 15.0
const BOTTOM := -14.0
const X_WEST := -24.0
const X_EAST := 84.0       # the rear camera stands at x = 86

var _strip: Node3D


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "qwantani_noon_puresky", "sky_energy": 1.0,
		"sun_energy": 1.7, "sun_color": Color(1.0, 0.97, 0.92),
		"ambient_energy": 0.55, "exposure": 0.82, "white": 6.0,
		"fog_density": 0.0007, "fog_color": Color(0.74, 0.82, 0.93), "fog_aerial": 0.55, "fog_sky_affect": 0.0,
		# the rear and side elevations look across the backdrop valleys at the ranges 100-600 m away:
		# the same depth haze as the FPV keeps them reading as distant
		"ortho_fog": true,
		"ssao": true, "ssil": true, "glow": true, "saturation": 1.04, "contrast": 1.05,
		"shadow_distance": 150.0,
		"horizon_y": 0.5, "card_ground": "aerial_grass_rock", "card_ground_tint": Color(0.8, 0.85, 0.75),
	})


func build_scene(scene: Dictionary) -> void:
	var prof := _profile(scene)
	_strip = Alpine.ridge_strip(prof[0], prof[1], Z_BACK, Z_FRONT, BOTTOM, LANE, scene)
	content.add_child(_strip)
	content.add_child(Alpine.boulders(_strip.get_meta("profile"), Z_BACK, Z_FRONT, X_WEST + 6.0, X_EAST - 1.5, 5150))
	_build_clouds(scene.get("clouds", []))
	content.add_child(Alpine.backdrop(Z_BACK, Z_FRONT, X_WEST, X_EAST))
	Goat.build(self, content)


func _profile(scene: Dictionary) -> Array:
	## Box2D lane profile (exact, densified to 0.25 m) plus a summit plateau to the west and the
	## valley floor to the east.
	var pts: Array = scene.get("profile", [])
	var valley: Array = scene.get("valley", [65.0, 100.0, 0.5])
	var summit: Array = scene.get("summit", [25.0, 12.0])
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	var y_top: float = summit[1]
	# West: summit plateau continues, then a gentle descent (seen far ahead in the FPV and rear views)
	var x0: float = pts[0][0] if pts.size() > 0 else 20.0
	var x := X_WEST
	while x < x0 - 0.01:
		var t := clampf((x - X_WEST) / (x0 - 8.0 - X_WEST), 0.0, 1.0)
		xs.append(x)
		ys.append(lerpf(y_top - 5.0, y_top, smoothstep(0.0, 1.0, t)))
		x += 1.0 if x < x0 - 12.0 else 0.25
	for i in pts.size():
		var a: Array = pts[i]
		if i > 0:
			var p: Array = pts[i - 1]
			var n := int(ceil((float(a[0]) - float(p[0])) / 0.25))
			for s in range(1, n):
				var tt := float(s) / n
				xs.append(lerpf(p[0], a[0], tt))
				ys.append(lerpf(p[1], a[1], tt))
		xs.append(a[0])
		ys.append(a[1])
	# Meadow edge drops to the valley floor (Box2D: vertical step at x = 65)
	var vx: float = valley[0]
	var vy: float = valley[2]
	xs.append(vx + 0.03)
	ys.append(vy)
	x = vx + 0.25
	while x < X_EAST + 0.01:
		xs.append(x)
		ys.append(vy)
		x += 0.5
	return [xs, ys]


func _build_clouds(clouds: Array) -> void:
	## The 10 GymBox cloud boxes (4.5 x 1.4 x 2.5 m) as soft puffs. A cutaway in the top view only:
	## they float between that camera and the goat (GymBox draws them under the terrain there).
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Clouds")
	var mat := Alpine.cloud_material()
	for ci in clouds.size():
		var c: Array = clouds[ci]
		var rng := RandomNumberGenerator.new()
		rng.seed = 900 + ci
		var cn := M.node(root, Vector3(c[0], c[1], c[2]))
		var w: float = c[3]
		var h: float = c[4]
		var d: float = c[5]
		for k in 14:
			var u := rng.randf_range(-1.0, 1.0) if k > 4 else (k / 4.0 - 0.5) * 2.0
			var rad := lerpf(0.45, 0.8, 1.0 - absf(u)) * h * rng.randf_range(0.8, 1.15)
			var p := Vector3(u * (w * 0.5 - rad * 0.6), rng.randf_range(-0.15, 0.25) * h + (1.0 - absf(u)) * 0.2 * h,
				rng.randf_range(-0.5, 0.5) * (d * 0.5 - rad * 0.5))
			var m := M.mi(cn, M.sphere(1.0, 20, 10), mat, p)
			m.scale = Vector3(rad * 1.25, rad, rad * 1.1)
			m.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			m.set_meta("no_mask", true)
	set_views(root, ["rear", "side", "fpv", "chase"])


func chase_pose() -> Transform3D:
	var p: Vector3 = objects["torso"].global_position if objects.has("torso") else Vector3(62, 3, 0)
	return Transform3D(Basis(), p + Vector3(3.5, 1.6, 5.5)).looking_at(p + Vector3(-0.8, 0.2, 0), Vector3.UP)
