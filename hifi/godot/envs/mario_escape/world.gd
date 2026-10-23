extends EnvWorld
## Mario Escape: a bright storybook-but-photoreal platformer meadow. A dirt path runs along the
## GymBox ground boxes, real chasms cut across the meadow at the four pits, brick platforms with
## plank decks, a stone west wall, a steel flagpole with a rippling gold flag, trees, shrubs and
## rolling hills behind. The runner is an original stocky adventurer animated from the state.
## Python: hifi/exporters/mario_escape.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Moving ids: player (mask class "player"); flag (class "goal", static); wall0..wallN, the west
## wall's blocks (no mask class, cut away per view through state["vis"]). p: vx, vy, g (grounded),
## fr (facing right), step, vmax.
## Cross-view consistency: every scenery object (trees, shrubs, hills, the far meadow) is drawn in
## every view whose frame contains it. The only exceptions are cutaways of what stands between a
## camera and the player: the near-side tree row in the side elevation, the west wall in the rear
## view and its blocks in front of the FPV camera (state["vis"]); the rear view's near plane is a
## section just behind the player (exporter).

const Hero := preload("res://envs/mario_escape/hero.gd")
const Level := preload("res://envs/mario_escape/level.gd")
const Trees := preload("res://envs/mario_escape/trees.gd")

const X_WEST := -24.0
const X_EAST := 112.0

var _hero := {}


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "sunflowers_puresky", "sky_energy": 1.0,
		"sun_energy": 1.35, "sun_color": Color(1.0, 0.96, 0.9),
		"ambient_energy": 0.5, "exposure": 0.95, "white": 6.0,
		"fog_density": 0.0009, "fog_color": Color(0.72, 0.8, 0.9), "fog_aerial": 0.5, "fog_sky_affect": 0.0,
		"ssao": true, "ssil": true, "glow": true, "saturation": 1.1, "contrast": 1.05,
		"shadow_distance": 140.0,
		"horizon_y": 0.0, "card_ground": "leafy_grass", "card_ground_tint": Color(0.8, 0.95, 0.7),
	})


func build_scene(scene: Dictionary) -> void:
	Level.ground(content, scene.get("ground", []), scene.get("pits", []), X_WEST, X_EAST)
	for b in scene.get("platforms", []):
		Level.platform(content, b)
	var flag := Level.flagpole(content, scene.get("flag", [70.0, 10.0, 0.1]))
	register("flag", flag, "goal")
	var blocks: Array = Level.wall(content, scene.get("wall", [-0.4, 0.0, 0.0, 18.0]), scene.get("wall_blocks", []))
	for i in blocks.size():
		register("wall%d" % i, blocks[i])
	_build_scenery(scene.get("pits", []))
	_hero = Hero.build(content)
	register("player", _hero["root"], "player")


func _on_ground(x: float, pits: Array, margin: float) -> bool:
	for pr in pits:
		if x > float(pr[0]) - margin and x < float(pr[1]) + margin:
			return false
	return true


func _build_scenery(pits: Array) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 4711
	# Scenery is drawn in every view whose frame contains it (one consistent world). The rear
	# elevation's near plane cuts away whatever lies west of the player; beyond him the tree rows
	# stack at full size along the frame's edges, clear of the path he runs on.
	# Broadleaf trees behind the path (never over a chasm)
	var xf: Array = []
	var x := X_WEST + 2.0
	while x < X_EAST - 2.0:
		x += rng.randf_range(5.0, 9.0)
		var tx := x + rng.randf_range(-1.0, 1.0)
		if not _on_ground(tx, pits, 3.0):
			continue
		var z := rng.randf_range(-9.0, -26.0)
		var s := rng.randf_range(0.75, 1.15)
		xf.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s, s)), Vector3(tx, 0.0, z)))
	var trees := Trees.forest(xf, 11)
	content.add_child(trees)
	# A second, sparser row further back
	var xf2: Array = []
	x = X_WEST
	while x < X_EAST:
		x += rng.randf_range(4.0, 7.0)
		if not _on_ground(x, pits, 3.0):
			continue
		var s2 := rng.randf_range(0.9, 1.3)
		xf2.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s2, s2, s2)), Vector3(x, 0.0, rng.randf_range(-30.0, -38.0))))
	var trees2 := Trees.forest(xf2, 12)
	content.add_child(trees2)
	# Trees on the near side of the path (z 8..15): a cutaway in the side elevation only, where the
	# whole row stands between its camera (z = 24) and the player
	var xf3: Array = []
	x = X_WEST
	while x < X_EAST:
		x += rng.randf_range(6.0, 11.0)
		if not _on_ground(x, pits, 3.5):
			continue
		var s3 := rng.randf_range(0.8, 1.2)
		xf3.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s3, s3, s3)), Vector3(x, 0.0, rng.randf_range(8.0, 15.0))))
	var trees3 := Trees.forest(xf3, 13)
	content.add_child(trees3)
	set_views(trees3, ["top", "rear", "fpv", "chase"])
	# Low shrubs along the back verge
	var shrubs := Node3D.new()
	shrubs.name = "Shrubs"
	content.add_child(shrubs)
	x = X_WEST + 4.0
	while x < X_EAST - 4.0:
		x += rng.randf_range(3.0, 6.0)
		if not _on_ground(x, pits, 1.5):
			continue
		var sh := Assets.fitted_model("shrub_03", Vector3(1.4, 0.9, 1.4) * rng.randf_range(0.7, 1.2))
		if sh == null:
			break
		sh.position = Vector3(x, 0.0, rng.randf_range(-3.6, -6.5))
		sh.rotation = Vector3(0, rng.randf() * TAU, 0)
		shrubs.add_child(sh)
	# Rolling hills far behind and in front of the level
	var back := Level.hills(-140.0, -48.0, -120.0, 230.0, 21, 11.0)
	content.add_child(back)
	var front := Level.hills(40.0, 160.0, -120.0, 230.0, 22, 9.0)
	content.add_child(front)
	# Meadow beyond the level slabs: four rectangles around them
	var far := Node3D.new()
	far.name = "FarMeadow"
	content.add_child(far)
	var gm := Level.meadow_material()
	for r in [[-300.0, 400.0, -300.0, Level.Z_BACK], [-300.0, 400.0, Level.Z_FRONT, 300.0],
			[-300.0, X_WEST, Level.Z_BACK, Level.Z_FRONT], [X_EAST, 400.0, Level.Z_BACK, Level.Z_FRONT]]:
		var w: float = r[1] - r[0]
		var d: float = r[3] - r[2]
		var pm := M.mi(far, M.plane(Vector2(w, d)), gm, Vector3(r[0] + w * 0.5, -0.02, r[2] + d * 0.5))
		pm.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	if not _hero.is_empty():
		Hero.pose(_hero, state.get("p", {}))


func chase_pose() -> Transform3D:
	var p: Vector3 = objects["player"].global_position if objects.has("player") else Vector3(2, 1, 0)
	return Transform3D(Basis(), p + Vector3(-3.5, 1.8, 5.5)).looking_at(p + Vector3(1.0, 0.2, 0), Vector3.UP)
