extends EnvWorld
## MultiCar Racing: CarRacing track (asphalt, kerbs, grass, grandstand, trees) and open-wheel cars.
## Python: hifi/exporters/multicar_racing.py. Top-down env: Box2D (x, y) -> Godot (x, h, -y).
## Moving ids: ego (and opponent). p.cars[i]: wheels [steer, phase, skid, on_grass] x4, brake,
## speed, roll, pitch. p.skids: [[on_grass, x0, y0, x1, y1, ...], ...] (CarRacing skid trails).

const Track := preload("res://envs/multicar_racing/track.gd")
const Car := preload("res://envs/multicar_racing/car.gd")
const SKID_Y := 0.033

var _cars := {}
var _skid_mesh: ImmediateMesh
var _skid_mat: StandardMaterial3D
var _mud_mat: StandardMaterial3D


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_48d_partly_cloudy_puresky", "sky_energy": 1.0,
		"sun_energy": 1.5, "shadow_distance": 260.0, "exposure": 1.0,
		"fog_density": 0.0007, "fog_color": Color(0.72, 0.79, 0.88), "fog_aerial": 0.6, "fog_sky_affect": 0.25,
		"ssao": true, "ssil": true,
		"card_ground": "aerial_grass_rock", "card_ground_tint": Color(0.82, 0.98, 0.72), "card_ground_scale": 0.12,
	})


func build_scene(scene: Dictionary) -> void:
	var track := Track.new()
	track.name = "Track"
	content.add_child(track)
	track.build(scene)
	# One consistent world: the scenery (gantry, grandstand, trees, bushes) is drawn in every view
	# whose frame contains it. The rear / side elevations cut away what stands between their camera
	# and the car with a near-plane section just in front of the car (exporters/multicar_racing.py).
	_cars.clear()
	for c in scene.get("cars", []):
		var car := Car.new()
		car.name = String(c["name"])
		content.add_child(car)
		var rgb: Array = c.get("color", [0.8, 0.1, 0.1])
		car.setup(Color(rgb[0], rgb[1], rgb[2]))
		register(String(c["name"]), car, "car")
		_cars[String(c["name"])] = car
	_skid_mesh = ImmediateMesh.new()
	var skids := M.mi(content, _skid_mesh)
	skids.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	skids.set_meta("no_mask", true)
	_skid_mat = M.mat(Color(0.03, 0.03, 0.03, 0.75), 0.9, 0.0, {"alpha": 0.75})
	_mud_mat = M.mat(Color(0.25, 0.2, 0.08, 0.8), 1.0, 0.0, {"alpha": 0.8})


func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	for c in p.get("cars", []):
		var car = _cars.get(String(c.get("name", "")))
		if car != null:
			car.apply(c, sim_t)
	_draw_skids(p.get("skids", []))


func _draw_skids(skids: Array) -> void:
	_skid_mesh.clear_surfaces()
	for grass in [0, 1]:
		var verts := PackedVector3Array()
		for s in skids:
			if int(s[0]) != grass or s.size() < 5:
				continue
			var pts: Array = []
			for k in range(1, s.size() - 1, 2):
				pts.append(Vector3(float(s[k]), SKID_Y, -float(s[k + 1])))
			for k in pts.size() - 1:
				var d: Vector3 = (pts[k + 1] - pts[k])
				if d.length() < 1e-4:
					continue
				var side := d.normalized().cross(Vector3.UP) * 0.16
				var q := [pts[k] - side, pts[k] + side, pts[k + 1] + side, pts[k + 1] - side]
				for idx in [0, 2, 1, 0, 3, 2]:
					verts.append(q[idx])
		if verts.is_empty():
			continue
		_skid_mesh.surface_begin(Mesh.PRIMITIVE_TRIANGLES, _mud_mat if grass == 1 else _skid_mat)
		for v in verts:
			_skid_mesh.surface_set_normal(Vector3.UP)
			_skid_mesh.surface_add_vertex(v)
		_skid_mesh.surface_end()


func chase_pose() -> Transform3D:
	var car: Node3D = _cars.get("ego")
	if car == null:
		return super.chase_pose()
	var fwd := -car.global_transform.basis.z
	var p := car.global_position
	return Transform3D(Basis(), p - fwd * 10.0 + Vector3.UP * 3.6).looking_at(p + fwd * 6.0 + Vector3.UP, Vector3.UP)
