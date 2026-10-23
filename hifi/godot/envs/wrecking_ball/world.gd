extends EnvWorld
## Wrecking Ball: crawler crane with a lattice boom swinging a steel ball at two pillars of precast
## concrete blocks on a demolition site; jersey barrier between the crane and the site.
## Python: hifi/exporters/wrecking_ball.py. Side-view env: Box2D (x, y) = Godot (x, y), z = depth.
## Moving ids: chassis, boom (class "crane"), ball ("ball"), block0..5 ("target_block" for the
## upper two of each pillar, "base_block" for the bottom ones). Posed here each step: hydraulics
## ("crane", luffing rams), cable ("cable": steel rope rebuilt in place from p.cable + shackle).
## Static: barrier ("barrier"). p: cable (7 points), knocked[6], tread (chassis x). dyn: dust.

const Parts := preload("res://envs/excavator/parts.gd")

const GROUND := 2.0
const LANE := 8.0
const X0 := -70.0           # terrain strip start (the rear camera sits just behind it)
const X1 := 240.0
const Z_BACK := -90.0
const Z_FRONT := 60.0
const BOOM_RAM_BASE := Vector3(2.35, -0.1, 0.0)     # chassis
const BOOM_RAM_END := Vector3(-1.0, -0.36, 0.0)     # boom
const BOOM_RAM_Z := 0.62
const CABLE_R := 0.035

var _chassis: Node3D
var _boom: Node3D
var _tracks: Array = []
var _rams: Array = []
var _hoist_mesh := ArrayMesh.new()
var _smoke: Array = []
var _cable_mesh := ArrayMesh.new()
var _shackle: Node3D
var _block_mats: Array = []
var _noise := FastNoiseLite.new()

var _orange: Material
var _dark: Material
var _steel: Material
var _shoe: Material
var _chrome: Material
var _pin: Material
var _glass: Material
var _rubber: Material


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_38d_partly_cloudy_puresky", "sun_energy": 2.3, "sun_color": Color(1.0, 0.96, 0.9),
		"ambient_energy": 0.95, "exposure": 0.95, "white": 7.0, "ssao": true, "ssil": true, "glow": true,
		"saturation": 1.05, "contrast": 1.05, "shadow_distance": 180.0,
		"horizon_y": GROUND, "card_ground": "gravel", "card_ground_tint": Color(0.85, 0.82, 0.78),
	})


func build_scene(scene: Dictionary) -> void:
	_make_materials()
	_build_terrain()
	_build_backdrop()
	_build_site(scene)
	_build_barrier(scene.get("barrier", [11.0, 2.5, 0.25, 0.5]))
	_chassis = _build_chassis()
	register("chassis", _chassis, "crane")
	_boom = _build_boom()
	register("boom", _boom, "crane")
	register("hydraulics", _build_hydraulics(), "crane")
	register("ball", _build_ball(), "ball")
	register("cable", _build_cable(), "cable")
	var targets: Array = scene.get("target", [0, 1, 1, 0, 1, 1])
	var half: Array = scene.get("block_half", [0.6, 0.7])
	for i in targets.size():
		register("block%d" % i, _build_block(i, int(targets[i]) == 1, half), "target_block" if int(targets[i]) == 1 else "base_block")


func _make_materials() -> void:
	_orange = Parts.grime_paint(Color(0.9, 0.5, 0.06), {"grime": 0.42, "bottom": 1.0, "low": 2.0, "high": 3.0, "coat": 0.6, "seed": 31})
	_dark = Parts.grime_paint(Color(0.07, 0.07, 0.075), {"grime": 0.3, "bottom": 1.0, "low": 2.0, "high": 2.9, "rough": 0.55, "metal": 0.3, "coat": 0.1, "seed": 32})
	_steel = Parts.grime_paint(Color(0.2, 0.2, 0.21), {"grime": 0.65, "bottom": 1.0, "low": 2.0, "high": 2.8, "rough": 0.45, "metal": 0.8, "coat": 0.0, "seed": 33})
	_shoe = Parts.grime_paint(Color(0.1, 0.095, 0.09), {"grime": 0.85, "bottom": 1.0, "low": 2.0, "high": 2.7, "rough": 0.6, "metal": 0.6, "coat": 0.0, "seed": 34, "scale": 1.6})
	_chrome = M.mat(Color(0.86, 0.87, 0.9), 0.08, 1.0)
	_pin = M.mat(Color(0.3, 0.3, 0.31), 0.35, 0.9)
	_glass = M.glass(Color(0.08, 0.11, 0.13), 0.62)
	_rubber = M.mat(Color(0.03, 0.03, 0.03), 0.75, 0.0)


# ---------------------------------------------------------------- terrain / scenery

func _height(x: float, z: float) -> float:
	## Flat lot (|z| < LANE), low rubble mounds behind, a shallow dip in front that never rises
	## above the lot (nothing hides the crane from the side camera).
	if z < -LANE:
		var t := clampf((-z - LANE) / 6.0, 0.0, 1.0)
		var fade := clampf((z - Z_BACK) / 25.0, 0.0, 1.0)
		return GROUND + t * t * (3.0 - 2.0 * t) * fade * (0.8 + 0.7 * _noise.get_noise_2d(x, z))
	if z > LANE:
		var t := clampf((z - LANE) / (Z_FRONT - LANE), 0.0, 1.0)
		return GROUND - sin(PI * t) * (0.6 + 0.35 * _noise.get_noise_2d(x + 40.0, z))
	return GROUND


func _build_terrain() -> void:
	_noise.seed = 9
	_noise.frequency = 0.05
	_noise.fractal_octaves = 3
	var xs := PackedFloat32Array()
	for i in int(X1 - X0) + 1:
		xs.append(X0 + i)
	var top := Parts.ground_blend("gravel", "dirt", {"a_uv": 0.35, "b_uv": 0.25, "lane": 5.5, "soft": 3.0,
		"a_tint": Color(0.95, 0.92, 0.88), "b_tint": Color(0.9, 0.86, 0.8), "mask_freq": 0.03, "ruts": 0.7,
		"rut_z": Vector2(1.2, 0.45), "seed": 51, "lane_bias": 0.9})
	var sec := Assets.pbr("excavated_soil_wall", {"uv": 0.3, "tint": Color(0.82, 0.78, 0.74)})
	content.add_child(Parts.heightfield(xs, Z_BACK, Z_FRONT, 0.75, _height, -3.0, top, sec))
	# Ground beyond the strip, in every view (one consistent world; the ortho elevations see it
	# edge-on). No shadow: it would only shade the strip's soil cross-sections below it.
	var ring := _ground_ring(Rect2(X0, Z_BACK, X1 - X0, Z_FRONT - Z_BACK), 2500.0, GROUND - 0.04,
		Parts.matte(Assets.pbr("dry_ground_01", {"uv": 0.12, "world": true, "tint": Color(0.82, 0.78, 0.72)})))
	ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(ring)


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
	## City blocks and trees around the site. Cross-view consistency: drawn in every view whose
	## frame contains them (the ortho rear / side elevations show them full size; their far planes
	## in exporters/wrecking_ball.py reach past them). No tree stands in the fixed rear / side
	## cameras' sight lines to the lot, so nothing has to be cut away.
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Backdrop")
	var rng := RandomNumberGenerator.new()
	rng.seed = 5
	var facades := [_facade("rectangular_facade_tiles", 0.25, Color(0.9, 0.88, 0.85), 3.4, 1.8),
		_facade("exterior_wall_cladding", 0.3, Color(0.8, 0.82, 0.85), 3.2, 1.4),
		_facade("brick_wall_02", 0.4, Color(0.95, 0.9, 0.88), 3.0, 2.2),
		_facade("concrete_wall_008", 0.2, Color(0.82, 0.82, 0.8), 3.6, 1.6)]
	var roof := M.mat(Color(0.25, 0.25, 0.26), 0.8, 0.1)
	# Rings of buildings at 120..300 m, leaving the lot open
	for i in 70:
		var a := rng.randf() * TAU
		var d := rng.randf_range(120.0, 320.0)
		var c := Vector3(20.0 + cos(a) * d * 1.3, GROUND, sin(a) * d)
		var w := rng.randf_range(14.0, 34.0)
		var dpt := rng.randf_range(14.0, 30.0)
		var h := rng.randf_range(10.0, 48.0) * (1.0 if d > 160.0 else 0.6)
		# Low-rise blocks (6-10 m) where the ortho elevations look: along the lane beyond the site
		# (rear view) and behind the lot (side view, crane x -60..60), so they keep sky above the
		# skyline; the towers stand elsewhere in the ring.
		if (c.x > 0.0 and absf(c.z) < 40.0) or (c.z < 0.0 and c.x > -90.0 and c.x < 90.0):
			h = minf(h, 6.0 + 2.0 * (i % 3))
		var b := M.mi(root, M.box(Vector3(w, h, dpt)), facades[i % facades.size()], c + Vector3(0, h * 0.5, 0), Vector3(0, rng.randf_range(-12, 12), 0))
		b.name = "Building"
		M.mi(root, M.box(Vector3(w * 1.02, 0.6, dpt * 1.02)), roof, c + Vector3(0, h + 0.3, 0), b.rotation_degrees)
	# Trees between the lot and the buildings
	var xf: Array = []
	for i in 60:
		var a := rng.randf() * TAU
		var d := rng.randf_range(70.0, 115.0)
		var p := Vector3(20.0 + cos(a) * d * 1.3, 0.0, sin(a) * d)
		p.y = _height(p.x, p.z) - 0.1
		var t := Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * rng.randf_range(0.9, 1.6)), p)
		# keep the rear camera's corridor (x < 30 along the lane) and the side camera's
		# (between the lot and z = 70) clear
		if (p.x < 30.0 and absf(p.z) < 22.0) or (p.z > 0.0 and p.z < 75.0 and absf(p.x - 5.0) < 55.0):
			continue
		xf.append(t)
	root.add_child(Terrain.tree_multimesh(xf, "broadleaf", 3))
	# Half-demolished building shell beyond the pillars
	_build_shell(root, Vector3(42.0, GROUND, -4.0))


static var _facade_shader: Shader


func _facade(tex_name: String, uv: float, tint: Color, floor_h: float, bay: float) -> ShaderMaterial:
	## Building wall: Poly Haven texture (world triplanar) with window bands per floor.
	if _facade_shader == null:
		_facade_shader = Shader.new()
		_facade_shader.code = """
shader_type spatial;
uniform sampler2D wall : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform vec4 tint : source_color = vec4(1.0);
uniform float uv_scale = 0.25;
uniform float floor_h = 3.4;
uniform float bay = 1.8;
uniform vec4 glass : source_color = vec4(0.12, 0.15, 0.19, 1.0);
varying vec3 wp;
varying vec3 wn;
void vertex() {
	wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	wn = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
}
void fragment() {
	vec3 an = abs(wn);
	float h = an.x > an.z ? wp.z : wp.x;
	vec2 tuv = an.y > 0.7 ? wp.xz : vec2(h, wp.y);
	vec3 c = texture(wall, tuv * uv_scale).rgb * tint.rgb;
	float fy = fract((wp.y - 1.0) / floor_h);
	float fx = fract(h / bay);
	float win = (an.y < 0.5 && wp.y > 3.0) ? step(0.38, fy) * step(fy, 0.82) * step(0.14, fx) * step(fx, 0.86) : 0.0;
	float lit = step(0.93, fract(sin(dot(floor(vec2(h / bay, (wp.y - 1.0) / floor_h)), vec2(12.9898, 78.233))) * 43758.5453));
	ALBEDO = mix(c, glass.rgb + lit * vec3(0.25, 0.22, 0.15), win);
	ROUGHNESS = mix(0.85, 0.08, win);
	METALLIC = mix(0.0, 0.5, win);
}
"""
	var m := ShaderMaterial.new()
	m.shader = _facade_shader
	var t := Parts.tex(tex_name, "diff")
	m.set_shader_parameter("wall", t if t != null else Assets.noise_texture(3, 0.05, false))
	m.set_shader_parameter("tint", tint)
	m.set_shader_parameter("uv_scale", uv)
	m.set_shader_parameter("floor_h", floor_h)
	m.set_shader_parameter("bay", bay)
	return m


func _build_shell(parent: Node3D, pos: Vector3) -> void:
	var root := M.node(parent, pos, Vector3.ZERO, "Shell")
	var conc := Assets.pbr("concrete_wall_008", {"uv": 0.4, "tint": Color(0.82, 0.82, 0.8)})
	var rebar := M.mat(Color(0.3, 0.18, 0.1), 0.7, 0.6)
	var floors := 4
	for f in floors:
		var y := f * 3.4
		var span := 18.0 - f * 3.0
		M.mi(root, M.box(Vector3(10.0, 0.3, span), 0.02), conc, Vector3(0, y + 3.25, -(18.0 - span) * 0.5))
		for cx in [-4.6, 0.0, 4.6]:
			for k in int(span / 4.5) + 1:
				var z := 9.0 - 4.5 * k - 0.0 - (18.0 - span)
				if z < -9.0:
					continue
				M.mi(root, M.box(Vector3(0.45, 3.1, 0.45), 0.01), conc, Vector3(cx, y + 1.55, z))
		# exposed rebar at the broken edge
		for k in 8:
			var z := span - 9.0 + 0.3
			M.mi(root, M.cyl(0.015, 0.015, 0.9, 5), rebar, Vector3(-4.0 + k * 1.1, y + 3.2, z), Vector3(70 + k * 5, 0, 0))
	# rubble at the foot of the broken side
	var rng := RandomNumberGenerator.new()
	rng.seed = 12
	for i in 16:
		M.mi(root, Parts.rock_mesh(Vector3(1.6, 0.8, 1.3), 60 + i % 5, 1.0), conc, Vector3(rng.randf_range(-5.0, 5.0), -0.1, rng.randf_range(9.5, 14.0)),
			Vector3(rng.randf_range(-10, 10), rng.randf() * 360.0, 0), Vector3.ONE * rng.randf_range(0.6, 1.4))


func _build_site(scene: Dictionary) -> void:
	var site_x := float(scene.get("site_x", 20.0))
	var rng := RandomNumberGenerator.new()
	rng.seed = 23
	# Old floor slab under the pillars (flush with the ground)
	var slab := Assets.pbr("concrete_floor_02", {"uv": 0.35, "tint": Color(0.9, 0.9, 0.88)})
	M.mi(content, M.box(Vector3(9.0, 0.4, 7.0), 0.04), slab, Vector3(site_x + 1.0, GROUND - 0.195, 0), Vector3.ZERO).name = "Slab"
	# Rubble piles and debris behind the lot
	var props := M.node(content, Vector3.ZERO, Vector3.ZERO, "Props")
	var gravel := Assets.pbr("gravel", {"uv": 0.6, "tint": Color(0.75, 0.73, 0.7)})
	var rubble := Assets.pbr("concrete_floor_worn_001", {"uv": 0.9, "tint": Color(0.8, 0.79, 0.76)})
	for i in 26:
		var x := rng.randf_range(-12.0, 45.0)
		var z := rng.randf_range(-18.0, -10.0)
		var s := rng.randf_range(0.4, 1.3)
		M.mi(props, Parts.rock_mesh(Vector3(1.2, 0.6, 0.9), 40 + i % 6, 1.0), rubble, Vector3(x, _height(x, z) - 0.08, z),
			Vector3(rng.randf_range(-8, 8), rng.randf() * 360.0, rng.randf_range(-8, 8)), Vector3(s, s, s))
	# Temporary site fence along the back of the lot
	var post := M.mat(Color(0.55, 0.56, 0.58), 0.4, 0.9)
	var mesh := M.mat(Color(0.32, 0.33, 0.34), 0.5, 0.8, {"alpha": 0.16, "cull": false})
	for k in 26:
		var x := -16.0 + k * 3.5
		M.mi(props, M.cyl(0.03, 0.03, 2.0, 6), post, Vector3(x, GROUND + 1.0, -9.0))
		M.mi(props, M.box(Vector3(0.7, 0.14, 0.35), 0.02), M.mat(Color(0.42, 0.41, 0.39), 0.8), Vector3(x, GROUND + 0.07, -9.0))
		M.mi(props, M.box(Vector3(3.45, 0.03, 0.02)), post, Vector3(x + 1.75, GROUND + 1.95, -9.0))
		M.mi(props, M.box(Vector3(3.45, 0.03, 0.02)), post, Vector3(x + 1.75, GROUND + 0.08, -9.0))
		var panel := M.mi(props, M.box(Vector3(3.45, 1.9, 0.01)), mesh, Vector3(x + 1.75, GROUND + 1.0, -9.0))
		panel.set_meta("no_mask", true)
		panel.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# Waste skip and barrels near the fence
	var skip := Parts.grime_paint(Color(0.12, 0.26, 0.16), {"grime": 0.7, "coat": 0.1, "rough": 0.6, "seed": 38})
	var sk := M.node(props, Vector3(6.0, GROUND, -13.0), Vector3(0, 8, 0), "Skip")
	var outer := M.poly([[-1.8, 0.0], [1.8, 0.0], [2.2, 1.2], [-2.2, 1.2]])
	var inner := M.poly([[-1.7, 0.12], [1.7, 0.12], [2.06, 1.2], [-2.06, 1.2]])
	for sz in [-1.0, 1.0]:
		M.mi(sk, M.prism(outer, sz * 0.85 - 0.04, sz * 0.85 + 0.04), skip)
	M.mi(sk, M.prism(Geometry2D.clip_polygons(outer, inner)[0], -0.85, 0.85), skip)
	M.mi(sk, M.box(Vector3(3.5, 0.6, 1.6)), Assets.pbr("gravel", {"uv": 0.8, "tint": Color(0.7, 0.68, 0.64)}), Vector3(0, 0.75, 0))
	for i in 3:
		var br := Assets.fitted_model("Barrel_01", Vector3(0.6, 0.9, 0.6))
		if br:
			br.position = Vector3(-6.0 + i * 0.7, GROUND, -7.4)
			props.add_child(br)
	# Cones along the front of the lot. Cross-view consistency: props and cones are drawn in every
	# view whose frame contains them. The cones are small and leave gaps in front of the pillars,
	# so the side view still sees the target blocks.
	var cones := M.node(content, Vector3.ZERO, Vector3.ZERO, "Cones")
	var cone_mat := M.mat(Color(0.95, 0.35, 0.05), 0.5)
	var band := M.mat(Color(0.95, 0.95, 0.95), 0.3)
	var pillars: Array = scene.get("pillar_x", [])
	for k in 8:
		var cx := -4.0 + k * 5.0
		var clear := true
		for px in pillars:
			clear = clear and absf(cx - float(px)) > 1.6
		if not clear:
			continue
		var c := M.node(cones, Vector3(cx, GROUND, 8.5))
		M.mi(c, M.cyl(0.04, 0.17, 0.7, 12), cone_mat, Vector3(0, 0.35, 0))
		M.mi(c, M.cyl(0.08, 0.11, 0.1, 12), band, Vector3(0, 0.45, 0))
		M.mi(c, M.box(Vector3(0.4, 0.04, 0.4)), cone_mat, Vector3(0, 0.02, 0))


func _build_barrier(b: Array) -> void:
	## Static site barrier: 0.5 x 1.0 m section, extruded z +-3 -> four 1.5 m jersey barriers.
	var root := M.node(content, Vector3(float(b[0]), float(b[1]) - float(b[3]), 0), Vector3.ZERO, "Barrier")
	var conc := Assets.pbr("concrete_floor_worn_001", {"uv": 0.6, "tint": Color(0.88, 0.87, 0.84)})
	for k in 4:
		var z := -2.25 + k * 1.5
		var m := Assets.fitted_model("concrete_road_barrier", Vector3(1.48, 1.0, 0.5), "stretch")
		if m != null:
			m.rotation = Vector3(0, PI * 0.5, 0)
			m.position = Vector3(0, 0, z)
			root.add_child(m)
		else:
			var prof := M.poly([[-0.25, 0.0], [0.25, 0.0], [0.2, 0.25], [0.1, 0.35], [0.08, 1.0], [-0.08, 1.0], [-0.1, 0.35], [-0.2, 0.25]])
			M.mi(root, M.prism(prof, z - 0.74, z + 0.74), conc)
	register("barrier", root, "barrier")


# ---------------------------------------------------------------- crane

func _build_chassis() -> Node3D:
	## Box2D chassis: 5 x 2 m (local y -1..1), extruded z +-1.6. Boom hinge at (1.5, 1.0).
	var root := M.node(content, Vector3(0, 3, 0), Vector3.ZERO, "Chassis")
	for zc in [-1.2, 1.2]:
		_tracks.append(Parts.track(root, {"half_len": 2.15, "cy": -0.68, "r": 0.3, "z": zc, "w": 0.74, "pitch": 0.21,
			"shoe": _shoe, "frame": _dark, "steel": _steel, "rollers": 7, "carriers": 2}))
	M.mi(root, M.box(Vector3(3.0, 0.36, 1.7), 0.05), _dark, Vector3(0, -0.66, 0))
	M.mi(root, M.cyl(0.9, 0.95, 0.1, 32), _dark, Vector3(0, -0.3, 0))
	# Deck, counterweight slabs, machinery house
	M.mi(root, M.box(Vector3(4.75, 0.18, 3.0), 0.04), _orange, Vector3(-0.08, -0.16, 0))
	for k in 3:
		M.mi(root, M.box(Vector3(0.74, 0.32, 3.0), 0.05), _dark if k == 1 else _orange, Vector3(-2.12, 0.1 + k * 0.33, 0))
	M.mi(root, M.box(Vector3(2.3, 0.92, 2.04), 0.06), _orange, Vector3(-0.6, 0.39, -0.48))
	for k in 5:
		M.mi(root, M.box(Vector3(1.6, 0.03, 0.02)), _dark, Vector3(-0.6, 0.12 + k * 0.12, -1.505))
	# +z face of the house (seen by the side camera): access door, louvres, handrail
	M.mi(root, M.box(Vector3(0.55, 0.7, 0.02), 0.01), _orange, Vector3(-1.25, 0.37, 0.545))
	M.mi(root, M.box(Vector3(0.04, 0.12, 0.03)), _dark, Vector3(-1.04, 0.4, 0.56))
	for k in 4:
		M.mi(root, M.box(Vector3(0.75, 0.03, 0.02)), _dark, Vector3(-0.2, 0.2 + k * 0.1, 0.55))
	M.mi(root, M.box(Vector3(4.7, 0.06, 0.04), 0.01), _dark, Vector3(-0.08, -0.08, 1.5))
	for k in 4:
		M.mi(root, M.box(Vector3(0.9, 0.03, 0.06)), _dark, Vector3(-0.9, 0.86, -0.85 + k * 0.18))
	M.mi(root, M.cyl(0.06, 0.06, 0.25, 10), _pin, Vector3(-1.3, 0.97, -1.1))
	_smoke = Parts.exhaust(root, Vector3(-1.3, 1.12, -1.1))
	# Boom foot pedestal and pin (1.5, 1.0)
	var foot := M.poly([[0.8, -0.08], [2.15, -0.08], [1.95, 0.5], [1.72, 1.1], [1.28, 1.1], [1.05, 0.5]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(foot, sz * 0.56 - 0.05, sz * 0.56 + 0.05), _orange)
	M.mi(root, M.box(Vector3(1.2, 0.55, 1.0), 0.04), _orange, Vector3(1.5, 0.2, 0))
	Parts.pin(root, Vector3(1.5, 1.0, 0), 0.08, 1.26, _pin)
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.26, 0.3, 0.1), 0.02), _orange, Vector3(BOOM_RAM_BASE.x, BOOM_RAM_BASE.y, sz * (BOOM_RAM_Z + 0.12)))
		M.mi(root, M.box(Vector3(0.5, 0.2, 0.22), 0.02), _orange, Vector3(BOOM_RAM_BASE.x - 0.05, BOOM_RAM_BASE.y - 0.08, sz * BOOM_RAM_Z))
	# Cab at the front of the +z side (towards the side camera), roof at y = 0.98
	var cab := Parts.cab(root, Vector3(1.55, 1.05, 0.72), _orange, _glass, M.mat(Color(0.05, 0.05, 0.05), 0.7))
	cab.position = Vector3(1.35, -0.07, 1.12)
	M.mi(cab, M.box(Vector3(0.14, 0.06, 0.1), 0.01), M.emissive(Color(1.0, 0.45, 0.05), 1.5), Vector3(-0.55, 1.07, 0.2))
	# Walkway rails on the house
	var rail := M.mat(Color(0.9, 0.85, 0.2), 0.5, 0.3)
	M.mi(root, M.tube(PackedVector3Array([Vector3(-2.3, 0.97, 1.4), Vector3(-2.3, 0.97, 0.7), Vector3(0.5, 0.97, 0.7)]), 0.02, 6), rail)
	for x in [-2.3, -1.0, 0.4]:
		M.mi(root, M.cyl(0.015, 0.015, 0.32, 6), rail, Vector3(x, 0.82, 0.7))
	return root


func _build_boom() -> Node3D:
	## Box2D boom: 7 x 0.6 m (local x -3.5..3.5), extruded z +-0.5; hinge at -3.3, sheave at +3.3.
	## Lattice: four chords, laced on all faces, solid foot and head sections.
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Boom")
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var x0 := -2.7
	var x1 := 2.75
	var hy := 0.25
	var hz := 0.4
	var chords := [Vector2(hy, hz), Vector2(hy, -hz), Vector2(-hy, hz), Vector2(-hy, -hz)]
	for c in chords:
		st.append_from(M.tube(PackedVector3Array([Vector3(x0, c.x, c.y), Vector3(x1, c.x, c.y)]), 0.045, 8), 0, Transform3D())
	var panels := 11
	var dx := (x1 - x0) / panels
	for k in panels:
		var xa := x0 + k * dx
		var xb := xa + dx
		var flip := 1.0 if k % 2 == 0 else -1.0
		# side faces (z = +-hz): zig-zag in y; top/bottom faces (y = +-hy): zig-zag in z
		for sz in [-1.0, 1.0]:
			st.append_from(M.tube(PackedVector3Array([Vector3(xa, -hy * flip, sz * hz), Vector3(xb, hy * flip, sz * hz)]), 0.022, 5), 0, Transform3D())
		for sy in [-1.0, 1.0]:
			st.append_from(M.tube(PackedVector3Array([Vector3(xa, sy * hy, -hz * flip), Vector3(xb, sy * hy, hz * flip)]), 0.022, 5), 0, Transform3D())
		# battens
		st.append_from(M.tube(PackedVector3Array([Vector3(xa, hy, -hz), Vector3(xa, hy, hz)]), 0.02, 5), 0, Transform3D())
		st.append_from(M.tube(PackedVector3Array([Vector3(xa, -hy, -hz), Vector3(xa, -hy, hz)]), 0.02, 5), 0, Transform3D())
	M.mi(root, st.commit(), _orange)
	# Foot section (solid, around the hinge pin) and head section with the sheave
	var foot := M.poly([[-2.6, -0.3], [-3.3, -0.24], [-3.5, -0.1], [-3.5, 0.12], [-3.3, 0.26], [-2.6, 0.3]])
	M.mi(root, M.prism(foot, -0.46, 0.46), _orange)
	Parts.pin(root, Vector3(-3.3, 0, 0), 0.12, 1.1, _pin)
	var head := M.poly([[2.65, -0.3], [3.1, -0.22], [3.5, -0.05], [3.5, 0.12], [3.25, 0.28], [2.65, 0.3]])
	for sz in [-1.0, 1.0]:
		M.mi(root, M.prism(head, sz * 0.3 - 0.05, sz * 0.3 + 0.05), _orange)
	M.mi(root, M.box(Vector3(0.3, 0.5, 0.6), 0.03), _orange, Vector3(2.75, 0, 0))
	var sheave := M.node(root, Vector3(3.3, 0, 0), Vector3.ZERO, "Sheave")
	M.mi(sheave, M.cyl(0.26, 0.26, 0.16, 24), M.mat(Color(0.15, 0.15, 0.16), 0.4, 0.8), Vector3.ZERO, Vector3(90, 0, 0))
	Parts.pin(sheave, Vector3.ZERO, 0.05, 0.72, _pin)
	# Ram lugs under the boom
	M.mi(root, M.prism(M.poly([[-1.35, -0.26], [-0.65, -0.26], [-0.85, -0.44], [-1.15, -0.44]]), -0.75, 0.75), _orange)
	Parts.pin(root, BOOM_RAM_END, 0.06, BOOM_RAM_Z * 2.0 + 0.2, _pin)
	return root


func _build_hydraulics() -> Node3D:
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Hydraulics")
	for sz in [-1.0, 1.0]:
		_rams.append(Parts.ram(root, {"r": 0.11, "rod": 0.06, "lb": 1.5, "lr": 1.7, "body": _orange, "chrome": _chrome, "pin": _pin}))
	# Hoist rope: winch drum in the house -> over the boom foot -> along the boom to the head sheave
	M.mi(root, _hoist_mesh, M.mat(Color(0.18, 0.18, 0.18), 0.35, 0.9)).name = "HoistRope"
	_update_rams()
	return root


func _build_ball() -> Node3D:
	var root := M.node(content, Vector3(7, 4, 0), Vector3.ZERO, "Ball")
	var rust := Assets.pbr("rusty_metal_02", {"uv": 1.1, "tint": Color(0.36, 0.33, 0.31), "metallic": 0.9, "roughness": 0.85})
	M.mi(root, M.sphere(0.7, 32, 16), rust)
	# cast seam band (rotates with the ball, shows its spin)
	M.mi(root, M.cyl(0.705, 0.705, 0.05, 32), M.mat(Color(0.12, 0.11, 0.1), 0.6, 0.8), Vector3.ZERO, Vector3(0, 0, 90))
	return root


func _build_cable() -> Node3D:
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Cable")
	M.tube(PackedVector3Array([Vector3(6.9, 7.2, 0), Vector3(6.9, 4.7, 0)]), CABLE_R, 6, _cable_mesh)
	var rope := M.mat(Color(0.2, 0.2, 0.2), 0.35, 0.9, {"normal_noise": 1.5, "noise_scale": 6.0})
	var cm := M.mi(root, _cable_mesh, rope)
	cm.name = "Rope"
	# Swivel + shackle at the ball end (oriented along the last cable segment each step)
	_shackle = M.node(root, Vector3.ZERO, Vector3.ZERO, "Shackle")
	var forged := M.mat(Color(0.22, 0.21, 0.2), 0.45, 0.9)
	M.mi(_shackle, M.cyl(0.07, 0.07, 0.22, 12), forged, Vector3(0, 0.16, 0))
	M.mi(_shackle, M.box(Vector3(0.2, 0.06, 0.06), 0.015), forged, Vector3(0, 0.04, 0))
	for sx in [-1.0, 1.0]:
		M.mi(_shackle, M.box(Vector3(0.04, 0.14, 0.05), 0.01), forged, Vector3(sx * 0.08, -0.02, 0))
	M.mi(_shackle, M.box(Vector3(0.16, 0.08, 0.1), 0.02), forged, Vector3(0, -0.06, 0))
	return root


func _build_block(i: int, target: bool, half: Array) -> Node3D:
	## Precast concrete block 1.2 x 1.4 x 1.6 m (Box2D half (0.6, 0.7), extruded z +-0.8).
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Block%d" % i)
	var hx := float(half[0])
	var hy := float(half[1])
	var conc: BaseMaterial3D = Assets.pbr("concrete_wall_008", {"uv": 0.55, "tint": Color(0.84, 0.83, 0.8)}).duplicate()
	_block_mats.append(conc)
	M.mi(root, M.box(Vector3(hx * 2.0, hy * 2.0, 1.6), 0.05), conc)
	# Lifting anchors on top, chamfer shadow lines
	for sx in [-1.0, 1.0]:
		M.mi(root, M.cyl(0.04, 0.04, 0.02, 8), _pin, Vector3(sx * hx * 0.5, hy + 0.003, 0))
	if target:
		# spray-painted demolition cross on the faces seen by the crane / side camera
		var paint := M.mat(Color(0.95, 0.32, 0.04), 0.7, 0.0)
		for face in [[Vector3(-hx - 0.004, 0, 0), Vector3(0, 90, 0)], [Vector3(0, 0, 0.804), Vector3.ZERO], [Vector3(0, 0, -0.804), Vector3.ZERO]]:
			var f := M.node(root, face[0], face[1])
			for s in [-1.0, 1.0]:
				M.mi(f, M.box(Vector3(0.07, hy * 1.25, 0.004)), paint, Vector3.ZERO, Vector3(0, 0, s * 33.0))
	return root


# ---------------------------------------------------------------- per step

func spawn_dyn(kind: String, d: Dictionary) -> Node3D:
	if kind != "dust":
		return null
	var holder := Node3D.new()
	holder.name = "Dust"
	Vfx.puff(holder, Color(0.42, 0.4, 0.37), abs(str(d.get("id", "")).hash()) % 1000, 7)
	return holder


func update_dyn(node: Node3D, a: Dictionary, b: Dictionary, alpha: float) -> void:
	var t := lerpf(float(a.get("t", 0.0)), float(b.get("t", 0.0)), alpha)
	if node.get_child_count() > 0:
		Vfx.set_puff(node.get_child(0), t, 1.6, 0.75)


func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var tread := float(p.get("tread", _chassis.position.x))
	for t in _tracks:
		Parts.set_track(t, tread)
	_update_rams()
	Parts.set_exhaust(_smoke, sim_t, float(p.get("load", 0.0)))
	var cable: Array = p.get("cable", [])
	if cable.size() >= 2:
		var pts := PackedVector3Array()
		for c in cable:
			pts.append(Vector3(float(c[0]), float(c[1]), 0.0))
		M.tube(pts, CABLE_R, 6, _cable_mesh)
		var end := pts[pts.size() - 1]
		var dir := (pts[pts.size() - 2] - end).normalized()
		var x := dir.cross(Vector3(0, 0, 1)).normalized()
		_shackle.transform = Transform3D(Basis(x, dir, x.cross(dir)), end)
	var knocked: Array = p.get("knocked", [])
	for i in mini(knocked.size(), _block_mats.size()):
		_block_mats[i].albedo_color = Color(0.78, 0.74, 0.69) if int(knocked[i]) == 1 else Color(0.84, 0.83, 0.8)


func _update_rams() -> void:
	if _chassis == null or _boom == null:
		return
	var ct := _chassis.transform
	var bt := _boom.transform
	for i in _rams.size():
		var z := BOOM_RAM_Z * (1.0 if i == 0 else -1.0)
		Parts.set_ram(_rams[i], ct * (BOOM_RAM_BASE + Vector3(0, 0, z)), bt * (BOOM_RAM_END + Vector3(0, 0, z)))
	M.tube(PackedVector3Array([ct * Vector3(0.62, 0.62, 0.0), bt * Vector3(-3.05, 0.34, 0.0), bt * Vector3(3.2, 0.32, 0.0), bt * Vector3(3.42, 0.2, 0.0)]),
		0.022, 5, _hoist_mesh)


func chase_pose() -> Transform3D:
	var p := _chassis.global_position if _chassis else Vector3(0, 3, 0)
	return Transform3D(Basis(), p + Vector3(-7.0, 4.5, 13.0)).looking_at(p + Vector3(5.0, 1.5, 0), Vector3.UP)
