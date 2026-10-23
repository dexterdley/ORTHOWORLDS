extends RefCounted
## Cheap wind-driven field dressing for the catapult world (preload("res://envs/catapult_war/nature.gd")):
##   grass(xforms, wind, opts)   MultiMesh of procedural blade clumps bending with sim_time and the wind
##   banner(parent, opts)        pole + cloth that streams downwind and ripples (vertex shader)
##   set_banner(b, wind)         points the cloth downwind, droop from the wind strength
## Everything moves with the global `sim_time` shader parameter only (deterministic).

static var _grass_shader: Shader
static var _cloth_shader: Shader


static func clump_mesh(blades: int, height: float, seed: int) -> ArrayMesh:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in blades:
		var a := rng.randf() * TAU
		var r := rng.randf_range(0.0, 0.12)
		var base := Vector3(cos(a) * r, 0.0, sin(a) * r)
		var yaw := rng.randf() * TAU
		var side := Vector3(cos(yaw), 0, sin(yaw)) * rng.randf_range(0.012, 0.022)
		var lean := Vector3(cos(a), 0, sin(a)) * rng.randf_range(0.03, 0.14)
		var h := height * rng.randf_range(0.55, 1.0)
		var tip := base + lean + Vector3(0, h, 0)
		var mid := base + lean * 0.35 + Vector3(0, h * 0.5, 0)
		var n := Vector3(0, 1, 0)
		# two-segment blade: 3 triangles, normals tilted up so both sides catch the light
		for tri in [[base - side, base + side, mid + side * 0.7], [base - side, mid + side * 0.7, mid - side * 0.7], [mid - side * 0.7, mid + side * 0.7, tip]]:
			for v in tri:
				st.set_normal((n + (v - base).normalized() * 0.2).normalized())
				st.set_uv(Vector2(0, v.y / height))
				st.add_vertex(v)
	return st.commit()


static func flower_mesh(stems: int, height: float, seed: int) -> ArrayMesh:
	## Wildflowers: thin stems with small cross-quad heads (UV.x = 1 marks the petals).
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in stems:
		var a := rng.randf() * TAU
		var base := Vector3(cos(a), 0, sin(a)) * rng.randf_range(0.0, 0.1)
		var h := height * rng.randf_range(0.6, 1.0)
		var tip := base + Vector3(rng.randf_range(-0.04, 0.04), h, rng.randf_range(-0.04, 0.04))
		var side := Vector3(0.006, 0, 0.006)
		for v in [base - side, base + side, tip]:
			st.set_normal(Vector3.UP)
			st.set_uv(Vector2(0, v.y / height))
			st.add_vertex(v)
		var r := rng.randf_range(0.025, 0.045)
		for q in [[Vector3(-r, 0, 0), Vector3(r, 0, 0), Vector3(0, 0, -r), Vector3(0, 0, r)]]:
			for tri in [[tip + q[0], tip + q[2], tip + q[1]], [tip + q[0], tip + q[1], tip + q[3]]]:
				for v in tri:
					st.set_normal(Vector3.UP)
					st.set_uv(Vector2(1, 1))
					st.add_vertex(v + Vector3(0, 0.01, 0))
	return st.commit()


static func grass_material(wind: float, opts := {}) -> ShaderMaterial:
	if _grass_shader == null:
		_grass_shader = Shader.new()
		_grass_shader.code = """
shader_type spatial;
render_mode cull_disabled, diffuse_burley, specular_schlick_ggx;
global uniform float sim_time;
uniform vec4 base_col : source_color = vec4(0.16, 0.22, 0.07, 1.0);
uniform vec4 tip_col : source_color = vec4(0.45, 0.5, 0.2, 1.0);
uniform vec4 dry_col : source_color = vec4(0.55, 0.47, 0.25, 1.0);
uniform vec4 flower_col : source_color = vec4(0.95, 0.85, 0.2, 1.0);
uniform float wind = 0.0;
uniform float height = 0.5;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
varying float hgt;
varying float dry;
void vertex() {
	hgt = clamp(VERTEX.y / height, 0.0, 1.0);
	vec3 wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	dry = texture(noise_tex, wp.xz * 0.04).r;
	float gust = sin(sim_time * (1.6 + 1.2 * abs(wind)) + wp.x * 0.45 + wp.z * 0.3);
	vec3 bend = vec3(wind * 0.16 + gust * (0.035 + 0.06 * abs(wind)), 0.0, gust * 0.02);
	VERTEX += inverse(mat3(MODEL_MATRIX)) * bend * hgt * hgt;
}
void fragment() {
	vec3 c = mix(base_col.rgb, tip_col.rgb, hgt);
	c = mix(c, dry_col.rgb, smoothstep(0.45, 0.75, dry) * 0.8);
	c = mix(c, flower_col.rgb, step(0.5, UV.x));
	ALBEDO = c;
	ROUGHNESS = 0.85;
	SPECULAR = 0.25;
	BACKLIGHT = c * 0.6;
	AO = mix(0.45, 1.0, hgt);
	AO_LIGHT_AFFECT = 0.6;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _grass_shader
	m.set_shader_parameter("wind", wind)
	m.set_shader_parameter("height", float(opts.get("height", 0.5)))
	m.set_shader_parameter("noise_tex", Assets.noise_texture(77, 0.03, false))
	if opts.has("base"):
		m.set_shader_parameter("base_col", opts["base"])
	if opts.has("tip"):
		m.set_shader_parameter("tip_col", opts["tip"])
	if opts.has("dry"):
		m.set_shader_parameter("dry_col", opts["dry"])
	if opts.has("flower"):
		m.set_shader_parameter("flower_col", opts["flower"])
	return m


static func grass(xforms: Array, mat: Material, mesh: Mesh) -> MultiMeshInstance3D:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = mesh
	mm.instance_count = xforms.size()
	for i in xforms.size():
		mm.set_instance_transform(i, xforms[i])
	var mmi := MultiMeshInstance3D.new()
	mmi.name = "Grass"
	mmi.multimesh = mm
	mmi.material_override = mat
	mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	return mmi


static func cloth_material(col_a: Color, col_b: Color, wind: float, length: float) -> ShaderMaterial:
	if _cloth_shader == null:
		_cloth_shader = Shader.new()
		_cloth_shader.code = """
shader_type spatial;
render_mode cull_disabled, diffuse_burley;
global uniform float sim_time;
uniform vec4 col_a : source_color = vec4(0.6, 0.06, 0.05, 1.0);
uniform vec4 col_b : source_color = vec4(0.85, 0.7, 0.25, 1.0);
uniform float strength = 0.5;
uniform float len = 2.0;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
void vertex() {
	float u = clamp(VERTEX.x / len, 0.0, 1.0);
	float ph = VERTEX.x * 2.6 - sim_time * (3.0 + 7.0 * strength);
	float amp = (0.05 + 0.16 * strength) * u;
	VERTEX.z += sin(ph) * amp + sin(ph * 1.7 + VERTEX.y * 2.0) * amp * 0.35;
	NORMAL = normalize(vec3(-cos(ph) * amp * 2.6, 0.0, 1.0));
}
void fragment() {
	// heraldic banner: two-colour field with a chevron and a fringe
	vec2 uv = UV;
	float chev = step(abs(uv.y - 0.5) * 0.9 + uv.x * 0.55, 0.42);
	vec3 c = mix(col_a.rgb, col_b.rgb, chev);
	c = mix(c, col_b.rgb, step(0.93, uv.x));
	float weave = texture(noise_tex, uv * vec2(6.0, 3.0)).r;
	ALBEDO = c * (0.85 + 0.3 * weave);
	ROUGHNESS = 0.9;
	BACKLIGHT = c * 0.5;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _cloth_shader
	m.set_shader_parameter("col_a", col_a)
	m.set_shader_parameter("col_b", col_b)
	m.set_shader_parameter("strength", clampf(absf(wind) / 1.5, 0.0, 1.0))
	m.set_shader_parameter("len", length)
	m.set_shader_parameter("noise_tex", Assets.noise_texture(91, 0.08, false))
	return m


static func banner(parent: Node3D, pos: Vector3, opts: Dictionary) -> Node3D:
	## Wooden pole of height h with a cloth of size (length x width) hoisted at the top.
	var h := float(opts.get("height", 7.0))
	var length := float(opts.get("length", 2.2))
	var width := float(opts.get("width", 0.9))
	var wind := float(opts.get("wind", 0.0))
	var root := M.node(parent, pos, Vector3.ZERO, "Banner")
	var wood: Material = opts.get("wood", M.mat(Color(0.35, 0.25, 0.16), 0.8))
	var iron: Material = opts.get("iron", M.mat(Color(0.15, 0.14, 0.13), 0.5, 0.8))
	M.mi(root, M.cyl(0.06, 0.08, h, 10), wood, Vector3(0, h * 0.5, 0))
	M.mi(root, M.sphere(0.09, 10, 6), iron, Vector3(0, h + 0.05, 0))
	M.mi(root, M.cyl(0.02, 0.02, length * 0.55, 6), wood, Vector3(length * 0.25, h - 0.12, 0), Vector3(0, 0, 90))
	var hoist := M.node(root, Vector3(0, h - 0.12, 0), Vector3.ZERO, "Hoist")
	var pm := PlaneMesh.new()
	pm.orientation = PlaneMesh.FACE_Z
	pm.size = Vector2(length, width)
	pm.subdivide_width = 20
	pm.subdivide_depth = 6
	pm.center_offset = Vector3(length * 0.5, -width * 0.5, 0)
	var cloth := M.mi(hoist, pm, cloth_material(opts.get("col_a", Color(0.55, 0.06, 0.05)), opts.get("col_b", Color(0.85, 0.68, 0.22)), wind, length))
	cloth.name = "Cloth"
	cloth.extra_cull_margin = 0.5
	root.set_meta("hoist", hoist)
	set_banner(root, wind)
	return root


static func set_banner(root: Node3D, wind: float) -> void:
	## Downwind (+x for wind > 0), droop 70 deg in calm air to 12 deg in a strong wind.
	var hoist: Node3D = root.get_meta("hoist")
	var s := clampf(absf(wind) / 1.5, 0.0, 1.0)
	var droop := deg_to_rad(lerpf(70.0, 12.0, sqrt(s)))
	hoist.rotation = Vector3(0, 0.0 if wind >= 0.0 else PI, -droop)
