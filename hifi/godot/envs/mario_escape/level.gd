extends RefCounted
## Level geometry for the Mario Escape world (preloaded by world.gd, no class_name).
##   ground()     meadow slabs per GymBox ground segment (top at y = 0) with a dirt path along the
##                lane (|z| <= 1.5, the GymBox ground boxes), soil cross-sections, and real pits:
##                chasms cut across the whole meadow, dark walls, rocky floor, closed ends
##   platform()   brick blocks with a plank deck (top exactly at the Box2D top)
##   flagpole()   steel pole, stone plinth, gold finial, rippling gold flag (vertex shader, sim_time)
##   wall()       the west boundary wall: stone courses (one node per block), coping, merlons
##   hills()      rolling hills around the meadow (scenery: drawn in every view whose frame holds it)

const Z_BACK := -40.0
const Z_FRONT := 16.0
const DEPTH := 9.0          # slab depth below the meadow
const PIT_FLOOR := -6.0
const PATH_HW := 1.5        # GymBox ground / platform extrusion: z = -1.5 .. 1.5

static var _top_shader: Shader
static var _flag_shader: Shader


static func _tex(name: String, kind: String) -> Texture2D:
	var maps: Dictionary = Assets.index().get("textures", {}).get(name, {})
	return Assets.image_texture(maps[kind]) if maps.has(kind) else null


static func top_material() -> ShaderMaterial:
	if _top_shader == null:
		_top_shader = Shader.new()
		_top_shader.code = TOP_SHADER
	var m := ShaderMaterial.new()
	m.shader = _top_shader
	for ts in [["grass", "leafy_grass"], ["dirt", "dirt"], ["mud", "brown_mud_02"]]:
		m.set_shader_parameter(ts[0] + "_d", _tex(ts[1], "diff"))
		m.set_shader_parameter(ts[0] + "_n", _tex(ts[1], "nor_gl"))
		m.set_shader_parameter(ts[0] + "_a", _tex(ts[1], "arm"))
	m.set_shader_parameter("macro_tex", Assets.noise_texture(512, 0.015, false, 256))
	m.set_shader_parameter("path_hw", PATH_HW)
	return m


static func meadow_material() -> ShaderMaterial:
	## Lawn only (no path), for the hills and the meadow beyond the level.
	var m := top_material()
	m.set_shader_parameter("path_hw", -100.0)
	return m


static func soil_material(dark := false) -> BaseMaterial3D:
	if dark:
		return Assets.pbr("excavated_soil_wall", {"uv": 0.35, "tint": Color(0.36, 0.3, 0.26), "color": Color(0.2, 0.15, 0.11)})
	return Assets.pbr("excavated_soil_wall", {"uv": 0.35, "tint": Color(0.62, 0.52, 0.46), "color": Color(0.4, 0.29, 0.2)})


static func _quad(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, n: Vector3) -> void:
	var uv := func(p: Vector3) -> Vector2:
		return Vector2(p.z, p.y) if absf(n.x) > 0.5 else (Vector2(p.x, p.z) if absf(n.y) > 0.5 else Vector2(p.x, p.y))
	M._quad_uv(st, [a, b, c, d], n, [uv.call(a), uv.call(b), uv.call(c), uv.call(d)])


static func ground(parent: Node3D, segs: Array, pits: Array, x_west: float, x_east: float) -> Node3D:
	## segs: [[x0, x1, y0, y1], ...] GymBox ground boxes (top y1 = 0). The first and last slabs extend
	## to x_west / x_east (scenery beyond the level). Pit faces get the dark soil.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Ground")
	var top := SurfaceTool.new()
	top.begin(Mesh.PRIMITIVE_TRIANGLES)
	var soil := SurfaceTool.new()
	soil.begin(Mesh.PRIMITIVE_TRIANGLES)
	var pit := SurfaceTool.new()
	pit.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in segs.size():
		var g: Array = segs[i]
		var x0: float = x_west if i == 0 else float(g[0])
		var x1: float = x_east if i == segs.size() - 1 else float(g[1])
		var yt: float = g[3]
		var yb := yt - DEPTH
		# top: strips in z so the path edge has vertices (|z| = 1.5)
		var zs := [Z_BACK, -PATH_HW - 0.6, -PATH_HW, 0.0, PATH_HW, PATH_HW + 0.6, Z_FRONT]
		for k in zs.size() - 1:
			_quad(top, Vector3(x0, yt, zs[k]), Vector3(x1, yt, zs[k]), Vector3(x1, yt, zs[k + 1]), Vector3(x0, yt, zs[k + 1]), Vector3.UP)
		# front / back cross-sections
		_quad(soil, Vector3(x0, yb, Z_FRONT), Vector3(x1, yb, Z_FRONT), Vector3(x1, yt, Z_FRONT), Vector3(x0, yt, Z_FRONT), Vector3(0, 0, 1))
		_quad(soil, Vector3(x0, yb, Z_BACK), Vector3(x1, yb, Z_BACK), Vector3(x1, yt, Z_BACK), Vector3(x0, yt, Z_BACK), Vector3(0, 0, -1))
		# west / east faces: pit walls are dark, level ends are plain soil
		var west_pit := i > 0
		var east_pit := i < segs.size() - 1
		_quad(pit if west_pit else soil, Vector3(x0, yb, Z_BACK), Vector3(x0, yb, Z_FRONT), Vector3(x0, yt, Z_FRONT), Vector3(x0, yt, Z_BACK), Vector3(-1, 0, 0))
		_quad(pit if east_pit else soil, Vector3(x1, yb, Z_BACK), Vector3(x1, yb, Z_FRONT), Vector3(x1, yt, Z_FRONT), Vector3(x1, yt, Z_BACK), Vector3(1, 0, 0))
		# inner face at the slab's east end, facing -X (enclosed, so invisible from outside): the rear
		# elevation's near plane opens the slab the player stands on and sees this cross-section
		var xi := x1 - 0.02
		_quad(soil, Vector3(xi, yb, Z_BACK), Vector3(xi, yb, Z_FRONT), Vector3(xi, yt - 0.02, Z_FRONT), Vector3(xi, yt - 0.02, Z_BACK), Vector3(-1, 0, 0))
	for pr in pits:
		var a: float = pr[0]
		var b: float = pr[1]
		# chasm floor and its closed ends (back wall faces the side camera, front wall faces inwards)
		_quad(pit, Vector3(a, PIT_FLOOR, Z_BACK), Vector3(b, PIT_FLOOR, Z_BACK), Vector3(b, PIT_FLOOR, Z_FRONT), Vector3(a, PIT_FLOOR, Z_FRONT), Vector3.UP)
		_quad(pit, Vector3(a, PIT_FLOOR - 3.0, Z_BACK), Vector3(b, PIT_FLOOR - 3.0, Z_BACK), Vector3(b, 0.0, Z_BACK), Vector3(a, 0.0, Z_BACK), Vector3(0, 0, 1))
		_quad(pit, Vector3(a, PIT_FLOOR, Z_FRONT), Vector3(b, PIT_FLOOR, Z_FRONT), Vector3(b, 0.0, Z_FRONT), Vector3(a, 0.0, Z_FRONT), Vector3(0, 0, -1))
	top.generate_tangents()
	for pair in [[top, top_material(), "Top"], [soil, soil_material(), "Soil"], [pit, soil_material(true), "PitWalls"]]:
		var mi := MeshInstance3D.new()
		mi.name = pair[2]
		mi.mesh = (pair[0] as SurfaceTool).commit()
		mi.material_override = pair[1]
		root.add_child(mi)
	# loose rocks on the chasm floors
	var rng := RandomNumberGenerator.new()
	rng.seed = 77
	var rock := M.mat(Color(0.2, 0.18, 0.17), 0.9, 0.0, {"normal_noise": 1.5, "noise_scale": 2.0})
	for pr in pits:
		for k in 26:
			var x := rng.randf_range(float(pr[0]) + 0.2, float(pr[1]) - 0.2)
			var z := rng.randf_range(Z_BACK + 1.0, Z_FRONT - 1.0)
			var s := rng.randf_range(0.2, 0.55)
			var r := M.mi(root, M.sphere(s, 8, 5), rock, Vector3(x, PIT_FLOOR + s * 0.25, z))
			r.scale = Vector3(1.0, 0.6, 0.9)
	return root


static func platform(parent: Node3D, b: Array) -> Node3D:
	## b = [x0, x1, y0, y1]: brick body with a plank deck whose top is the Box2D top.
	var x0: float = b[0]
	var x1: float = b[1]
	var y0: float = b[2]
	var y1: float = b[3]
	var w := x1 - x0
	var root := M.node(parent, Vector3((x0 + x1) * 0.5, (y0 + y1) * 0.5, 0), Vector3.ZERO, "Platform")
	var brick := Assets.pbr("red_brick", {"uv": 0.9, "tint": Color(0.8, 0.72, 0.68), "color": Color(0.55, 0.25, 0.18)})
	var deck := Assets.pbr("weathered_planks", {"uv": 0.9, "tint": Color(0.82, 0.66, 0.5), "color": Color(0.5, 0.36, 0.24)})
	var h := y1 - y0
	M.mi(root, M.box(Vector3(w - 0.04, h - 0.1, PATH_HW * 2.0 - 0.04), 0.02), brick, Vector3(0, -0.05, 0))
	M.mi(root, M.box(Vector3(w + 0.06, 0.12, PATH_HW * 2.0 + 0.06), 0.025), deck, Vector3(0, h * 0.5 - 0.06, 0))
	# inner cross-section at the east end, facing -X (see ground()): keeps the block visible in the
	# rear elevation when its near plane cuts through it under the player
	M.mi(root, M.plane(Vector2(h - 0.04, PATH_HW * 2.0 - 0.06)), brick, Vector3(w * 0.5 - 0.04, -0.02, 0), Vector3(0, 0, 90))
	return root


static func flagpole(parent: Node3D, f: Array) -> Node3D:
	var x: float = f[0]
	var top: float = f[1]
	var root := M.node(parent, Vector3(x, 0, 0), Vector3.ZERO, "Flagpole")
	var steel := M.mat(Color(0.78, 0.79, 0.8), 0.28, 1.0)
	var gold := M.mat(Color(1.0, 0.76, 0.3), 0.22, 1.0)
	var stone := Assets.pbr("concrete_block_wall", {"uv": 1.0, "tint": Color(0.85, 0.83, 0.8), "color": Color(0.6, 0.6, 0.58)})
	M.mi(root, M.box(Vector3(0.7, 0.5, 0.7), 0.04), stone, Vector3(0, 0.25, 0))
	M.mi(root, M.cyl(0.075, 0.09, top - 0.4, 16), steel, Vector3(0, 0.4 + (top - 0.4) * 0.5, 0))
	M.mi(root, M.sphere(0.16, 20, 10), gold, Vector3(0, top + 0.12, 0))
	# flag: GymBox quad (x, top) (x+2.5, top-1) (x+2.5, top-2) (x, top-2) at z = 0, rippling
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nu := 14
	var nv := 5
	for i in nu + 1:
		for k in nv + 1:
			var u := float(i) / nu
			var v := float(k) / nv
			var xx := 0.08 + u * 2.42
			var y_hi := top - u * 1.0
			var yy := lerpf(top - 2.0, y_hi, v)
			st.set_uv(Vector2(u, v))
			st.set_normal(Vector3(0, 0, 1))
			st.add_vertex(Vector3(xx, yy, 0))
	for i in nu:
		for k in nv:
			var a := i * (nv + 1) + k
			var b := a + nv + 1
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(b)
			st.add_index(a)
			st.add_index(a + 1)
			st.add_index(b + 1)
	if _flag_shader == null:
		_flag_shader = Shader.new()
		_flag_shader.code = FLAG_SHADER
	var fm := ShaderMaterial.new()
	fm.shader = _flag_shader
	fm.set_shader_parameter("cloth_n", _tex("denim_fabric", "nor_gl"))
	var flag := MeshInstance3D.new()
	flag.name = "Flag"
	flag.mesh = st.commit()
	flag.material_override = fm
	root.add_child(flag)
	return root


static func wall(parent: Node3D, w: Array, blocks: Array) -> Array:
	## w = [x0, x1, y0, y1]; GymBox extrudes it over z = -2.5 .. 2.5. Built from the exporter's
	## blocks ([x0, x1, y0, y1, z0, z1, kind]: "s" stone course, "c" coping / merlon) so the FPV can
	## cut away exactly the blocks between its camera and the player. The stone is world-mapped,
	## so the unbevelled courses read as one seamless wall. Returns the block nodes in order.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "WestWall")
	var stone := Assets.pbr("stone_wall", {"uv": 0.5, "world": true, "tint": Color(0.88, 0.86, 0.82), "color": Color(0.55, 0.53, 0.5)})
	var cap := Assets.pbr("concrete_block_wall", {"uv": 0.8, "world": true, "tint": Color(0.8, 0.78, 0.75), "color": Color(0.55, 0.55, 0.52)})
	if blocks.is_empty():
		blocks = [[w[0], w[1], 0.0, w[3], -2.5, 2.5, "s"]]
	var out: Array = []
	for b in blocks:
		var size := Vector3(float(b[1]) - float(b[0]), float(b[3]) - float(b[2]), float(b[5]) - float(b[4]))
		var c := Vector3(float(b[0]) + float(b[1]), float(b[2]) + float(b[3]), float(b[4]) + float(b[5])) * 0.5
		var is_cap: bool = String(b[6]) == "c"
		out.append(M.mi(root, M.box(size, 0.03 if is_cap else 0.0), cap if is_cap else stone, c))
	return out


static func hills(z_lo: float, z_hi: float, x_lo: float, x_hi: float, seed: int, amp: float) -> MeshInstance3D:
	## Rolling grassy hills: a heightfield of seeded smooth bumps over the given rectangle.
	var n := FastNoiseLite.new()
	n.seed = seed
	n.frequency = 0.022
	n.fractal_octaves = 3
	var step := 3.0
	var nx := int((x_hi - x_lo) / step) + 1
	var nz := int((z_hi - z_lo) / step) + 1
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in nx:
		for k in nz:
			var x := x_lo + i * step
			var z := z_lo + k * step
			# fade to the meadow level at the rectangle's border
			var e := minf(minf(x - x_lo, x_hi - x), minf(z - z_lo, z_hi - z))
			var fade := smoothstep(0.0, 18.0, e)
			var h := amp * fade * pow(0.5 + 0.5 * n.get_noise_2d(x, z), 1.6)
			st.set_uv(Vector2(x, z))
			st.add_vertex(Vector3(x, h - 0.05, z))
	for i in nx - 1:
		for k in nz - 1:
			var a := i * nz + k
			var b := a + nz
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)
	st.generate_normals()
	st.generate_tangents()
	var mi := MeshInstance3D.new()
	mi.name = "Hills"
	mi.mesh = st.commit()
	mi.material_override = meadow_material()
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	return mi


const TOP_SHADER := """
shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;
uniform sampler2D grass_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D dirt_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D dirt_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D dirt_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D mud_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D mud_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D mud_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D macro_tex : filter_linear_mipmap, repeat_enable;
uniform float path_hw = 1.5;
uniform vec3 grass_tint = vec3(0.86, 1.02, 0.7);
uniform vec3 lush = vec3(0.04, 0.095, 0.014);   // linear albedo
uniform float lushness = 0.8;
uniform vec3 dirt_tint = vec3(0.78, 0.66, 0.54);
uniform vec3 mud_tint = vec3(0.85, 0.75, 0.65);
varying vec3 wp;
void vertex() {
	wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 p = wp.xz;
	float m1 = texture(macro_tex, p * 0.03).r;
	float m2 = texture(macro_tex, p * 0.17 + vec2(0.3, 0.7)).r;
	float d = abs(wp.z) - path_hw;
	// the GymBox ground boxes: a trodden earth path, grassy verge, lawn beyond
	float path = 1.0 - smoothstep(-0.12, 0.1, d + (m2 - 0.5) * 0.25);
	float rut = smoothstep(0.35, 0.0, abs(abs(wp.z) - 0.55)) * 0.6 * path;
	vec2 pg = p * 0.45;
	vec2 pd = p * 0.5;
	vec2 pm = p * 0.6;
	// lush lawn: the photo texture's luminance detail carried on a saturated green
	vec3 g0 = texture(grass_d, pg).rgb;
	float gl = dot(g0, vec3(0.3, 0.59, 0.11));
	vec3 g = mix(g0 * grass_tint, lush * (gl / 0.32), lushness) * mix(0.82, 1.12, m1) * mix(0.92, 1.06, m2);
	vec3 dd = texture(dirt_d, pd).rgb * dirt_tint;
	vec3 mm = texture(mud_d, pm).rgb * mud_tint;
	vec3 path_c = mix(dd, mm, clamp(rut + (m1 - 0.5) * 0.6, 0.0, 1.0));
	ALBEDO = mix(g, path_c, path);
	vec3 n_path = mix(texture(dirt_n, pd).rgb, texture(mud_n, pm).rgb, rut);
	NORMAL_MAP = mix(texture(grass_n, pg).rgb, n_path, path);
	vec3 a_path = mix(texture(dirt_a, pd).rgb, texture(mud_a, pm).rgb, rut);
	vec3 arm = mix(texture(grass_a, pg).rgb, a_path, path);
	AO = arm.r;
	AO_LIGHT_AFFECT = 0.4;
	// grass seen at grazing angles (side / rear elevations) must not mirror the bright sky
	ROUGHNESS = clamp(arm.g, 0.8, 1.0);
	METALLIC = 0.0;
	SPECULAR = 0.1;
}
"""

const FLAG_SHADER := """
shader_type spatial;
render_mode cull_disabled, diffuse_burley;
global uniform float sim_time;
uniform sampler2D cloth_n : hint_normal, filter_linear_mipmap, repeat_enable;
void vertex() {
	float u = UV.x;
	float w = sin(u * 7.0 - sim_time * 4.2) * 0.13 + sin(u * 13.0 - sim_time * 6.1 + UV.y * 2.0) * 0.04;
	VERTEX.z += w * u;
	NORMAL = normalize(vec3(-0.9 * u * cos(u * 7.0 - sim_time * 4.2), 0.0, 1.0));
}
void fragment() {
	vec3 gold = vec3(0.62, 0.36, 0.025);   // linear albedo
	ALBEDO = gold * (0.85 + 0.15 * UV.y);
	NORMAL_MAP = texture(cloth_n, UV * vec2(5.0, 2.0)).rgb;
	ROUGHNESS = 0.55;
	METALLIC = 0.15;
	BACKLIGHT = vec3(0.2, 0.12, 0.01);
}
"""
