extends RefCounted
## Heavy-machinery building blocks shared by the excavator and wrecking-ball worlds
## (preload("res://envs/excavator/parts.gd")). Side-view convention: parts are built in a Box2D
## body-local frame, local (x, y) -> Godot (x, y, z), z = extrusion depth.
##
##   grime_paint(color, opts)        painted steel with dirt / wear (object-space triplanar noise)
##   stripes(a, b, width)            diagonal hazard stripes
##   ground_blend(a, b, opts)        terrain material blending two Poly Haven sets by noise + lane
##   matte(mat)                      copy of a material with weak grazing reflections (huge planes)
##   heightfield(xs, z0, z1, ...)    side-view ground from a height callable, with soil sections
##   rock_mass(half, opts)           noise-displaced rock block (no cracks between faces)
##   rock_mesh(size, seed, angular)  cached low-poly scatter rock (base on y = 0)
##   track(parent, opts)             crawler track: frame, rollers, sprocket/idler + MultiMesh shoes
##   set_track(track, phase)         moves the shoes along the loop (phase = distance driven)
##   ram(parent, opts) / set_ram()   hydraulic cylinder between two world points
##   cab(parent, size, paint, glass) operator cab with glazing, roof and rails
##   exhaust(parent, top) / set_exhaust(puffs, sim_t, load)   looping deterministic stack smoke
## The catapult world also preloads this file (ground_blend, heightfield, rock_mesh, grime_paint).

static var _grime_shader: Shader
static var _ground_shader: Shader


static func matte(m: BaseMaterial3D, spec := 0.25) -> BaseMaterial3D:
	var c: BaseMaterial3D = m.duplicate()
	c.metallic_specular = spec
	c.metallic = 0.0
	return c


static func tex(name: String, map: String) -> Texture2D:
	var maps: Dictionary = Assets.index().get("textures", {}).get(name, {})
	return Assets.image_texture(maps[map]) if maps.has(map) else null


# ================================================================ materials

static func grime_paint(color: Color, opts := {}) -> ShaderMaterial:
	## opts: grime (0..1 overall dirt), bottom (0..1) with low / high (world y where the dirt
	##       splashed up from the ground is full / gone),
	##       rough, metal, scale (noise tiles per metre), dirt (Color), coat (clearcoat 0..1),
	##       dirt_tex (Poly Haven texture name for the dirt layer)
	if _grime_shader == null:
		_grime_shader = Shader.new()
		_grime_shader.code = """
shader_type spatial;
render_mode blend_mix, cull_back, diffuse_burley, specular_schlick_ggx;
uniform vec4 base_color : source_color = vec4(0.8, 0.55, 0.05, 1.0);
uniform vec4 dirt_color : source_color = vec4(0.32, 0.26, 0.19, 1.0);
uniform float grime = 0.4;
uniform float low_y = 2.0;
uniform float high_y = 2.6;
uniform float bottom_amt = 0.0;
uniform float rough = 0.42;
uniform float metal = 0.15;
uniform float coat = 0.5;
uniform float nscale = 0.7;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform sampler2D dirt_tex : source_color, repeat_enable, filter_linear_mipmap;
uniform sampler2D dirt_nor : hint_normal, repeat_enable, filter_linear_mipmap;
varying vec3 lp;
varying vec3 ln;
varying float wy;
void vertex() { lp = VERTEX; ln = NORMAL; wy = (MODEL_MATRIX * vec4(VERTEX, 1.0)).y; }
vec4 tri(sampler2D t, vec3 p, vec3 w) {
	return texture(t, p.zy) * w.x + texture(t, p.xz) * w.y + texture(t, p.xy) * w.z;
}
void fragment() {
	vec3 w = pow(abs(ln), vec3(4.0));
	w /= (w.x + w.y + w.z + 1e-5);
	vec3 p = lp * nscale;
	float n1 = tri(noise_tex, p, w).r;
	float n2 = tri(noise_tex, p * 4.3 + 0.37, w).r;
	float bottom = (1.0 - smoothstep(low_y, high_y, wy)) * bottom_amt;
	float up = clamp(ln.y, 0.0, 1.0);
	float g = clamp((n1 * 0.75 + n2 * 0.45 - 0.7) * 3.0 + bottom * 1.2 + up * 0.15, 0.0, 1.0) * grime;
	vec3 dirt = tri(dirt_tex, p * 1.6, w).rgb * dirt_color.rgb * 1.8;
	vec3 paint = base_color.rgb * (0.94 + 0.12 * n2);
	ALBEDO = mix(paint, dirt, g);
	ROUGHNESS = mix(rough + 0.15 * n2, 0.95, g);
	METALLIC = mix(metal, 0.0, g);
	CLEARCOAT = coat * (1.0 - g);
	CLEARCOAT_ROUGHNESS = 0.25;
	vec3 nm = tri(dirt_nor, p * 1.6, w).rgb;
	NORMAL_MAP = mix(vec3(0.5, 0.5, 1.0), nm, g * 0.8);
}
"""
	var m := ShaderMaterial.new()
	m.shader = _grime_shader
	m.set_shader_parameter("base_color", color)
	m.set_shader_parameter("dirt_color", opts.get("dirt", Color(0.3, 0.25, 0.18)))
	m.set_shader_parameter("grime", float(opts.get("grime", 0.4)))
	m.set_shader_parameter("low_y", float(opts.get("low", 2.0)))
	m.set_shader_parameter("high_y", float(opts.get("high", 2.6)))
	m.set_shader_parameter("bottom_amt", float(opts.get("bottom", 0.0)))
	m.set_shader_parameter("rough", float(opts.get("rough", 0.42)))
	m.set_shader_parameter("metal", float(opts.get("metal", 0.15)))
	m.set_shader_parameter("coat", float(opts.get("coat", 0.5)))
	m.set_shader_parameter("nscale", float(opts.get("scale", 0.7)))
	m.set_shader_parameter("noise_tex", Assets.noise_texture(int(opts.get("seed", 11)), 0.03, false))
	var dt := tex(String(opts.get("dirt_tex", "dirt")), "diff")
	m.set_shader_parameter("dirt_tex", dt if dt != null else Assets.noise_texture(5, 0.05, false))
	var dn := tex(String(opts.get("dirt_tex", "dirt")), "nor_gl")
	if dn != null:
		m.set_shader_parameter("dirt_nor", dn)
	return m


static var _stripe_shader: Shader


static func stripes(a: Color, b: Color, width := 0.12) -> ShaderMaterial:
	## Diagonal hazard stripes in object space (for flat warning plates).
	if _stripe_shader == null:
		_stripe_shader = Shader.new()
		_stripe_shader.code = """
shader_type spatial;
uniform vec4 col_a : source_color = vec4(0.9, 0.6, 0.05, 1.0);
uniform vec4 col_b : source_color = vec4(0.03, 0.03, 0.03, 1.0);
uniform float width = 0.12;
varying vec3 lp;
void vertex() { lp = VERTEX; }
void fragment() {
	float s = step(0.5, fract((lp.y + lp.z + lp.x) / (2.0 * width)));
	ALBEDO = mix(col_a.rgb, col_b.rgb, s);
	ROUGHNESS = 0.55;
	CLEARCOAT = 0.3;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _stripe_shader
	m.set_shader_parameter("col_a", a)
	m.set_shader_parameter("col_b", b)
	m.set_shader_parameter("width", width)
	return m


static func ground_blend(a: String, b: String, opts := {}) -> ShaderMaterial:
	## Two Poly Haven sets blended by world-space noise; `a` dominates inside |z| < lane.
	## opts: a_uv, b_uv (tiles per metre), a_tint, b_tint, lane, soft, mask_freq, seed, macro
	if _ground_shader == null:
		_ground_shader = Shader.new()
		_ground_shader.code = """
shader_type spatial;
render_mode blend_mix, cull_back, diffuse_burley, specular_schlick_ggx;
uniform sampler2D a_alb : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D a_nrm : hint_normal, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D a_arm : hint_default_white, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D b_alb : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D b_nrm : hint_normal, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D b_arm : hint_default_white, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D mask_tex : repeat_enable, filter_linear_mipmap;
uniform vec4 a_tint : source_color = vec4(1.0);
uniform vec4 b_tint : source_color = vec4(1.0);
uniform float a_uv = 0.4;
uniform float b_uv = 0.3;
uniform float mask_uv = 0.02;
uniform float lane = 5.0;
uniform float soft = 3.0;
uniform float lane_bias = 0.8;
uniform float macro = 0.25;
uniform float ruts = 0.0;
uniform vec2 rut_z = vec2(1.15, 0.35);
varying vec3 wp;
void vertex() { wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
void fragment() {
	vec2 uv = wp.xz;
	float n = texture(mask_tex, uv * mask_uv).r;
	float n2 = texture(mask_tex, uv * mask_uv * 5.3 + 0.21).r;
	float lw = 1.0 - smoothstep(lane, lane + soft, abs(wp.z));
	float m = clamp((n * 0.7 + n2 * 0.5 - 0.6) * 4.0 + 0.5 + (lw - 0.5) * 2.0 * lane_bias, 0.0, 1.0);
	vec2 ua = uv * a_uv;
	vec2 ub = uv * b_uv;
	vec3 ca = texture(a_alb, ua).rgb * a_tint.rgb;
	vec3 cb = texture(b_alb, ub).rgb * b_tint.rgb;
	vec3 oa = texture(a_arm, ua).rgb;
	vec3 ob = texture(b_arm, ub).rgb;
	float mac = mix(1.0 - macro, 1.0 + macro, texture(mask_tex, uv * mask_uv * 0.37 + 0.5).r);
	// compacted wheel / track ruts along the lane
	float rz = abs(abs(wp.z) - rut_z.x);
	float rut = ruts * (1.0 - smoothstep(rut_z.y * 0.5, rut_z.y, rz)) * (0.6 + 0.4 * n2);
	vec3 col = mix(cb, ca, m) * mac * (1.0 - 0.28 * rut);
	ALBEDO = col;
	vec3 o = mix(ob, oa, m);
	AO = o.r;
	AO_LIGHT_AFFECT = 0.3;
	ROUGHNESS = clamp(o.g - 0.1 * rut, 0.0, 1.0);
	METALLIC = 0.0;
	SPECULAR = 0.3;
	NORMAL_MAP = mix(texture(b_nrm, ub).rgb, texture(a_nrm, ua).rgb, m);
	NORMAL_MAP_DEPTH = 1.0 - 0.6 * rut;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _ground_shader
	for k in [["a", a], ["b", b]]:
		var nm: String = k[1]
		var albedo := tex(nm, "diff")
		m.set_shader_parameter(k[0] + "_alb", albedo if albedo != null else Assets.noise_texture(nm.hash() % 1000, 0.05, false))
		var nrm := tex(nm, "nor_gl")
		if nrm != null:
			m.set_shader_parameter(k[0] + "_nrm", nrm)
		var arm := tex(nm, "arm")
		if arm != null:
			m.set_shader_parameter(k[0] + "_arm", arm)
	m.set_shader_parameter("mask_tex", Assets.noise_texture(int(opts.get("seed", 21)), 0.02, false))
	m.set_shader_parameter("a_tint", opts.get("a_tint", Color.WHITE))
	m.set_shader_parameter("b_tint", opts.get("b_tint", Color.WHITE))
	m.set_shader_parameter("a_uv", float(opts.get("a_uv", 0.4)))
	m.set_shader_parameter("b_uv", float(opts.get("b_uv", 0.3)))
	m.set_shader_parameter("mask_uv", float(opts.get("mask_freq", 0.02)))
	m.set_shader_parameter("lane", float(opts.get("lane", 5.0)))
	m.set_shader_parameter("soft", float(opts.get("soft", 3.0)))
	m.set_shader_parameter("lane_bias", float(opts.get("lane_bias", 0.8)))
	m.set_shader_parameter("macro", float(opts.get("macro", 0.25)))
	m.set_shader_parameter("ruts", float(opts.get("ruts", 0.0)))
	m.set_shader_parameter("rut_z", opts.get("rut_z", Vector2(1.15, 0.35)))
	return m


# ================================================================ terrain

static func heightfield(xs: PackedFloat32Array, z0: float, z1: float, dz: float, height: Callable, bottom: float,
		top_mat: Material, sec_mat: Material) -> Node3D:
	## Side-view ground: top surface y = height(x, z) over xs x [z0, z1] (UV = world x, z), with
	## cross-section faces dropping to `bottom` on all four sides (seen by the ortho elevations).
	var root := Node3D.new()
	root.name = "Terrain"
	var nz := maxi(1, int(ceil((z1 - z0) / dz)))
	var nx := xs.size()
	var hts := PackedFloat32Array()
	hts.resize(nx * (nz + 1))
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in nx:
		for j in nz + 1:
			var z := lerpf(z0, z1, float(j) / nz)
			var y: float = height.call(xs[i], z)
			hts[i * (nz + 1) + j] = y
			st.set_uv(Vector2(xs[i], z))
			st.add_vertex(Vector3(xs[i], y, z))
	for i in nx - 1:
		for j in nz:
			var a := i * (nz + 1) + j
			var b := a + nz + 1
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)
	st.generate_normals()
	st.generate_tangents()
	var top := MeshInstance3D.new()
	top.name = "Top"
	top.mesh = st.commit()
	top.material_override = top_mat
	root.add_child(top)
	var sec := SurfaceTool.new()
	sec.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in nx - 1:
		for f in [[nz, z1, 1.0], [0, z0, -1.0]]:
			var j: int = f[0]
			var zf: float = f[1]
			Terrain._quad(sec, [Vector3(xs[i], bottom, zf), Vector3(xs[i + 1], bottom, zf),
				Vector3(xs[i + 1], hts[(i + 1) * (nz + 1) + j], zf), Vector3(xs[i], hts[i * (nz + 1) + j], zf)], Vector3(0, 0, f[2]))
	for e in [[0, -1.0], [nx - 1, 1.0]]:
		var i: int = e[0]
		for j in nz:
			var za := lerpf(z0, z1, float(j) / nz)
			var zb := lerpf(z0, z1, float(j + 1) / nz)
			Terrain._quad(sec, [Vector3(xs[i], bottom, za), Vector3(xs[i], bottom, zb), Vector3(xs[i], hts[i * (nz + 1) + j + 1], zb),
				Vector3(xs[i], hts[i * (nz + 1) + j], za)], Vector3(e[1], 0, 0))
	sec.generate_tangents()
	var side := MeshInstance3D.new()
	side.name = "Section"
	side.mesh = sec.commit()
	side.material_override = sec_mat
	root.add_child(side)
	return root


# ================================================================ rock

static func rock_mass(half: Vector3, opts := {}) -> ArrayMesh:
	## Box of half extents `half` (centred on the origin) with every vertex pushed along a shared
	## noise field, so faces stay crack-free. opts: amp (m), freq, seed, n (grid cells per metre),
	## keep (Array of [axis, sign]: faces that must stay inside their plane, e.g. contact faces),
	## round (corner rounding 0..1)
	var amp := float(opts.get("amp", 0.25))
	var noise := FastNoiseLite.new()
	noise.seed = int(opts.get("seed", 5))
	noise.frequency = float(opts.get("freq", 0.6))
	noise.fractal_octaves = 4
	var keep: Array = opts.get("keep", [])
	var rnd := float(opts.get("round", 0.35))
	var per_m := float(opts.get("n", 3.0))
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for a in 3:
		var u := (a + 1) % 3
		var v := (a + 2) % 3
		for s in [-1.0, 1.0]:
			var nu := maxi(2, int(ceil(half[u] * 2.0 * per_m)))
			var nv := maxi(2, int(ceil(half[v] * 2.0 * per_m)))
			var grid: Array = []
			for i in nu + 1:
				var row: Array = []
				for j in nv + 1:
					var p := Vector3.ZERO
					p[a] = s * half[a]
					p[u] = lerpf(-half[u], half[u], float(i) / nu)
					p[v] = lerpf(-half[v], half[v], float(j) / nv)
					row.append(_rock_vertex(p, half, noise, amp, rnd, keep))
				grid.append(row)
			var n := Vector3.ZERO
			n[a] = s
			for i in nu:
				for j in nv:
					var q00: Vector3 = grid[i][j]
					var q10: Vector3 = grid[i + 1][j]
					var q11: Vector3 = grid[i + 1][j + 1]
					var q01: Vector3 = grid[i][j + 1]
					_tri_auto(st, q00, q10, q11, n)
					_tri_auto(st, q00, q11, q01, n)
	st.index()
	st.generate_normals()
	return st.commit()


static var _rock_cache := {}


static func rock_mesh(size: Vector3, seed: int, angular := 0.0) -> ArrayMesh:
	## Low-poly scatter rock (a few hundred triangles instead of a 66k-triangle scan), base flat on
	## y = 0. angular 0 = weathered boulder, 1 = broken concrete / blasted chunk.
	var key := "%s:%d:%.2f" % [size, seed, angular]
	if _rock_cache.has(key):
		return _rock_cache[key]
	var h := size * 0.5
	var m := rock_mass(h, {"amp": minf(h.x, minf(h.y, h.z)) * lerpf(0.5, 0.22, angular), "freq": lerpf(0.9, 2.2, angular) / maxf(h.x, h.z),
		"seed": seed, "n": clampf(5.0 / maxf(size.x, size.z), 1.5, 6.0), "round": lerpf(0.7, 0.05, angular), "keep": [[1, -1.0]]})
	# shift so the flat base sits on y = 0
	var st := SurfaceTool.new()
	st.append_from(m, 0, Transform3D(Basis(), Vector3(0, h.y, 0)))
	var out := st.commit()
	_rock_cache[key] = out
	return out


static func _rock_vertex(p: Vector3, half: Vector3, noise: FastNoiseLite, amp: float, rnd: float, keep: Array) -> Vector3:
	var q := p / half
	# Superellipsoid-ish rounding: pull corners towards the centre
	var corner := (absf(q.x) * absf(q.y) + absf(q.y) * absf(q.z) + absf(q.x) * absf(q.z)) / 3.0
	var dir := q.normalized()
	var d := clampf(noise.get_noise_3dv(p) * 2.2, -1.0, 1.0) * amp - rnd * amp * 2.0 * corner * corner
	var out := p + dir * d
	for k in keep:
		var ax: int = k[0]
		var sg: float = k[1]
		if sg > 0.0:
			out[ax] = minf(out[ax], half[ax])
		else:
			out[ax] = maxf(out[ax], -half[ax])
		if absf(p[ax] - sg * half[ax]) < 1e-4 and k.size() > 2:
			# kept face: relief only inwards (k[2] = depth in metres)
			out[ax] = sg * half[ax] - sg * minf(absf(noise.get_noise_3dv(p * 1.7 + Vector3(9, 9, 9))) * 2.5, 1.0) * float(k[2])
	return out


static func _tri_auto(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, n: Vector3) -> void:
	if (b - a).cross(c - a).dot(n) > 0.0:
		var t := b
		b = c
		c = t
	st.add_vertex(a)
	st.add_vertex(b)
	st.add_vertex(c)


# ================================================================ crawler track

static func track(parent: Node3D, opts: Dictionary) -> Dictionary:
	## Crawler track loop in the XY plane centred on (0, cy) at depth z, width w (along z).
	## opts: half_len (sprocket/idler x), cy, r (end radius of the shoe path), z, w, pitch,
	##       shoe (Material), frame (Material), steel (Material), rollers (int), carriers (int)
	var hl := float(opts.get("half_len", 1.6))
	var cy := float(opts.get("cy", -0.7))
	var r := float(opts.get("r", 0.27))
	var z := float(opts.get("z", 1.1))
	var w := float(opts.get("w", 0.6))
	var shoe_t := float(opts.get("shoe_t", 0.06))
	var shoe_mat: Material = opts.get("shoe")
	var frame_mat: Material = opts.get("frame")
	var steel: Material = opts.get("steel")
	var per := 4.0 * hl + TAU * r
	var n := int(round(per / float(opts.get("pitch", 0.19))))
	var root := M.node(parent, Vector3(0, 0, z), Vector3.ZERO, "Track")
	# Shoe: plate + grouser + link lugs (local x along the loop, +y outwards)
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var pitch := per / n
	st.append_from(M.box(Vector3(pitch * 0.92, shoe_t * 0.55, w), 0.008), 0, Transform3D(Basis(), Vector3(0, shoe_t * 0.2, 0)))
	st.append_from(M.box(Vector3(0.035, shoe_t * 0.5, w * 0.98), 0.006), 0, Transform3D(Basis(), Vector3(-pitch * 0.18, shoe_t * 0.6, 0)))
	for sz in [-1.0, 1.0]:
		st.append_from(M.box(Vector3(pitch * 0.95, shoe_t * 0.9, w * 0.13), 0.01), 0,
			Transform3D(Basis(), Vector3(0, -shoe_t * 0.35, sz * w * 0.22)))
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = st.commit()
	mm.instance_count = n
	var mmi := MultiMeshInstance3D.new()
	mmi.name = "Shoes"
	mmi.multimesh = mm
	mmi.material_override = shoe_mat
	root.add_child(mmi)
	# Sprocket (rear, toothed) and idler (front)
	var rr := r - shoe_t * 0.5 - 0.01
	var spr := M.node(root, Vector3(-hl, cy, 0), Vector3.ZERO, "Sprocket")
	M.mi(spr, M.cyl(rr * 0.92, rr * 0.92, w * 0.45, 20), steel, Vector3.ZERO, Vector3(90, 0, 0))
	M.mi(spr, M.cyl(rr * 0.45, rr * 0.45, w * 0.75, 14), frame_mat, Vector3.ZERO, Vector3(90, 0, 0))
	for k in 9:
		var a := TAU * k / 9.0
		M.mi(spr, M.box(Vector3(0.07, 0.08, w * 0.4), 0.01), steel, Vector3(cos(a) * rr * 0.95, sin(a) * rr * 0.95, 0), Vector3(0, 0, rad_to_deg(a) + 90.0))
	var idl := M.node(root, Vector3(hl, cy, 0), Vector3.ZERO, "Idler")
	M.mi(idl, M.cyl(rr, rr, w * 0.5, 22), steel, Vector3.ZERO, Vector3(90, 0, 0))
	M.mi(idl, M.cyl(rr * 0.55, rr * 0.55, w * 0.7, 14), frame_mat, Vector3.ZERO, Vector3(90, 0, 0))
	# Track frame between the rollers
	var inner_top := cy + r - shoe_t * 0.5
	var inner_bot := cy - r + shoe_t * 0.5
	M.mi(root, M.box(Vector3(hl * 2.0 - rr * 1.2, (inner_top - inner_bot) * 0.5, w * 0.62), 0.03), frame_mat,
		Vector3(0, cy + 0.02, 0))
	# Bottom rollers and top carrier rollers
	var nb := int(opts.get("rollers", 5))
	var rb := (inner_top - inner_bot) * 0.24
	for i in nb:
		var x := lerpf(-hl + rr * 1.15, hl - rr * 1.15, float(i) / maxf(nb - 1, 1))
		M.mi(root, M.cyl(rb, rb, w * 0.55, 14), steel, Vector3(x, inner_bot + rb, 0), Vector3(90, 0, 0))
	var nc := int(opts.get("carriers", 2))
	for i in nc:
		var x := lerpf(-hl * 0.45, hl * 0.45, float(i) / maxf(nc - 1, 1))
		M.mi(root, M.cyl(rb * 0.75, rb * 0.75, w * 0.3, 12), steel, Vector3(x, inner_top - rb * 0.75, 0), Vector3(90, 0, 0))
	var tr := {"mm": mm, "n": n, "per": per, "hl": hl, "cy": cy, "r": r, "root": root}
	set_track(tr, 0.0)
	return tr


static func set_track(tr: Dictionary, phase: float) -> void:
	var mm: MultiMesh = tr["mm"]
	var n: int = tr["n"]
	var per: float = tr["per"]
	var hl: float = tr["hl"]
	var cy: float = tr["cy"]
	var r: float = tr["r"]
	var top_len := 2.0 * hl
	var arc := PI * r
	for i in n:
		var s := fposmod(i * per / n + phase, per)
		var pos: Vector3
		var ang: float   # rotation about z of the shoe (local +y = outward)
		if s < top_len:
			pos = Vector3(-hl + s, cy + r, 0)
			ang = 0.0
		elif s < top_len + arc:
			var th := PI * 0.5 - (s - top_len) / r
			pos = Vector3(hl + cos(th) * r, cy + sin(th) * r, 0)
			ang = th - PI * 0.5
		elif s < 2.0 * top_len + arc:
			pos = Vector3(hl - (s - top_len - arc), cy - r, 0)
			ang = PI
		else:
			var th := -PI * 0.5 - (s - 2.0 * top_len - arc) / r
			pos = Vector3(-hl + cos(th) * r, cy + sin(th) * r, 0)
			ang = th - PI * 0.5
		mm.set_instance_transform(i, Transform3D(Basis(Vector3(0, 0, 1), ang), pos))


# ================================================================ hydraulic ram

static func ram(parent: Node3D, opts: Dictionary) -> Node3D:
	## Cylinder from A to B (set with set_ram): barrel at A, chrome rod ending at B, pin eyes.
	## opts: r (barrel radius), rod (rod radius), lb (barrel length), lr (rod length), body, chrome, pin
	var r := float(opts.get("r", 0.09))
	var rr := float(opts.get("rod", 0.05))
	var lb := float(opts.get("lb", 1.5))
	var lr := float(opts.get("lr", 1.5))
	var body: Material = opts.get("body")
	var chrome: Material = opts.get("chrome")
	var pin: Material = opts.get("pin", body)
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Ram")
	var barrel := M.node(root, Vector3.ZERO, Vector3.ZERO, "Barrel")
	M.mi(barrel, M.cyl(r, r, lb - 0.12, 16), body, Vector3(0, lb * 0.5 + 0.06, 0))
	M.mi(barrel, M.cyl(r * 1.15, r * 1.15, 0.1, 16), body, Vector3(0, lb - 0.05, 0))
	M.mi(barrel, M.cyl(r * 1.1, r * 1.1, 0.08, 16), body, Vector3(0, 0.1, 0))
	M.mi(barrel, M.cyl(r * 0.8, r * 0.8, r * 2.4, 14), pin, Vector3.ZERO, Vector3(90, 0, 0))
	var rod := M.node(root, Vector3(0, lb, 0), Vector3.ZERO, "Rod")
	M.mi(rod, M.cyl(rr, rr, lr, 14), chrome, Vector3(0, -lr * 0.5, 0))
	M.mi(rod, M.cyl(rr * 1.5, rr * 1.5, rr * 2.6, 12), pin, Vector3.ZERO, Vector3(90, 0, 0))
	M.mi(rod, M.box(Vector3(rr * 2.2, rr * 2.4, rr * 1.8), 0.01), pin, Vector3(0, -rr * 1.2, 0))
	root.set_meta("lb", lb)
	root.set_meta("lr", lr)
	return root


static func set_ram(root: Node3D, a: Vector3, b: Vector3) -> void:
	var d := b - a
	var l := d.length()
	if l < 1e-4:
		return
	var y := d / l
	var x := y.cross(Vector3(0, 0, 1))
	if x.length() < 1e-4:
		x = Vector3(1, 0, 0)
	x = x.normalized()
	var z := x.cross(y).normalized()
	root.transform = Transform3D(Basis(x, y, z), a)
	var rod: Node3D = root.get_node("Rod")
	rod.position = Vector3(0, l, 0)
	var barrel: Node3D = root.get_node("Barrel")
	var lb: float = root.get_meta("lb")
	# Barrel never pokes past the rod eye when the ram is short
	barrel.scale = Vector3(1, minf(1.0, (l - 0.12) / lb), 1)


# ================================================================ cab

static func cab(parent: Node3D, size: Vector3, frame: Material, glass: Material, dark: Material) -> Node3D:
	## Operator cab, base on y = 0, centred in x/z, front towards +x. Glazing on front, sides, rear.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Cab")
	var h := size.y
	var sx := size.x * 0.5
	var sz := size.z * 0.5
	var post := 0.06
	# Side profile: sloped windscreen at the front
	var prof := PackedVector2Array([Vector2(-sx, 0), Vector2(sx, 0), Vector2(sx, h * 0.42), Vector2(sx - h * 0.16, h * 0.97),
		Vector2(sx - h * 0.2, h), Vector2(-sx + 0.04, h), Vector2(-sx, h * 0.96)])
	# Lower panel (opaque), glass box above, roof
	M.mi(root, M.prism(_clip_poly(prof, h * 0.36), -sz, sz), frame)
	var glass_prof := _band_poly(prof, h * 0.36, h * 0.93)
	M.mi(root, M.prism(glass_prof, -sz + 0.015, sz - 0.015), glass)
	var roof := M.mi(root, M.box(Vector3(size.x - h * 0.18, 0.07, size.z + 0.04), 0.025), frame, Vector3(-h * 0.09, h - 0.02, 0))
	roof.name = "Roof"
	# Pillars (corners) and mid-door post
	for zz in [-sz + post * 0.5, sz - post * 0.5]:
		M.mi(root, M.box(Vector3(post, h * 0.62, post), 0.01), frame, Vector3(-sx + post * 0.5, h * 0.66, zz))
		var fp := M.mi(root, M.box(Vector3(post, h * 0.62, post), 0.01), frame, Vector3(sx - h * 0.08, h * 0.66, zz))
		fp.rotation = Vector3(0, 0, atan2(h * 0.16, h * 0.55))
		M.mi(root, M.box(Vector3(post * 0.8, h * 0.58, post * 0.7), 0.01), frame, Vector3(-sx * 0.15, h * 0.65, zz))
	# Interior silhouette: seat back and console (seen through the glass)
	M.mi(root, M.box(Vector3(0.12, h * 0.45, size.z * 0.42), 0.03), dark, Vector3(-sx * 0.35, h * 0.55, 0))
	M.mi(root, M.box(Vector3(0.3, h * 0.12, size.z * 0.5), 0.03), dark, Vector3(-sx * 0.2, h * 0.36, 0))
	return root


static func _clip_poly(prof: PackedVector2Array, y_max: float) -> PackedVector2Array:
	var clip := PackedVector2Array([Vector2(-100, -1), Vector2(100, -1), Vector2(100, y_max), Vector2(-100, y_max)])
	var res := Geometry2D.intersect_polygons(prof, clip)
	return res[0] if res.size() > 0 else prof


static func _band_poly(prof: PackedVector2Array, y0: float, y1: float) -> PackedVector2Array:
	var clip := PackedVector2Array([Vector2(-100, y0), Vector2(100, y0), Vector2(100, y1), Vector2(-100, y1)])
	var res := Geometry2D.intersect_polygons(prof, clip)
	return res[0] if res.size() > 0 else prof


# ================================================================ exhaust

static func exhaust(parent: Node3D, top: Vector3, n := 3) -> Array:
	## n staggered smoke puffs above an exhaust stack (parent-local `top`); animate with set_exhaust.
	var puffs: Array = []
	for k in n:
		var p := Vfx.puff(parent, Color(0.17, 0.17, 0.17), 300 + k * 7, 4)
		p.position = top
		p.set_meta("top", top)
		puffs.append(p)
	return puffs


static func set_exhaust(puffs: Array, t: float, load: float) -> void:
	## Deterministic loop driven by sim time; denser smoke under load.
	for k in puffs.size():
		var p: Node3D = puffs[k]
		var ph := fposmod(t * 0.85 + float(k) / puffs.size(), 1.0)
		var top: Vector3 = p.get_meta("top")
		p.position = top + Vector3(-0.55 * ph * ph, 1.4 * ph, 0.0)
		Vfx.set_puff(p, ph, 0.22 + 0.25 * load, 0.1 + 0.45 * load)


# ================================================================ misc

static func pin(parent: Node3D, pos: Vector3, r: float, length: float, mat: Material) -> MeshInstance3D:
	## Pin / boss along z.
	return M.mi(parent, M.cyl(r, r, length, 16), mat, pos, Vector3(90, 0, 0))


static func plate(parent: Node3D, poly: PackedVector2Array, z0: float, z1: float, mat: Material) -> MeshInstance3D:
	return M.mi(parent, M.prism(poly, z0, z1), mat)


static func hose(parent: Node3D, pts: PackedVector3Array, r: float, mat: Material) -> MeshInstance3D:
	return M.mi(parent, M.tube(pts, r, 6), mat)
