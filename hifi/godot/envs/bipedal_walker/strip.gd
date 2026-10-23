extends RefCounted
## Terrain helpers of the BipedalWalker family (Bipedal Walker, Parkour Runner):
##   Profile         the Box2D terrain profile y(x) (piecewise linear, as the Box2D edge chain)
##   extend()        profile continued beyond the 0..92.9 m course so no view sees a void
##   field()         heightfield surface over xs x zs from a height Callable h(x, z)
##   wall()          vertical cross-section face at constant z down to `bottom`


class Profile:
	var xs: PackedFloat32Array
	var ys: PackedFloat32Array

	func _init(px: PackedFloat32Array, py: PackedFloat32Array) -> void:
		xs = px
		ys = py

	func h(x: float) -> float:
		if xs.is_empty():
			return 0.0
		if x <= xs[0]:
			return ys[0]
		var i := xs.bsearch(x)
		if i >= xs.size():
			return ys[-1]
		var t := (x - xs[i - 1]) / maxf(xs[i] - xs[i - 1], 1e-6)
		return lerpf(ys[i - 1], ys[i], t)


static func extend(tx: Array, ty: Array, far_hills := 14.0) -> Array:
	## [xs, ys]: the course profile, a gentle low continuation for ~150 m on both ends (seen by the
	## ortho views near the course ends) and rolling hills further out (FPV horizon only: the rear
	## view's sky backdrop sits 150 m ahead of the walker and hides them).
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	var y0: float = float(ty[0])
	var y1: float = float(ty[-1])
	var x0: float = float(tx[0])
	var x1: float = float(tx[-1])
	var left: Array = []
	var d := 2.5
	while d <= 400.0:
		left.append(d)
		d += 2.5 if d < 60.0 else (5.0 if d < 150.0 else 12.0)
	for i in range(left.size() - 1, -1, -1):
		var dd: float = left[i]
		xs.append(x0 - dd)
		ys.append(y0 + _ext(dd, 1.7, far_hills))
	for i in tx.size():
		xs.append(float(tx[i]))
		ys.append(float(ty[i]))
	for dd in left:
		xs.append(x1 + float(dd))
		ys.append(y1 + _ext(float(dd), 2.3, far_hills))
	return [xs, ys]


static func _ext(d: float, ph: float, hills: float) -> float:
	var near := -0.35 * smoothstep(0.0, 30.0, d) + 0.12 * sin(d * 0.21 + ph) * smoothstep(0.0, 8.0, d)
	var far := hills * smoothstep(165.0, 330.0, d) * (0.65 + 0.35 * sin(d * 0.023 + ph))
	return near + far


static func field(xs: PackedFloat32Array, zs: PackedFloat32Array, h: Callable) -> ArrayMesh:
	## Surface grid (xs[i], h(x, z), zs[j]); xs and zs increasing. UVs are world metres (x, z).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nz := zs.size()
	for i in xs.size():
		for j in nz:
			var x := xs[i]
			var z := zs[j]
			st.set_uv(Vector2(x, z))
			st.add_vertex(Vector3(x, float(h.call(x, z)), z))
	for i in xs.size() - 1:
		for j in nz - 1:
			var a := i * nz + j
			var b := a + nz
			# clockwise seen from above (+Y): Godot front faces
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)
	st.generate_normals()
	st.generate_tangents()
	return st.commit()


static func wall(xs: PackedFloat32Array, z: float, h: Callable, bottom, facing: float, top_off := 0.0) -> ArrayMesh:
	## Vertical face at z from `bottom` up to h(x, z) + top_off, normal (0, 0, facing).
	## bottom may be a Callable too (layered cross-sections).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := Vector3(0, 0, facing)
	for i in xs.size() - 1:
		var xa := xs[i]
		var xb := xs[i + 1]
		var ya := float(h.call(xa, z)) + top_off
		var yb := float(h.call(xb, z)) + top_off
		var ba: float = float(bottom.call(xa, z)) if bottom is Callable else float(bottom)
		var bb: float = float(bottom.call(xb, z)) if bottom is Callable else float(bottom)
		M._quad_uv(st, [Vector3(xa, ba, z), Vector3(xb, bb, z), Vector3(xb, yb, z), Vector3(xa, ya, z)], n,
			[Vector2(xa, ba), Vector2(xb, bb), Vector2(xb, yb), Vector2(xa, ya)])
	st.generate_tangents()
	return st.commit()


static func end_cap(x: float, zs: PackedFloat32Array, h: Callable, bottom: float, facing: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := Vector3(facing, 0, 0)
	for j in zs.size() - 1:
		var za := zs[j]
		var zb := zs[j + 1]
		M._quad_uv(st, [Vector3(x, bottom, za), Vector3(x, bottom, zb), Vector3(x, float(h.call(x, zb)), zb), Vector3(x, float(h.call(x, za)), za)], n,
			[Vector2(za, bottom), Vector2(zb, bottom), Vector2(zb, 1.0), Vector2(za, 1.0)])
	return st.commit()


static func span(a: float, b: float, step: float) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var n := maxi(1, int(ceil((b - a) / step)))
	for j in n + 1:
		out.append(lerpf(a, b, float(j) / n))
	return out


# ---------------------------------------------------------------- ground material

static var _ground_shader: Shader


static func ground_material(a: String, b: String, opts := {}) -> ShaderMaterial:
	## Ground mapped on world xz: Poly Haven set `a` sampled twice (scale and a rotated 0.77x copy,
	## blended by noise, which hides the tiling), set `b` mixed in by a macro noise mask, plus a
	## macro brightness variation. opts: scale_a, scale_b, tint_a, tint_b, mix (0..1 share of b),
	## macro (noise scale per metre), macro_amp, normal, rough.
	if _ground_shader == null:
		_ground_shader = Shader.new()
		_ground_shader.code = """
shader_type spatial;
uniform sampler2D alb_a : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D nor_a : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D orm_a : filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D alb_b : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D nor_b : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D orm_b : filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D macro : filter_linear_mipmap, repeat_enable;
uniform bool has_orm_a = false;
uniform bool has_orm_b = false;
uniform float scale_a = 0.25;
uniform float scale_b = 0.4;
uniform vec3 tint_a : source_color = vec3(1.0);
uniform vec3 tint_b : source_color = vec3(1.0);
uniform float mix_b = 0.4;
uniform float macro_scale = 0.02;
uniform float macro_amp = 0.25;
uniform float normal_k = 1.0;
uniform float rough_k = 1.0;
varying vec3 wpos;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 p = wpos.xz;
	float n1 = texture(macro, p * macro_scale).r;
	float n2 = texture(macro, p * macro_scale * 4.3 + vec2(0.37, 0.71)).r;
	float n3 = texture(macro, p * macro_scale * 0.31 + vec2(0.13, 0.5)).r;
	vec2 pa = p * scale_a;
	vec2 pr = mat2(vec2(0.766, 0.643), vec2(-0.643, 0.766)) * p * scale_a * 0.77 + vec2(0.31, 0.17);
	float wr = smoothstep(0.4, 0.6, n2);
	vec3 ca = mix(texture(alb_a, pa).rgb, texture(alb_a, pr).rgb, wr) * tint_a;
	vec3 na = mix(texture(nor_a, pa).rgb, texture(nor_a, pr).rgb, wr);
	vec3 oa = has_orm_a ? mix(texture(orm_a, pa).rgb, texture(orm_a, pr).rgb, wr) : vec3(1.0, 0.85, 0.0);
	vec2 pb = p * scale_b;
	vec3 cb = texture(alb_b, pb).rgb * tint_b;
	vec3 nb = texture(nor_b, pb).rgb;
	vec3 ob = has_orm_b ? texture(orm_b, pb).rgb : vec3(1.0, 0.85, 0.0);
	float wb = smoothstep(0.88 - mix_b, 1.12 - mix_b, n1 * 0.7 + n3 * 0.3);
	ALBEDO = mix(ca, cb, wb) * (1.0 - macro_amp * 0.5 + macro_amp * (0.6 * n3 + 0.4 * n2));
	NORMAL_MAP = mix(na, nb, wb);
	NORMAL_MAP_DEPTH = normal_k;
	vec3 o = mix(oa, ob, wb);
	AO = o.r;
	AO_LIGHT_AFFECT = 0.4;
	ROUGHNESS = clamp(o.g * rough_k, 0.05, 1.0);
	METALLIC = 0.0;
	SPECULAR = 0.35;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _ground_shader
	var tex: Dictionary = Assets.index().get("textures", {})
	for pair in [["a", a], ["b", b]]:
		var maps: Dictionary = tex.get(pair[1], {})
		if maps.has("diff"):
			m.set_shader_parameter("alb_" + pair[0], Assets.image_texture(maps["diff"]))
		else:
			m.set_shader_parameter("alb_" + pair[0], Assets.noise_texture(String(pair[1]).hash() % 1000, 0.05, false))
		if maps.has("nor_gl"):
			m.set_shader_parameter("nor_" + pair[0], Assets.image_texture(maps["nor_gl"]))
		if maps.has("arm"):
			m.set_shader_parameter("orm_" + pair[0], Assets.image_texture(maps["arm"]))
			m.set_shader_parameter("has_orm_" + pair[0], true)
	m.set_shader_parameter("macro", Assets.noise_texture(913, 0.02, false, 256))
	m.set_shader_parameter("scale_a", float(opts.get("scale_a", 0.25)))
	m.set_shader_parameter("scale_b", float(opts.get("scale_b", 0.4)))
	m.set_shader_parameter("tint_a", opts.get("tint_a", Color(1, 1, 1)))
	m.set_shader_parameter("tint_b", opts.get("tint_b", Color(1, 1, 1)))
	m.set_shader_parameter("mix_b", float(opts.get("mix", 0.4)))
	m.set_shader_parameter("macro_scale", float(opts.get("macro", 0.02)))
	m.set_shader_parameter("macro_amp", float(opts.get("macro_amp", 0.25)))
	m.set_shader_parameter("normal_k", float(opts.get("normal", 1.0)))
	m.set_shader_parameter("rough_k", float(opts.get("rough", 1.0)))
	return m
