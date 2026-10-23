extends RefCounted
## Procedural broadleaf trees for the Mario Escape world (preloaded by world.gd, no class_name).
## Poly Haven tree scans are millions of triangles, too heavy for software rendering, so each tree
## is a bark-textured trunk with tapered limbs and a crown of noise-displaced leaf clumps: an
## opaque dark core plus an alpha-scissored outer shell that breaks the silhouette into leaves.
## Three seeded variants, instanced with MultiMesh.

static var _leaf_shader: Shader


static func forest(xforms: Array, seed: int) -> Node3D:
	var root := Node3D.new()
	root.name = "Trees"
	var bark := Assets.pbr("bark_brown_02", {"uv": 1.4, "tint": Color(0.8, 0.74, 0.68), "color": Color(0.3, 0.22, 0.16)})
	var core := leaf_material(false)
	var shell := leaf_material(true)
	var nv := 3
	for v in nv:
		var parts := _variant(seed * 13 + v)
		var mine: Array = []
		for i in xforms.size():
			if i % nv == v:
				mine.append(xforms[i])
		if mine.is_empty():
			continue
		for pm in [[parts[0], bark], [parts[1], core], [parts[2], shell]]:
			var mm := MultiMesh.new()
			mm.transform_format = MultiMesh.TRANSFORM_3D
			mm.mesh = pm[0]
			mm.instance_count = mine.size()
			for k in mine.size():
				mm.set_instance_transform(k, mine[k])
			var mmi := MultiMeshInstance3D.new()
			mmi.multimesh = mm
			mmi.material_override = pm[1]
			root.add_child(mmi)
	return root


static func leaf_material(cut: bool) -> ShaderMaterial:
	if _leaf_shader == null:
		_leaf_shader = Shader.new()
		_leaf_shader.code = LEAF_SHADER
	var m := ShaderMaterial.new()
	m.shader = _leaf_shader
	m.set_shader_parameter("noise_tex", Assets.noise_texture(909, 0.05, false, 256))
	m.set_shader_parameter("cut", 0.47 if cut else 0.0)
	m.set_shader_parameter("shade", 1.0 if cut else 0.62)
	return m


static func _variant(seed: int) -> Array:
	## [trunk+limbs mesh, crown core mesh, crown shell mesh], base at y = 0, ~7-9 m tall.
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var trunk_h := rng.randf_range(2.6, 3.4)
	var crown_c := Vector3(0, trunk_h + rng.randf_range(2.0, 2.6), 0)
	var crown_r := Vector3(rng.randf_range(2.4, 3.0), rng.randf_range(1.9, 2.4), rng.randf_range(2.4, 3.0))
	var wood := SurfaceTool.new()
	wood.begin(Mesh.PRIMITIVE_TRIANGLES)
	var lean := Vector3(rng.randf_range(-0.25, 0.25), 0, rng.randf_range(-0.25, 0.25))
	_tube(wood, PackedVector3Array([Vector3(0, -0.2, 0), Vector3(0, trunk_h * 0.5, 0) + lean * 0.4, Vector3(0, trunk_h, 0) + lean]),
		0.34, 0.22, 10)
	var top := Vector3(0, trunk_h, 0) + lean
	var core := SurfaceTool.new()
	core.begin(Mesh.PRIMITIVE_TRIANGLES)
	var shell := SurfaceTool.new()
	shell.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nb := 5
	for b in nb:
		var a := TAU * (b + rng.randf_range(-0.2, 0.2)) / nb
		var up := rng.randf_range(0.5, 0.95)
		var tip := crown_c + Vector3(cos(a) * crown_r.x * 0.62, (up - 0.55) * crown_r.y, sin(a) * crown_r.z * 0.62)
		var mid := top.lerp(tip, 0.5) + Vector3(0, 0.35, 0)
		_tube(wood, PackedVector3Array([top, mid, tip]), 0.17, 0.05, 7)
	# leaf clumps: around the crown ellipsoid surface and its centre, bigger at the top
	var n := FastNoiseLite.new()
	n.seed = seed
	n.frequency = 1.4
	var clumps := 15
	for c in clumps:
		var dir := Vector3(rng.randf_range(-1, 1), rng.randf_range(-0.55, 1.0), rng.randf_range(-1, 1)).normalized()
		var pos := crown_c + dir * crown_r * rng.randf_range(0.45, 0.75)
		var rad := rng.randf_range(1.0, 1.45) * (1.0 + 0.15 * dir.y)
		_blob(core, pos, rad * 0.84, n, 1, c)
		_blob(shell, pos, rad, n, 2, c)
	_blob(core, crown_c, crown_r.y * 0.95, n, 1, 99)
	wood.generate_normals()
	core.generate_normals()
	shell.generate_normals()
	return [wood.commit(), core.commit(), shell.commit()]


static func _blob(st: SurfaceTool, c: Vector3, r: float, n: FastNoiseLite, subdiv: int, salt: int) -> void:
	var t := (1.0 + sqrt(5.0)) / 2.0
	var verts: Array = [Vector3(-1, t, 0), Vector3(1, t, 0), Vector3(-1, -t, 0), Vector3(1, -t, 0),
		Vector3(0, -1, t), Vector3(0, 1, t), Vector3(0, -1, -t), Vector3(0, 1, -t),
		Vector3(t, 0, -1), Vector3(t, 0, 1), Vector3(-t, 0, -1), Vector3(-t, 0, 1)]
	var faces: Array = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
		[10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
		[8, 6, 7], [9, 8, 1]]
	for k in verts.size():
		verts[k] = verts[k].normalized()
	for it in subdiv:
		var cache := {}
		var nf: Array = []
		for f in faces:
			var m: Array = []
			for e in [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]]:
				var key := "%d_%d" % [mini(e[0], e[1]), maxi(e[0], e[1])]
				if not cache.has(key):
					verts.append(((verts[e[0]] + verts[e[1]]) * 0.5).normalized())
					cache[key] = verts.size() - 1
				m.append(cache[key])
			nf.append([f[0], m[0], m[2]])
			nf.append([f[1], m[1], m[0]])
			nf.append([f[2], m[2], m[1]])
			nf.append([m[0], m[1], m[2]])
		faces = nf
	var pts: Array = []
	for v in verts:
		var d: Vector3 = v
		var rr := r * (1.0 + 0.22 * n.get_noise_3d(d.x + salt * 3.1, d.y, d.z))
		var p := c + d * rr
		p.y = maxf(p.y, c.y - r * 0.55)          # flatter clump bottoms
		pts.append(p)
	for f in faces:
		var a: Vector3 = pts[f[0]]
		var b: Vector3 = pts[f[1]]
		var cc: Vector3 = pts[f[2]]
		var out := (a + b + cc) / 3.0 - c
		M._tri(st, a, b, cc, out.normalized())


static func _tube(st: SurfaceTool, points: PackedVector3Array, r0: float, r1: float, segs: int) -> void:
	var n := points.size()
	var rings: Array = []
	var prev_n := Vector3(1, 0, 0)
	for i in n:
		var tgt := (points[mini(i + 1, n - 1)] - points[maxi(i - 1, 0)]).normalized()
		var nx := (prev_n - tgt * prev_n.dot(tgt))
		if nx.length() < 1e-3:
			nx = tgt.cross(Vector3.FORWARD)
		nx = nx.normalized()
		prev_n = nx
		var bx := tgt.cross(nx)
		var rad := lerpf(r0, r1, float(i) / (n - 1))
		var ring: Array = []
		for s in segs:
			var a := TAU * s / segs
			var d := nx * cos(a) + bx * sin(a)
			ring.append([points[i] + d * rad, d])
		rings.append(ring)
	for i in n - 1:
		for s in segs:
			var s1 := (s + 1) % segs
			var a: Array = rings[i][s]
			var b: Array = rings[i][s1]
			var c: Array = rings[i + 1][s1]
			var d: Array = rings[i + 1][s]
			var fn: Vector3 = (a[1] + b[1] + c[1] + d[1]).normalized()
			M._tri(st, a[0], b[0], c[0], fn)
			M._tri(st, a[0], c[0], d[0], fn)


const LEAF_SHADER := """
shader_type spatial;
render_mode diffuse_lambert_wrap, specular_schlick_ggx, cull_disabled;
uniform sampler2D noise_tex : filter_linear_mipmap, repeat_enable;
uniform float cut = 0.47;
uniform float shade = 1.0;
uniform vec3 leaf_dark = vec3(0.01, 0.03, 0.005);     // linear albedo
uniform vec3 leaf_light = vec3(0.05, 0.12, 0.016);
varying vec3 lp;
varying vec3 ln;
void vertex() {
	lp = VERTEX;
	ln = NORMAL;
}
float tri(vec3 p, vec3 w, float s) {
	return texture(noise_tex, p.zy * s).r * w.x + texture(noise_tex, p.xz * s).r * w.y + texture(noise_tex, p.xy * s).r * w.z;
}
void fragment() {
	vec3 w = pow(abs(ln), vec3(4.0));
	w /= dot(w, vec3(1.0));
	float n1 = tri(lp, w, 0.9);
	float n2 = tri(lp, w, 4.3);
	float v = n1 * 0.45 + n2 * 0.55;
	float height = clamp((lp.y - 3.0) / 5.0, 0.0, 1.0);
	ALBEDO = mix(leaf_dark, leaf_light, smoothstep(0.35, 0.75, v)) * shade * (0.75 + 0.35 * height);
	ALPHA = cut > 0.0 ? v + 0.08 : 1.0;
	ALPHA_SCISSOR_THRESHOLD = max(cut, 0.01);
	NORMAL = normalize(NORMAL + (vec3(n2, n1, 1.0 - n2) - 0.5) * 0.9);
	ROUGHNESS = 0.9;
	SPECULAR = 0.12;
	BACKLIGHT = vec3(0.05, 0.09, 0.01);
}
"""
