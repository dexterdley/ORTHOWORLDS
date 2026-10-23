extends RefCounted
## Alpine terrain for the Mountain Goat world (preloaded by world.gd, no class_name).
##   ridge_strip()   the GymBox mountain extruded across z: exact Box2D profile in the lane
##                   (|z| <= LANE), rocky shoulders falling away on both sides (so the side view's
##                   skyline is the lane edge, as in GymBox), cliff cross-sections and end caps
##   backdrop()      distant valleys and snowy ranges around the ridge (in every view whose frame holds them)
##   boulders()      low-poly procedural boulders (MultiMesh; Poly Haven rock scans are too heavy
##                   for software rendering)
##   terrain_material()  one shader for grass / scree / snow / cliff: lane cover from the per-segment
##                   GymBox snow flags, shoulder cover from altitude, slope and macro noise


class RidgeProfile:
	## Terrain height: Box2D profile y(x) inside |z| <= lane; outside it the shoulders drop by a
	## seeded, always-positive amount (the lane edge is the local crest), never below the valley
	## floor: the ridge rises out of a flat valley plain at the GymBox valley height.
	var xs: PackedFloat32Array
	var ys: PackedFloat32Array
	var lane := 1.2
	var floor_y := 0.5
	var n1 := FastNoiseLite.new()
	var n2 := FastNoiseLite.new()
	var n3 := FastNoiseLite.new()

	func _init(px: PackedFloat32Array, py: PackedFloat32Array, lane_hw: float, valley_y: float) -> void:
		xs = px
		ys = py
		lane = lane_hw
		floor_y = valley_y
		n1.seed = 31
		n1.frequency = 0.35
		n2.seed = 77
		n2.frequency = 0.045
		n3.seed = 5
		n3.frequency = 0.13
		n3.fractal_octaves = 3

	func base(x: float) -> float:
		if x <= xs[0]:
			return ys[0]
		var i := xs.bsearch(x)
		if i >= xs.size():
			return ys[-1]
		var t := (x - xs[i - 1]) / maxf(xs[i] - xs[i - 1], 1e-6)
		return lerpf(ys[i - 1], ys[i], t)

	func offset(x: float, z: float) -> float:
		var d := absf(z) - lane
		if d <= 0.0:
			return 0.0
		var ramp := clampf(d / 0.9, 0.0, 1.0)
		var big := 0.75 + 0.5 * (0.5 + 0.5 * n2.get_noise_2d(x, z))
		var small := 0.5 + 0.5 * n1.get_noise_2d(x, z)
		var gully := minf(d, 8.0) / 8.0 * 1.3 * (0.5 + 0.5 * n3.get_noise_2d(x, z))
		var h := 0.3 * pow(d, 1.25) * big + ramp * (0.5 * small + 0.12) + gully
		# smooth min with the depth down to the valley floor
		var lim := maxf(base(x) - floor_y, 0.0)
		var k := 0.6
		var t := clampf(0.5 + 0.5 * (lim - h) / k, 0.0, 1.0)
		return -maxf(lerpf(lim, h, t) - k * t * (1.0 - t), 0.0)

	func height(x: float, z: float) -> float:
		return base(x) + offset(x, z)


static func _zs(z0: float, z1: float, lane: float) -> PackedFloat32Array:
	## Fine columns near the lane, coarse towards the edges; +-lane exactly on a column.
	var out: Array = []
	var z := -lane
	while z <= lane + 1e-4:
		out.append(z)
		z += lane / 3.0
	for side in [-1.0, 1.0]:
		var lim: float = z0 if side < 0.0 else z1
		var d := 0.0
		while true:
			d += 0.3 if d < 2.0 else (0.6 if d < 6.0 else 1.0)
			var zz: float = side * (lane + d)
			if (side < 0.0 and zz <= lim) or (side > 0.0 and zz >= lim):
				out.append(lim)
				break
			out.append(zz)
	out.sort()
	return PackedFloat32Array(out)


static func _grid(st: SurfaceTool, nx: int, nz: int) -> void:
	for i in nx - 1:
		for j in nz - 1:
			var a := i * nz + j
			var b := a + nz
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)


static func ridge_strip(xs: PackedFloat32Array, ys: PackedFloat32Array, z0: float, z1: float, bottom: float,
		lane: float, scene: Dictionary) -> Node3D:
	var root := Node3D.new()
	root.name = "Ridge"
	var prof := RidgeProfile.new(xs, ys, lane, float(scene.get("valley", [0.0, 0.0, 0.5])[2]))
	root.set_meta("profile", prof)
	var zs := _zs(z0, z1, lane)
	var nz := zs.size()
	var nx := xs.size()
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var front := PackedFloat32Array()
	var back := PackedFloat32Array()
	var west := PackedFloat32Array()
	var east := PackedFloat32Array()
	for i in nx:
		for j in nz:
			var y := prof.height(xs[i], zs[j])
			if j == 0:
				back.append(y)
			if j == nz - 1:
				front.append(y)
			if i == 0:
				west.append(y)
			if i == nx - 1:
				east.append(y)
			st.set_uv(Vector2(xs[i], zs[j]))
			st.add_vertex(Vector3(xs[i], y, zs[j]))
	_grid(st, nx, nz)
	st.generate_normals()
	st.generate_tangents()
	var top := MeshInstance3D.new()
	top.name = "Top"
	top.mesh = st.commit()
	top.material_override = terrain_material(scene, lane, 9.5, 7.5)
	root.add_child(top)
	# Cross-sections (front z1, back z0) and end caps (west, east): cliff rock
	var sec := SurfaceTool.new()
	sec.begin(Mesh.PRIMITIVE_TRIANGLES)
	for face in [[z1, 1.0, front], [z0, -1.0, back]]:
		var zf: float = face[0]
		var hy: PackedFloat32Array = face[2]
		for i in nx - 1:
			M._quad_uv(sec, [Vector3(xs[i], bottom, zf), Vector3(xs[i + 1], bottom, zf), Vector3(xs[i + 1], hy[i + 1], zf),
				Vector3(xs[i], hy[i], zf)], Vector3(0, 0, face[1]),
				[Vector2(xs[i], bottom), Vector2(xs[i + 1], bottom), Vector2(xs[i + 1], hy[i + 1]), Vector2(xs[i], hy[i])])
	for end in [[0, -1.0, west], [nx - 1, 1.0, east]]:
		var xe: float = xs[end[0]]
		var hz: PackedFloat32Array = end[2]
		for j in nz - 1:
			M._quad_uv(sec, [Vector3(xe, bottom, zs[j]), Vector3(xe, bottom, zs[j + 1]), Vector3(xe, hz[j + 1], zs[j + 1]),
				Vector3(xe, hz[j], zs[j])], Vector3(end[1], 0, 0),
				[Vector2(zs[j], bottom), Vector2(zs[j + 1], bottom), Vector2(zs[j + 1], hz[j + 1]), Vector2(zs[j], hz[j])])
	var side := MeshInstance3D.new()
	side.name = "Section"
	side.mesh = sec.commit()
	side.material_override = cliff_material()
	root.add_child(side)
	return root


static func cliff_material() -> BaseMaterial3D:
	return Assets.pbr("cliff_side", {"uv": 0.16, "tint": Color(0.6, 0.55, 0.52), "color": Color(0.36, 0.3, 0.26)})


static func backdrop(z0: float, z1: float, xw: float, xe: float) -> Node3D:
	## Valleys around the ridge rising to snowy ranges 150-400 m away, with pine forests below the
	## tree line. Near the ridge it stays just below the GymBox valley floor, so the ridge rises from it.
	var root := Node3D.new()
	root.name = "Backdrop"
	var nb := FastNoiseLite.new()
	nb.seed = 4242
	nb.frequency = 0.0045
	nb.fractal_octaves = 5
	var nr := FastNoiseLite.new()
	nr.seed = 4243
	nr.frequency = 0.007
	nr.fractal_type = FastNoiseLite.FRACTAL_RIDGED
	nr.fractal_octaves = 3
	var nf := FastNoiseLite.new()
	nf.seed = 99
	nf.frequency = 0.02
	var noises := [nb, nr, nf]
	var step := 10.0
	var gx0 := -500.0
	var gx1 := 380.0
	var gz0 := -480.0
	var gz1 := 480.0
	var nx := int((gx1 - gx0) / step) + 1
	var nz := int((gz1 - gz0) / step) + 1
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in nx:
		for j in nz:
			var x := gx0 + i * step
			var z := gz0 + j * step
			st.set_uv(Vector2(x, z))
			st.add_vertex(Vector3(x, _bd_height(x, z, z0, z1, xw, xe, noises), z))
	_grid(st, nx, nz)
	st.generate_normals()
	st.generate_tangents()
	var mi := MeshInstance3D.new()
	mi.name = "Ranges"
	mi.mesh = st.commit()
	var m := terrain_material({}, -1000.0, 70.0, 6.0)
	m.set_shader_parameter("uv_grass", 0.05)
	m.set_shader_parameter("uv_rock", 0.02)
	m.set_shader_parameter("uv_snow", 0.04)
	m.set_shader_parameter("uv_cliff", 0.02)
	m.set_shader_parameter("macro_scale", 0.15)
	m.set_shader_parameter("rock_tint", Vector3(0.62, 0.57, 0.52))
	mi.material_override = m
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	root.add_child(mi)
	# Pine forests on the valley floors and lower slopes
	var rng := RandomNumberGenerator.new()
	rng.seed = 2024
	var xforms: Array = []
	var tries := 0
	while xforms.size() < 300 and tries < 6000:
		tries += 1
		var x := rng.randf_range(-320.0, 280.0)
		var z := rng.randf_range(-320.0, 320.0)
		var d := _dist_out(x, z, z0, z1, xw, xe)
		if d < 28.0 or d > 190.0:
			continue
		var h := _bd_height(x, z, z0, z1, xw, xe, noises)
		if h > 12.0 or nf.get_noise_2d(x * 1.7, z * 1.7) < -0.1:
			continue
		var s := rng.randf_range(0.8, 1.4)
		xforms.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s * rng.randf_range(0.95, 1.3), s)),
			Vector3(x, h - 0.3, z)))
	var trees := Terrain.tree_multimesh(xforms, "pine", 7)
	for c in trees.get_children():
		if c is GeometryInstance3D:
			c.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	root.add_child(trees)
	return root


static func _dist_out(x: float, z: float, z0: float, z1: float, xw: float, xe: float) -> float:
	var dz := maxf(0.0, maxf(z0 - z, z - z1))
	var dx := maxf(0.0, maxf(xw - x, x - xe))
	return sqrt(dz * dz + dx * dx)


static func _bd_height(x: float, z: float, z0: float, z1: float, xw: float, xe: float, noises: Array) -> float:
	var d := _dist_out(x, z, z0, z1, xw, xe)
	var rise := smoothstep(30.0, 260.0, d)
	var body: float = 0.5 + 0.5 * noises[0].get_noise_2d(x, z)
	var ridge: float = 0.5 + 0.5 * noises[1].get_noise_2d(x, z)
	# The western range (ahead of the climb) is the tallest
	var west := 1.0 + 0.6 * smoothstep(0.0, -260.0, x)
	var fine: float = noises[2].get_noise_2d(x, z)
	return -0.4 + 0.6 * fine + rise * west * (25.0 + 90.0 * body + 70.0 * ridge * rise)


static func boulders(prof, z_back: float, z_front: float, x0: float, x1: float, seed: int) -> Node3D:
	## Boulders on the shoulders: never in the lane, never tall enough in front (z > 0) to hide the
	## goat in the side view, never between the goat and the rear / FPV cameras' lines of sight.
	var root := Node3D.new()
	root.name = "Boulders"
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var variants := 4
	var xfs: Array = []
	for v in variants:
		xfs.append([])
	for i in 140:
		var x := rng.randf_range(x0, x1)
		var back := i % 3 != 0
		var z := rng.randf_range(z_back + 1.0, -2.4) if back else rng.randf_range(2.8, minf(z_front - 0.8, 4.8))
		var s := rng.randf_range(0.3, 1.5) if back else rng.randf_range(0.25, 0.7)
		if rng.randf() < 0.55:
			s *= 0.45
		var y: float = prof.height(x, z)
		if not back and y + s * 0.75 > prof.base(x) - 0.35:
			continue
		var b := Basis(Vector3.UP, rng.randf() * TAU) * Basis(Vector3.RIGHT, rng.randf_range(-0.25, 0.25))
		b = b.scaled(Vector3(s * rng.randf_range(0.9, 1.4), s * rng.randf_range(0.55, 0.9), s))
		xfs[i % variants].append(Transform3D(b, Vector3(x, y - 0.12 * s, z)))
	var mat := Assets.pbr("rock_boulder_dry", {"uv": 1.1, "tint": Color(0.6, 0.57, 0.53), "color": Color(0.45, 0.43, 0.4)})
	for v in variants:
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = rock_mesh(seed * 7 + v)
		mm.instance_count = xfs[v].size()
		for k in xfs[v].size():
			mm.set_instance_transform(k, xfs[v][k])
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.material_override = mat
		root.add_child(mmi)
	return root


static func rock_mesh(seed: int) -> ArrayMesh:
	## Noise-displaced icosphere (radius ~0.5, flattened base), smooth-shaded.
	var t := (1.0 + sqrt(5.0)) / 2.0
	var verts: Array = [Vector3(-1, t, 0), Vector3(1, t, 0), Vector3(-1, -t, 0), Vector3(1, -t, 0),
		Vector3(0, -1, t), Vector3(0, 1, t), Vector3(0, -1, -t), Vector3(0, 1, -t),
		Vector3(t, 0, -1), Vector3(t, 0, 1), Vector3(-t, 0, -1), Vector3(-t, 0, 1)]
	var faces: Array = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
		[10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
		[8, 6, 7], [9, 8, 1]]
	for k in verts.size():
		verts[k] = verts[k].normalized()
	for it in 3:
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
	var n := FastNoiseLite.new()
	n.seed = seed
	n.frequency = 0.9
	n.fractal_octaves = 3
	var cell := FastNoiseLite.new()
	cell.seed = seed + 1
	cell.noise_type = FastNoiseLite.TYPE_CELLULAR
	cell.frequency = 1.1
	cell.cellular_return_type = FastNoiseLite.RETURN_DISTANCE2_SUB
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var pos: Array = []
	for v in verts:
		var r := 0.5 * (1.0 + 0.32 * n.get_noise_3dv(v * 1.3) + 0.22 * cell.get_noise_3dv(v * 1.6))
		var p: Vector3 = v * r
		if p.y < -0.18:
			p.y = -0.18 + (p.y + 0.18) * 0.25
		pos.append(p)
	for p in pos:
		st.set_uv(Vector2(p.x + p.z, p.y))
		st.add_vertex(p)
	for f in faces:
		# outward winding for Godot (clockwise seen from outside)
		var a: Vector3 = pos[f[0]]
		var b: Vector3 = pos[f[1]]
		var c: Vector3 = pos[f[2]]
		if (b - a).cross(c - a).dot(a + b + c) > 0.0:
			st.add_index(f[0])
			st.add_index(f[2])
			st.add_index(f[1])
		else:
			st.add_index(f[0])
			st.add_index(f[1])
			st.add_index(f[2])
	st.generate_normals()
	return st.commit()


static func _tex(name: String, kind: String) -> Texture2D:
	var maps: Dictionary = Assets.index().get("textures", {}).get(name, {})
	return Assets.image_texture(maps[kind]) if maps.has(kind) else null


static var _terrain_shader: Shader
static var _cloud_shader: Shader


static func terrain_material(scene: Dictionary, lane: float, snow_line: float, grass_line: float) -> ShaderMaterial:
	if _terrain_shader == null:
		_terrain_shader = Shader.new()
		_terrain_shader.code = TERRAIN_SHADER
	var m := ShaderMaterial.new()
	m.shader = _terrain_shader
	for ts in [["grass", "rocky_terrain_02"], ["rock", "aerial_rocks_02"], ["snow", "snow_02"], ["cliff", "cliff_side"]]:
		m.set_shader_parameter(ts[0] + "_d", _tex(ts[1], "diff"))
		m.set_shader_parameter(ts[0] + "_n", _tex(ts[1], "nor_gl"))
		m.set_shader_parameter(ts[0] + "_a", _tex(ts[1], "arm"))
	m.set_shader_parameter("macro_tex", Assets.noise_texture(611, 0.012, false, 256))
	# Lane cover per GymBox segment: snow where the segment's mean height is above the snow line
	var snow: Array = scene.get("snow", [])
	var n := maxi(snow.size(), 1)
	var img := Image.create(n, 1, false, Image.FORMAT_R8)
	for i in snow.size():
		img.set_pixel(i, 0, Color(float(snow[i][2]), 0, 0))
	m.set_shader_parameter("flags_tex", ImageTexture.create_from_image(img))
	m.set_shader_parameter("flags_x0", float(snow[0][0]) if snow.size() > 0 else 20.0)
	m.set_shader_parameter("flags_x1", float(snow[-1][1]) if snow.size() > 0 else 65.0)
	m.set_shader_parameter("flags_seg", float(scene.get("seg", 1.25)))
	m.set_shader_parameter("flags_n", n)
	m.set_shader_parameter("lane", lane)
	m.set_shader_parameter("snow_line", snow_line)
	m.set_shader_parameter("grass_line", grass_line)
	return m


static func cloud_material() -> ShaderMaterial:
	if _cloud_shader == null:
		_cloud_shader = Shader.new()
		_cloud_shader.code = CLOUD_SHADER
	var m := ShaderMaterial.new()
	m.shader = _cloud_shader
	m.set_shader_parameter("noise_tex", Assets.noise_texture(777, 0.03, false, 256))
	return m


const TERRAIN_SHADER := """
shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

uniform sampler2D grass_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D grass_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rock_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rock_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D rock_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D snow_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D snow_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D snow_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D cliff_d : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D cliff_n : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D cliff_a : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D macro_tex : filter_linear_mipmap, repeat_enable;
uniform sampler2D flags_tex : filter_nearest;
uniform float flags_x0 = 20.0;
uniform float flags_x1 = 65.0;
uniform float flags_seg = 1.25;
uniform int flags_n = 36;
uniform float lane = 1.2;
uniform float snow_line = 9.5;
uniform float grass_line = 7.5;
uniform float uv_grass = 0.33;
uniform float uv_rock = 0.11;
uniform float uv_snow = 0.3;
uniform float uv_cliff = 0.16;
uniform float macro_scale = 1.0;
uniform vec3 grass_tint = vec3(0.92, 1.04, 0.82);
uniform vec3 rock_tint = vec3(0.8, 0.72, 0.63);
uniform float rock_sat = 0.3;
uniform vec3 snow_tint = vec3(0.98, 0.99, 1.0);
uniform vec3 cliff_tint = vec3(0.7, 0.64, 0.59);
uniform float cliff_sat = 0.5;

varying vec3 wp;
varying vec3 wn;

void vertex() {
	wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	wn = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
}

float lane_snow(float x) {
	if (x < flags_x0) return 1.0;
	int i = clamp(int(floor((x - flags_x0) / flags_seg)), 0, flags_n - 1);
	return texelFetch(flags_tex, ivec2(i, 0), 0).r;
}

void fragment() {
	vec2 p = UV;
	float m1 = texture(macro_tex, p * 0.021 * macro_scale).r;
	float m2 = texture(macro_tex, p * 0.093 * macro_scale + vec2(0.37, 0.11)).r;
	// geometric face normal: smoothed vertex normals understate short steep banks
	vec3 fn = normalize(cross(dFdx(wp), dFdy(wp)));
	float slope = 1.0 - min(abs(fn.y), clamp(wn.y, 0.0, 1.0));
	float d = abs(wp.z) - lane;
	float in_lane = (wp.x <= flags_x1 + 0.02) ? 1.0 - smoothstep(-0.05, 0.08, d + (m2 - 0.5) * 0.12) : 0.0;
	float ls = lane_snow(wp.x);
	float alt = wp.y + (m1 - 0.5) * 3.0;
	float sn = smoothstep(snow_line - 1.0, snow_line + 1.2, alt + (m2 - 0.5) * 2.0) * (1.0 - smoothstep(0.42, 0.75, slope));
	float gr = (1.0 - smoothstep(grass_line - 2.5, grass_line + 1.0, alt)) * (1.0 - smoothstep(0.2, 0.4, slope));
	gr *= smoothstep(0.48, 0.64, m2 + 0.22 * (1.0 - smoothstep(0.0, 3.0, d))) * (1.0 - sn);
	// cliff rock on steep faces (the lane itself is never steeper than ~30 deg except the meadow edge)
	float cl = smoothstep(0.32, 0.55, slope) * (1.0 - in_lane) + smoothstep(0.45, 0.6, slope) * in_lane;
	float w_snow = mix(sn, ls, in_lane) * (1.0 - cl);
	float w_grass = mix(gr, 1.0 - ls, in_lane) * (1.0 - cl);
	float w_cliff = cl;
	float w_rock = max(1.0 - w_snow - w_grass - w_cliff, 0.0);
	vec2 pg = p * uv_grass;
	vec2 pr = p * uv_rock;
	vec2 ps = p * uv_snow;
	vec2 pc = (abs(fn.x) > abs(fn.z) ? wp.zy : wp.xy) * uv_cliff;
	vec3 rc = texture(rock_d, pr).rgb;
	rc = mix(vec3(dot(rc, vec3(0.333))), rc, rock_sat) * rock_tint;
	vec3 alb = texture(grass_d, pg).rgb * grass_tint * w_grass + rc * w_rock
		+ texture(snow_d, ps).rgb * snow_tint * w_snow
		+ mix(vec3(dot(texture(cliff_d, pc).rgb, vec3(0.333))), texture(cliff_d, pc).rgb, cliff_sat) * cliff_tint * w_cliff;
	alb *= mix(0.84, 1.1, m1) * mix(0.94, 1.04, m2);
	vec3 nrm = texture(grass_n, pg).rgb * w_grass + texture(rock_n, pr).rgb * w_rock + texture(snow_n, ps).rgb * w_snow
		+ vec3(0.5, 0.5, 1.0) * w_cliff;
	vec3 arm = texture(grass_a, pg).rgb * w_grass + texture(rock_a, pr).rgb * w_rock + texture(snow_a, ps).rgb * w_snow
		+ texture(cliff_a, pc).rgb * w_cliff;
	ALBEDO = alb;
	NORMAL_MAP = nrm;
	NORMAL_MAP_DEPTH = 1.0;
	AO = arm.r;
	AO_LIGHT_AFFECT = 0.5;
	ROUGHNESS = clamp(arm.g, 0.35, 1.0);
	METALLIC = 0.0;
	SPECULAR = 0.35;
}
"""

const CLOUD_SHADER := """
shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_back, diffuse_lambert_wrap, specular_disabled, shadows_disabled;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
varying vec3 wp;
void vertex() {
	wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	float f = clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	float n = texture(noise_tex, wp.xy * 0.21 + wp.zy * 0.13).r;
	float n2 = texture(noise_tex, wp.xz * 0.55 + wp.yx * 0.41).r;
	vec3 wnrm = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	ALBEDO = mix(vec3(0.72, 0.75, 0.82), vec3(1.0), smoothstep(-0.6, 0.5, wnrm.y));
	float a = f * 1.15 + (n - 0.5) * 1.2 + (n2 - 0.5) * 0.7 - 0.45;
	ALPHA = clamp(a * 1.6, 0.0, 1.0) * 0.8;
	BACKLIGHT = vec3(0.45);
}
"""
