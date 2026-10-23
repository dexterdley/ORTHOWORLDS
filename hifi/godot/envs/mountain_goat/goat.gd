extends RefCounted
## White mountain goat fitted to the GymBox Box2D bodies (preloaded by world.gd, no class_name).
## Every part is built in its body-local frame (Box2D local (lx, ly) -> Godot (lx, ly, z)); the goat
## faces -X (west).
##   torso  box 1.2 x 0.7: barrel, shoulder hump, chest, rump, belly fur, neck, tail
##   head   circle r 0.22 at torso (-0.7, +0.5): cranium, snout, nose, eyes, ears, beard
##   horn0/1  Box2D triangle (0,0) (-0.15,0.3) (0.05,0.255) on top of the head: tapered keratin horns
##   leg0..7  thigh 0.16 x 0.45 / shin 0.14 x 0.45 boxes: fur upper leg, slim cannon, black cloven hoof
##            (near legs at +z, far legs at -z, under the body)

const LEG_Z := 0.2
const SHELLS := 4          # fur shells (alpha-scissored hanging strands) over the long-haired parts
const SHELL_STEP := 0.009

static var _ell_cache := {}
static var _shell_shader: Shader
static var _shell_mats: Array = []


static func fur_material(uv: float, stretch: float, normal: float, tint: Color) -> BaseMaterial3D:
	## Off-white curly fleece, stretched vertically on the flanks so the curls read as hanging hair.
	var m: BaseMaterial3D = Assets.pbr("curly_teddy_natural", {"uv": uv, "tint": tint, "color": tint, "normal": normal}).duplicate()
	m.uv1_scale = Vector3(uv, uv * stretch, uv)
	m.roughness = 1.0
	m.rim_enabled = true
	m.rim = 0.35
	m.rim_tint = 0.6
	return m


static func build(w, parent: Node3D) -> Node3D:
	var fur := fur_material(1.6, 0.42, 1.8, Color(0.8, 0.8, 0.78))
	var fur_fine := fur_material(3.2, 0.7, 1.2, Color(0.8, 0.8, 0.79))
	var horn := M.mat(Color(0.075, 0.068, 0.062), 0.42, 0.0, {"clearcoat": 0.35, "normal_noise": 0.6, "noise_scale": 6.0})
	var hoof := M.mat(Color(0.055, 0.05, 0.048), 0.5)
	var nose := M.mat(Color(0.05, 0.045, 0.045), 0.32)
	var eye := M.mat(Color(0.03, 0.02, 0.012), 0.06, 0.0, {"clearcoat": 1.0, "clearcoat_roughness": 0.02})

	# ---------------------------------------------------------------- torso (box 1.2 x 0.7, faces -X)
	var torso := M.node(parent, Vector3.ZERO, Vector3.ZERO, "GoatTorso")
	_ef(torso, Vector3(0.58, 0.28, 0.27), Vector3(0.04, 0.0, 0), -3.0, fur)          # barrel
	_ef(torso, Vector3(0.36, 0.35, 0.29), Vector3(-0.32, 0.06, 0), -10.0, fur)       # shoulder hump
	_ef(torso, Vector3(0.28, 0.32, 0.26), Vector3(-0.46, -0.06, 0), 0.0, fur)        # deep chest
	_ef(torso, Vector3(0.31, 0.29, 0.27), Vector3(0.37, 0.02, 0), 0.0, fur)          # rump
	_ef(torso, Vector3(0.5, 0.17, 0.255), Vector3(-0.05, -0.21, 0), 0.0, fur)        # shaggy belly skirt
	_ef(torso, Vector3(0.17, 0.27, 0.16), Vector3(-0.6, 0.27, 0), 38.0, fur)         # thick neck
	_ef(torso, Vector3(0.2, 0.2, 0.19), Vector3(-0.64, 0.05, 0), 15.0, fur)          # brisket
	_e(torso, Vector3(0.05, 0.1, 0.055), Vector3(0.63, 0.15, 0), -35.0, fur_fine)   # tail
	w.register("torso", torso, "goat")

	# ---------------------------------------------------------------- head (circle r 0.22)
	var head := M.node(parent, Vector3.ZERO, Vector3.ZERO, "GoatHead")
	_e(head, Vector3(0.16, 0.145, 0.125), Vector3(0.03, 0.06, 0), 0.0, fur_fine)    # cranium
	_e(head, Vector3(0.21, 0.092, 0.088), Vector3(-0.12, -0.04, 0), 33.0, fur_fine)  # long face, forward-down
	_e(head, Vector3(0.075, 0.068, 0.07), Vector3(-0.265, -0.13, 0), 33.0, fur_fine)  # muzzle
	_e(head, Vector3(0.1, 0.1, 0.1), Vector3(0.08, -0.03, 0), 0.0, fur_fine)        # cheeks / jaw
	_e(head, Vector3(0.035, 0.03, 0.045), Vector3(-0.325, -0.16, 0), 33.0, nose)    # nose pad
	_ef(head, Vector3(0.06, 0.14, 0.05), Vector3(-0.14, -0.21, 0), 20.0, fur)        # beard
	for sz in [-1.0, 1.0]:
		M.mi(head, M.sphere(0.02, 12, 6), eye, Vector3(-0.07, 0.055, sz * 0.098))
		var ear := _e(head, Vector3(0.035, 0.095, 0.016), Vector3(0.09, 0.14, sz * 0.095), 0.0, fur_fine)
		ear.rotation_degrees = Vector3(sz * -48.0, 0.0, -35.0)
	w.register("head", head, "goat")

	# ---------------------------------------------------------------- horns (Box2D horn frame: base at the origin)
	var pts := PackedVector3Array([Vector3(0.015, -0.085, 0), Vector3(-0.006, 0.02, 0), Vector3(-0.02, 0.11, 0),
		Vector3(-0.018, 0.19, 0), Vector3(0.0, 0.25, 0), Vector3(0.034, 0.295, 0)])
	var horn_mesh := _taper_tube(pts, 0.03, 0.004, 10)
	for k in 2:
		var hn := M.node(parent, Vector3.ZERO, Vector3.ZERO, "GoatHorn%d" % k)
		M.mi(hn, horn_mesh, horn, Vector3(0, 0, 0.058 if k == 0 else -0.058))
		w.register("horn%d" % k, hn, "goat")

	# ---------------------------------------------------------------- legs
	for i in 8:
		var thigh := i % 2 == 0
		var near := i < 4
		var rear := (i / 2) % 2 == 1
		var z := LEG_Z if near else -LEG_Z
		var leg := M.node(parent, Vector3.ZERO, Vector3.ZERO, "GoatLeg%d" % i)
		var g := M.node(leg, Vector3(0, 0, z))
		if thigh:
			# upper leg: long-haired "pantaloons" reaching into the body, tapering to the knee
			var up := Vector3(0.125, 0.27, 0.11) if rear else Vector3(0.11, 0.25, 0.1)
			_ef(g, up, Vector3(0.0, 0.1, 0), 0.0, fur)
			M.mi(g, M.cyl(0.075, 0.062, 0.3, 14), fur, Vector3(0, -0.075, 0))
			_e(g, Vector3(0.064, 0.072, 0.062), Vector3(0, -0.215, 0), 0.0, fur_fine)
		else:
			M.mi(g, M.cyl(0.06, 0.047, 0.34, 12), fur_fine, Vector3(0, 0.04, 0))
			_e(g, Vector3(0.05, 0.055, 0.048), Vector3(0, -0.14, 0), 0.0, fur_fine)
			# black cloven hoof, sole exactly on the Box2D shin's bottom edge (local y = -0.225)
			for s in [-1.0, 1.0]:
				var hf := M.mi(g, M.cyl(0.024, 0.036, 0.075, 10), hoof, Vector3(-0.012, -0.1875, s * 0.019))
				hf.scale = Vector3(1.25, 1.0, 0.85)
			M.mi(g, M.sphere(0.012, 8, 4), hoof, Vector3(0.035, -0.135, 0.015))
			M.mi(g, M.sphere(0.012, 8, 4), hoof, Vector3(0.035, -0.135, -0.015))
		w.register("leg%d" % i, leg, "goat")
	return torso


static func _e(parent: Node3D, r: Vector3, pos: Vector3, rot_z_deg: float, mat: Material) -> MeshInstance3D:
	return M.mi(parent, ellipsoid(r), mat, pos, Vector3(0, 0, rot_z_deg))


static func _ef(parent: Node3D, r: Vector3, pos: Vector3, rot_z_deg: float, mat: Material) -> MeshInstance3D:
	## Long-haired part: the ellipsoid plus SHELLS fur shells (no shadows, no mask twins).
	var base := _e(parent, r, pos, rot_z_deg, mat)
	for k in SHELLS:
		var sh := M.mi(base, ellipsoid(r + Vector3.ONE * SHELL_STEP * (k + 1)), shell_material(k))
		sh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		sh.set_meta("no_mask", true)
	return base


static func shell_material(k: int) -> ShaderMaterial:
	if _shell_shader == null:
		_shell_shader = Shader.new()
		_shell_shader.code = SHELL_SHADER
		_shell_mats.clear()
		var tex := Assets.noise_texture(321, 0.09, false, 256)
		for i in SHELLS:
			var m := ShaderMaterial.new()
			m.shader = _shell_shader
			m.set_shader_parameter("strands", tex)
			m.set_shader_parameter("shell_t", (i + 1.0) / SHELLS)
			_shell_mats.append(m)
	return _shell_mats[k]


const SHELL_SHADER := """
shader_type spatial;
render_mode diffuse_lambert_wrap, specular_disabled, cull_back, depth_draw_opaque;
uniform sampler2D strands : filter_linear_mipmap, repeat_enable;
uniform float shell_t = 0.25;
uniform vec3 tint = vec3(0.62, 0.62, 0.6);   // linear albedo
varying vec3 lp;
varying vec3 ln;
void vertex() {
	lp = VERTEX;
	ln = NORMAL;
}
void fragment() {
	vec3 w = pow(abs(ln), vec3(3.0));
	w /= dot(w, vec3(1.0));
	// strands hang down: stretched along y on the flanks
	float sx = texture(strands, vec2(lp.z * 13.0, lp.y * 2.6)).r;
	float sz = texture(strands, vec2(lp.x * 13.0, lp.y * 2.6)).r;
	float sy = texture(strands, lp.xz * 8.0).r;
	float s = sx * w.x + sy * w.y + sz * w.z;
	ALBEDO = tint * mix(0.84, 1.02, shell_t) * mix(0.9, 1.05, s);
	ALPHA = s;
	ALPHA_SCISSOR_THRESHOLD = 0.42 + 0.3 * shell_t;
	ROUGHNESS = 1.0;
	AO = mix(0.75, 1.0, shell_t);
	BACKLIGHT = vec3(0.25);
}
"""


static func ellipsoid(r: Vector3) -> Mesh:
	## Ellipsoid mesh with baked radii (correct normals, unstretched object-space triplanar fur).
	var key := "%.3f,%.3f,%.3f" % [r.x, r.y, r.z]
	if _ell_cache.has(key):
		return _ell_cache[key]
	var sm := SphereMesh.new()
	sm.radius = 1.0
	sm.height = 2.0
	sm.radial_segments = 28
	sm.rings = 14
	var am := ArrayMesh.new()
	am.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, sm.get_mesh_arrays())
	var mdt := MeshDataTool.new()
	mdt.create_from_surface(am, 0)
	for i in mdt.get_vertex_count():
		var v := mdt.get_vertex(i)
		mdt.set_vertex(i, v * r)
		var n := mdt.get_vertex_normal(i)
		mdt.set_vertex_normal(i, Vector3(n.x / r.x, n.y / r.y, n.z / r.z).normalized())
	am.clear_surfaces()
	mdt.commit_to_surface(am)
	_ell_cache[key] = am
	return am


static func _taper_tube(points: PackedVector3Array, r0: float, r1: float, segs := 10) -> ArrayMesh:
	## Tube along a polyline with the radius tapering from r0 to r1, closed at both ends.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := points.size()
	var total := 0.0
	var acc := PackedFloat32Array([0.0])
	for i in range(1, n):
		total += points[i].distance_to(points[i - 1])
		acc.append(total)
	var rings: Array = []
	var prev_n := Vector3(0, 0, 1)
	for i in n:
		var tgt := (points[mini(i + 1, n - 1)] - points[maxi(i - 1, 0)]).normalized()
		var nx := (prev_n - tgt * prev_n.dot(tgt)).normalized()
		prev_n = nx
		var bx := tgt.cross(nx)
		var rad := lerpf(r0, r1, pow(acc[i] / total, 1.3))
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
			M._tri_n(st, a[0], b[0], c[0], a[1], b[1], c[1], Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), fn)
			M._tri_n(st, a[0], c[0], d[0], a[1], c[1], d[1], Vector2(0, 0), Vector2(1, 1), Vector2(0, 1), fn)
	for end in [[0, -1.0], [n - 1, 1.0]]:
		var i: int = end[0]
		var tgt := (points[mini(i + 1, n - 1)] - points[maxi(i - 1, 0)]).normalized() * float(end[1])
		for s in segs:
			var a: Array = rings[i][s]
			var b: Array = rings[i][(s + 1) % segs]
			M._tri(st, points[i], a[0], b[0], tgt)
	return st.commit()
