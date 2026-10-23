extends RefCounted
## Four-link walker mech fitted to the BipedalWalker Box2D bodies (gymnasium bipedal_walker.py):
##   hull   HULL_POLY / 30: pentagon 2.13 x 0.57 m, GymBox extrusion z +-1.0
##   thigh  box 0.267 x 1.133 m, shin box 0.213 x 1.133 m, extrusion 0.4 m (legs[0..1] z 1.1..1.5,
##          legs[2..3] z -1.5..-1.1); the hip sits at hull (0, -0.267), the knee at the thigh's
##          bottom centre. Python poses each leg node at its extrusion centre (z = +-1.3).
## Each part is built in its Box2D body frame (side-view: local (lx, ly) -> (lx, ly, z)) and stays
## inside the Box2D shape (a few cm of trim at most), so feet touch the ground exactly where the
## box corners do. The thigh is a fork (two plates) and the shin a single central beam, so the two
## links nest at the knee instead of interpenetrating when it bends.

const HULL := [Vector2(-1.0, 0.3), Vector2(0.2, 0.3), Vector2(1.13333, 0.03333), Vector2(1.13333, -0.26667), Vector2(-1.0, -0.26667)]
const HIP := Vector2(0.0, -0.26667)
const HALF := 0.56667            # half leg length (LEG_H / 2)


static func palette(kind: String) -> Dictionary:
	## GymBox colours: Bipedal hull (127, 51, 229) / (76, 76, 127), legs (178, 101, 152) and
	## (128, 51, 102); Parkour runner: teal hull, navy thighs, white shins, skin-tone head.
	if kind == "runner":
		return {"body": Color8(0, 158, 150), "trim": Color8(0, 98, 94), "accent": Color8(235, 235, 240),
			"thigh": [Color8(45, 55, 82), Color8(70, 80, 108)], "shin": [Color8(232, 232, 236), Color8(196, 196, 206)],
			"joint": Color8(28, 30, 36), "head": Color8(238, 196, 152), "eye": Color(1.0, 0.45, 0.12)}
	return {"body": Color8(92, 40, 192), "trim": Color8(56, 58, 102), "accent": Color8(214, 212, 222),
		"thigh": [Color8(134, 74, 118), Color8(94, 42, 80)], "shin": [Color8(134, 74, 118), Color8(94, 42, 80)],
		"joint": Color8(30, 30, 36), "eye": Color(0.15, 0.85, 1.0)}


# ================================================================ materials

static var _paint_shader: Shader


static func painted(c: Color, rough := 0.55, coat := 0.25, grime := 0.7, scale := 1.1) -> Material:
	## Weathered painted steel: own colour, modulated by the (desaturated) wear and grime of Poly
	## Haven blue_metal_plate, with its panel normals and roughness. Object-space triplanar, so the
	## wear sticks to the moving part.
	if _paint_shader == null:
		_paint_shader = Shader.new()
		_paint_shader.code = """
shader_type spatial;
uniform vec3 paint : source_color = vec3(0.5);
uniform sampler2D wear : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D orm : filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool has_tex = false;
uniform float scale = 1.0;
uniform float grime = 0.5;
uniform float rough = 0.55;
uniform float coat = 0.25;
varying vec3 lp;
varying vec3 lw;
void vertex() {
	lp = VERTEX * scale;
	lw = pow(abs(NORMAL), vec3(4.0));
	lw /= (lw.x + lw.y + lw.z);
	TANGENT = normalize(vec3(0.0, 0.0, -1.0) * abs(NORMAL.x) + vec3(1.0, 0.0, 0.0) * abs(NORMAL.y) + vec3(1.0, 0.0, 0.0) * abs(NORMAL.z));
	BINORMAL = normalize(vec3(0.0, 1.0, 0.0) * abs(NORMAL.x) + vec3(0.0, 0.0, -1.0) * abs(NORMAL.y) + vec3(0.0, 1.0, 0.0) * abs(NORMAL.z));
}
vec3 tri(sampler2D t) {
	return texture(t, lp.zy).rgb * lw.x + texture(t, lp.xz).rgb * lw.y + texture(t, lp.xy).rgb * lw.z;
}
void fragment() {
	if (has_tex) {
		vec3 w = tri(wear);
		float l = dot(w, vec3(0.3, 0.55, 0.15));
		float k = clamp(l / 0.045, 0.45, 1.7);
		ALBEDO = paint * mix(1.0, k, grime);
		NORMAL_MAP = tri(nor);
		NORMAL_MAP_DEPTH = 0.7;
		vec3 o = tri(orm);
		ROUGHNESS = clamp(rough * (0.6 + 0.6 * o.g), 0.08, 1.0);
		AO = o.r;
		AO_LIGHT_AFFECT = 0.3;
	} else {
		ALBEDO = paint;
		ROUGHNESS = rough;
	}
	METALLIC = 0.15;
	SPECULAR = 0.5;
	CLEARCOAT = coat;
	CLEARCOAT_ROUGHNESS = 0.3;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _paint_shader
	var maps: Dictionary = Assets.index().get("textures", {}).get("blue_metal_plate", {})
	if maps.has("diff") and maps.has("nor_gl") and maps.has("arm"):
		m.set_shader_parameter("has_tex", true)
		m.set_shader_parameter("wear", Assets.image_texture(maps["diff"]))
		m.set_shader_parameter("nor", Assets.image_texture(maps["nor_gl"]))
		m.set_shader_parameter("orm", Assets.image_texture(maps["arm"]))
	m.set_shader_parameter("paint", c)
	m.set_shader_parameter("scale", scale)
	m.set_shader_parameter("grime", grime)
	m.set_shader_parameter("rough", rough)
	m.set_shader_parameter("coat", coat)
	return m


static func mats(pal: Dictionary) -> Dictionary:
	return {
		"dark": M.mat(pal.get("joint", Color(0.08, 0.08, 0.09)), 0.42, 0.75, {"normal_noise": 0.5, "noise_scale": 3.0}),
		"steel": M.mat(Color(0.6, 0.61, 0.63), 0.28, 1.0),
		"chrome": M.mat(Color(0.92, 0.92, 0.94), 0.07, 1.0),
		"rubber": M.mat(Color(0.035, 0.035, 0.038), 0.88, 0.0, {"normal_noise": 0.8, "noise_scale": 6.0}),
		"glass": M.mat(Color(0.015, 0.02, 0.03), 0.04, 0.5, {"clearcoat": 1.0}),
		"eye": M.emissive(pal.get("eye", Color(0.2, 0.9, 1.0)), 5.0),
		"lamp": M.emissive(Color(1.0, 0.92, 0.75), 4.0),
		"tail": M.emissive(Color(1.0, 0.08, 0.04), 3.0),
	}


# ================================================================ helpers

static func slab(parent: Node3D, p: PackedVector2Array, za: float, zb: float, mat: Material, bevel := 0.0) -> MeshInstance3D:
	if bevel > 0.0:
		return M.mi(parent, bevel_prism(p, minf(za, zb), maxf(za, zb), bevel), mat)
	return M.mi(parent, M.prism(p, minf(za, zb), maxf(za, zb)), mat)


static func bevel_prism(p: PackedVector2Array, z0: float, z1: float, b: float) -> ArrayMesh:
	## Convex polygon extruded z0..z1 with the cap edges chamfered by b (catches highlights).
	var n := p.size()
	var ccw := M.signed_area(p) > 0.0
	b = minf(b, (z1 - z0) * 0.45)
	var en: Array = []        # outward edge normals
	for i in n:
		var e := p[(i + 1) % n] - p[i]
		var nn := Vector2(e.y, -e.x).normalized()
		en.append(nn if ccw else -nn)
	var inn := PackedVector2Array()
	for i in n:
		var na: Vector2 = en[(i - 1 + n) % n]
		var nb: Vector2 = en[i]
		var bis := (na + nb)
		inn.append(p[i] - bis * b / maxf(1.0 + na.dot(nb), 0.2))
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var idx := Geometry2D.triangulate_polygon(inn)
	for k in range(0, idx.size(), 3):
		var a := inn[idx[k]]
		var c := inn[idx[k + 1]]
		var d := inn[idx[k + 2]]
		M._tri(st, Vector3(a.x, a.y, z1), Vector3(c.x, c.y, z1), Vector3(d.x, d.y, z1), Vector3(0, 0, 1))
		M._tri(st, Vector3(a.x, a.y, z0), Vector3(c.x, c.y, z0), Vector3(d.x, d.y, z0), Vector3(0, 0, -1))
	for i in n:
		var j := (i + 1) % n
		var nrm := Vector3(en[i].x, en[i].y, 0.0)
		var pa := p[i]
		var pb := p[j]
		var ia := inn[i]
		var ib := inn[j]
		M._quad(st, [Vector3(pa.x, pa.y, z0 + b), Vector3(pb.x, pb.y, z0 + b), Vector3(pb.x, pb.y, z1 - b), Vector3(pa.x, pa.y, z1 - b)], nrm)
		M._quad(st, [Vector3(pa.x, pa.y, z1 - b), Vector3(pb.x, pb.y, z1 - b), Vector3(ib.x, ib.y, z1), Vector3(ia.x, ia.y, z1)], (nrm + Vector3(0, 0, 1)).normalized())
		M._quad(st, [Vector3(pa.x, pa.y, z0 + b), Vector3(pb.x, pb.y, z0 + b), Vector3(ib.x, ib.y, z0), Vector3(ia.x, ia.y, z0)], (nrm + Vector3(0, 0, -1)).normalized())
	return st.commit()


static func inset(p: PackedVector2Array, d: float) -> PackedVector2Array:
	var r := Geometry2D.offset_polygon(p, -d, Geometry2D.JOIN_MITER)
	return r[0] if r.size() > 0 else p


static func zcyl(parent: Node3D, r: float, length: float, mat: Material, pos: Vector3, segs := 24) -> MeshInstance3D:
	## Cylinder with its axis along Z.
	return M.mi(parent, M.cyl(r, r, length, segs), mat, pos, Vector3(90, 0, 0))


static func xcyl(parent: Node3D, r: float, length: float, mat: Material, pos: Vector3, segs := 16) -> MeshInstance3D:
	return M.mi(parent, M.cyl(r, r, length, segs), mat, pos, Vector3(0, 0, 90))


# ================================================================ hull

static func build_hull(parent: Node3D, pal: Dictionary, head := false) -> Node3D:
	var root := Node3D.new()
	root.name = "Hull"
	parent.add_child(root)
	var mt := mats(pal)
	var body := painted(pal["body"], 0.5)
	var trim := painted(pal["trim"], 0.6, 0.2)
	var accent := painted(pal["accent"], 0.5)
	var poly := PackedVector2Array(HULL)
	# Core shell z +-0.9 and stepped side armour out to the GymBox +-1.0 (the step reads as a panel line)
	slab(root, poly, -0.93, 0.93, body, 0.04)
	var in1 := inset(poly, 0.03)
	var in2 := inset(poly, 0.085)
	for s in [-1.0, 1.0]:
		slab(root, in1, 0.92 * s, 0.965 * s, trim)
		slab(root, in2, 0.955 * s, 1.0 * s, body, 0.02)
		M.mi(root, M.box(Vector3(1.7, 0.035, 0.012), 0.004), trim, Vector3(-0.1, -0.16, 1.0 * s))
		# side number plate and vents on the armour
		M.mi(root, M.box(Vector3(0.36, 0.13, 0.012), 0.004), accent, Vector3(-0.55, 0.06, 1.0 * s))
		for k in 4:
			M.mi(root, M.box(Vector3(0.05, 0.16, 0.012), 0.004), mt["dark"], Vector3(0.12 + k * 0.085, -0.04, 1.0 * s))
	# Dorsal deck: hatch, vent slats, sensor mast with a beacon
	M.mi(root, M.box(Vector3(0.82, 0.03, 1.36), 0.012), trim, Vector3(-0.42, 0.3 + 0.008, 0))
	for k in 6:
		M.mi(root, M.box(Vector3(0.035, 0.022, 0.9), 0.006), mt["dark"], Vector3(-0.74 + k * 0.075, 0.325, 0))
	M.mi(root, M.cyl(0.013, 0.016, 0.34, 8), mt["steel"], Vector3(-0.86, 0.47, -0.66))
	M.mi(root, M.sphere(0.028, 12, 6), mt["tail"], Vector3(-0.86, 0.645, -0.66))
	M.mi(root, M.cyl(0.05, 0.06, 0.05, 12), mt["dark"], Vector3(-0.86, 0.325, -0.66))
	# Sensor visor on the sloped nose (A -> B), with a glowing eye bar
	var a := Vector2(0.2, 0.3)
	var b := Vector2(1.13333, 0.03333)
	var d := (b - a)
	var ang := rad_to_deg(atan2(d.y, d.x))
	var n := Vector2(-d.y, d.x).normalized()
	var c := a + d * 0.52 + n * 0.006
	M.mi(root, M.box(Vector3(0.62, 0.026, 1.56), 0.01), mt["glass"], Vector3(c.x, c.y, 0), Vector3(0, 0, ang))
	var e := a + d * 0.5 + n * 0.022
	M.mi(root, M.box(Vector3(0.07, 0.012, 1.2), 0.004), mt["eye"], Vector3(e.x, e.y, 0), Vector3(0, 0, ang))
	# Nose: bumper grille and headlights on the vertical front face
	M.mi(root, M.box(Vector3(0.03, 0.16, 1.2), 0.008), mt["dark"], Vector3(1.13333, -0.12, 0))
	for s in [-1.0, 1.0]:
		xcyl(root, 0.055, 0.03, mt["lamp"], Vector3(1.14, -0.11, 0.78 * s))
		xcyl(root, 0.07, 0.02, mt["chrome"], Vector3(1.13, -0.11, 0.78 * s))
	# Tail: power-pack grille and tail lights (what the Rear and FPV cameras see)
	M.mi(root, M.box(Vector3(0.03, 0.3, 0.9), 0.01), mt["dark"], Vector3(-1.0, 0.0, 0))
	for k in 5:
		M.mi(root, M.box(Vector3(0.012, 0.022, 0.84), 0.004), mt["steel"], Vector3(-1.018, -0.1 + k * 0.05, 0))
	for s in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.02, 0.06, 0.18), 0.006), mt["tail"], Vector3(-1.005, 0.17, 0.72 * s))
		M.mi(root, M.box(Vector3(0.02, 0.05, 0.14), 0.006), mt["lamp"], Vector3(-1.005, -0.17, 0.74 * s))
	# Hip axle through the belly, hip actuators between hull and thighs
	zcyl(root, 0.1, 2.0, mt["dark"], Vector3(HIP.x, HIP.y, 0))
	for s in [-1.0, 1.0]:
		zcyl(root, 0.17, 0.12, mt["dark"], Vector3(HIP.x, HIP.y, 1.04 * s))
		zcyl(root, 0.11, 0.124, mt["steel"], Vector3(HIP.x, HIP.y, 1.04 * s))
		# hydraulic feed lines from the belly to the actuator
		var line := PackedVector3Array([Vector3(-0.25, -0.2, 0.92 * s), Vector3(-0.2, -0.27, 0.98 * s), Vector3(-0.1, -0.3, 1.0 * s)])
		M.mi(root, M.tube(line, 0.018, 6), mt["rubber"])
	if head:
		_build_head(root, pal, mt)
	return root


static func _build_head(root: Node3D, pal: Dictionary, mt: Dictionary) -> void:
	## Parkour runner head: the GymBox square at hull-local (0.133..0.6, 0.3..0.767), z +-0.6,
	## as a wide sensor helmet on a neck that rests on the sloped deck (nothing floats).
	var shell := painted(pal.get("head", Color8(238, 196, 152)), 0.45, 0.5)
	var hood := painted(pal["body"], 0.6, 0.2)
	M.mi(root, M.cyl(0.13, 0.15, 0.16, 18), mt["dark"], Vector3(0.37, 0.29, 0))
	M.mi(root, M.box(Vector3(0.462, 0.455, 1.18), 0.1), shell, Vector3(0.3667, 0.535, 0))
	# hood / collar over the back of the head
	M.mi(root, M.box(Vector3(0.2, 0.36, 1.2), 0.06), hood, Vector3(0.235, 0.49, 0))
	# visor band and eyes on the front face (x = 0.6)
	M.mi(root, M.box(Vector3(0.03, 0.13, 0.98), 0.01), mt["glass"], Vector3(0.597, 0.58, 0))
	for s in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.012, 0.035, 0.14), 0.004), mt["eye"], Vector3(0.614, 0.585, 0.2 * s))
	# ear pods
	for s in [-1.0, 1.0]:
		zcyl(root, 0.11, 0.04, mt["dark"], Vector3(0.38, 0.55, 0.585 * s))


# ================================================================ legs

static func build_thigh(parent: Node3D, pal: Dictionary, side: int) -> Node3D:
	## side 0 = legs[0] (z +1.3, outer face +z), 1 = legs[2] (z -1.3, outer face -z).
	var root := Node3D.new()
	root.name = "Thigh%d" % side
	parent.add_child(root)
	var mt := mats(pal)
	var paint := painted(pal["thigh"][side], 0.5)
	var plate := PackedVector2Array([Vector2(-0.128, 0.5), Vector2(0.128, 0.5), Vector2(0.104, -0.5), Vector2(-0.104, -0.5)])
	for sz in [-1.0, 1.0]:
		slab(root, plate, sz * 0.13, sz * 0.2, paint, 0.016)
		slab(root, inset(plate, 0.045), sz * 0.2, sz * 0.207, painted(pal["thigh"][side] * 0.62, 0.6))
	# web between the plates (upper half only: the shin swings in the lower half of the fork)
	M.mi(root, M.box(Vector3(0.13, 0.5, 0.26), 0.02), mt["dark"], Vector3(-0.025, 0.2, 0))
	# hip hub, knee pin with caps
	zcyl(root, 0.15, 0.4, mt["dark"], Vector3(0, HALF, 0))
	zcyl(root, 0.075, 0.42, mt["steel"], Vector3(0, HALF, 0))
	zcyl(root, 0.08, 0.42, mt["steel"], Vector3(0, -HALF, 0))
	for sz in [-1.0, 1.0]:
		zcyl(root, 0.115, 0.02, paint, Vector3(0, -HALF, 0.205 * sz))
	# hydraulic ram along the front edge
	M.mi(root, M.cyl(0.034, 0.034, 0.42, 12), mt["dark"], Vector3(0.1, 0.13, 0))
	M.mi(root, M.cyl(0.017, 0.017, 0.34, 10), mt["chrome"], Vector3(0.1, -0.22, 0))
	return root


static func build_shin(parent: Node3D, pal: Dictionary, side: int) -> Node3D:
	var root := Node3D.new()
	root.name = "Shin%d" % side
	parent.add_child(root)
	var mt := mats(pal)
	var paint := painted(pal["shin"][side], 0.45)
	var beam := PackedVector2Array([Vector2(-0.092, 0.5), Vector2(0.092, 0.5), Vector2(0.072, -0.42), Vector2(-0.072, -0.42)])
	slab(root, beam, -0.11, 0.11, paint, 0.02)
	# knee hub (inside the thigh fork), shin guard, ankle and struts
	zcyl(root, 0.11, 0.24, mt["dark"], Vector3(0, HALF, 0))
	M.mi(root, M.box(Vector3(0.03, 0.5, 0.17), 0.01), mt["dark"], Vector3(0.085, 0.08, 0))
	zcyl(root, 0.058, 0.34, mt["steel"], Vector3(0, -0.44, 0))
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.12, 0.1, 0.035), 0.01), mt["steel"], Vector3(0, -0.48, 0.155 * sz))
	# foot pad: its sole is the Box2D box's bottom edge (y = -HALF), so it touches the ground exactly
	M.mi(root, M.box(Vector3(0.22, 0.085, 0.38), 0.025), mt["rubber"], Vector3(0, -HALF + 0.0425, 0))
	M.mi(root, M.box(Vector3(0.18, 0.03, 0.34), 0.01), mt["dark"], Vector3(0, -HALF + 0.095, 0))
	return root
