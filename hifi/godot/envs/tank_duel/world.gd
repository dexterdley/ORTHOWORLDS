extends EnvWorld
## Tank Duel: two compact main battle tanks in a walled desert arena (concrete perimeter wall,
## concrete / gabion / sandstone cover blocks), wasteland sky.
## Python: hifi/exporters/tank_duel.py. Top-down env: Box2D (x, y) at height h -> Godot (x, h, -y),
## body angle -> yaw. Tank local frame: +X forward, -Z = Box2D left (+y).
## Moving ids: player, enemy (hulls). Static, hideable by the elevation section: obs<i>, wall<i>,
## and the desert props scn<i> (no mask class; their cutaway is computed here from state["cams"]).
## One consistent world: every object (props, walls, hills) is drawn in every view whose frame
## contains it; only cutaways (wholly between an elevation camera and the player) are removed.
## p.tanks[i] = [turret yaw rel. hull, hit flash 0..1, left track m, right track m, steps since fire,
##               forward speed m/s].
## Transient: shell (tracer), boom (tank hit fireball), dust (wall / obstacle hit).

const TOP_HULL := 1.0           # GymBox hull top
const TURRET_TOP := 1.5         # GymBox turret top
const BARREL_Y := 1.25          # GymBox barrel axis (1.15..1.35)
const TRACK_Z := 0.72           # track centre line
const TRACK_W := 0.42
const ROAD_R := 0.24
const ROAD_Y := 0.29
const ROAD_X := [-0.95, -0.475, 0.0, 0.475, 0.95]
const IDLER := Vector3(1.27, 0.38, 0.16)      # x, y, r
const SPROCKET := Vector3(-1.28, 0.38, 0.17)

const TRACK_SHADER := """
shader_type spatial;
render_mode cull_back;
instance uniform float track_phase = 0.0;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform vec3 dust : source_color = vec3(0.62, 0.53, 0.38);
void fragment() {
	float u = UV.x + track_phase;
	float f = fract(u / 0.15);
	float plate = smoothstep(0.03, 0.12, f) * (1.0 - smoothstep(0.78, 0.9, f));
	float n = texture(noise_tex, vec2(u * 0.31, UV.y * 0.9)).r;
	float rib = smoothstep(0.42, 0.5, fract(UV.y * 4.0 + 0.25)) * 0.35;
	vec3 steel = vec3(0.19, 0.18, 0.17) * (0.7 + 0.6 * n) * (1.0 - rib);
	vec3 col = mix(vec3(0.025), steel, plate);
	col = mix(col, dust * (0.7 + 0.3 * n), smoothstep(0.45, 0.85, n) * 0.55 * plate);
	ALBEDO = col;
	METALLIC = 0.5 * plate;
	ROUGHNESS = mix(0.95, 0.5, plate * (1.0 - n * 0.4));
}
"""

const GROUND_SHADER := """
shader_type spatial;
uniform sampler2D a_alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D a_nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D a_arm : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D b_alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D b_nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D b_arm : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform float a_scale = 0.3;
uniform float b_scale = 0.22;
uniform vec3 a_tint : source_color = vec3(1.0);
uniform vec3 b_tint : source_color = vec3(1.0);
varying vec3 wpos;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 p = wpos.xz;
	float m = texture(noise_tex, p * 0.011).r * 0.75 + texture(noise_tex, p * 0.047 + 0.31).r * 0.35;
	float t = smoothstep(0.52, 0.75, m) * 0.85;
	vec2 ua = p * a_scale;
	vec2 ub = p * b_scale;
	float macro = 0.86 + 0.26 * texture(noise_tex, p * 0.0037 + 0.71).r;
	ALBEDO = mix(texture(a_alb, ua).rgb * a_tint, texture(b_alb, ub).rgb * b_tint, t) * macro;
	NORMAL_MAP = mix(texture(a_nor, ua).rgb, texture(b_nor, ub).rgb, t);
	vec3 arm = mix(texture(a_arm, ua).rgb, texture(b_arm, ub).rgb, t);
	AO = arm.r;
	AO_LIGHT_AFFECT = 0.3;
	ROUGHNESS = clamp(arm.g * 1.05, 0.0, 1.0);
	METALLIC = 0.0;
}
"""

const HESCO_SHADER := """
shader_type spatial;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform vec3 cloth : source_color = vec3(0.5, 0.43, 0.31);
varying vec3 lpos;
varying vec3 lnrm;
void vertex() {
	lpos = VERTEX;
	lnrm = NORMAL;
}
void fragment() {
	vec2 q = abs(lnrm.x) > 0.5 ? lpos.zy : (abs(lnrm.z) > 0.5 ? lpos.xy : lpos.xz);
	vec2 g = abs(fract(q / 0.075) - 0.5);
	float wire = 1.0 - smoothstep(0.03, 0.08, min(g.x, g.y));
	float n = texture(noise_tex, q * 0.35).r;
	float bulge = 0.5 + 0.5 * sin(q.x * 3.14159 / 1.0) * sin(q.y * 3.14159 / 1.0);
	vec3 base = cloth * (0.72 + 0.4 * n) * (0.85 + 0.15 * bulge);
	ALBEDO = mix(base, vec3(0.16, 0.15, 0.13), wire * 0.85);
	METALLIC = wire * 0.6;
	ROUGHNESS = mix(0.95, 0.45, wire);
}
"""

var _tanks: Array = []          # per tank: Dictionary of nodes / materials
var _scene_ok := false
var _props: Array = []          # [id, world AABB] of the desert props outside the walls
const PROP_CUT_REACH := 2.4     # lateral half-extent of the player in the elevations (hull + barrel)


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_48d_partly_cloudy_puresky", "sky_energy": 1.0, "sky_rotation": 150.0,
		"sun_from_hdri": true, "sun_elev": 48.0, "sun_azim": 220.0,
		"sun_energy": 2.4, "sun_color": Color(1.0, 0.94, 0.84),
		"ambient_energy": 1.0, "exposure": 0.95, "white": 6.0,
		"fog_density": 0.0016, "fog_color": Color(0.86, 0.82, 0.74), "fog_aerial": 0.6, "fog_sky_affect": 0.0,
		"saturation": 1.04, "contrast": 1.06, "shadow_distance": 110.0,
		"card_ground": "sand_01", "card_ground_tint": Color(0.74, 0.68, 0.56), "card_ground_scale": 0.3,
		"card_eye": 1.4,
		# Depth haze in the ortho views too: the elevations show the hill ring 140-300 m away at
		# full size, and the haze sets it back behind the arena.
		"ortho_fog": true,
	})
	# Ortho views have no aerial perspective: depth haze that leaves the arena (< ~105 m from an
	# elevation camera) clear and sets the hill ring (> ~115 m) back like the FPV's aerial haze.
	if lk.env_ortho != null:
		lk.env_ortho.fog_mode = Environment.FOG_MODE_DEPTH
		lk.env_ortho.fog_depth_begin = 95.0
		lk.env_ortho.fog_depth_end = 300.0
		lk.env_ortho.fog_depth_curve = 0.6
		lk.env_ortho.fog_density = 0.85


func build_scene(scene: Dictionary) -> void:
	_tanks.clear()
	_build_ground()
	_build_walls(scene)
	_build_obstacles(scene)
	_build_scenery()
	_build_hills()
	for i in 2:
		var t := _build_tank(i)
		_tanks.append(t)
		register("player" if i == 0 else "enemy", t["root"], "player" if i == 0 else "enemy")
	_scene_ok = true


# ================================================================ helpers

static func _g(p: Vector2, h: float) -> Vector3:
	return Vector3(p.x, h, -p.y)


static func _tex(name: String, map: String) -> Texture2D:
	var t: Dictionary = Assets.index().get("textures", {}).get(name, {})
	var path := str(t.get(map, ""))
	return Assets.image_texture(path) if path != "" else null


static func _loft(b: PackedVector2Array, t: PackedVector2Array, h0: float, h1: float) -> ArrayMesh:
	## Box2D-local polygons (same vertex count): bottom b at h0, top t at h1 -> sloped solid.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := b.size()
	var ccw := M.signed_area(b) > 0.0
	var it := Geometry2D.triangulate_polygon(t)
	for i in range(0, it.size(), 3):
		M._tri(st, _g(t[it[i]], h1), _g(t[it[i + 1]], h1), _g(t[it[i + 2]], h1), Vector3.UP)
	var ib := Geometry2D.triangulate_polygon(b)
	for i in range(0, ib.size(), 3):
		M._tri(st, _g(b[ib[i]], h0), _g(b[ib[i + 1]], h0), _g(b[ib[i + 2]], h0), Vector3.DOWN)
	for i in n:
		var j := (i + 1) % n
		var q := [_g(b[i], h0), _g(b[j], h0), _g(t[j], h1), _g(t[i], h1)]
		var e := b[j] - b[i]
		var nn := Vector2(e.y, -e.x).normalized() * (1.0 if ccw else -1.0)
		var a0: Vector3 = q[0]
		var a1: Vector3 = q[1]
		var a3: Vector3 = q[3]
		var nrm: Vector3 = (a1 - a0).cross(a3 - a0).normalized()
		if nrm.dot(Vector3(nn.x, 0.0, -nn.y)) < 0.0:
			nrm = -nrm
		M._quad(st, q, nrm)
	return st.commit()


static func _band(loop: PackedVector2Array, thick: float, z0: float, z1: float) -> ArrayMesh:
	## Closed track belt: loop (x, y) in the tank's side plane, extruded over z0..z1.
	## UV.x = arc length along the loop (CCW seen from +Z), UV.y = z.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := loop.size()
	var nrm: Array = []
	for i in n:
		var a := loop[(i - 1 + n) % n]
		var b := loop[(i + 1) % n]
		var e := (b - a).normalized()
		nrm.append(Vector2(e.y, -e.x))
	var s := 0.0
	for i in n:
		var j := (i + 1) % n
		var pa := loop[i]
		var pb := loop[j]
		var na: Vector2 = nrm[i]
		var nb: Vector2 = nrm[j]
		var l := pa.distance_to(pb)
		var ia := pa - na * thick
		var ib := pb - nb * thick
		var fo := Vector3((na + nb).x, (na + nb).y, 0.0).normalized()
		# outer
		M._tri_n(st, Vector3(pa.x, pa.y, z0), Vector3(pb.x, pb.y, z0), Vector3(pb.x, pb.y, z1),
			Vector3(na.x, na.y, 0), Vector3(nb.x, nb.y, 0), Vector3(nb.x, nb.y, 0),
			Vector2(s, z0), Vector2(s + l, z0), Vector2(s + l, z1), fo)
		M._tri_n(st, Vector3(pa.x, pa.y, z0), Vector3(pb.x, pb.y, z1), Vector3(pa.x, pa.y, z1),
			Vector3(na.x, na.y, 0), Vector3(nb.x, nb.y, 0), Vector3(na.x, na.y, 0),
			Vector2(s, z0), Vector2(s + l, z1), Vector2(s, z1), fo)
		# inner
		M._tri_n(st, Vector3(ia.x, ia.y, z0), Vector3(ib.x, ib.y, z0), Vector3(ib.x, ib.y, z1),
			Vector3(-na.x, -na.y, 0), Vector3(-nb.x, -nb.y, 0), Vector3(-nb.x, -nb.y, 0),
			Vector2(s, z0), Vector2(s + l, z0), Vector2(s + l, z1), -fo)
		M._tri_n(st, Vector3(ia.x, ia.y, z0), Vector3(ib.x, ib.y, z1), Vector3(ia.x, ia.y, z1),
			Vector3(-na.x, -na.y, 0), Vector3(-nb.x, -nb.y, 0), Vector3(-na.x, -na.y, 0),
			Vector2(s, z0), Vector2(s + l, z1), Vector2(s, z1), -fo)
		# side walls
		for zz in [z0, z1]:
			var sn := Vector3(0, 0, -1.0 if zz == z0 else 1.0)
			var z: float = zz
			M._tri_n(st, Vector3(pa.x, pa.y, z), Vector3(pb.x, pb.y, z), Vector3(ib.x, ib.y, z), sn, sn, sn,
				Vector2(s, 0.0), Vector2(s + l, 0.0), Vector2(s + l, thick), sn)
			M._tri_n(st, Vector3(pa.x, pa.y, z), Vector3(ib.x, ib.y, z), Vector3(ia.x, ia.y, z), sn, sn, sn,
				Vector2(s, 0.0), Vector2(s + l, thick), Vector2(s, thick), sn)
		s += l
	return st.commit()


static func _track_loop() -> PackedVector2Array:
	## Convex outline (x, y) around road wheels, idler and sprocket, plus the belt thickness.
	var pts := PackedVector2Array()
	var circles: Array = []
	for x in ROAD_X:
		circles.append(Vector3(x, ROAD_Y, ROAD_R))
	circles.append(IDLER)
	circles.append(SPROCKET)
	for c in circles:
		var cc: Vector3 = c
		for k in 28:
			var a := TAU * k / 28.0
			pts.append(Vector2(cc.x + cos(a) * (cc.z + 0.05), cc.y + sin(a) * (cc.z + 0.05)))
	var hull := Geometry2D.convex_hull(pts)
	if hull.size() > 1 and hull[0].distance_to(hull[hull.size() - 1]) < 1e-5:
		hull.remove_at(hull.size() - 1)
	if M.signed_area(hull) < 0.0:
		hull.reverse()
	# resample to even spacing so the tread pattern scrolls smoothly
	var out := PackedVector2Array()
	var n := hull.size()
	for i in n:
		var a := hull[i]
		var b := hull[(i + 1) % n]
		var k := maxi(1, int(ceil(a.distance_to(b) / 0.06)))
		for s in k:
			out.append(a.lerp(b, float(s) / k))
	return out


func _paint(idx: int) -> BaseMaterial3D:
	var m: BaseMaterial3D
	if idx == 0:
		m = Assets.pbr("green_metal_rust", {"uv": 0.7, "tint": Color(0.64, 0.7, 0.44), "color": Color(0.24, 0.3, 0.18), "metallic": 0.1}).duplicate()
	else:
		m = Assets.pbr("rusty_painted_metal", {"uv": 0.55, "tint": Color(0.72, 0.54, 0.45), "color": Color(0.5, 0.2, 0.17), "metallic": 0.12}).duplicate()
	m.metallic_specular = 0.25
	m.emission_enabled = true
	m.emission = Color(1.0, 0.78, 0.5)
	m.emission_energy_multiplier = 0.0
	return m


# ================================================================ ground, walls, cover

func _build_ground() -> void:
	var sm := ShaderMaterial.new()
	var sh := Shader.new()
	sh.code = GROUND_SHADER
	sm.shader = sh
	for pair in [["a", "sand_01"], ["b", "dry_ground_01"]]:
		var k: String = pair[0]
		var nm: String = pair[1]
		sm.set_shader_parameter(k + "_alb", _tex(nm, "diff"))
		sm.set_shader_parameter(k + "_nor", _tex(nm, "nor_gl"))
		sm.set_shader_parameter(k + "_arm", _tex(nm, "arm"))
	sm.set_shader_parameter("noise_tex", Assets.noise_texture(77, 0.02, false))
	sm.set_shader_parameter("a_tint", Color(0.93, 0.86, 0.72))
	sm.set_shader_parameter("b_tint", Color(1.05, 0.92, 0.74))
	sm.set_shader_parameter("a_scale", 0.32)
	sm.set_shader_parameter("b_scale", 0.2)
	var g := MeshInstance3D.new()
	g.name = "Ground"
	var pm := PlaneMesh.new()
	pm.size = Vector2(460, 460)
	g.mesh = pm
	g.material_override = sm
	g.position = Vector3(20, 0, -15)
	g.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(g)


func _build_walls(scene: Dictionary) -> void:
	var conc := Assets.pbr("concrete_block_wall", {"uv": 0.5, "world": true, "tint": Color(1.55, 1.45, 1.3), "color": Color(0.55, 0.52, 0.48)})
	var cap := Assets.pbr("concrete_floor_02", {"uv": 0.6, "world": true, "tint": Color(0.9, 0.87, 0.82), "color": Color(0.62, 0.6, 0.56)})
	var stain := M.mat(Color(0.32, 0.27, 0.2), 0.95, 0.0, {"alpha": 0.35})
	var walls: Array = scene.get("walls", [])
	var W: float = scene.get("W", 40.0)
	var H: float = scene.get("H", 30.0)
	for i in walls.size():
		var poly := M.poly(walls[i])
		var mn := Vector2(INF, INF)
		var mx := Vector2(-INF, -INF)
		for p in poly:
			mn = Vector2(minf(mn.x, p.x), minf(mn.y, p.y))
			mx = Vector2(maxf(mx.x, p.x), maxf(mx.y, p.y))
		var c := (mn + mx) * 0.5
		var sz := mx - mn
		var root := M.node(content, _g(c, 0.0), Vector3.ZERO, "Wall%d" % i)
		M.mi(root, M.box(Vector3(sz.x, 2.38, sz.y), 0.02), conc, Vector3(0, 1.19, 0))
		M.mi(root, M.box(Vector3(sz.x + 0.08, 0.12, sz.y + 0.08), 0.02), cap, Vector3(0, 2.44, 0))
		# outward normal (away from the arena centre): buttresses on the outside only
		var outward := Vector2(0, 0)
		if sz.x > sz.y:
			outward = Vector2(0, 1.0 if c.y > H * 0.5 else -1.0)
		else:
			outward = Vector2(1.0 if c.x > W * 0.5 else -1.0, 0)
		var length := maxf(sz.x, sz.y)
		var along := Vector2(1, 0) if sz.x > sz.y else Vector2(0, 1)
		var nb := int(length / 5.0)
		for k in range(1, nb):
			var o := along * (-length * 0.5 + k * length / nb) + outward * (minf(sz.x, sz.y) * 0.5 + 0.2)
			var bs := Vector3(0.45 if along.x > 0.5 else 0.4, 2.1, 0.4 if along.x > 0.5 else 0.45)
			M.mi(root, M.box(bs, 0.03), conc, Vector3(o.x, 1.05, -o.y))
		# dust stain along the inner foot
		var inner := -outward * (minf(sz.x, sz.y) * 0.5 + 0.004)
		var st := M.mi(root, M.box(Vector3(sz.x if along.x > 0.5 else 0.005, 0.5, sz.y if along.y > 0.5 else 0.005)), stain,
			Vector3(inner.x, 0.25, -inner.y))
		st.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		register("wall%d" % i, root, "wall")
	# corner posts (GymBox leaves the 0.5 m corner squares open)
	for cx in [-0.25, W + 0.25]:
		for cy in [-0.25, H + 0.25]:
			var post := M.node(content, Vector3(cx, 0, -cy), Vector3.ZERO, "Corner")
			M.mi(post, M.box(Vector3(0.72, 2.6, 0.72), 0.03), conc, Vector3(0, 1.3, 0))
			M.mi(post, M.box(Vector3(0.84, 0.14, 0.84), 0.03), cap, Vector3(0, 2.62, 0))


func _build_obstacles(scene: Dictionary) -> void:
	var obs: Array = scene.get("obstacles", [])
	var conc := Assets.pbr("concrete_wall_008", {"uv": 0.45, "world": true, "tint": Color(0.92, 0.89, 0.84), "color": Color(0.62, 0.6, 0.56)})
	var stone := Assets.pbr("large_sandstone_blocks", {"uv": 0.35, "world": true, "tint": Color(1.0, 0.9, 0.74), "color": Color(0.66, 0.58, 0.45)})
	var sand := Assets.pbr("sand_01", {"uv": 0.5, "world": true, "tint": Color(0.7, 0.6, 0.45)})
	var hesco := ShaderMaterial.new()
	var hs := Shader.new()
	hs.code = HESCO_SHADER
	hesco.shader = hs
	hesco.set_shader_parameter("noise_tex", Assets.noise_texture(31, 0.05, false))
	var frame := M.mat(Color(0.2, 0.19, 0.17), 0.5, 0.7)
	for i in obs.size():
		var o: Dictionary = obs[i]
		var c: Array = o["c"]
		var hf: Array = o["half"]
		var sx := 2.0 * float(hf[0])
		var sz := 2.0 * float(hf[1])
		var root := M.node(content, Vector3(float(c[0]), 0, -float(c[1])), Vector3.ZERO, "Obs%d" % i)
		match str(o.get("kind", "concrete")):
			"hesco":
				# Gabion cells (~1 m): wire mesh + geotextile, sand fill slightly below the rim
				M.mi(root, M.box(Vector3(sx, 1.96, sz), 0.04), hesco, Vector3(0, 0.98, 0))
				M.mi(root, M.box(Vector3(sx - 0.1, 0.06, sz - 0.1), 0.03), sand, Vector3(0, 1.95, 0))
				var nx := maxi(1, int(round(sx)))
				var nz := maxi(1, int(round(sz)))
				for k in range(0, nx + 1):
					for side in [-1.0, 1.0]:
						var x := -sx * 0.5 + k * sx / nx
						M.mi(root, M.box(Vector3(0.05, 2.0, 0.05)), frame, Vector3(x, 1.0, side * (sz * 0.5 + 0.005)))
				for k in range(1, nz):
					for side in [-1.0, 1.0]:
						var z := -sz * 0.5 + k * sz / nz
						M.mi(root, M.box(Vector3(0.05, 2.0, 0.05)), frame, Vector3(side * (sx * 0.5 + 0.005), 1.0, z))
			"rock":
				# Sandstone masonry: two courses of big blocks with uneven tops (top <= 2.0 m)
				M.mi(root, M.box(Vector3(sx, 1.1, sz), 0.06), stone, Vector3(0, 0.55, 0))
				var rng := RandomNumberGenerator.new()
				rng.seed = 911 + i
				var nb := maxi(1, int(round(sx / 0.9)))
				for k in nb:
					var w := sx / nb
					var hgt := rng.randf_range(0.72, 0.9)
					M.mi(root, M.box(Vector3(w - 0.03, hgt, sz - rng.randf_range(0.0, 0.12)), 0.07), stone,
						Vector3(-sx * 0.5 + (k + 0.5) * w, 1.1 + hgt * 0.5, rng.randf_range(-0.04, 0.04)))
			_:
				# Cast concrete blocks, two courses, chamfered
				M.mi(root, M.box(Vector3(sx, 0.98, sz), 0.05), conc, Vector3(0, 0.49, 0))
				M.mi(root, M.box(Vector3(sx - 0.04, 0.98, sz - 0.04), 0.05), conc, Vector3(0, 1.49, 0))
				for lx in [-0.3, 0.3]:
					M.mi(root, M.box(Vector3(0.06, 0.04, 0.06), 0.01), frame, Vector3(lx * sx, 1.995, 0))
		register("obs%d" % i, root, "obstacle")


func _build_scenery() -> void:
	## Desert clutter outside the walls (low-poly Poly Haven models; software rendering budget):
	## rocks, dead quiver trees, barrels, tyres, barriers, crates. Shown in every view whose frame
	## contains them; each is registered (no mask class) as scn<i> so _cut_props() can remove it
	## from an elevation view while it stands between that camera and the player (cutaway).
	var rng := RandomNumberGenerator.new()
	rng.seed = 4242
	var kinds := [["rock_07", 1.8, "fit"], ["rock_09", 2.0, "fit"], ["rock_07", 1.1, "fit"], ["dead_quiver_trunk", 2.8, "height"],
		["Barrel_01", 0.95, "height"], ["old_tyre", 1.0, "fit"], ["rock_09", 1.2, "fit"], ["concrete_road_barrier_02", 2.0, "fit"],
		["wooden_crate_01", 1.0, "fit"], ["Barrel_01", 0.95, "height"]]
	var root := M.node(content, Vector3.ZERO, Vector3.ZERO, "Scenery")
	_props.clear()
	var placed := 0
	var tries := 0
	while placed < 34 and tries < 600:
		tries += 1
		var near := placed % 3 != 0
		var x := rng.randf_range(-14.0, 54.0) if near else rng.randf_range(-40.0, 80.0)
		var y := rng.randf_range(-12.0, 42.0) if near else rng.randf_range(-35.0, 65.0)
		if x > -2.4 and x < 42.4 and y > -2.4 and y < 32.4:
			continue
		var k: Array = kinds[placed % kinds.size()]
		var s: float = float(k[1]) * rng.randf_range(0.75, 1.25)
		var m := Assets.fitted_model(str(k[0]), Vector3(s, s, s), str(k[2]))
		placed += 1
		if m == null:
			continue
		m.position = Vector3(x, -0.03 * s, -y)
		m.rotation.y = rng.randf() * TAU
		root.add_child(m)
		var id := "scn%d" % _props.size()
		register(id, m)
		_props.append([id, _world_aabb(m)])


static func _world_aabb(n: Node3D) -> AABB:
	var box := AABB()
	var first := true
	for gi in n.find_children("*", "VisualInstance3D", true, false) + ([n] if n is VisualInstance3D else []):
		var b: AABB = gi.global_transform * gi.get_aabb()
		box = b if first else box.merge(b)
		first = false
	return box


func _cut_props(cams: Array) -> void:
	## Cutaway of the props in the rear / side elevations: a prop wholly between the camera and
	## the player (look point) is removed from that view when it overlaps the player's screen
	## extent or reaches past the near plane (it would be sliced); every other prop stays.
	var cut := {}
	for c in cams:
		var vi := ["rear", "side"].find(String(c.get("view", "")))
		if vi < 0:
			continue
		var pos := Vector3(float(c["pos"][0]), float(c["pos"][1]), float(c["pos"][2]))
		var look := Vector3(float(c["look"][0]), float(c["look"][1]), float(c["look"][2]))
		var up := Vector3(float(c["up"][0]), float(c["up"][1]), float(c["up"][2]))
		var fwd := (look - pos).normalized()
		var right := fwd.cross(up).normalized()
		var look_d := (look - pos).dot(fwd)
		var near := float(c.get("near", 0.05))
		for pr in _props:
			var box: AABB = pr[1]
			var zmin := INF
			var zmax := -INF
			var xmin := INF
			var xmax := -INF
			for k in 8:
				var q := box.get_endpoint(k) - pos
				zmin = minf(zmin, q.dot(fwd))
				zmax = maxf(zmax, q.dot(fwd))
				xmin = minf(xmin, q.dot(right))
				xmax = maxf(xmax, q.dot(right))
			if zmax < look_d and (zmin < near or (xmax > -PROP_CUT_REACH and xmin < PROP_CUT_REACH)):
				cut[pr[0]] = int(cut.get(pr[0], 0)) | (1 << (vi + 1))
	for pr in _props:
		set_hidden_mask(pr[0], int(cut.get(pr[0], 0)))


func _build_hills() -> void:
	## Ring of low dunes / mesas far outside the arena, drawn in every view whose frame contains
	## it (the ortho elevations' far planes reach past it, see exporters/tank_duel.py).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var noise := FastNoiseLite.new()
	noise.seed = 5
	noise.frequency = 0.011
	var na := 160
	var radii := [140.0, 175.0, 210.0, 250.0, 300.0]
	var centre := Vector3(20, 0, -15)
	var grid: Array = []
	for ri in radii.size():
		var row: Array = []
		for ai in na + 1:
			var a := TAU * ai / na
			var rr: float = radii[ri]
			var p := centre + Vector3(cos(a) * rr, 0, sin(a) * rr)
			var h := 0.0
			if ri > 0 and ri < radii.size() - 1:
				# <= ~8 m: the true-scale elevations (frame top 14 m) see the ring at full height, so
				# taller dunes would replace their whole sky with hillside
				h = 1.5 + 4.5 * maxf(0.0, noise.get_noise_2d(p.x, p.z) + 0.25) + 0.8 * noise.get_noise_2d(p.x * 3.0, p.z * 3.0)
			elif ri == radii.size() - 1:
				h = 1.0
			p.y = h - 0.5
			row.append(p)
		grid.append(row)
	for ri in radii.size() - 1:
		for ai in na:
			var a: Vector3 = grid[ri][ai]
			var b: Vector3 = grid[ri][ai + 1]
			var c: Vector3 = grid[ri + 1][ai + 1]
			var d: Vector3 = grid[ri + 1][ai]
			st.set_uv(Vector2(a.x, a.z) * 0.05)
			st.add_vertex(a)
			st.set_uv(Vector2(c.x, c.z) * 0.05)
			st.add_vertex(c)
			st.set_uv(Vector2(b.x, b.z) * 0.05)
			st.add_vertex(b)
			st.set_uv(Vector2(a.x, a.z) * 0.05)
			st.add_vertex(a)
			st.set_uv(Vector2(d.x, d.z) * 0.05)
			st.add_vertex(d)
			st.set_uv(Vector2(c.x, c.z) * 0.05)
			st.add_vertex(c)
	st.generate_normals()
	var mi := MeshInstance3D.new()
	mi.name = "Hills"
	mi.mesh = st.commit()
	mi.material_override = Assets.pbr("dry_ground_01", {"uv": 0.04, "world": true, "tint": Color(0.95, 0.83, 0.66)})
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(mi)
	# winding check: Godot culls back faces; show both sides of the far terrain
	var m2: BaseMaterial3D = mi.material_override.duplicate()
	m2.cull_mode = BaseMaterial3D.CULL_DISABLED
	mi.material_override = m2


# ================================================================ tank

func _build_tank(idx: int) -> Dictionary:
	var root := Node3D.new()
	root.name = "Tank%d" % idx
	content.add_child(root)
	var paint := _paint(idx)
	var dark := M.mat(Color(0.07, 0.07, 0.065), 0.75, 0.4)
	var steel := M.mat(Color(0.32, 0.31, 0.3), 0.42, 0.85, {"normal_noise": 0.5, "noise_scale": 2.0})
	var rubber := M.mat(Color(0.045, 0.045, 0.045), 0.88, 0.0)
	var grate := Assets.pbr("metal_grate_rusty", {"uv": 4.0, "tint": Color(0.3, 0.28, 0.26), "color": Color(0.15, 0.14, 0.13)})
	var glass := M.glass(Color(0.05, 0.07, 0.08), 0.85)
	var lamp := M.emissive(Color(1.0, 0.9, 0.7), 0.6)
	var canvas := Assets.pbr("denim_fabric", {"uv": 1.5, "tint": Color(0.55, 0.5, 0.36), "color": Color(0.45, 0.4, 0.3)})

	# ---- running gear: tracks, road wheels, idler, sprocket (both sides)
	var track_mat := ShaderMaterial.new()
	var tsh := Shader.new()
	tsh.code = TRACK_SHADER
	track_mat.shader = tsh
	track_mat.set_shader_parameter("noise_tex", Assets.noise_texture(12, 0.08, false))
	var loop := _track_loop()
	var band := _band(loop, 0.05, -TRACK_W * 0.5, TRACK_W * 0.5)
	var tracks: Array = []
	var wheels: Array = [[], []]
	for side in 2:
		var z := TRACK_Z * (-1.0 if side == 0 else 1.0)    # side 0 = Box2D left = Godot -Z
		var tr := M.mi(root, band, track_mat, Vector3(0, 0, z))
		tracks.append(tr)
		for x in ROAD_X:
			wheels[side].append([_wheel(root, Vector3(x, ROAD_Y, z), ROAD_R, TRACK_W * 0.86, rubber, steel, dark), ROAD_R])
		wheels[side].append([_wheel(root, Vector3(IDLER.x, IDLER.y, z), IDLER.z, TRACK_W * 0.6, steel, steel, dark), IDLER.z])
		wheels[side].append([_sprocket(root, Vector3(SPROCKET.x, SPROCKET.y, z), SPROCKET.z, steel, dark), SPROCKET.z])
		# side skirt (armour plate hanging from the fender)
		var sk := PackedVector2Array([Vector2(-1.42, 0.3), Vector2(1.18, 0.3), Vector2(1.36, 0.5), Vector2(1.36, 0.64), Vector2(-1.42, 0.64)])
		var zs := (0.985 if side == 1 else -0.985)
		M.mi(root, M.prism(sk, zs - 0.022, zs + 0.022), paint)
		for k in 4:
			M.mi(root, M.box(Vector3(0.02, 0.33, 0.05)), dark, Vector3(-0.78 + k * 0.62, 0.47, zs + (0.008 if side == 1 else -0.008)))

	# ---- lower hull between the tracks + upper hull over the full width
	var low := PackedVector2Array([Vector2(-1.38, 0.24), Vector2(1.18, 0.24), Vector2(1.47, 0.6), Vector2(-1.46, 0.62)])
	M.mi(root, M.prism(low, -0.5, 0.5), paint)
	var up := PackedVector2Array([Vector2(-1.48, 0.62), Vector2(1.5, 0.62), Vector2(1.5, 0.68), Vector2(0.72, TOP_HULL),
		Vector2(-1.40, TOP_HULL), Vector2(-1.49, 0.9)])
	M.mi(root, M.prism(up, -0.995, 0.995), paint)
	# engine deck grilles and exhausts
	for sz in [-0.45, 0.45]:
		M.mi(root, M.box(Vector3(0.6, 0.02, 0.6), 0.005), dark, Vector3(-1.02, TOP_HULL + 0.01, sz))
		for k in 5:
			M.mi(root, M.box(Vector3(0.07, 0.03, 0.56), 0.008), paint, Vector3(-1.26 + k * 0.12, TOP_HULL + 0.025, sz))
	for sz in [-0.55, 0.55]:
		M.mi(root, M.box(Vector3(0.05, 0.16, 0.42), 0.01), grate, Vector3(-1.5, 0.8, sz), Vector3(0, 0, -8))
		M.mi(root, M.box(Vector3(0.06, 0.04, 0.44), 0.01), dark, Vector3(-1.51, 0.9, sz), Vector3(0, 0, -8))
	for sz in [-0.9, 0.9]:
		M.mi(root, M.box(Vector3(0.04, 0.05, 0.07), 0.01), M.emissive(Color(0.9, 0.1, 0.05), 0.8), Vector3(-1.49, 0.93, sz))
	M.mi(root, M.box(Vector3(0.1, 0.1, 0.12), 0.02), dark, Vector3(-1.52, 0.5, 0))
	# jerry cans on the rear fenders
	for sz in [-0.86, 0.86]:
		M.mi(root, M.box(Vector3(0.14, 0.24, 0.1), 0.02), M.mat(Color(0.2, 0.24, 0.16), 0.7, 0.2), Vector3(-1.3, TOP_HULL + 0.12, sz))
	# headlights + guards on the glacis corners
	for sz in [-0.82, 0.82]:
		var hl := M.mi(root, M.cyl(0.07, 0.07, 0.08, 12), dark, Vector3(1.3, 0.8, sz), Vector3(0, 0, -62))
		M.mi(hl, M.cyl(0.055, 0.055, 0.01, 12), lamp, Vector3(0, 0.045, 0))
	# driver's periscopes, tow hooks, spare track links on the glacis, fender stowage boxes
	M.mi(root, M.box(Vector3(0.12, 0.06, 0.3), 0.01), glass, Vector3(0.8, TOP_HULL - 0.02, -0.3), Vector3(0, 0, -22))
	for sz in [-0.55, 0.55]:
		M.mi(root, M.box(Vector3(0.12, 0.08, 0.1), 0.02), dark, Vector3(1.52, 0.6, sz))
	for k in 3:
		M.mi(root, M.box(Vector3(0.13, 0.03, 0.62), 0.01), track_mat, Vector3(1.08 + k * 0.13, 0.83 - k * 0.055, 0.2), Vector3(0, 0, -22))
	for sz in [-0.86, 0.86]:
		M.mi(root, M.box(Vector3(0.72, 0.13, 0.2), 0.02), paint, Vector3(-0.55, TOP_HULL + 0.065, sz))
	M.mi(root, M.box(Vector3(0.5, 0.11, 0.18), 0.03), canvas, Vector3(0.25, TOP_HULL + 0.055, 0.85))

	# ---- turret (yaw node at the hull centre)
	var turret := M.node(root, Vector3.ZERO, Vector3.ZERO, "Turret")
	M.mi(turret, M.cyl(0.52, 0.55, 0.05, 32), dark, Vector3(0, TOP_HULL + 0.025, 0))
	var tb := PackedVector2Array([Vector2(0.62, 0.3), Vector2(0.62, -0.3), Vector2(0.38, -0.5), Vector2(-0.52, -0.5),
		Vector2(-0.6, -0.4), Vector2(-0.6, 0.4), Vector2(-0.52, 0.5), Vector2(0.38, 0.5)])
	var tt := PackedVector2Array([Vector2(0.48, 0.22), Vector2(0.48, -0.22), Vector2(0.3, -0.42), Vector2(-0.47, -0.43),
		Vector2(-0.55, -0.36), Vector2(-0.55, 0.36), Vector2(-0.47, 0.43), Vector2(0.3, 0.42)])
	M.mi(turret, _loft(tb, tt, TOP_HULL + 0.04, TURRET_TOP), paint)
	# mantlet + gun
	M.mi(turret, M.box(Vector3(0.2, 0.3, 0.46), 0.04), paint, Vector3(0.64, BARREL_Y, 0))
	var barrel := M.node(turret, Vector3(0.7, BARREL_Y, 0), Vector3.ZERO, "Barrel")
	M.mi(barrel, M.cyl(0.11, 0.115, 0.36, 18), paint, Vector3(0.18, 0, 0), Vector3(0, 0, -90))
	M.mi(barrel, M.cyl(0.066, 0.08, 1.0, 18), steel, Vector3(0.85, 0, 0), Vector3(0, 0, -90))
	M.mi(barrel, M.cyl(0.1, 0.1, 0.26, 18), paint, Vector3(0.68, 0, 0), Vector3(0, 0, -90))
	M.mi(barrel, M.cyl(0.085, 0.085, 0.14, 12), dark, Vector3(1.24, 0, 0), Vector3(0, 0, -90))
	var flash := Vfx.flame(barrel, 1.1, 0.2, Color(1.0, 0.62, 0.22))
	flash.position = Vector3(1.33, 0, 0)
	flash.rotation_degrees = Vector3(0, 0, 90)
	var smoke := Vfx.puff(barrel, Color(0.62, 0.58, 0.52), 7 + idx, 6)
	smoke.position = Vector3(1.6, 0.05, 0)
	smoke.visible = false
	# commander's cupola (right rear), loader's hatch (left), periscopes, antenna
	var cup := M.node(turret, Vector3(-0.18, TURRET_TOP, 0.2))
	M.mi(cup, M.cyl(0.19, 0.21, 0.1, 20), paint, Vector3(0, 0.05, 0))
	M.mi(cup, M.cyl(0.17, 0.17, 0.035, 20), dark, Vector3(0, 0.115, 0))
	for k in 6:
		var a := TAU * k / 6.0
		M.mi(cup, M.box(Vector3(0.05, 0.045, 0.07), 0.005), glass, Vector3(cos(a) * 0.19, 0.075, sin(a) * 0.19), Vector3(0, -rad_to_deg(a), 0))
	M.mi(turret, M.cyl(0.15, 0.16, 0.035, 18), paint, Vector3(-0.15, TURRET_TOP + 0.017, -0.24))
	M.mi(turret, M.box(Vector3(0.1, 0.07, 0.14), 0.01), dark, Vector3(0.22, TURRET_TOP + 0.03, -0.18))
	M.mi(turret, M.cyl(0.006, 0.01, 1.0, 6), dark, Vector3(-0.48, TURRET_TOP + 0.5, -0.33))
	M.mi(turret, M.cyl(0.03, 0.03, 0.05, 8), dark, Vector3(-0.48, TURRET_TOP + 0.02, -0.33))
	# smoke grenade dischargers on the front cheeks
	for sgn in [-1.0, 1.0]:
		for k in 3:
			M.mi(turret, M.cyl(0.035, 0.035, 0.16, 10), dark, Vector3(0.38 - k * 0.07, 1.33, sgn * (0.47 + k * 0.012)),
				Vector3(-sgn * 55.0, 0, -25))
	# rear stowage basket with a canvas tarp
	M.mi(turret, M.box(Vector3(0.14, 0.2, 0.86), 0.02), canvas, Vector3(-0.66, 1.2, 0))
	M.mi(turret, M.box(Vector3(0.16, 0.02, 0.9)), dark, Vector3(-0.66, 1.31, 0))

	# dust kicked up behind the tracks (phase follows the track distance: deterministic)
	var dust: Array = []
	for side in 2:
		var d := Vfx.puff(root, Color(0.8, 0.7, 0.52), 40 + idx * 2 + side, 5)
		d.position = Vector3(-1.62, 0.15, TRACK_Z * (-1.0 if side == 0 else 1.0))
		d.visible = false
		dust.append(d)
	return {"root": root, "turret": turret, "barrel": barrel, "flash": flash, "smoke": smoke,
		"paint": paint, "tracks": tracks, "wheels": wheels, "dust": dust}


func _wheel(parent: Node3D, pos: Vector3, rad: float, width: float, tyre: Material, rim: Material, hub: Material) -> Node3D:
	var n := M.node(parent, pos)
	var spin := M.node(n, Vector3.ZERO, Vector3.ZERO, "Spin")
	M.mi(spin, M.cyl(rad, rad, width, 22), tyre, Vector3.ZERO, Vector3(90, 0, 0))
	M.mi(spin, M.cyl(rad * 0.74, rad * 0.74, width + 0.02, 18), rim, Vector3.ZERO, Vector3(90, 0, 0))
	# hub with lugs (makes the rotation readable)
	var hubm := M.mi(spin, M.cyl(rad * 0.24, rad * 0.24, width + 0.07, 6), hub, Vector3.ZERO, Vector3(90, 0, 0))
	hubm.name = "Hub"
	return spin


func _sprocket(parent: Node3D, pos: Vector3, rad: float, steel: Material, hub: Material) -> Node3D:
	var n := M.node(parent, pos)
	var spin := M.node(n, Vector3.ZERO, Vector3.ZERO, "Spin")
	M.mi(spin, M.cyl(rad * 0.86, rad * 0.86, 0.3, 20), steel, Vector3.ZERO, Vector3(90, 0, 0))
	for k in 10:
		var a := TAU * k / 10.0
		M.mi(spin, M.box(Vector3(0.06, 0.07, 0.26), 0.01), steel, Vector3(cos(a) * rad * 0.9, sin(a) * rad * 0.9, 0), Vector3(0, 0, rad_to_deg(a)))
	M.mi(spin, M.cyl(rad * 0.3, rad * 0.3, 0.36, 8), hub, Vector3.ZERO, Vector3(90, 0, 0))
	return spin


# ================================================================ per step

func on_state(state: Dictionary, prev: Dictionary, alpha: float) -> void:
	if not _scene_ok:
		return
	_cut_props(state.get("cams", []))
	var cur: Array = state.get("p", {}).get("tanks", [])
	var old: Array = prev.get("p", {}).get("tanks", cur)
	for i in mini(cur.size(), _tanks.size()):
		var c: Array = cur[i]
		var o: Array = old[i] if i < old.size() else c
		var t: Dictionary = _tanks[i]
		var turret: Node3D = t["turret"]
		turret.rotation.y = lerp_angle(float(o[0]), float(c[0]), alpha)
		var paint: BaseMaterial3D = t["paint"]
		paint.emission_energy_multiplier = 2.2 * float(c[1])
		# track scroll and wheel spin
		for side in 2:
			var ph := lerpf(float(o[2 + side]), float(c[2 + side]), alpha)
			var tr: MeshInstance3D = t["tracks"][side]
			tr.set_instance_shader_parameter("track_phase", ph)
			for w in t["wheels"][side]:
				var spin: Node3D = w[0]
				spin.rotation.z = -ph / float(w[1])
		# dust behind the tracks while driving forward
		var spd := float(c[5]) if c.size() > 5 else 0.0
		var kd := clampf((spd - 0.6) / 3.0, 0.0, 1.0)
		for side in 2:
			var d: Node3D = t["dust"][side]
			if kd > 0.0:
				var ph := lerpf(float(o[2 + side]), float(c[2 + side]), alpha)
				Vfx.set_puff(d, fposmod(ph / 1.4 + side * 0.5, 1.0), 0.45 + 0.5 * kd, 0.4 * kd)
			else:
				d.visible = false
		# firing: muzzle flash, smoke, barrel recoil
		var age := int(c[4])
		var barrel: Node3D = t["barrel"]
		var k := exp(-float(age) / 2.5) if age < 14 else 0.0
		barrel.position.x = 0.7 - 0.2 * k
		var flame: Node3D = t["flash"]
		Vfx.set_flame(flame, [1.0, 0.6, 0.25][age] if age < 3 else 0.0)
		var smoke: Node3D = t["smoke"]
		if age < 18:
			smoke.visible = true
			Vfx.set_puff(smoke, (age + 1) / 19.0, 0.8, 0.35)
		else:
			smoke.visible = false


func spawn_dyn(kind: String, d: Dictionary) -> Node3D:
	var root := Node3D.new()
	match kind:
		"shell":
			var col := Color(1.0, 0.72, 0.15) if int(d.get("o", 0)) == 0 else Color(1.0, 0.3, 0.22)
			Vfx.glow_capsule(root, 0.5, 0.05, col, 6.0)
		"boom":
			Vfx.explosion(root)
		"dust":
			Vfx.puff(root, Color(0.8, 0.71, 0.53), int(str(d.get("id", "0")).hash()) % 97, 6)
		_:
			return null
	return root


func update_dyn(node: Node3D, a: Dictionary, b: Dictionary, alpha: float) -> void:
	if not b.has("t") or node.get_child_count() == 0:
		return
	var t := lerpf(float(a.get("t", b["t"])), float(b["t"]), alpha)
	var fx: Node3D = node.get_child(0)
	if fx.name == "Explosion":
		Vfx.set_explosion(fx, t, float(b.get("r", 1.8)), Color(0.24, 0.21, 0.18))
	else:
		Vfx.set_puff(fx, t, float(b.get("r", 0.7)), 0.85)


func chase_pose() -> Transform3D:
	if _tanks.is_empty():
		return Transform3D(Basis(), Vector3(5, 6, -5))
	var r: Node3D = _tanks[0]["root"]
	var fwd := r.global_transform.basis.x
	var p := r.global_position
	return Transform3D(Basis(), p - fwd * 7.5 + Vector3(0, 4.0, 0)).looking_at(p + fwd * 3.0 + Vector3(0, 1.0, 0), Vector3.UP)
