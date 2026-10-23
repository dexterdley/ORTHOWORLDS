extends EnvWorld
## Rescue Ambulance: a modern city grid (asphalt roads, paved blocks, textured mid-rise buildings,
## street lamps, park outskirts, a distant skyline), street obstacles (striped roadblocks, a yellow
## city bus, debris piles), the casualty site and a white ambulance with a flashing light bar.
## Python: hifi/exporters/rescue_ambulance.py. Top-down env: Box2D (x, y) at height h = Godot (x, h, -y).
## Moving ids: car (hull), wheel0..3 (wheel bodies, steer included). p: wheels [[phase, steer]],
## v, gas, brake, car [x, y, angle] (drives the per-view section caps), rear_cut, side_cut.
##
## Section views: the Rear / Side cameras put their near plane on GymBox's cut plane (3 m behind /
## 1.6 m right of the car). Everything the plane opens (ground, buildings, obstacles) gets a cap
## mesh seen only by that view, rebuilt every step from the car pose (meta "section_cap", exempt
## from the cross-view check).
## One consistent world: every other object (buildings, lamps, props, trees, the distant skyline)
## is drawn in every view whose frame contains it; the only thing removed is what the section cut
## clips in front of the car (between an elevation camera and its cut plane).

const City := preload("res://envs/rescue_ambulance/city.gd")
const Veh := preload("res://envs/rescue_ambulance/vehicles.gd")
const CAP_EPS := 0.02
const CAP_HALF := 34.0      # half width of the ground cap along the cut line (view is 40 m wide)

var _amb := {}
var _spins: Array = []
var _spin_sign: Array = []
var _site := {}
var _ambers: Array = []
var _blds: Array = []       # [x0, y0, x1, y1, h, family, tint]
var _obs: Array = []        # {"poly": PackedVector2Array, "h": float, "inv": Transform3D, "caps": {view: ArrayMesh}}
var _bcap := {}             # view -> ArrayMesh (building caps, under the registered buildings node)
var _gcap := {}             # view -> ArrayMesh (ground cross-section)
var _wall_mats: Array = []
var _soil_mats: Array = []
var _last_car := []
var _roads: Array = []      # road rects (ground cap layers)
var _blocks: Array = []     # paved block rects


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_48d_partly_cloudy_puresky", "sky_energy": 1.0,
		"sun_energy": 1.55, "sun_color": Color(1.0, 0.96, 0.9), "ambient_energy": 1.1,
		"exposure": 1.0, "white": 6.0, "ssao": true, "ssil": false, "glow": true,
		"shadow_distance": 170.0, "fog_density": 0.0011, "fog_color": Color(0.74, 0.79, 0.86),
		"fog_aerial": 0.5, "fog_sky_affect": 0.0, "saturation": 1.04, "contrast": 1.05,
		"card_ground": "leafy_grass", "card_ground_tint": Color(0.75, 0.8, 0.7), "card_ground_scale": 0.12,
	})


# ================================================================ scene

func build_scene(scene: Dictionary) -> void:
	_blds.clear()
	_obs.clear()
	_bcap.clear()
	_gcap.clear()
	_ambers.clear()
	_spins.clear()
	_spin_sign.clear()
	_last_car = []
	var blocks: Array = scene.get("blocks", [])
	_roads = scene.get("roads", [])
	_blocks = blocks
	var grass := Assets.pbr("leafy_grass", {"uv": 0.2, "world": true, "tint": Color(0.5, 0.74, 0.42), "color": Color(0.25, 0.4, 0.18)})
	var soil := Assets.pbr("excavated_soil_wall", {"uv": 0.25, "world": true, "color": Color(0.35, 0.27, 0.2)})
	var ground := Terrain.ground_slab(Vector2.ZERO, Vector2(1000.0, 1000.0), 3.0, grass, soil, 0.0)
	content.add_child(ground)
	City.roads(content, scene.get("roads", []), float(scene.get("road_r", 6.0)), float(scene.get("pitch", 42.0)))
	City.markings(content, scene.get("dashes", []), scene.get("roads", []), float(scene.get("road_r", 6.0)))
	City.blocks(content, blocks)
	_build_buildings(scene, blocks)
	City.lamps(content, blocks)
	City.props(content, blocks)
	City.trees(content, float(scene.get("city_half", 90.0)))
	City.skyline(content)      # all views: the elevation far planes reach past it (exporter)
	_build_ground_caps()
	_build_victim(scene)
	_build_obstacles(scene.get("obstacles", []))
	_build_car()


func _build_buildings(scene: Dictionary, blocks: Array) -> void:
	var node := Node3D.new()
	node.name = "Buildings"
	content.add_child(node)
	var blds: Array = scene.get("buildings", [])
	City.buildings(node, blds, blocks)
	_wall_mats.clear()
	for f in City.FAMILIES.size():
		_wall_mats.append(City.wall_material(f))
	for b in blds:
		var c := Color8(int(b[5][0]), int(b[5][1]), int(b[5][2]))
		var fam := City.family_of(c)
		_blds.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), fam, City.tint_of(c, fam)])
	for v in ["rear", "side"]:
		var mi := MeshInstance3D.new()
		mi.name = "Cap_" + v
		var am := ArrayMesh.new()
		mi.mesh = am
		node.add_child(mi)
		set_views(mi, [v])
		mi.set_meta("section_cap", true)
		_bcap[v] = am
	register("buildings", node, "building")


func _build_ground_caps() -> void:
	## Section of the ground under the cut: surface layer (asphalt / pavers / turf) over a base
	## course and soil. One surface per material, rebuilt with the cut every step.
	_soil_mats = [
		Assets.pbr("asphalt_02", {"uv": 0.3, "world": true, "tint": Color(0.5, 0.5, 0.52), "color": Color(0.2, 0.2, 0.2)}),
		Assets.pbr("concrete_floor_02", {"uv": 0.6, "world": true, "tint": Color(1.1, 1.12, 1.15), "color": Color(0.6, 0.6, 0.58)}),
		Assets.pbr("leafy_grass", {"uv": 0.6, "world": true, "tint": Color(0.4, 0.6, 0.3), "color": Color(0.2, 0.33, 0.13)}),
		Assets.pbr("gravel", {"uv": 0.7, "world": true, "tint": Color(1.2, 1.25, 1.35), "color": Color(0.45, 0.43, 0.4)}),
		Assets.pbr("excavated_soil_wall", {"uv": 0.2, "world": true, "tint": Color(0.72, 0.66, 0.6), "color": Color(0.35, 0.27, 0.2)}),
	]
	for v in ["rear", "side"]:
		var mi := MeshInstance3D.new()
		mi.name = "GroundCap_" + v
		var am := ArrayMesh.new()
		mi.mesh = am
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		content.add_child(mi)
		set_views(mi, [v])
		mi.set_meta("section_cap", true)
		_gcap[v] = am


static func _rect_poly(q: Array) -> PackedVector2Array:
	return PackedVector2Array([Vector2(q[0], q[1]), Vector2(q[2], q[1]), Vector2(q[2], q[3]), Vector2(q[0], q[3])])


func _surface_spans(c: Vector2, d: Vector2, kept: Vector2) -> Array:
	## [[t0, t1, kind]] along the cut line covering [-CAP_HALF, CAP_HALF]; kind 0 road, 1 block, 2 turf.
	var marks: Array = []
	for q in _roads:
		var tt := _cut(_rect_poly(q), c, d, kept)
		if tt.y > tt.x and tt.y > -CAP_HALF and tt.x < CAP_HALF:
			marks.append([tt.x, tt.y, 0])
	for q in _blocks:
		var tt := _cut(_rect_poly(q), c, d, kept)
		if tt.y > tt.x and tt.y > -CAP_HALF and tt.x < CAP_HALF:
			marks.append([tt.x, tt.y, 1])
	# sample boundaries, classify each piece by priority road > block > turf
	var cuts := [-CAP_HALF, CAP_HALF]
	for m in marks:
		cuts.append(clampf(m[0], -CAP_HALF, CAP_HALF))
		cuts.append(clampf(m[1], -CAP_HALF, CAP_HALF))
	cuts.sort()
	var out: Array = []
	for i in cuts.size() - 1:
		var a: float = cuts[i]
		var b: float = cuts[i + 1]
		if b - a < 1e-4:
			continue
		var mid := (a + b) * 0.5
		var kind := 2
		for m in marks:
			if mid > m[0] and mid < m[1]:
				kind = mini(kind, int(m[2]))
		if not out.is_empty() and int(out[-1][2]) == kind and absf(float(out[-1][1]) - a) < 1e-4:
			out[-1][1] = b
		else:
			out.append([a, b, kind])
	return out


func _build_victim(scene: Dictionary) -> void:
	var vic: Array = scene.get("victim", [0.0, 0.0, 1.6])
	var cd: Array = scene.get("victim_dir", [1.0, 0.0])
	_site = Veh.casualty_site(content, float(vic[2]), Vector2(float(cd[0]), float(cd[1])))
	var root: Node3D = _site["root"]
	root.position = Vector3(float(vic[0]), 0.0, -float(vic[1]))
	register("victim", root, "victim")


func _build_obstacles(obs: Array) -> void:
	for o in obs:
		var kind := String(o["kind"])
		var idx := int(o["idx"])
		var x := float(o["x"])
		var y := float(o["y"])
		var a := float(o["angle"])
		var w := float(o["w"])
		var h := float(o["h"])
		var hh := float(o["height"])
		var node := Node3D.new()
		node.name = "Obstacle%d_%s" % [idx, kind]
		content.add_child(node)
		node.transform = Transform3D(Basis(Vector3.UP, a), Vector3(x, 0.0, -y))
		var cap_mat: Material
		match kind:
			"bus":
				Veh.bus(node, w, h, hh)
				cap_mat = M.paint(Color(0.93, 0.66, 0.08), 0.7, 0.2)
			"roadblock":
				var rb := Veh.roadblock(node, w, h, hh)
				_ambers.append(rb["amber"])
				cap_mat = Assets.pbr("concrete_wall_008", {"uv": 0.8, "world": true, "color": Color(0.7, 0.7, 0.68)})
			_:
				Veh.debris(node, idx, w, h, hh)
				cap_mat = M.mat(Color(0.42, 0.36, 0.3), 0.9) if idx % 3 != 1 else Assets.pbr("weathered_planks", {"uv": 1.0, "world": true, "color": Color(0.45, 0.33, 0.2)})
		var caps := {}
		for v in ["rear", "side"]:
			var mi := MeshInstance3D.new()
			mi.name = "Cap_" + v
			var am := ArrayMesh.new()
			mi.mesh = am
			mi.material_override = cap_mat
			node.add_child(mi)
			set_views(mi, [v])
			mi.set_meta("section_cap", true)
			caps[v] = am
		var ca := cos(a)
		var sa := sin(a)
		var poly := PackedVector2Array()
		for l in [Vector2(-w, -h), Vector2(w, -h), Vector2(w, h), Vector2(-w, h)]:
			var lx: float = l.x * 0.5
			var ly: float = l.y * 0.5
			poly.append(Vector2(x + lx * ca - ly * sa, y + lx * sa + ly * ca))
		_obs.append({"poly": poly, "h": hh, "inv": node.transform.affine_inverse(), "caps": caps})
		register("obs%d" % idx, node, kind)


func _build_car() -> void:
	_amb = Veh.ambulance(content)
	register("car", _amb["root"], "car")
	for i in 4:
		var wn := Node3D.new()
		wn.name = "Wheel%d" % i
		content.add_child(wn)
		var left := i % 2 == 0     # car_dynamics order: FL, FR, RL, RR (x < 0 = left)
		_spins.append(Veh.ambulance_wheel(wn, left))
		_spin_sign.append(1.0 if left else -1.0)
		register("wheel%d" % i, wn, "car")


# ================================================================ per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var wl: Array = p.get("wheels", [])
	for i in mini(wl.size(), _spins.size()):
		(_spins[i] as Node3D).rotation = Vector3(_spin_sign[i] * float(wl[i][0]), 0.0, 0.0)
	# Light bar: two blue groups alternating (3 Hz double flash), from sim time
	var ph := fposmod(sim_t * 2.0, 1.0)
	var a_on := (ph < 0.12) or (ph > 0.2 and ph < 0.32)
	var b_on := (ph > 0.5 and ph < 0.62) or (ph > 0.7 and ph < 0.82)
	if not _amb.is_empty():
		(_amb["la"] as StandardMaterial3D).emission_energy_multiplier = 9.0 if a_on else 0.35
		(_amb["lb"] as StandardMaterial3D).emission_energy_multiplier = 9.0 if b_on else 0.35
		(_amb["ga"] as OmniLight3D).light_energy = 2.5 if a_on else 0.0
		(_amb["gb"] as OmniLight3D).light_energy = 2.5 if b_on else 0.0
		(_amb["brake"] as StandardMaterial3D).emission_energy_multiplier = 5.0 if float(p.get("brake", 0.0)) > 0.05 else 0.7
	if not _site.is_empty():
		var bon := fposmod(sim_t * 1.25, 1.0) < 0.45
		(_site["beacon"] as StandardMaterial3D).emission_energy_multiplier = 8.0 if bon else 0.6
		(_site["light"] as OmniLight3D).light_energy = 2.0 if bon else 0.0
	var amber_on := fposmod(sim_t * 1.0 + 0.3, 1.0) < 0.5
	for m in _ambers:
		(m as StandardMaterial3D).emission_energy_multiplier = 6.0 if amber_on else 0.4
	var car: Array = p.get("car", [])
	if car.size() == 3 and car != _last_car:
		_last_car = car.duplicate()
		_update_caps(float(car[0]), float(car[1]), float(car[2]), float(p.get("rear_cut", 3.0)), float(p.get("side_cut", 1.6)))


# ================================================================ section caps

func _update_caps(x: float, y: float, a: float, rear_cut: float, side_cut: float) -> void:
	var f := Vector2(-sin(a), cos(a))
	var r := Vector2(cos(a), sin(a))
	var c := Vector2(x, y)
	# Rear: cut plane rear_cut behind the car, kept side +f, line along r
	_build_caps("rear", c - f * (rear_cut - CAP_EPS), r, f)
	# Side: cut plane side_cut right of the car, kept side -r, line along f
	_build_caps("side", c + r * (side_cut - CAP_EPS), f, -r)


static func _cut(poly: PackedVector2Array, c: Vector2, d: Vector2, kept: Vector2) -> Vector2:
	## Parameter range (t0, t1) along c + t d where the cut line crosses the convex polygon (t0 > t1: none).
	var t0 := INF
	var t1 := -INF
	var n := poly.size()
	for i in n:
		var p := poly[i]
		var q := poly[(i + 1) % n]
		var sp := (p - c).dot(kept)
		var sq := (q - c).dot(kept)
		if (sp > 0.0) == (sq > 0.0):
			continue
		var pt := p + (q - p) * (sp / (sp - sq))
		var t := (pt - c).dot(d)
		t0 = minf(t0, t)
		t1 = maxf(t1, t)
	return Vector2(t0, t1)


static func _cap_quad(st: SurfaceTool, c: Vector2, d: Vector2, t0: float, t1: float, z0: float, z1: float, nrm: Vector3, col: Color, xf := Transform3D()) -> void:
	var a := c + d * t0
	var b := c + d * t1
	var n := (xf.basis * nrm).normalized()
	City.quad(st, xf * Vector3(a.x, z0, -a.y), xf * Vector3(b.x, z0, -b.y), xf * Vector3(b.x, z1, -b.y), xf * Vector3(a.x, z1, -a.y), n, col)


func _build_caps(view: String, c: Vector2, d: Vector2, kept: Vector2) -> void:
	var nrm := Vector3(-kept.x, 0.0, kept.y)     # facing the camera (Godot)
	# Ground cross-section: asphalt / gravel / soil layers below the cut
	var gm: ArrayMesh = _gcap.get(view)
	if gm != null:
		gm.clear_surfaces()
		var sts: Array = []
		var used: Array = []
		for k in 5:
			sts.append(City.new_st())
			used.append(false)
		# kind -> [surface material, top, bottom of the surface layer, bottom of the base course]
		var lay := {0: [0, 0.015, -0.22, -0.6], 1: [1, City.WALK_Y, -0.18, -0.55], 2: [2, 0.0, -0.12, -0.12]}
		for sp in _surface_spans(c, d, kept):
			var L: Array = lay[int(sp[2])]
			_cap_quad(sts[L[0]], c, d, sp[0], sp[1], L[2], L[1], nrm, Color.WHITE)
			used[L[0]] = true
			if L[3] < L[2]:
				_cap_quad(sts[3], c, d, sp[0], sp[1], L[3], L[2], nrm, Color.WHITE)
				used[3] = true
			_cap_quad(sts[4], c, d, sp[0], sp[1], -14.0, L[3], nrm, Color.WHITE)
			used[4] = true
		for k in 5:
			if used[k]:
				(sts[k] as SurfaceTool).commit(gm)
				gm.surface_set_material(gm.get_surface_count() - 1, _soil_mats[k])
	# Buildings, one surface per facade family
	var bm: ArrayMesh = _bcap.get(view)
	if bm != null:
		bm.clear_surfaces()
		var sts: Array = []
		var used: Array = []
		for f in City.FAMILIES.size():
			sts.append(City.new_st())
			used.append(false)
		for b in _blds:
			var poly := PackedVector2Array([Vector2(b[0], b[1]), Vector2(b[2], b[1]), Vector2(b[2], b[3]), Vector2(b[0], b[3])])
			var tt := _cut(poly, c, d, kept)
			if tt.y - tt.x > 0.005:
				_cap_quad(sts[b[5]], c, d, tt.x, tt.y, 0.0, float(b[4]), nrm, (b[6] as Color) * 0.92)
				used[b[5]] = true
		for f in sts.size():
			if used[f]:
				(sts[f] as SurfaceTool).commit(bm)
				bm.surface_set_material(bm.get_surface_count() - 1, _wall_mats[f])
	# Obstacles (caps live in each obstacle's local frame)
	for o in _obs:
		var om: ArrayMesh = o["caps"].get(view)
		if om == null:
			continue
		om.clear_surfaces()
		var tt := _cut(o["poly"], c, d, kept)
		if tt.y - tt.x > 0.005:
			var st := City.new_st()
			_cap_quad(st, c, d, tt.x, tt.y, 0.0, float(o["h"]), nrm, Color.WHITE, o["inv"])
			st.commit(om)


func chase_pose() -> Transform3D:
	var car: Node3D = _amb.get("root")
	if car == null:
		return super.chase_pose()
	var t := car.global_transform
	var p := t.origin
	return Transform3D(Basis(), p + t.basis.z * 10.0 + Vector3(0, 4.5, 0)).looking_at(p + Vector3(0, 1.2, 0), Vector3.UP)
