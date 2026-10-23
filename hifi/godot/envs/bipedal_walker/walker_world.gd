extends EnvWorld
## Shared world of the BipedalWalker family: Bipedal Walker (envs/bipedal_walker/world.gd) and
## Parkour Runner (envs/parkour_runner/world.gd) extend this script and add their look.
## Python: hifi/exporters/bipedal_walker.py / parkour_runner.py. Side-view env: Box2D (x, y) = Godot
## (x, y), z = depth. Moving ids: hull, leg0..leg3 (class "walker"); p: flag (0 orange, 1 green,
## 2 red), hx, hy, scroll, contact[2].
##
## Terrain: the GymBox terrain is the Box2D profile extruded over z +-10. Here that "lane" band
## follows the profile exactly (and is split into pieces where Parkour tags segments with an
## obstacle: crest / pit), with a soil cross-section face at z = +10 that the Side camera sees like
## GymBox's slab. Behind (z < -10) a displaced back band carries the scenery; in front (z > 10) a
## dropping front band; distant ridges on both sides.
## Cross-view consistency: one world, every object is drawn in every view whose frame contains it.
## Cutaways (things between a camera and the walker) are the only exceptions:
##   Side  the front group (z > LANE) is hidden there (FRONT_VIEWS): it lies behind / at the Side
##         camera (z = 10.6, near plane z = 10.4) and would only shadow the soil cross-section
##   Rear  the camera's near plane is a section cut REAR_CUT behind the hull (see the exporter),
##         capped with a cross-section (RearCap, meta "section_cap")
## Ortho far planes reach all in-frame scenery (Side: the back band to BACK_Z; the core sky card at
## 0.97 far sits behind it). Ortho skies come from that card (Look "horizon_y" = SKY_HORIZON).

const Robot := preload("res://envs/bipedal_walker/robot.gd")
const Strip := preload("res://envs/bipedal_walker/strip.gd")

const LANE := 10.0             # GymBox terrain extrusion |z| <= 10: exact Box2D profile
const NEAR := 18.0             # near bands LANE..NEAR (split per obstacle like the lane)
const BACK_Z := -160.0         # back band (all views) z range: BACK_Z .. -LANE
const FRONT_Z := 160.0         # front band: LANE .. FRONT_Z
const RIDGE_Z := 460.0         # distant ridges out to +-RIDGE_Z
const BOTTOM := -2.0           # cross-sections go down to here (below the frame bottom y = 0)
const SKY_HORIZON := 3.4       # Look "horizon_y": ground height the core ortho sky card sits on
## Views of the front group (z > LANE): all but Side, where it is the cutaway in front of the walker
const FRONT_VIEWS := ["top", "rear", "fpv", "chase"]

var prof                       # Strip.Profile: exact Box2D profile + extensions
var xs := PackedFloat32Array()
var ys := PackedFloat32Array()
var course := Vector2(0.0, 92.87)
var pieces: Array = []         # lane pieces: {x0, x1, tag, caps: [MeshInstance3D, ...]}
var box_caps: Array = []       # obstacle cross-sections for the rear cut: {x0, x1, node}
var obs_nodes := {}            # obstacle index -> Node3D (registered "obs<i>")
var obs_kind := {}             # obstacle index -> kind
var hull_node: Node3D
var legs: Array = []
var flag_root: Node3D
var flag_cloth: MeshInstance3D
var flag_mat: StandardMaterial3D
var view_w := 20.0
var view_h := 13.333           # ortho frame height: the Side frame spans y = 0 .. view_h
var rear_cut := 1.8
var noise := FastNoiseLite.new()
var noise_low := FastNoiseLite.new()
var _cap_layers := {}          # kind -> [[thickness, world-triplanar material], ...] top to bottom


# ================================================================ hooks (override in the env worlds)

func robot_kind() -> String:
	return "bipedal"


func with_head() -> bool:
	return false


func lane_material(_tag_kind: String) -> Material:
	## Top surface of the lane band; tag_kind "" for plain ground, else the obstacle kind owning it.
	return M.mat(Color(0.3, 0.42, 0.2), 0.9)


func section_layers() -> Array:
	## Cross-section strata at z = +10 and on the rear cut, top to bottom: [[thickness, material], ...];
	## the last layer runs down to BOTTOM.
	return [[1.0, M.mat(Color(0.35, 0.25, 0.18), 0.95)]]


func section_layers_for(_tag_kind: String) -> Array:
	## Strata under a lane piece owned by an obstacle kind ("" = plain ground).
	return section_layers()


func cap_layers(kind: String) -> Array:
	if not _cap_layers.has(kind):
		var out: Array = []
		for l in section_layers_for(kind):
			out.append([float(l[0]), world_triplanar(l[1])])
		_cap_layers[kind] = out
	return _cap_layers[kind]


func band_material() -> Material:
	return lane_material("")


func near_band_material(_tag_kind: String) -> Material:
	## Ground of the near bands (|z| in LANE..NEAR) where an obstacle owns the x range (e.g. the
	## tapering ends of a Parkour crest); "" = plain.
	return band_material()


func band_tag(_xm: float) -> int:
	## Obstacle owning the near-band segment centred at x = xm, or -1.
	return -1


func lane_piece_extra(_parent: Node3D, _px: PackedFloat32Array, _tag: int) -> void:
	## Extra geometry for a lane piece (e.g. Parkour pit end walls).
	pass


func band_rough() -> Vector2:
	## (rise of the back band, drop of the front band) in metres
	return Vector2(0.9, 1.2)


func band_height(x: float, z: float) -> float:
	## Ground height outside the lane: the Box2D profile plus displacement growing away from it.
	var base: float = prof.h(x)
	var dist := absf(z) - LANE
	if dist <= 0.0:
		return base
	var ramp := smoothstep(0.0, 3.0, dist)
	var n := 0.5 + 0.5 * noise.get_noise_2d(x, z)
	var r := band_rough()
	var ridge := 38.0 * smoothstep(150.0, 320.0, absf(z)) * (0.55 + 0.45 * noise_low.get_noise_2d(x, z))
	if z < 0.0:
		return base + ramp * r.x * n + ridge
	return base - ramp * r.y * n + ridge * 0.8


func prepare(_scene: Dictionary) -> void:
	## Called once the profile exists, before any terrain is built (e.g. derived profiles).
	pass


func build_obstacles(_scene: Dictionary) -> void:
	pass


func build_scenery(_scene: Dictionary) -> void:
	pass


# ================================================================ build

func build_scene(scene: Dictionary) -> void:
	noise.seed = 11
	noise.frequency = 0.06
	noise_low.seed = 5
	noise_low.frequency = 0.008
	view_w = float(scene.get("W", 20.0))
	view_h = float(scene.get("H", 13.333))
	rear_cut = float(scene.get("rear_cut", 1.8))
	var tx: Array = scene.get("tx", [])
	var ty: Array = scene.get("ty", [])
	if tx.size() < 2:
		tx = [0.0, 92.87]
		ty = [3.33, 3.33]
	course = Vector2(float(tx[0]), float(tx[-1]))
	var ext := Strip.extend(tx, ty, 30.0)
	xs = ext[0]
	ys = ext[1]
	prof = Strip.Profile.new(xs, ys)
	pieces.clear()
	box_caps.clear()
	obs_nodes.clear()
	obs_kind.clear()
	legs.clear()
	for ob in scene.get("obstacles", []):
		obs_kind[int(ob["i"])] = str(ob["kind"])
	prepare(scene)
	_build_lane(scene.get("tags", []), tx.size())
	_build_bands()
	build_obstacles(scene)
	build_scenery(scene)
	for i in obs_nodes:
		register("obs%d" % i, obs_nodes[i], obs_kind.get(i, "obstacle"))
	_build_robot()
	_build_flag(scene)


func obs_node(i: int) -> Node3D:
	if not obs_nodes.has(i):
		var n := Node3D.new()
		n.name = "Obstacle%d" % i
		content.add_child(n)
		obs_nodes[i] = n
	return obs_nodes[i]


static func world_triplanar(m: Material) -> Material:
	## Same material with world-space triplanar UVs (for meshes that are scaled every step).
	if m is BaseMaterial3D:
		var d: BaseMaterial3D = m.duplicate()
		d.uv1_triplanar = true
		d.uv1_world_triplanar = true
		return d
	return m


# ---------------------------------------------------------------- terrain

func _build_lane(tags: Array, n_course: int) -> void:
	# Index of the first course point inside xs (left extension points come first)
	var off := 0
	while off < xs.size() and xs[off] < course.x - 1e-4:
		off += 1
	var seg_tag := func(k: int) -> int:
		var i := k - off
		return int(tags[i]) if (i >= 0 and i < tags.size() and i < n_course - 1) else -1
	var zs := Strip.span(-LANE, LANE, 2.5)
	var plain := Node3D.new()
	plain.name = "Lane"
	content.add_child(plain)
	var hf := func(x: float, _z: float) -> float: return prof.h(x)
	var k0 := 0
	var nseg := xs.size() - 1
	while k0 < nseg:
		var t: int = seg_tag.call(k0)
		var k1 := k0
		while k1 + 1 < nseg and int(seg_tag.call(k1 + 1)) == t:
			k1 += 1
		var px := xs.slice(k0, k1 + 2)
		var parent: Node3D = plain if t < 0 else obs_node(t)
		var kind: String = "" if t < 0 else String(obs_kind.get(t, ""))
		var top := M.mi(parent, Strip.field(px, zs, hf), lane_material(kind))
		top.name = "LaneTop"
		_build_section(parent, px, LANE, 1.0, hf, kind)
		lane_piece_extra(parent, px, t)
		pieces.append({"x0": px[0], "x1": px[-1], "tag": t, "kind": kind, "caps": _make_cap(parent, kind)})
		k0 = k1 + 1


func _build_section(parent: Node3D, px: PackedFloat32Array, z: float, facing: float, hf: Callable, kind := "") -> void:
	var layers := section_layers_for(kind)
	var depth := 0.0
	for li in layers.size():
		var l: Array = layers[li]
		var top_d := depth
		var last := li == layers.size() - 1
		depth += float(l[0])
		var d0 := top_d
		var d1 := depth
		var top_f := func(x: float, zz: float) -> float: return float(hf.call(x, zz)) - d0
		var bot = BOTTOM
		if not last:
			bot = func(x: float, zz: float) -> float: return float(hf.call(x, zz)) - d1
		var w := M.mi(parent, Strip.wall(px, z, top_f, bot, facing), l[1])
		w.name = "Section%d" % li


func _make_cap(parent: Node3D, kind := "") -> Array:
	## Rear-view cross-section boxes of a lane piece (one per stratum), parked until the cut is inside.
	var out: Array = []
	for l in cap_layers(kind):
		var mi := M.mi(parent, M.box(Vector3(0.02, 1.0, 1.0)), l[1], Vector3(0, -1000, 0))
		mi.name = "RearCap"
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		mi.set_meta("section_cap", true)
		set_views(mi, ["rear"])
		out.append(mi)
	return out


func add_box_cap(parent: Node3D, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, mat: Material) -> void:
	## Rear-view cross-section of an obstacle box (shown while the rear cut passes through it).
	var mi := M.mi(parent, M.box(Vector3(0.02, 1.0, 1.0)), world_triplanar(mat), Vector3(0, -1000, 0))
	mi.name = "RearCap"
	mi.scale = Vector3(1.0, maxf(y1 - y0, 0.01), maxf(z1 - z0, 0.01))
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mi.set_meta("cap_y", (y0 + y1) * 0.5)
	mi.set_meta("cap_z", (z0 + z1) * 0.5)
	mi.set_meta("section_cap", true)
	set_views(mi, ["rear"])
	box_caps.append({"x0": x0, "x1": x1, "node": mi})


func _build_bands() -> void:
	var hf := func(x: float, z: float) -> float: return band_height(x, z)
	# Near bands (|z| in LANE..NEAR): split where an obstacle owns the x range
	var nb := Strip.span(-NEAR, -LANE, 1.0)
	var nf := Strip.span(LANE, NEAR, 1.0)
	var k0 := 0
	var nseg := xs.size() - 1
	while k0 < nseg:
		var t := band_tag(0.5 * (xs[k0] + xs[k0 + 1]))
		var k1 := k0
		while k1 + 1 < nseg and band_tag(0.5 * (xs[k1 + 1] + xs[k1 + 2])) == t:
			k1 += 1
		var px := xs.slice(k0, k1 + 2)
		var mat := near_band_material("" if t < 0 else String(obs_kind.get(t, "")))
		var b := M.mi(content, Strip.field(px, nb, hf), mat)
		b.name = "NearBack"
		var f := M.mi(content, Strip.field(px, nf, hf), mat)
		f.name = "NearFront"
		set_views(f, FRONT_VIEWS)
		k0 = k1 + 1
	# Far bands
	var far_back := Strip.span(BACK_Z, -40.0, 6.0)
	far_back.remove_at(far_back.size() - 1)
	far_back.append_array(Strip.span(-40.0, -NEAR, 2.0))
	var back := M.mi(content, Strip.field(xs, far_back, hf), band_material())
	back.name = "BackBand"
	var front_z := Strip.span(NEAR, 40.0, 2.0)
	front_z.remove_at(front_z.size() - 1)
	front_z.append_array(Strip.span(40.0, FRONT_Z, 6.0))
	var front := M.mi(content, Strip.field(xs, front_z, hf), band_material())
	front.name = "FrontBand"
	set_views(front, FRONT_VIEWS)
	# distant ridges: coarse. The back ridge starts at BACK_Z, just inside the Side view's sky card,
	# so that view shows its foot; the rest stands behind the card
	var rx := PackedFloat32Array()
	var x := xs[0]
	while x <= xs[-1]:
		rx.append(x)
		x += 10.0
	var far_m := world_triplanar(band_material())
	var r1 := M.mi(content, Strip.field(rx, Strip.span(-RIDGE_Z, BACK_Z, 20.0), hf), far_m)
	var r2 := M.mi(content, Strip.field(rx, Strip.span(FRONT_Z, RIDGE_Z, 20.0), hf), far_m)
	for r in [r1, r2]:
		r.name = "Ridge"
		r.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	set_views(r2, FRONT_VIEWS)


func plants_split(parent: Node3D, kind: String, xforms: Array, seed: int, height: float, col: Color, vis_end := 0.0) -> void:
	## Flora.plants() (envs/bipedal_walker/flora.gd) in a back group (z <= 0: every view) and a front
	## group (z > 0: FRONT_VIEWS, hidden from the Side view as the cutaway in front of the walker).
	var flora := preload("res://envs/bipedal_walker/flora.gd")
	var back: Array = []
	var front: Array = []
	for t in xforms:
		(front if (t as Transform3D).origin.z > 0.0 else back).append(t)
	flora.plants(parent, kind, back, seed, height, col, vis_end)
	for mmi in flora.plants(parent, kind, front, seed, height, col, vis_end):
		set_views(mmi, FRONT_VIEWS)


func ground(x: float, z: float) -> float:
	## Ground height anywhere (lane: exact profile).
	return prof.h(x) if absf(z) <= LANE else band_height(x, z)


# ---------------------------------------------------------------- robot, flag

func _build_robot() -> void:
	var pal := Robot.palette(robot_kind())
	hull_node = Robot.build_hull(content, pal, with_head())
	register("hull", hull_node, "walker")
	# legs[0] thigh / legs[1] shin are the i = -1 pair (z +1.3), legs[2] / legs[3] the i = +1 pair (z -1.3)
	for i in 4:
		var side := i / 2
		var n: Node3D = Robot.build_thigh(content, pal, side) if i % 2 == 0 else Robot.build_shin(content, pal, side)
		n.position = Vector3(5.0, 4.0, 1.3 if side == 0 else -1.3)
		register("leg%d" % i, n, "walker")
		legs.append(n)
	hull_node.position = Vector3(4.67, 5.6, 0)


func _build_flag(scene: Dictionary) -> void:
	## GymBox start flag: pole x 1.4..1.433, y 3.33..5.0; pennant (x, y2) (x, y2 - 0.33) (x + 0.83, y2 - 0.17)
	var f: Array = scene.get("flag", [1.4, 3.333, 5.0])
	var x := float(f[0])
	var y1 := float(f[1])
	var y2 := float(f[2])
	var gy: float = prof.h(x + 0.0167)
	flag_root = M.node(content, Vector3(x + 0.0167, gy, 0))
	flag_root.name = "Flag"
	var steel := M.mat(Color(0.78, 0.79, 0.8), 0.25, 1.0)
	var dark := M.mat(Color(0.06, 0.06, 0.065), 0.5, 0.4)
	var top := y2 - gy
	M.mi(flag_root, M.cyl(0.02, 0.024, top + 0.05, 10), steel, Vector3(0, (top + 0.05) * 0.5, 0))
	M.mi(flag_root, M.sphere(0.035, 12, 6), steel, Vector3(0, top + 0.07, 0))
	M.mi(flag_root, M.cyl(0.07, 0.09, 0.08, 14), dark, Vector3(0, 0.03, 0))
	flag_mat = M.mat(Color8(230, 51, 0), 0.85, 0.0, {"cull": false})
	flag_cloth = M.mi(flag_root, _pennant(0.8167, 0.3333, y2 - gy), flag_mat)
	flag_cloth.position = Vector3(0.0167, 0, 0)


static func _pennant(length: float, height: float, top: float) -> ArrayMesh:
	## Triangular pennant from the pole (local x = 0) with a baked wave along z.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nu := 12
	var nv := 4
	for i in nu + 1:
		var u := float(i) / nu
		var hi := top - u * height * 0.5
		var lo := top - height + u * height * 0.5
		var z := 0.09 * sin(u * PI * 1.6) * (0.25 + u)
		for j in nv + 1:
			var v := float(j) / nv
			st.set_uv(Vector2(u, v))
			st.add_vertex(Vector3(u * length, lerpf(lo, hi, v), z))
	for i in nu:
		for j in nv:
			var a := i * (nv + 1) + j
			var b := a + nv + 1
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(b)
			st.add_index(a)
			st.add_index(a + 1)
			st.add_index(b + 1)
	st.generate_normals()
	return st.commit()


# ================================================================ per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var hx := float(p.get("hx", hull_node.position.x if hull_node else 4.67))
	_update_rear_cut(hx - rear_cut)
	if flag_mat:
		var k := int(p.get("flag", 0))
		flag_mat.albedo_color = [Color8(230, 51, 0), Color8(50, 205, 50), Color8(200, 30, 30)][clampi(k, 0, 2)]
		flag_cloth.rotation = Vector3(0, 0.12 * sin(sim_t * 1.7), 0.03 * sin(sim_t * 2.3))


func _update_rear_cut(xc: float) -> void:
	var hc: float = prof.h(xc)
	var xcap := xc + 0.012
	for pc in pieces:
		var caps: Array = pc["caps"]
		var layers := cap_layers(String(pc["kind"]))
		var inside: bool = xc >= float(pc["x0"]) and xc < float(pc["x1"])
		var depth := 0.0
		for li in caps.size():
			var mi: MeshInstance3D = caps[li]
			if not inside:
				mi.position = Vector3(0, -1000, 0)
				continue
			var thick: float = float(layers[li][0])
			var y_top := hc - depth
			var y_bot := BOTTOM if li == caps.size() - 1 else maxf(hc - depth - thick, BOTTOM)
			depth += thick
			mi.scale = Vector3(1.0, maxf(y_top - y_bot, 0.005), 2.0 * LANE + 2.0)
			mi.position = Vector3(xcap, (y_top + y_bot) * 0.5, 0)
	for bc in box_caps:
		var mi: MeshInstance3D = bc["node"]
		if xc >= float(bc["x0"]) and xc <= float(bc["x1"]):
			mi.position = Vector3(xcap + 0.004, float(mi.get_meta("cap_y")), float(mi.get_meta("cap_z")))
		else:
			mi.position = Vector3(0, -1000, 0)


func chase_pose() -> Transform3D:
	var p := hull_node.global_position if hull_node else Vector3(5, 5, 0)
	return Transform3D(Basis(), p + Vector3(-3.5, 1.6, 6.0)).looking_at(p + Vector3(0.6, -0.4, 0), Vector3.UP)
