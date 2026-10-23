extends EnvWorld
## Drone Dogfight: two ducted racing quadcopters over an outdoor drone-racing arena (mowed turf,
## padded perimeter walls, safety netting on steel poles, floodlight masts, pit tents).
## Python: hifi/exporters/drone_dogfight.py. Top-down env: Box2D (x, y) at height h = Godot (x, h, -y).
## Moving ids: player, opponent (alive flags in p.alive; velocities p.pv / p.ov drive a cosmetic
## bank). dyn: "bolt" (glowing tracer, class "bolt"), "explosion" (phase t, radius r).

const ALT := 5.0
# Box2D arm boxes (centre, angle) of DroneDogfightEnv._make_drone; rotors sit at the outer arm tips
const ARMS := [[0.45, 0.30, 0.785], [-0.45, 0.30, -0.785], [0.45, -0.30, -0.785], [-0.45, -0.30, 0.785]]
const TIP := Vector2(0.697, 0.547)
const DUCT_R := 0.42

var _drones: Array = []
var _tilts: Array = []
var _prop_mat: ShaderMaterial
var _net_mat: ShaderMaterial


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_48d_partly_cloudy_puresky", "card_ground": "leafy_grass", "card_ground_tint": Color(0.6, 0.75, 0.45), "sun_energy": 1.7, "sun_color": Color(1.0, 0.97, 0.92),
		"ambient_energy": 0.85, "exposure": 1.0, "white": 6.0, "glow": true, "ssao": true, "ssil": true,
		"fog_density": 0.0012, "fog_color": Color(0.78, 0.84, 0.92), "fog_aerial": 0.5, "fog_sky_affect": 0.0,
		"saturation": 1.06, "contrast": 1.05, "shadow_distance": 90.0,
	})


func build_scene(scene: Dictionary) -> void:
	var W: float = scene.get("W", 25.0)
	var H: float = scene.get("H", 20.0)
	var wt: float = scene.get("wall_t", 0.5)
	var wh: float = scene.get("wall_h", 3.0)
	_build_ground(W, H)
	_build_markings(W, H)
	_build_walls(W, H, wt, wh)
	_build_nets(W, H, wh)
	_build_surroundings(W, H)
	_drones.clear()
	_tilts.clear()
	for i in 2:
		var d := _build_drone(i == 0)
		register(["player", "opponent"][i], d, ["player", "opponent"][i])
		_drones.append(d)
		_tilts.append(d.get_node("Tilt"))


# ================================================================ arena

func _build_ground(W: float, H: float) -> void:
	# Surrounding field and a concrete apron around the arena
	var grass := Assets.pbr("leafy_grass", {"uv": 0.25, "world": true, "tint": Color(0.62, 0.74, 0.44), "color": Color(0.3, 0.42, 0.2)})
	var soil := Assets.pbr("dirt", {"uv": 0.5, "color": Color(0.3, 0.24, 0.18)})
	content.add_child(Terrain.ground_slab(Vector2(W * 0.5, H * 0.5), Vector2(260, 260), 1.0, grass, soil, -0.03))
	var apron := Assets.pbr("concrete_floor_worn_001", {"uv": 0.3, "world": true, "tint": Color(0.86, 0.86, 0.84), "color": Color(0.55, 0.55, 0.53)})
	content.add_child(Terrain.ground_slab(Vector2(W * 0.5, H * 0.5), Vector2(W + 9.0, H + 9.0), 0.2, apron, apron, 0.0))
	# Arena turf: 2 m mowing checker aligned with GymBox's floor tiles (light where (i + j) is even)
	var light := Assets.pbr("leafy_grass", {"uv": 0.35, "world": true, "tint": Color(0.58, 0.86, 0.4), "color": Color(0.32, 0.6, 0.28)})
	var dark := Assets.pbr("leafy_grass", {"uv": 0.35, "world": true, "tint": Color(0.42, 0.68, 0.3), "color": Color(0.25, 0.5, 0.22)})
	for parity in 2:
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var nx := int(ceil(W / 2.0))
		var ny := int(ceil(H / 2.0))
		for i in nx:
			for j in ny:
				if (i + j) % 2 != parity:
					continue
				var x0 := i * 2.0
				var x1 := minf(x0 + 2.0, W)
				var y0 := j * 2.0
				var y1 := minf(y0 + 2.0, H)
				M._quad_uv(st, [Vector3(x0, 0.006, -y0), Vector3(x1, 0.006, -y0), Vector3(x1, 0.006, -y1), Vector3(x0, 0.006, -y1)],
					Vector3.UP, [Vector2(x0, y0), Vector2(x1, y0), Vector2(x1, y1), Vector2(x0, y1)])
		var mi := MeshInstance3D.new()
		mi.name = "Turf%d" % parity
		mi.mesh = st.commit()
		mi.material_override = light if parity == 0 else dark
		content.add_child(mi)


func _build_markings(W: float, H: float) -> void:
	var paint := M.mat(Color(0.93, 0.93, 0.9), 0.75, 0.0)
	var y := 0.012
	# Centre line and circle
	M.mi(content, M.box(Vector3(0.12, 0.004, H - 1.2)), paint, Vector3(W * 0.5, y, -H * 0.5))
	M.mi(content, M.lathe(PackedVector2Array([Vector2(3.06, y), Vector2(2.94, y)]), 64), paint, Vector3(W * 0.5, 0, -H * 0.5))
	# Home pads (spawn sides): square outline + circle
	for px in [4.0, W - 4.0]:
		var pad := M.node(content, Vector3(px, 0, -H * 0.5))
		for s in [-1.0, 1.0]:
			M.mi(pad, M.box(Vector3(2.4, 0.004, 0.1)), paint, Vector3(0, y, s * 1.15))
			M.mi(pad, M.box(Vector3(0.1, 0.004, 2.4)), paint, Vector3(s * 1.15, y, 0))
		M.mi(pad, M.lathe(PackedVector2Array([Vector2(0.85, y), Vector2(0.75, y)]), 40), paint)


func _build_walls(W: float, H: float, wt: float, wh: float) -> void:
	## Box2D boundary boxes (0.5 m thick) as padded crash walls, 3 m tall (GymBox render height).
	var pad := Assets.pbr("corrugated_iron", {"uv": 0.5, "tint": Color(0.74, 0.8, 0.92), "color": Color(0.55, 0.6, 0.68), "metallic": 0.6})
	var cap := M.mat(Color(0.9, 0.91, 0.93), 0.3, 0.7)
	var base := M.mat(Color(0.12, 0.12, 0.13), 0.7, 0.2)
	var led := M.emissive(Color(0.15, 0.85, 1.0), 2.5)
	# South / north span the full width; west / east only the inner length (no overlapping corners)
	var defs := [[W * 0.5, wt * 0.5, W, wt, Vector2(0, 1)], [W * 0.5, H - wt * 0.5, W, wt, Vector2(0, -1)],
		[wt * 0.5, H * 0.5, wt, H - 2.0 * wt, Vector2(1, 0)], [W - wt * 0.5, H * 0.5, wt, H - 2.0 * wt, Vector2(-1, 0)]]
	for d in defs:
		var sx: float = d[2]
		var sy: float = d[3]
		var inward: Vector2 = d[4]
		var root := M.node(content, Vector3(d[0], 0, -d[1]))
		M.mi(root, M.box(Vector3(sx, 0.25, sy), 0.01), base, Vector3(0, 0.125, 0))
		M.mi(root, M.box(Vector3(sx - 0.02, wh - 0.35, sy - 0.02), 0.05), pad, Vector3(0, 0.25 + (wh - 0.35) * 0.5, 0))
		M.mi(root, M.box(Vector3(sx + 0.04, 0.1, sy + 0.04), 0.02), cap, Vector3(0, wh - 0.05, 0))
		# LED strip on the inner face
		var len_ := maxf(sx, sy) - 0.6
		var off := Vector3(inward.x * (sx * 0.5 + 0.005), wh - 0.32, -inward.y * (sy * 0.5 + 0.005))
		var strip_size := Vector3(len_, 0.06, 0.02) if absf(inward.y) > 0.5 else Vector3(0.02, 0.06, len_)
		var s := M.mi(root, M.box(strip_size), led, off)
		s.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


func _net_material() -> ShaderMaterial:
	if _net_mat == null:
		var sh := Shader.new()
		sh.code = """
shader_type spatial;
render_mode blend_mix, cull_disabled, depth_draw_never, shadows_disabled;
uniform vec4 net_color : source_color = vec4(0.06, 0.065, 0.07, 1.0);
uniform float cell = 0.1;
uniform float line_w = 0.008;
void fragment() {
	vec2 g = UV / cell;
	vec2 fw = max(fwidth(g), vec2(1e-4));
	vec2 d = abs(fract(g - 0.5) - 0.5);
	float lw = line_w / cell;
	vec2 cov = clamp((0.5 * lw - d) / fw + 0.5, 0.0, 1.0);
	float a = max(cov.x, cov.y);
	float avg = 1.0 - (1.0 - lw) * (1.0 - lw);
	a = mix(a, avg * 1.1, smoothstep(0.35, 1.0, max(fw.x, fw.y)));
	ALBEDO = net_color.rgb;
	ROUGHNESS = 0.9;
	ALPHA = clamp(a, 0.0, 1.0) * net_color.a;
}
"""
		_net_mat = ShaderMaterial.new()
		_net_mat.shader = sh
	return _net_mat


func _build_nets(W: float, H: float, wh: float) -> void:
	## Safety netting on steel poles above the walls. One consistent world: every net shows in
	## every view whose frame contains it, except the cutaway: the south net is removed from the
	## side view and the east net from the rear view, because each lies wholly between that
	## elevation camera and the drones (hifi/tools/check_views.py counts these as cutaways).
	var steel := M.mat(Color(0.62, 0.64, 0.67), 0.35, 0.85)
	var top := wh + 6.5
	var sides := {
		"south": [Vector2(0, -0.05), Vector2(W, -0.05), ["top", "rear", "fpv", "chase"]],
		"north": [Vector2(0, H + 0.05), Vector2(W, H + 0.05), ["top", "rear", "side", "fpv", "chase"]],
		"west": [Vector2(-0.05, 0), Vector2(-0.05, H), ["top", "rear", "side", "fpv", "chase"]],
		"east": [Vector2(W + 0.05, 0), Vector2(W + 0.05, H), ["top", "side", "fpv", "chase"]],
	}
	for k in sides:
		var a: Vector2 = sides[k][0]
		var b: Vector2 = sides[k][1]
		var grp := M.node(content, Vector3.ZERO, Vector3.ZERO, "Net_" + k)
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var L := a.distance_to(b)
		var n := Vector3(-(b - a).y, 0, -(b - a).x).normalized()
		M._quad_uv(st, [Vector3(a.x, wh, -a.y), Vector3(b.x, wh, -b.y), Vector3(b.x, top, -b.y), Vector3(a.x, top, -a.y)], n,
			[Vector2(0, wh), Vector2(L, wh), Vector2(L, top), Vector2(0, top)])
		var net := MeshInstance3D.new()
		net.mesh = st.commit()
		net.material_override = _net_material()
		net.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		net.set_meta("no_mask", true)
		grp.add_child(net)
		# Poles every ~5 m on the outer face, top cable
		var np := int(round(L / 5.0))
		for i in np + 1:
			var p := a.lerp(b, float(i) / np)
			M.mi(grp, M.cyl(0.05, 0.06, top + 0.1, 10), steel, Vector3(p.x, (top + 0.1) * 0.5, -p.y))
		var cable := PackedVector3Array([Vector3(a.x, top, -a.y), Vector3(b.x, top, -b.y)])
		M.mi(grp, M.tube(cable, 0.02, 6), steel)
		set_views(grp, sides[k][2])


func _build_surroundings(W: float, H: float) -> void:
	var steel := M.mat(Color(0.55, 0.57, 0.6), 0.4, 0.85)
	var dark := M.mat(Color(0.1, 0.1, 0.11), 0.5, 0.6)
	var lamp := M.mat(Color(0.95, 0.95, 0.9), 0.2, 0.0, {"emission": Color(1.0, 0.96, 0.88), "emission_energy": 1.2})
	var centre := Vector2(W * 0.5, H * 0.5)
	# Floodlight masts at the corners
	for c in [Vector2(-3.0, -3.0), Vector2(W + 3.0, -3.0), Vector2(-3.0, H + 3.0), Vector2(W + 3.0, H + 3.0)]:
		var mast := M.node(content, Vector3(c.x, 0, -c.y))
		M.mi(mast, M.cyl(0.6, 0.7, 0.5, 16), M.mat(Color(0.5, 0.5, 0.48), 0.8, 0.0), Vector3(0, 0.25, 0))
		M.mi(mast, M.cyl(0.14, 0.26, 16.0, 14), steel, Vector3(0, 8.25, 0))
		var head := M.node(mast, Vector3(0, 16.2, 0))
		var to_c: Vector2 = centre - c
		head.rotation = Vector3(0, atan2(to_c.y, to_c.x) - PI * 0.5, 0)
		M.mi(head, M.box(Vector3(2.6, 0.12, 0.12)), steel, Vector3(0, 0.0, 0))
		M.mi(head, M.box(Vector3(2.6, 0.12, 0.12)), steel, Vector3(0, 1.0, 0))
		for i in 3:
			for j in 2:
				var lp := Vector3(-0.85 + i * 0.85, 0.25 + j * 0.55, -0.12)
				var box := M.mi(head, M.box(Vector3(0.62, 0.42, 0.22), 0.03), dark, lp, Vector3(-25, 0, 0))
				M.mi(box, M.box(Vector3(0.54, 0.34, 0.02)), lamp, Vector3(0, 0, -0.115))
	# Pit tents behind the north wall
	var canopy_cols := [Color(0.9, 0.9, 0.88), Color(0.1, 0.32, 0.62), Color(0.9, 0.9, 0.88)]
	for i in 3:
		var tp := Vector2(5.0 + i * 7.0, H + 7.5)
		_tent(Vector3(tp.x, 0, -tp.y), canopy_cols[i], 0.15 * (i - 1))
	_tent(Vector3(-8.0, 0, -H * 0.5 + 3.0), Color(0.75, 0.12, 0.1), PI * 0.5)
	# Distant tree line, drawn in every view whose frame contains it (the ortho elevations show it
	# at full size behind the arena; their far planes reach past it, see exporters/drone_dogfight.py).
	# Trees between an elevation camera and the arena fall in front of its near plane (cutaway).
	var rng := RandomNumberGenerator.new()
	rng.seed = 2024
	var broad: Array = []
	var pine: Array = []
	for i in 150:
		var a := rng.randf() * TAU
		var d := rng.randf_range(42.0, 95.0)
		var s := rng.randf_range(0.9, 1.6)
		var t := Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s), Vector3(centre.x + cos(a) * d, 0, -centre.y + sin(a) * d))
		(pine if i % 3 == 0 else broad).append(t)
	for grp in [[broad, "broadleaf", 5], [pine, "pine", 9]]:
		var trees := Terrain.tree_multimesh(grp[0], grp[1], grp[2])
		content.add_child(trees)


func _tent(pos: Vector3, col: Color, yaw: float) -> void:
	var root := M.node(content, pos, Vector3(0, rad_to_deg(yaw), 0), "Tent")
	var frame := M.mat(Color(0.75, 0.76, 0.78), 0.35, 0.85)
	var cloth := M.mat(col, 0.85, 0.0, {"cull": false})
	for sx in [-1.0, 1.0]:
		for sz in [-1.0, 1.0]:
			M.mi(root, M.cyl(0.03, 0.03, 2.3, 8), frame, Vector3(sx * 1.45, 1.15, sz * 1.45))
	var roof := M.mi(root, M.cyl(0.0, 2.15, 0.9, 4), cloth, Vector3(0, 2.75, 0), Vector3(0, 45, 0))
	roof.name = "Roof"
	for k in 4:
		var a := k * PI * 0.5
		M.mi(root, M.box(Vector3(3.0, 0.25, 0.01)), cloth, Vector3(sin(a) * 1.5, 2.18, cos(a) * 1.5), Vector3(0, rad_to_deg(a), 0))
	M.mi(root, M.box(Vector3(1.6, 0.05, 0.7)), M.mat(Color(0.85, 0.85, 0.83), 0.6, 0.1), Vector3(0, 0.75, -0.6))
	for sx in [-0.7, 0.7]:
		M.mi(root, M.cyl(0.02, 0.02, 0.75, 6), frame, Vector3(sx, 0.375, -0.6))


# ================================================================ drones

func _prop_material() -> ShaderMaterial:
	if _prop_mat == null:
		var sh := Shader.new()
		sh.code = """
shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_disabled, shadows_disabled;
global uniform float sim_time;
instance uniform vec4 prop_tint : source_color = vec4(1.0, 0.84, 0.0, 1.0);
instance uniform float prop_phase = 0.0;
uniform float radius = 0.38;
varying vec3 lp;
void vertex() { lp = VERTEX; }
void fragment() {
	float rr = length(lp.xz) / radius;
	float a = atan(lp.z, lp.x) + prop_phase + sim_time * 23.0;
	float streak = 0.5 + 0.5 * cos(3.0 * a);
	float blade = smoothstep(0.12, 0.3, rr) * (1.0 - smoothstep(0.93, 1.0, rr));
	float tip = smoothstep(0.84, 0.95, rr) * (1.0 - smoothstep(0.95, 1.0, rr));
	ALBEDO = mix(vec3(0.03), prop_tint.rgb, 0.85);
	ROUGHNESS = 0.35;
	ALPHA = clamp(blade * (0.48 + 0.18 * streak) + tip * 0.35, 0.0, 0.92);
}
"""
		_prop_mat = ShaderMaterial.new()
		_prop_mat.shader = sh
	return _prop_mat


func _build_drone(player: bool) -> Node3D:
	## Drone local frame: nose = local +X, Box2D local (lx, ly) -> (lx, h, -ly), h = 0 at DRONE_ALT.
	var root := Node3D.new()
	root.name = "Player" if player else "Opponent"
	content.add_child(root)
	var tilt := M.node(root, Vector3.ZERO, Vector3.ZERO, "Tilt")
	var main_c := Color8(0, 220, 255) if player else Color8(255, 60, 60)
	var accent := Color8(255, 200, 30) if player else Color8(205, 208, 214)
	var paint := M.paint(main_c.darkened(0.06), 0.9, 0.15)
	var accent_paint := M.mat(accent, 0.28, 0.85, {"clearcoat": 0.6})
	var carbon := M.mat(Color(0.03, 0.032, 0.036), 0.3, 0.3, {"clearcoat": 0.7, "clearcoat_roughness": 0.2, "normal_noise": 0.5, "noise_scale": 9.0})
	var black := M.mat(Color(0.02, 0.02, 0.022), 0.45, 0.2)
	var alu := M.mat(Color(0.72, 0.73, 0.75), 0.25, 1.0)
	var glass := M.glass(Color(0.04, 0.06, 0.09), 0.82)
	var led := M.emissive(main_c.lightened(0.15), 4.0)

	# Canopy shell over the Box2D canopy box (1.2 x 0.5 m), paint with clearcoat
	var prof := PackedVector2Array([Vector2(0.0, -0.6), Vector2(0.09, -0.592), Vector2(0.17, -0.55), Vector2(0.226, -0.45),
		Vector2(0.248, -0.25), Vector2(0.25, 0.05), Vector2(0.242, 0.28), Vector2(0.21, 0.44), Vector2(0.14, 0.555),
		Vector2(0.06, 0.594), Vector2(0.0, 0.6)])
	M.mi(tilt, M.lathe(prof, 32, "x"), paint, Vector3.ZERO, Vector3.ZERO, Vector3(1.0, 0.95, 1.0))
	# Carbon belly plate + LED strip under it
	M.mi(tilt, M.box(Vector3(0.95, 0.035, 0.38), 0.012), carbon, Vector3(-0.02, -0.22, 0))
	var belly := M.mi(tilt, M.box(Vector3(0.6, 0.018, 0.05)), led, Vector3(-0.12, -0.243, 0))
	belly.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# Tinted canopy dome (GymBox cockpit disc at local (0.1, 0), r 0.28)
	M.mi(tilt, M.sphere(0.17, 24, 12), glass, Vector3(0.16, 0.165, 0), Vector3.ZERO, Vector3(1.6, 0.55, 1.15))
	# Nose camera
	M.mi(tilt, M.box(Vector3(0.1, 0.1, 0.13), 0.015), black, Vector3(0.58, 0.02, 0))
	M.mi(tilt, M.cyl(0.036, 0.04, 0.05, 16), glass, Vector3(0.64, 0.02, 0), Vector3(0, 0, -90))
	# Gun pod under the nose (bolts leave 1 m ahead of the centre along the heading)
	M.mi(tilt, M.box(Vector3(0.34, 0.1, 0.14), 0.02), black, Vector3(0.3, -0.27, 0))
	M.mi(tilt, M.cyl(0.04, 0.04, 0.5, 14), alu, Vector3(0.62, -0.28, 0), Vector3(0, 0, -90))
	M.mi(tilt, M.cyl(0.055, 0.05, 0.08, 14), black, Vector3(0.88, -0.28, 0), Vector3(0, 0, -90))
	var muzzle := M.mi(tilt, M.cyl(0.03, 0.03, 0.012, 12), led, Vector3(0.925, -0.28, 0), Vector3(0, 0, -90))
	muzzle.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# Landing skids
	for s in [-1.0, 1.0]:
		M.mi(tilt, M.tube(PackedVector3Array([Vector3(-0.5, -0.5, s * 0.3), Vector3(-0.44, -0.55, s * 0.3),
			Vector3(0.38, -0.55, s * 0.3), Vector3(0.46, -0.5, s * 0.3)]), 0.018, 8), carbon)
		for x in [-0.28, 0.24]:
			M.mi(tilt, M.tube(PackedVector3Array([Vector3(x, -0.22, s * 0.12), Vector3(x, -0.55, s * 0.3)]), 0.014, 6), carbon)
	# Battery pack strapped on top
	M.mi(tilt, M.box(Vector3(0.42, 0.12, 0.22), 0.025), M.mat(Color(0.16, 0.17, 0.18), 0.5, 0.1), Vector3(-0.14, 0.27, 0))
	M.mi(tilt, M.box(Vector3(0.43, 0.02, 0.06), 0.005), accent_paint, Vector3(-0.14, 0.335, 0))
	M.mi(tilt, M.box(Vector3(0.05, 0.14, 0.235)), black, Vector3(-0.02, 0.27, 0))
	# GPS mast, antennas (rear, swept back) and tail light
	M.mi(tilt, M.cyl(0.01, 0.012, 0.2, 6), black, Vector3(-0.42, 0.3, 0))
	M.mi(tilt, M.cyl(0.055, 0.06, 0.03, 16), M.mat(Color(0.85, 0.86, 0.88), 0.4, 0.1), Vector3(-0.42, 0.41, 0))
	for s in [-1.0, 1.0]:
		var pts := PackedVector3Array([Vector3(-0.47, 0.15, s * 0.07), Vector3(-0.55, 0.32, s * 0.1), Vector3(-0.6, 0.5, s * 0.12)])
		M.mi(tilt, M.tube(pts, 0.008, 6), black)
		M.mi(tilt, M.cyl(0.016, 0.016, 0.06, 8), black, Vector3(-0.6, 0.51, s * 0.12))
	var tail := M.mi(tilt, M.sphere(0.03, 12, 6), led, Vector3(-0.6, 0.02, 0))
	tail.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	# Carbon arms exactly over the Box2D arm fixtures (0.7 x 0.12 m), LED strip under each
	for a in ARMS:
		var arm := M.node(tilt, Vector3(a[0], -0.06, -a[1]), Vector3(0, rad_to_deg(a[2]), 0))
		M.mi(arm, M.box(Vector3(0.7, 0.045, 0.12), 0.012), carbon)
		var strip := M.mi(arm, M.box(Vector3(0.42, 0.012, 0.035)), led, Vector3(0.05, -0.028, 0))
		strip.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# Ducted rotors at the outer arm tips (GymBox rotor discs: gold player / grey opponent)
	var duct_prof := PackedVector2Array([Vector2(DUCT_R + 0.02, 0.0), Vector2(DUCT_R + 0.022, 0.07), Vector2(DUCT_R + 0.01, 0.1),
		Vector2(DUCT_R - 0.005, 0.098), Vector2(DUCT_R - 0.012, 0.06), Vector2(DUCT_R - 0.006, 0.006), Vector2(DUCT_R + 0.02, 0.0)])
	var duct_mesh := M.lathe(duct_prof, 40)
	var disc_mesh := M.lathe(PackedVector2Array([Vector2(0.385, 0.0), Vector2(0.0, 0.0)]), 32)
	var k := 0
	for sx in [1.0, -1.0]:
		for sy in [1.0, -1.0]:
			var tip := M.node(tilt, Vector3(sx * TIP.x, 0, -sy * TIP.y), Vector3.ZERO, "Rotor%d" % k)
			# Duct mount: cross of carbon bars under the duct, motor on top
			for ang in [45.0 * sx * sy, 45.0 * sx * sy + 90.0]:
				M.mi(tip, M.box(Vector3(2.0 * DUCT_R + 0.04, 0.03, 0.05), 0.008), carbon, Vector3(0, -0.05, 0), Vector3(0, ang, 0))
			M.mi(tip, duct_mesh, accent_paint, Vector3(0, -0.035, 0))
			M.mi(tip, M.cyl(0.07, 0.075, 0.035, 18), black, Vector3(0, -0.02, 0))
			M.mi(tip, M.cyl(0.08, 0.085, 0.07, 20), alu, Vector3(0, 0.03, 0))
			M.mi(tip, M.cyl(0.03, 0.045, 0.03, 12), accent_paint, Vector3(0, 0.08, 0))
			var disc := M.mi(tip, disc_mesh, _prop_material(), Vector3(0, 0.07, 0))
			disc.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			disc.set_instance_shader_parameter("prop_tint", accent.darkened(0.1) if player else Color(0.55, 0.56, 0.58))
			disc.set_instance_shader_parameter("prop_phase", k * 1.3 + (0.0 if player else 0.7))
			k += 1
	return root


# ================================================================ per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var alive: Array = p.get("alive", [1, 1])
	for i in _drones.size():
		var d: Node3D = _drones[i]
		d.visible = int(alive[i]) == 1
		var v: Array = p.get(["pv", "ov"][i], [0.0, 0.0])
		var vel := Vector3(float(v[0]), 0.0, -float(v[1]))
		var tilt: Node3D = _tilts[i]
		var speed := vel.length()
		if speed < 0.05:
			tilt.basis = Basis()
			continue
		# Cosmetic bank: the quad leans into its direction of travel (never moves its position)
		var ang := clampf(speed * 0.028, 0.0, 0.3)
		var axis := Vector3.UP.cross(vel / speed)
		var local_axis := (d.basis.inverse() * axis).normalized()
		tilt.basis = Basis(local_axis, ang)


func spawn_dyn(kind: String, d: Dictionary) -> Node3D:
	var root := Node3D.new()
	if kind == "bolt":
		var c := Color(1.0, 0.08, 1.0) if int(d.get("c", 0)) == 0 else Color(1.0, 0.38, 0.1)
		var g := Vfx.glow_capsule(root, 1.6, 0.055, c, 7.0)
		# Saturated core so the tracer keeps its GymBox colour (magenta / orange) after tonemapping
		for ch in g.get_children():
			if ch is MeshInstance3D and not ch.has_meta("no_mask"):
				ch.material_override = M.emissive(c, 2.2)
		return root
	if kind == "explosion":
		Vfx.explosion(root)
		return root
	root.free()
	return null


func update_dyn(node: Node3D, _a: Dictionary, b: Dictionary, _alpha: float) -> void:
	var ex := node.get_node_or_null("Explosion")
	if ex != null:
		Vfx.set_explosion(ex, float(b.get("t", 0.0)), float(b.get("r", 1.0)) * 1.15)


func chase_pose() -> Transform3D:
	var d: Node3D = _drones[0] if _drones.size() > 0 else null
	if d == null:
		return Transform3D(Basis(), Vector3(12, 8, 10))
	var fwd := d.global_transform.basis.x
	fwd.y = 0.0
	fwd = fwd.normalized()
	var p := d.global_position
	return Transform3D(Basis(), p - fwd * 4.0 + Vector3(0, 1.4, 0)).looking_at(p + fwd * 2.0, Vector3.UP)
