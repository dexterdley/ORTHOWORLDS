extends EnvWorld
## Toxic Gas Escape: an orange utility rover racing a toxic gas front through an abandoned concrete
## facility maze, overcast sky. Python: hifi/exporters/toxic_gas_escape.py.
## Top-down env: Box2D (x, y) at height h = Godot (x, h, -y); body angle = yaw about +Y.
## Moving id: car (registered "car"). Static ids: w<i> walls ("wall"), gate / door / exit_pad ("exit").
## p: gas_x (hard gas front, metres), phase (wheel spin, rad), steer (front-wheel angle, rad), speed.
## One consistent world: every object is drawn in every view whose frame contains it. Exceptions are
## cutaways only: walls / gate wholly on the viewer's side of the car (state["vis"]), and exterior
## props / sheds South of the facility (Rear) or East of it (Side), which always stand between that
## elevation camera and the car (_cutaway_views).

const Gas := preload("res://envs/toxic_gas_escape/gas.gd")

const WALL_H := 3.0
const GAS_X0 := -26.0         # gas body West end (beyond the views; the front starts behind the West wall)
const GAS_TOP := 3.35         # a little above the walls: wall tops get a light tint in Top
const GAS_Y0 := -30.0         # Box2D y span of the gas body: covers the Top view; the North face stays inside
const GAS_Y1 := 44.0          # North end of the gas body (past the North yard; the Rear ray starts at the car)
const WHEEL_R := 0.21
const WHEEL_W := 0.17
const WHEELS := [Vector2(0.6, 0.42), Vector2(0.6, -0.42), Vector2(-0.6, 0.42), Vector2(-0.6, -0.42)]  # front first

var _car: Node3D
var _wheel_steer: Array = []
var _wheel_spin: Array = []
var _spinners: Array = []     # [node, rad/s, phase]
var _car_beacon_light: OmniLight3D
var _gas: MeshInstance3D
var _gas_mat: ShaderMaterial
var _hazard: BaseMaterial3D


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "kloofendal_overcast_puresky", "sky_energy": 1.0,
		"sun_elev": 56.0, "sun_azim": 215.0, "sun_energy": 1.35, "sun_color": Color(1.0, 0.97, 0.92),
		"ambient_energy": 0.85, "card_ground": "asphalt_02", "card_ground_tint": Color(0.7, 0.7, 0.7), "exposure": 1.0, "white": 6.0, "glow": true, "ssao": true, "ssil": true,
		"saturation": 0.94, "contrast": 1.07, "shadow_distance": 90.0,
	})
	lk.sun.shadow_blur = 2.5


func build_scene(scene: Dictionary) -> void:
	_spinners.clear()
	_wheel_steer.clear()
	_wheel_spin.clear()
	var walls: Array = scene.get("walls", [])
	var aw := float(scene.get("W", 30.0))
	var ah := float(scene.get("H", 22.0))
	_hazard = _hazard_material()
	_build_ground(scene, aw, ah)
	_build_walls(walls)
	_build_floor_marks(walls, aw, ah)
	_build_grates(aw, ah)
	_build_exit(scene)
	_build_exterior(aw, ah)
	_car = _build_rover()
	register("car", _car, "car")
	_build_gas()


# ================================================================ ground

func _build_ground(scene: Dictionary, aw: float, ah: float) -> void:
	var s: Array = scene.get("slab", [-25.0, -3.0, 33.0, 40.0])
	var x0 := float(s[0])
	var y0 := float(s[1])
	var x1 := float(s[2])
	var y1 := float(s[3])
	var c := Vector3((x0 + x1) * 0.5, 0.0, -(y0 + y1) * 0.5)
	var size := Vector2(x1 - x0, y1 - y0)
	# Yard surface, then the section the Rear / Side views look at: 0.3 m slab over compacted gravel
	var yard := Assets.pbr("asphalt_02", {"uv": 0.09, "world": true, "tint": Color(0.72, 0.72, 0.71)})
	M.mi(content, M.plane(size), yard, c)
	var slab := Assets.pbr("concrete_wall_008", {"uv": 0.9, "world": true, "tint": Color(0.72, 0.72, 0.7)})
	M.mi(content, M.box(Vector3(size.x, 0.3, size.y)), slab, c + Vector3(0, -0.152, 0))
	var soil := Assets.pbr("gravel", {"uv": 0.8, "world": true, "tint": Color(0.62, 0.6, 0.56)})
	M.mi(content, M.box(Vector3(size.x, 2.7, size.y)), soil, c + Vector3(0, -1.65, 0))
	# Facility floor inside the maze
	var floor_mat := Assets.pbr("concrete_floor_worn_001", {"uv": 0.3, "world": true, "tint": Color(0.82, 0.83, 0.84)})
	M.mi(content, M.plane(Vector2(aw, ah)), floor_mat, Vector3(aw * 0.5, 0.003, -ah * 0.5))
	# Far yard beyond the section slab (edge-on, so invisible, in the Rear / Side elevations)
	M.mi(content, M.plane(Vector2(900, 900)), yard, Vector3(aw * 0.5, -0.03, -ah * 0.5))
	_build_stains(aw, ah)


func _build_stains(aw: float, ah: float) -> void:
	## Oil stains / grime decals on the maze floor (seeded).
	var rng := RandomNumberGenerator.new()
	rng.seed = 5150
	var tex := _blotch_texture()
	for i in 34:
		var d := Decal.new()
		d.texture_albedo = tex
		var sz := rng.randf_range(0.8, 3.2)
		d.size = Vector3(sz, 0.6, sz * rng.randf_range(0.6, 1.0))
		d.modulate = Color(0.2, 0.19, 0.16, rng.randf_range(0.2, 0.42)) if i % 3 != 0 else Color(0.3, 0.26, 0.15, 0.3)
		d.position = Vector3(rng.randf_range(1.0, aw - 1.0), 0.05, -rng.randf_range(1.0, ah - 1.0))
		d.rotation = Vector3(0, rng.randf() * TAU, 0)
		d.cull_mask = 0xFFFFF & ~(0b1111 << 11)
		content.add_child(d)


static func _blotch_texture() -> ImageTexture:
	var n := 128
	var noise := FastNoiseLite.new()
	noise.seed = 77
	noise.frequency = 0.035
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	for y in n:
		for x in n:
			var u := (x + 0.5) / n * 2.0 - 1.0
			var v := (y + 0.5) / n * 2.0 - 1.0
			var rr := sqrt(u * u + v * v)
			var a := clampf((1.0 - rr) * 2.2 + noise.get_noise_2d(x, y) * 1.4 - 0.2, 0.0, 1.0)
			img.set_pixel(x, y, Color(1, 1, 1, a * a))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


# ================================================================ maze walls

func _hazard_material() -> BaseMaterial3D:
	var base := Assets.pbr("concrete_wall_008", {"uv": 0.42, "world": true})
	var m: BaseMaterial3D = base.duplicate()
	m.albedo_texture = _stripe_texture()
	m.albedo_color = Color(1, 1, 1)
	m.uv1_scale = Vector3.ONE * 0.9     # 0.55 m stripe period
	return m


static func _stripe_texture() -> ImageTexture:
	var n := 128
	var noise := FastNoiseLite.new()
	noise.seed = 31
	noise.frequency = 0.06
	var dirt: Image = noise.get_seamless_image(n, n)
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	for y in n:
		for x in n:
			var yellow := fposmod(float(x + y), 64.0) < 32.0
			var c := Color(0.86, 0.62, 0.05) if yellow else Color(0.045, 0.045, 0.045)
			var w := 0.7 + 0.45 * dirt.get_pixel(x, y).r
			img.set_pixel(x, y, Color(c.r * w, c.g * w, c.b * w, 1.0))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


func _build_walls(walls: Array) -> void:
	var poured := Assets.pbr("concrete_wall_008", {"uv": 0.42, "world": true, "tint": Color(0.88, 0.9, 0.93)})
	var blocks := Assets.pbr("concrete_block_wall", {"uv": 0.5, "world": true, "tint": Color(1.55, 1.6, 1.68)})
	var cap := Assets.pbr("concrete_wall_008", {"uv": 0.8, "world": true, "tint": Color(1.02, 1.04, 1.07)})
	var pipe_mat := M.mat(Color(0.62, 0.5, 0.12), 0.45, 0.6, {"normal_noise": 0.6, "noise_scale": 2.0})
	var bracket := M.mat(Color(0.2, 0.2, 0.21), 0.5, 0.8)
	var box_mat := M.mat(Color(0.55, 0.57, 0.55), 0.5, 0.7)
	for i in walls.size():
		var w: Array = walls[i]
		var cx := float(w[0])
		var cy := float(w[1])
		var hw := float(w[2])
		var hh := float(w[3])
		var node := M.node(content, Vector3(cx, 0, -cy), Vector3.ZERO, "w%d" % i)
		# Runs overlap at junctions: shave 2 mm off each side of the thin axis and lift the copings of
		# North-South runs by 4 mm, so no two walls (distinct mask ids) ever share a coplanar face.
		var ns := hh > hw
		var shave := Vector3(0.004, 0.0, 0.0) if ns else Vector3(0.0, 0.0, 0.004)
		var lift := 0.004 if ns else 0.0
		M.mi(node, M.box(Vector3(2 * hw, WALL_H - 0.1, 2 * hh) - shave, 0.025), blocks if i < 4 else poured,
			Vector3(0, (WALL_H - 0.1) * 0.5, 0))
		M.mi(node, M.box(Vector3(2 * hw + 0.04, 0.1, 2 * hh + 0.04) - shave, 0.02), cap, Vector3(0, WALL_H - 0.05 + lift, 0))
		M.mi(node, M.box(Vector3(2 * hw + 0.02, 0.36, 2 * hh + 0.02) - shave, 0.01), _hazard, Vector3(0, 0.2 + lift, 0))
		# Gas pipe along the inner faces of the North / South outer walls (above car height)
		if i < 2:
			var side := 1.0 if i == 0 else -1.0     # inner face direction in Box2D y
			var z := -side * (hh + 0.13)
			var pts := PackedVector3Array([Vector3(-hw + 0.9, 2.45, z), Vector3(hw - 0.9, 2.45, z)])
			M.mi(node, M.tube(pts, 0.075, 12), pipe_mat)
			var k := -hw + 1.5
			while k < hw - 1.0:
				M.mi(node, M.box(Vector3(0.06, 0.06, 0.16)), bracket, Vector3(k, 2.45, -side * (hh + 0.07)))
				k += 3.0
		# Junction boxes on long internal walls (above car height)
		if i >= 4 and maxf(hw, hh) > 2.0:
			var along_x := hw > hh
			for sgn in [-1.0, 1.0]:
				var off := Vector3(0.0, 1.9, sgn * (hh + 0.06)) if along_x else Vector3(sgn * (hw + 0.06), 1.9, 0.0)
				var jb := M.mi(node, M.box(Vector3(0.36, 0.46, 0.12) if along_x else Vector3(0.12, 0.46, 0.36), 0.015), box_mat, off)
				jb.position.x += 0.0 if not along_x else (0.7 if sgn > 0 else -0.9)
				jb.position.z += 0.0 if along_x else (0.7 if sgn > 0 else -0.9)
		if i >= 4 and maxf(hw, hh) > 1.8 and i % 2 == 0:
			_sign(node, hw, hh, 1 if (i >> 1) % 2 == 0 else -1)
		register("w%d" % i, node, "wall")


func _build_floor_marks(walls: Array, aw: float, ah: float) -> void:
	## Worn yellow walkway lines 0.3 m off the wall faces (one batched mesh).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var off := 0.32
	var lw := 0.09
	var y := 0.007
	for i in walls.size():
		var w: Array = walls[i]
		var cx := float(w[0])
		var cy := float(w[1])
		var hw := float(w[2])
		var hh := float(w[3])
		var x0 := cx - hw - off
		var x1 := cx + hw + off
		var y0 := cy - hh - off
		var y1 := cy + hh + off
		var segs: Array = []
		if i == 0:
			segs = [[x0 + off * 2.0, y1, x1 - off * 2.0, y1]]
		elif i == 1:
			segs = [[x0 + off * 2.0, y0, x1 - off * 2.0, y0]]
		elif i == 2:
			segs = [[x1, y0 + off * 2.0, x1, y1 - off * 2.0]]
		elif i == 3:
			segs = [[x0, y0 + off * 2.0, x0, y1 - off * 2.0]]
		else:
			segs = [[x0, y0, x1, y0], [x0, y1, x1, y1], [x0, y0, x0, y1], [x1, y0, x1, y1]]
		for sgm in segs:
			var ax := clampf(float(sgm[0]), 0.9, aw - 0.9)
			var ay := clampf(float(sgm[1]), 0.9, ah - 0.9)
			var bx := clampf(float(sgm[2]), 0.9, aw - 0.9)
			var by := clampf(float(sgm[3]), 0.9, ah - 0.9)
			var qa: Array
			if absf(ay - by) < 1e-4:
				qa = [Vector3(ax - lw * 0.5, y, -(ay - lw * 0.5)), Vector3(bx + lw * 0.5, y, -(ay - lw * 0.5)),
					Vector3(bx + lw * 0.5, y, -(ay + lw * 0.5)), Vector3(ax - lw * 0.5, y, -(ay + lw * 0.5))]
			else:
				qa = [Vector3(ax - lw * 0.5, y, -(ay - lw * 0.5)), Vector3(ax + lw * 0.5, y, -(ay - lw * 0.5)),
					Vector3(ax + lw * 0.5, y, -(by + lw * 0.5)), Vector3(ax - lw * 0.5, y, -(by + lw * 0.5))]
			M._quad(st, qa, Vector3.UP)
	var paint := M.mat(Color(0.78, 0.6, 0.1), 0.65, 0.0, {"noise": 1.0, "noise_scale": 1.3})
	M.mi(content, st.commit(), paint)


func _sign(node: Node3D, hw: float, hh: float, side: int) -> void:
	## "DANGER - TOXIC GAS" placard flush on one long face of a wall (wall-local coordinates).
	var along_x := hw > hh
	var n := Vector3(0, 0, -float(side)) if along_x else Vector3(float(side), 0, 0)   # outward normal
	var depth := hh if along_x else hw
	var plate := M.node(node, n * (depth + 0.012) + Vector3(0, 1.55, 0))
	plate.rotation.y = atan2(n.x, n.z)       # local +Z -> outward normal
	M.mi(plate, M.box(Vector3(0.5, 0.62, 0.02), 0.004), M.mat(Color(0.88, 0.88, 0.85), 0.55, 0.05))
	var tri := PackedVector2Array([Vector2(-0.17, 0.02), Vector2(0.17, 0.02), Vector2(0.0, 0.3)])
	M.mi(plate, M.prism(tri, 0.011, 0.014), M.mat(Color(0.05, 0.05, 0.05), 0.6), Vector3(0, -0.03, 0))
	var inner := PackedVector2Array([Vector2(-0.13, 0.045), Vector2(0.13, 0.045), Vector2(0.0, 0.27)])
	M.mi(plate, M.prism(inner, 0.014, 0.017), M.mat(Color(0.92, 0.7, 0.05), 0.55), Vector3(0, -0.03, 0))
	M.mi(plate, M.box(Vector3(0.025, 0.09, 0.004)), M.mat(Color(0.05, 0.05, 0.05), 0.6), Vector3(0, 0.1, 0.018))
	M.mi(plate, M.box(Vector3(0.025, 0.022, 0.004)), M.mat(Color(0.05, 0.05, 0.05), 0.6), Vector3(0, 0.035, 0.018))
	for t in [["DANGER", 0.16, -0.12, Color(0.75, 0.05, 0.03)], ["TOXIC GAS", 0.12, -0.21, Color(0.05, 0.05, 0.05)]]:
		var l := Label3D.new()
		l.text = String(t[0])
		l.font_size = 48
		l.pixel_size = float(t[1]) / 48.0 * 0.6
		l.outline_size = 0
		l.modulate = t[3]
		l.shaded = true
		l.position = Vector3(0, float(t[2]), 0.0115)
		plate.add_child(l)


func _build_grates(aw: float, ah: float) -> void:
	## Floor drains: dark steel frame + grate, flush with the floor (y <= 1.2 cm).
	var grate := Assets.pbr("metal_grate_rusty", {"uv": 2.2, "world": true, "tint": Color(0.42, 0.42, 0.43)})
	var frame := M.mat(Color(0.2, 0.2, 0.21), 0.5, 0.8)
	var rng := RandomNumberGenerator.new()
	rng.seed = 808
	for i in 6:
		var p := Vector3(rng.randf_range(2.0, aw - 2.0), 0.0, -rng.randf_range(2.0, ah - 2.0))
		M.mi(content, M.box(Vector3(0.5, 0.01, 0.5), 0.0), frame, p + Vector3(0, 0.004, 0))
		M.mi(content, M.plane(Vector2(0.42, 0.42)), grate, p + Vector3(0, 0.0095, 0))


# ================================================================ exit

func _build_exit(scene: Dictionary) -> void:
	var ex: Array = scene.get("exit", [27.8, 11.0])
	var g: Array = scene.get("gate", [26.8, 0.8, 21.2])
	var exx := float(ex[0])
	var exy := float(ex[1])
	var gx := float(g[0])
	# Floor pad (r 1.6, like GymBox's gold disc): hazard ring + gold disc + arrow pointing East
	var pad_mat := StandardMaterial3D.new()
	pad_mat.albedo_texture = _pad_texture()
	pad_mat.roughness = 0.6
	pad_mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	var pad := Node3D.new()
	pad.name = "ExitPad"
	content.add_child(pad)
	M.mi(pad, _disc(1.6, 48, 1.7), pad_mat, Vector3(exx, 0.011, -exy))
	register("exit_pad", pad, "exit")
	# Cantilever arm from the East wall top with a hanging EXIT sign over the corridor at gx (GymBox's
	# arch position), green beacons on the East wall flanking the escape door
	var gate := Node3D.new()
	gate.name = "Gate"
	content.add_child(gate)
	var steel := M.mat(Color(0.32, 0.34, 0.36), 0.42, 0.85)
	var aw := float(scene.get("W", 30.0))
	var ay := exy + 1.1         # arm North of the pad centre so the pad stays readable in Top
	var x0 := gx - 0.15
	var x1 := aw - 0.25
	M.mi(gate, M.box(Vector3(x1 - x0, 0.16, 0.16), 0.012), steel, Vector3((x0 + x1) * 0.5, WALL_H + 0.08, -ay))
	M.mi(gate, M.box(Vector3(x1 - x0 - 0.1, 0.035, 0.17), 0.0), _hazard, Vector3((x0 + x1) * 0.5, WALL_H + 0.17, -ay))
	var brace := PackedVector3Array([Vector3(aw - 0.8, WALL_H - 0.75, -ay), Vector3(aw - 1.9, WALL_H, -ay)])
	M.mi(gate, M.tube(brace, 0.035, 8), steel)
	M.mi(gate, M.box(Vector3(0.04, 0.3, 0.22), 0.005), steel, Vector3(aw - 0.82, WALL_H - 0.72, -ay))
	var sign_mat := M.emissive(Color(0.1, 0.85, 0.3), 2.2)
	var frame := M.mat(Color(0.12, 0.12, 0.13), 0.5, 0.6)
	for dz in [-0.45, 0.45]:
		M.mi(gate, M.cyl(0.012, 0.012, 0.36, 6), steel, Vector3(gx, WALL_H - 0.18, -ay + dz))
	M.mi(gate, M.box(Vector3(0.1, 0.36, 1.2), 0.01), frame, Vector3(gx, WALL_H - 0.52, -ay))
	M.mi(gate, M.box(Vector3(0.11, 0.3, 1.12), 0.0), sign_mat, Vector3(gx, WALL_H - 0.52, -ay))
	for sx in [-1.0, 1.0]:
		var lbl := Label3D.new()
		lbl.text = "EXIT"
		lbl.font_size = 96
		lbl.outline_size = 0
		lbl.pixel_size = 0.0022
		lbl.modulate = Color(0.95, 1.0, 0.95)
		lbl.shaded = false
		lbl.position = Vector3(gx + sx * 0.058, WALL_H - 0.52, -ay)
		lbl.rotation_degrees = Vector3(0, 90.0 * sx, 0)
		gate.add_child(lbl)
	for dy in [-1.9, 1.9]:
		_beacon(gate, Vector3(aw - 0.4, WALL_H, -(exy + dy)), Color(0.15, 1.0, 0.35), 2.5)
	var gl := M.light(gate, Color(0.3, 1.0, 0.45), 1.0, 6.0, Vector3(gx - 0.3, WALL_H - 0.8, -ay))
	gl.name = "SignGlow"
	register("gate", gate, "exit")
	# Roller-shutter escape door on the East wall's inner face
	var door := Node3D.new()
	door.name = "Door"
	content.add_child(door)
	var dx := float(scene.get("W", 30.0)) - 0.8
	var shutter := Assets.pbr("corrugated_iron", {"uv": 0.9, "world": true, "tint": Color(0.55, 0.62, 0.58)})
	M.mi(door, M.box(Vector3(0.05, 2.5, 3.0), 0.01), shutter, Vector3(dx - 0.025, 1.25, -exy))
	for dz in [-1.58, 1.58]:
		M.mi(door, M.box(Vector3(0.1, 2.75, 0.14), 0.01), _hazard, Vector3(dx - 0.05, 1.375, -exy + dz))
	M.mi(door, M.box(Vector3(0.1, 0.22, 3.3), 0.01), steel, Vector3(dx - 0.05, 2.6, -exy))
	M.mi(door, M.box(Vector3(0.08, 0.18, 0.8), 0.0), sign_mat, Vector3(dx - 0.06, 2.82, -exy))
	var dl := Label3D.new()
	dl.text = "EXIT"
	dl.font_size = 64
	dl.pixel_size = 0.0022
	dl.outline_size = 0
	dl.shaded = false
	dl.position = Vector3(dx - 0.105, 2.82, -exy)
	dl.rotation_degrees = Vector3(0, -90, 0)
	door.add_child(dl)
	register("door", door, "exit")


func _beacon(parent: Node3D, pos: Vector3, col: Color, energy: float) -> void:
	var base := M.mat(Color(0.1, 0.1, 0.11), 0.5, 0.6)
	M.mi(parent, M.cyl(0.07, 0.08, 0.05, 14), base, pos + Vector3(0, 0.025, 0))
	var dome := M.mi(parent, M.sphere(0.065, 16, 8), M.mat(col, 0.1, 0.0, {"emission": col, "emission_energy": energy, "alpha": 0.85}),
		pos + Vector3(0, 0.08, 0))
	dome.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var rot := M.node(parent, pos + Vector3(0, 0.08, 0))
	M.mi(rot, M.box(Vector3(0.02, 0.07, 0.08)), M.emissive(col.lerp(Color.WHITE, 0.5), energy * 2.0), Vector3(0.025, 0, 0))
	var l := M.light(parent, col, 0.9, 4.5, pos + Vector3(0, 0.15, 0))
	l.name = "BeaconLight"
	_spinners.append([rot, 5.0, float(_spinners.size()) * 1.3])


static func _disc(radius: float, segs: int, uv_half: float) -> ArrayMesh:
	## Flat disc in the XZ plane with planar UVs (u along +X, v along +Z; uv_half metres = 0.5).
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for k in segs:
		var a0 := TAU * k / segs
		var a1 := TAU * (k + 1) / segs
		var p := [Vector3.ZERO, Vector3(cos(a0) * radius, 0, sin(a0) * radius), Vector3(cos(a1) * radius, 0, sin(a1) * radius)]
		var uv := []
		for q in p:
			uv.append(Vector2(q.x / (2.0 * uv_half) + 0.5, q.z / (2.0 * uv_half) + 0.5))
		M._tri_n(st, p[0], p[1], p[2], Vector3.UP, Vector3.UP, Vector3.UP, uv[0], uv[1], uv[2], Vector3.UP)
	return st.commit()


static func _pad_texture() -> ImageTexture:
	var n := 256
	var noise := FastNoiseLite.new()
	noise.seed = 9
	noise.frequency = 0.05
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	var gold := Color(0.85, 0.62, 0.08)
	var yellow := Color(0.9, 0.72, 0.08)
	var black := Color(0.04, 0.04, 0.04)
	for y in n:
		for x in n:
			var u := ((x + 0.5) / n * 2.0 - 1.0) * 1.7     # metres East
			var v := ((y + 0.5) / n * 2.0 - 1.0) * 1.7     # metres South
			var rr := sqrt(u * u + v * v)
			var c := gold
			if rr > 1.28:
				var k := int(floor((atan2(v, u) + PI) / TAU * 28.0))
				c = yellow if k % 2 == 0 else black
			elif rr > 1.18:
				c = black
			else:
				var head := u >= 0.05 and u <= 0.75 and absf(v) <= 0.62 * (0.75 - u) / 0.7
				var shaft := u >= -0.7 and u < 0.06 and absf(v) <= 0.17
				if head or shaft:
					c = black
			var w := 0.78 + 0.3 * (0.5 + 0.5 * noise.get_noise_2d(x, y))
			img.set_pixel(x, y, Color(c.r * w, c.g * w, c.b * w, 1.0))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


# ================================================================ exterior

func _cutaway_views(node: Node3D, x: float, y: float, aw: float) -> void:
	## Exterior items at Box2D (x, y): those South of the facility (y < 0) always lie between the
	## Rear camera and the car, those East of it (x > aw) between the Side camera and the car, so
	## they are cut away from that view (instead of being sliced by its near plane). The rest is
	## drawn in every view.
	var views := ["top", "fpv", "chase"]
	if y >= 0.0:
		views.append("rear")
	if x <= aw:
		views.append("side")
	if views.size() < 5:
		set_views(node, views)


func _build_exterior(aw: float, ah: float) -> void:
	var ext := M.node(content, Vector3.ZERO, Vector3.ZERO, "Exterior")
	var rng := RandomNumberGenerator.new()
	rng.seed = 4242
	var tints := [Color(0.55, 0.16, 0.1), Color(0.16, 0.3, 0.5), Color(0.2, 0.38, 0.24), Color(0.6, 0.45, 0.2), Color(0.45, 0.45, 0.47)]
	# Shipping containers around the facility: [x, y, yaw_deg] in Box2D metres
	var boxes := [[6.0, 26.0, 0.0], [13.2, 26.4, 0.0], [21.0, 27.5, 8.0], [9.0, -7.0, 0.0], [21.5, -8.0, -6.0],
		[37.5, 4.0, 90.0], [38.5, 15.5, 90.0], [-11.0, 3.0, 90.0], [-12.5, 15.0, 84.0], [-6.5, 28.0, 30.0], [40.0, 27.0, 0.0]]
	for i in boxes.size():
		var b: Array = boxes[i]
		var cont := _container(ext, tints[i % tints.size()])
		cont.position = Vector3(float(b[0]), 0, -float(b[1]))
		cont.rotation_degrees = Vector3(0, float(b[2]), 0)
		_cutaway_views(cont, float(b[0]), float(b[1]), aw)
	# Barrels, crates, utility boxes in clusters next to the containers
	var spots := [[3.0, 24.6], [17.0, 24.5], [12.0, -4.6], [35.0, 9.5], [-4.5, 8.0], [-4.0, 21.0], [26.0, -5.0], [34.5, 21.0]]
	for s in spots:
		for k in 5:
			var x := float(s[0]) + rng.randf_range(-1.3, 1.3)
			var y := float(s[1]) + rng.randf_range(-0.9, 0.9)
			var nm: String = ["Barrel_01", "Barrel_01", "wooden_crate_01", "metal_trash_can", "wooden_crate_02"][k]
			var sz := Vector3(0.62, 0.9, 0.62) if nm == "Barrel_01" else (Vector3(0.9, 0.9, 0.9) if nm.begins_with("wooden") else Vector3(0.6, 0.95, 0.6))
			var m := Assets.fitted_model(nm, sz)
			if m == null:
				continue
			m.position = Vector3(x, 0, -y)
			m.rotation = Vector3(0, rng.randf() * TAU, 0)
			ext.add_child(m)
			_cutaway_views(m, float(s[0]), float(s[1]), aw)
	var pole_mat := M.mat(Color(0.42, 0.43, 0.44), 0.45, 0.85)
	for s in [[-3.0, -4.5, 45.0], [33.0, -4.5, 135.0], [33.0, 25.5, 225.0], [-3.0, 25.5, 315.0]]:
		var mast := M.node(ext, Vector3(float(s[0]), 0, -float(s[1])), Vector3(0, float(s[2]), 0))
		M.mi(mast, M.cyl(0.08, 0.13, 7.0, 12), pole_mat, Vector3(0, 3.5, 0))
		M.mi(mast, M.cyl(0.25, 0.3, 0.3, 12), M.mat(Color(0.5, 0.5, 0.5), 0.8), Vector3(0, 0.15, 0))
		M.mi(mast, M.box(Vector3(1.1, 0.07, 0.07)), pole_mat, Vector3(0, 6.9, 0))
		for sx in [-0.5, 0.5]:
			var lamp := Assets.fitted_model("security_light", Vector3(0.45, 0.4, 0.45))
			if lamp != null:
				lamp.position = Vector3(sx, 6.75, 0)
				lamp.rotation_degrees = Vector3(0, -90, 0)
				mast.add_child(lamp)
		_cutaway_views(mast, float(s[0]), float(s[1]), aw)
	for s in [[30.0, -6.0], [-6.0, -2.0]]:
		var ub := Assets.fitted_model("utility_box_01", Vector3(1.2, 1.4, 0.6))
		if ub != null:
			ub.position = Vector3(float(s[0]), 0, -float(s[1]))
			ext.add_child(ub)
			_cutaway_views(ub, float(s[0]), float(s[1]), aw)
	# Distant warehouses + a stack (the Rear / Side far planes reach past them)
	var far := M.node(content, Vector3.ZERO, Vector3.ZERO, "Skyline")
	var clad := Assets.pbr("corrugated_iron", {"uv": 0.25, "world": true, "tint": Color(0.62, 0.64, 0.66)})
	var clad2 := Assets.pbr("rusty_metal_02", {"uv": 0.08, "world": true, "tint": Color(0.7, 0.66, 0.62)})
	var bricks := Assets.pbr("concrete_block_wall", {"uv": 0.15, "world": true, "tint": Color(0.6, 0.6, 0.58)})
	var sheds := [[aw * 0.5, ah + 70.0, 60.0, 18.0, 14.0], [aw + 75.0, ah * 0.5, 22.0, 70.0, 11.0], [-60.0, ah * 0.5 + 10.0, 26.0, 50.0, 16.0],
		[aw * 0.5 - 20.0, -75.0, 55.0, 24.0, 12.0], [aw + 60.0, ah + 60.0, 40.0, 30.0, 20.0], [-55.0, -50.0, 30.0, 30.0, 9.0]]
	for i in sheds.size():
		var sd: Array = sheds[i]
		var hgt := float(sd[4])
		var shed := M.node(far)
		M.mi(shed, M.box(Vector3(float(sd[2]), hgt, float(sd[3]))), [clad, clad2, bricks][i % 3], Vector3(float(sd[0]), hgt * 0.5, -float(sd[1])))
		M.mi(shed, M.box(Vector3(float(sd[2]) + 0.6, 0.6, float(sd[3]) + 0.6)), M.mat(Color(0.25, 0.26, 0.27), 0.7, 0.4),
			Vector3(float(sd[0]), hgt + 0.3, -float(sd[1])))
		_cutaway_views(shed, float(sd[0]), float(sd[1]), aw)
	var stack := Assets.pbr("concrete_wall_008", {"uv": 0.12, "world": true, "tint": Color(0.85, 0.84, 0.8)})
	var sp := Vector3(aw + 52.0, 0.0, -(ah + 40.0))
	var chimney := M.node(far)
	M.mi(chimney, M.cyl(1.6, 2.4, 34.0, 20), stack, sp + Vector3(0, 17.0, 0))
	for k in 2:
		M.mi(chimney, M.cyl(1.66 + 0.02 * k, 1.68 + 0.02 * k, 1.6, 20), M.mat(Color(0.62, 0.12, 0.08), 0.7),
			sp + Vector3(0, 32.0 - 3.2 * k, 0))
	_cutaway_views(chimney, sp.x, -sp.z, aw)


func _container(parent: Node3D, tint: Color) -> Node3D:
	var root := M.node(parent)
	var ribs := Assets.pbr("corrugated_iron", {"uv": 1.1, "tint": tint.lerp(Color.WHITE, 0.15), "color": tint})
	var trim := M.mat(tint * 0.6, 0.6, 0.5)
	M.mi(root, M.box(Vector3(6.06, 2.45, 2.44), 0.03), ribs, Vector3(0, 1.28, 0))
	M.mi(root, M.box(Vector3(6.1, 0.12, 2.48), 0.02), trim, Vector3(0, 2.53, 0))
	M.mi(root, M.box(Vector3(6.1, 0.12, 2.48), 0.02), trim, Vector3(0, 0.06, 0))
	for sx in [-3.03, 3.03]:
		M.mi(root, M.box(Vector3(0.06, 2.4, 2.3), 0.0), trim, Vector3(sx, 1.28, 0))
	return root


# ================================================================ rover

func _build_rover() -> Node3D:
	## Utility rover fitted to the Box2D fixtures (car-local x forward; local (lx, ly) -> (lx, h, -ly)):
	## chassis 1.7 x 0.9, cabin 0.7 x 0.6 at x = -0.15, wheels 0.44 x 0.18 at (+-0.6, +-0.42).
	var root := Node3D.new()
	root.name = "Rover"
	content.add_child(root)
	var paint := M.paint(Color(0.93, 0.36, 0.035), 0.72, 0.15)
	var roof := M.paint(Color(0.07, 0.11, 0.18), 0.6, 0.2)
	var dark := M.mat(Color(0.05, 0.05, 0.055), 0.6, 0.3)
	var steel := M.mat(Color(0.5, 0.51, 0.53), 0.35, 0.9)
	var glass := M.glass(Color(0.04, 0.06, 0.08), 0.62)
	# Chassis rail + lower tub between the wheels
	M.mi(root, M.box(Vector3(1.56, 0.12, 0.6), 0.02), dark, Vector3(0, 0.2, 0))
	M.mi(root, M.box(Vector3(1.66, 0.2, 0.62), 0.03), dark, Vector3(0, 0.35, 0))
	# Deck over the wheels (full footprint)
	M.mi(root, M.box(Vector3(1.7, 0.17, 0.9), 0.045), paint, Vector3(0, 0.535, 0))
	# Fender flares
	for w in WHEELS:
		M.mi(root, M.box(Vector3(0.56, 0.05, 0.1), 0.02), dark, Vector3(w.x, 0.465, -w.y * 1.07))
	# Hood (sloped) and front fascia
	var hood := PackedVector2Array([Vector2(0.2, 0.62), Vector2(0.84, 0.62), Vector2(0.84, 0.655), Vector2(0.72, 0.715), Vector2(0.2, 0.74)])
	M.mi(root, M.prism(hood, -0.42, 0.42), paint)
	M.mi(root, M.box(Vector3(0.07, 0.14, 0.86), 0.02), dark, Vector3(0.83, 0.32, 0))
	M.mi(root, M.box(Vector3(0.02, 0.09, 0.34), 0.0), dark, Vector3(0.852, 0.535, 0))
	for k in 5:
		M.mi(root, M.box(Vector3(0.008, 0.07, 0.012), 0.0), steel, Vector3(0.862, 0.535, -0.12 + 0.06 * k))
	var marker := M.emissive(Color(1.0, 0.5, 0.05), 1.5)
	for pz in [-0.453, 0.453]:
		M.mi(root, M.box(Vector3(0.06, 0.03, 0.006), 0.0), marker, Vector3(0.74, 0.535, pz))
	var head := M.emissive(Color(1.0, 0.95, 0.82), 4.0)
	for z in [-0.3, 0.3]:
		M.mi(root, M.cyl(0.05, 0.05, 0.02, 16), head, Vector3(0.853, 0.535, z), Vector3(0, 0, 90))
		M.mi(root, M.cyl(0.062, 0.062, 0.015, 16), steel, Vector3(0.846, 0.535, z), Vector3(0, 0, 90))
		var sp := SpotLight3D.new()
		sp.light_color = Color(1.0, 0.93, 0.8)
		sp.light_energy = 1.2
		sp.spot_range = 9.0
		sp.spot_angle = 30.0
		sp.position = Vector3(0.88, 0.535, z)
		sp.rotation_degrees = Vector3(-6.0, -90.0, 0)
		root.add_child(sp)
	# Cab: pillars, tinted glass, dark roof (GymBox's navy cabin top)
	M.mi(root, M.box(Vector3(0.66, 0.08, 0.6), 0.015), paint, Vector3(-0.15, 0.66, 0))
	var cab := PackedVector2Array([Vector2(-0.47, 0.695), Vector2(0.17, 0.695), Vector2(0.085, 0.89), Vector2(-0.47, 0.89)])
	M.mi(root, M.prism(cab, -0.28, 0.28), glass)
	for pz in [-0.301, 0.301]:
		M.mi(root, M.box(Vector3(0.012, 0.075, 0.004), 0.0), dark, Vector3(-0.17, 0.66, pz))
		M.mi(root, M.box(Vector3(0.05, 0.012, 0.006), 0.0), steel, Vector3(-0.05, 0.672, pz))
	for pz in [-0.285, 0.285]:
		M.mi(root, M.box(Vector3(0.05, 0.22, 0.04), 0.01), paint, Vector3(-0.465, 0.795, pz))
		M.mi(root, M.box(Vector3(0.04, 0.22, 0.04), 0.01), paint, Vector3(0.13, 0.79, pz), Vector3(0, 0, 23.5))
		M.mi(root, M.box(Vector3(0.58, 0.03, 0.035), 0.008), paint, Vector3(-0.17, 0.705, pz))
	M.mi(root, M.box(Vector3(0.66, 0.05, 0.62), 0.02), roof, Vector3(-0.19, 0.912, 0))
	for pz in [-0.35, 0.35]:
		M.mi(root, M.box(Vector3(0.03, 0.06, 0.07), 0.005), dark, Vector3(0.16, 0.78, pz))
		M.mi(root, M.box(Vector3(0.015, 0.012, 0.06), 0.0), dark, Vector3(0.16, 0.76, pz * 0.9))
	# Amber beacon on the roof
	M.mi(root, M.cyl(0.055, 0.06, 0.03, 14), dark, Vector3(-0.2, 0.962, 0))
	var amber := Color(1.0, 0.55, 0.05)
	var dome := M.mi(root, M.sphere(0.05, 14, 8), M.mat(amber, 0.15, 0.0, {"emission": amber, "emission_energy": 2.5, "alpha": 0.85}),
		Vector3(-0.2, 1.0, 0))
	dome.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var rot := M.node(root, Vector3(-0.2, 1.0, 0))
	M.mi(rot, M.box(Vector3(0.015, 0.05, 0.06)), M.emissive(Color(1.0, 0.8, 0.4), 6.0), Vector3(0.02, 0, 0))
	_spinners.append([rot, 7.0, 0.0])
	_car_beacon_light = M.light(root, amber, 0.3, 2.2, Vector3(-0.2, 1.08, 0))
	# Rear cargo bed with a gas-cylinder rack, hazard panel and tail lights
	var bed := M.mat(Color(0.2, 0.2, 0.21), 0.55, 0.7)
	M.mi(root, M.box(Vector3(0.34, 0.02, 0.8), 0.0), bed, Vector3(-0.67, 0.63, 0))
	for pz in [-0.41, 0.41]:
		M.mi(root, M.box(Vector3(0.34, 0.1, 0.03), 0.005), paint, Vector3(-0.67, 0.67, pz))
	M.mi(root, M.box(Vector3(0.03, 0.1, 0.85), 0.005), paint, Vector3(-0.835, 0.67, 0))
	var chest := M.mat(Color(0.16, 0.17, 0.18), 0.45, 0.75)
	M.mi(root, M.box(Vector3(0.24, 0.13, 0.5), 0.015), chest, Vector3(-0.64, 0.705, 0.1))
	M.mi(root, M.box(Vector3(0.25, 0.012, 0.51), 0.004), steel, Vector3(-0.64, 0.775, 0.1))
	var can := M.paint(Color(0.7, 0.06, 0.04), 0.5, 0.1)
	M.mi(root, M.box(Vector3(0.2, 0.16, 0.08), 0.02), can, Vector3(-0.66, 0.72, -0.27))
	M.mi(root, M.cyl(0.015, 0.015, 0.04, 8), dark, Vector3(-0.6, 0.815, -0.27))
	M.mi(root, M.box(Vector3(0.012, 0.08, 0.42), 0.0), _hazard, Vector3(-0.853, 0.535, 0))
	var tail := M.emissive(Color(0.9, 0.05, 0.03), 1.8)
	for z in [-0.34, 0.34]:
		M.mi(root, M.box(Vector3(0.02, 0.06, 0.1), 0.005), tail, Vector3(-0.853, 0.535, z))
	M.mi(root, M.box(Vector3(0.06, 0.12, 0.86), 0.02), dark, Vector3(-0.83, 0.32, 0))
	# Reflective side stripe, front push bar, whip antenna
	var stripe := M.mat(Color(0.85, 0.86, 0.86), 0.3, 0.2)
	M.mi(root, M.box(Vector3(1.5, 0.028, 0.906), 0.0), stripe, Vector3(-0.02, 0.5, 0))
	var bar_pts := PackedVector3Array([Vector3(0.86, 0.26, -0.36), Vector3(0.9, 0.3, -0.36), Vector3(0.92, 0.48, -0.32),
		Vector3(0.92, 0.48, 0.32), Vector3(0.9, 0.3, 0.36), Vector3(0.86, 0.26, 0.36)])
	M.mi(root, M.tube(bar_pts, 0.018, 8), dark)
	M.mi(root, M.cyl(0.004, 0.006, 0.6, 6), dark, Vector3(-0.8, 0.95, -0.38))
	M.mi(root, M.box(Vector3(0.02, 0.03, 0.02)), dark, Vector3(-0.8, 0.64, -0.38))
	# Wheels: steer node -> axle along the car's lateral axis -> spin (rotation.x)
	var tyre := M.mat(Color(0.045, 0.045, 0.045), 0.85, 0.0, {"normal_noise": 1.2, "noise_scale": 6.0})
	var rim := M.mat(Color(0.2, 0.2, 0.21), 0.35, 0.85)
	for i in WHEELS.size():
		var w: Vector2 = WHEELS[i]
		var steer := M.node(root, Vector3(w.x, WHEEL_R, -w.y))
		var axle := M.node(steer, Vector3.ZERO, Vector3(0, 90, 0))
		var spin := M.wheel(axle, WHEEL_R, WHEEL_W, tyre, rim, 5)
		_wheel_steer.append(steer)
		_wheel_spin.append(spin)
	return root


# ================================================================ gas

func _build_gas() -> void:
	_gas_mat = Gas.material()
	_gas = M.mi(content, M.box(Vector3.ONE), _gas_mat)
	_gas.name = "Gas"
	_gas.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_gas.set_meta("no_mask", true)
	_gas.visible = false


func _update_gas(gx: float) -> void:
	if gx <= GAS_X0 + 0.02:
		_gas.visible = false
		return
	_gas.visible = true
	var bmin := Vector3(GAS_X0, -0.6, -GAS_Y1)
	var bmax := Vector3(gx, GAS_TOP, -GAS_Y0)
	_gas.position = (bmin + bmax) * 0.5
	_gas.scale = bmax - bmin
	_gas_mat.set_shader_parameter("box_min", bmin)
	_gas_mat.set_shader_parameter("box_max", bmax)
	_gas_mat.set_shader_parameter("cut_rear_z", _car.position.z)
	_gas_mat.set_shader_parameter("cut_side_x", _car.position.x)


# ================================================================ per step

func on_state(state: Dictionary, _prev: Dictionary, _alpha: float) -> void:
	var p: Dictionary = state.get("p", {})
	var ph := float(p.get("phase", 0.0))
	var st := float(p.get("steer", 0.0))
	for i in _wheel_spin.size():
		(_wheel_spin[i] as Node3D).rotation.x = ph
		if i < 2:
			(_wheel_steer[i] as Node3D).rotation.y = st
	_update_gas(float(p.get("gas_x", -100.0)))
	for s in _spinners:
		(s[0] as Node3D).rotation.y = sim_t * float(s[1]) + float(s[2])
	if _car_beacon_light != null:
		_car_beacon_light.light_energy = 0.15 + 0.25 * maxf(0.0, sin(sim_t * 7.0))


func chase_pose() -> Transform3D:
	if _car == null:
		return Transform3D(Basis(), Vector3(15, 8, 0))
	var fwd := _car.global_transform.basis.x
	var p := _car.global_position
	return Transform3D(Basis(), p - fwd * 4.5 + Vector3(0, 2.4, 0)).looking_at(p + fwd * 1.5 + Vector3(0, 0.4, 0), Vector3.UP)
