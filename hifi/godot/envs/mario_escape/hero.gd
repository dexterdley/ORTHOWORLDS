extends RefCounted
## The runner: an original stocky adventurer (red work shirt, denim overalls with brass buttons,
## leather boots and gloves, plain red cap, brown hair, no moustache) fitted to the GymBox player
## box (0.76 x 1.5 m, feet at local y = -0.75, cap top ~1.55 m above the feet as in GymBox's stack:
## boots 0..0.22, overalls 0.22..0.72, shirt 0.72..1.08, face 1.08..1.42, cap 1.30..1.55).
## Built facing +X; pose() flips it for facing_right = 0 and animates it from the state only
## (run cycle phase = step_count * 0.45 like the GymBox arm swing, jump / fall poses in the air).

const FEET := -0.75


static var _ell_cache := {}


static func ell(r: Vector3) -> Mesh:
	## Ellipsoid with baked radii (correct normals, unstretched object-space textures).
	var key := "%.3f,%.3f,%.3f" % [r.x, r.y, r.z]
	if _ell_cache.has(key):
		return _ell_cache[key]
	var sm := SphereMesh.new()
	sm.radius = 1.0
	sm.height = 2.0
	sm.radial_segments = 24
	sm.rings = 12
	var am := ArrayMesh.new()
	am.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, sm.get_mesh_arrays())
	var mdt := MeshDataTool.new()
	mdt.create_from_surface(am, 0)
	for i in mdt.get_vertex_count():
		mdt.set_vertex(i, mdt.get_vertex(i) * r)
		var n := mdt.get_vertex_normal(i)
		mdt.set_vertex_normal(i, Vector3(n.x / r.x, n.y / r.y, n.z / r.z).normalized())
	am.clear_surfaces()
	mdt.commit_to_surface(am)
	_ell_cache[key] = am
	return am


static func _e(parent: Node3D, r: Vector3, pos: Vector3, mat: Material, rot := Vector3.ZERO) -> MeshInstance3D:
	return M.mi(parent, ell(r), mat, pos, rot)


static func build(parent: Node3D) -> Dictionary:
	var shirt := Assets.pbr("polar_fleece", {"uv": 5.0, "tint": Color(0.8, 0.13, 0.09), "color": Color(0.6, 0.08, 0.05), "normal": 1.5})
	var denim := Assets.pbr("denim_fabric", {"uv": 2.2, "tint": Color(0.82, 0.88, 1.0), "color": Color(0.16, 0.26, 0.48)})
	var cuff := Assets.pbr("denim_fabric", {"uv": 2.6, "tint": Color(1.0, 1.0, 1.05), "color": Color(0.3, 0.4, 0.6)})
	var boot := M.mat(Color(0.22, 0.12, 0.055), 0.45, 0.0, {"clearcoat": 0.3, "normal_noise": 0.4, "noise_scale": 8.0})
	var sole := M.mat(Color(0.07, 0.055, 0.045), 0.85)
	var glove := M.mat(Color(0.46, 0.3, 0.16), 0.55, 0.0, {"normal_noise": 0.5, "noise_scale": 10.0})
	var skin := M.mat(Color(0.82, 0.58, 0.45), 0.5, 0.0, {"rim": 0.3})
	var hair := M.mat(Color(0.18, 0.1, 0.045), 0.7, 0.0, {"normal_noise": 0.8, "noise_scale": 14.0})
	var cap := Assets.pbr("polar_fleece", {"uv": 6.0, "tint": Color(0.82, 0.12, 0.08), "color": Color(0.6, 0.08, 0.05)})
	var brass := M.mat(Color(0.8, 0.62, 0.32), 0.3, 1.0)
	var white := M.mat(Color(0.92, 0.9, 0.86), 0.3)
	var iris := M.mat(Color(0.12, 0.07, 0.03), 0.1, 0.0, {"clearcoat": 1.0})

	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Hero")
	var body := M.node(root, Vector3.ZERO, Vector3.ZERO, "Body")      # facing flip + planted-foot offset
	var j := {"root": root, "body": body}

	# ---------------------------------------------------------------- torso (pivot at the hips, y = -0.03)
	var torso := M.node(body, Vector3(0, -0.03, 0), Vector3.ZERO, "Torso")
	j["torso"] = torso
	_e(torso, Vector3(0.17, 0.15, 0.24), Vector3(0.0, 0.03, 0), denim)                 # seat / hips
	_e(torso, Vector3(0.175, 0.22, 0.245), Vector3(0.0, 0.24, 0), shirt)               # chest
	_e(torso, Vector3(0.16, 0.13, 0.22), Vector3(0.02, 0.11, 0), denim)                # overall waist
	_e(torso, Vector3(0.05, 0.12, 0.155), Vector3(0.125, 0.19, 0), denim)              # bib
	for sz in [-1.0, 1.0]:
		M.mi(torso, M.capsule(0.024, 0.42, 6), denim, Vector3(0.0, 0.34, sz * 0.12), Vector3(0, 0, 90))  # strap over the shoulder
		M.mi(torso, M.cyl(0.02, 0.02, 0.012, 10), brass, Vector3(0.172, 0.27, sz * 0.095), Vector3(0, 0, 90))
	M.mi(torso, M.cyl(0.07, 0.078, 0.1, 12), skin, Vector3(0, 0.44, 0))                     # neck

	# ---------------------------------------------------------------- head
	var head := M.node(torso, Vector3(0, 0.45, 0), Vector3.ZERO, "Head")
	j["head"] = head
	_e(head, Vector3(0.158, 0.172, 0.15), Vector3(0.0, 0.08, 0), skin)                 # skull / face
	_e(head, Vector3(0.11, 0.07, 0.115), Vector3(0.05, -0.02, 0), skin)                # jaw
	_e(head, Vector3(0.155, 0.15, 0.152), Vector3(-0.035, 0.1, 0), hair)               # hair (back)
	_e(head, Vector3(0.03, 0.036, 0.028), Vector3(0.158, 0.055, 0), skin, Vector3(0, 0, -15))   # nose
	for sz in [-1.0, 1.0]:
		_e(head, Vector3(0.012, 0.016, 0.02), Vector3(0.137, 0.105, sz * 0.056), white)
		M.mi(head, M.sphere(0.009, 8, 6), iris, Vector3(0.147, 0.105, sz * 0.056))
		M.mi(head, M.box(Vector3(0.024, 0.012, 0.046), 0.004), hair, Vector3(0.145, 0.142, sz * 0.058), Vector3(0, sz * 8.0, 0))
		_e(head, Vector3(0.022, 0.042, 0.016), Vector3(0.0, 0.08, sz * 0.152), skin)  # ears
		_e(head, Vector3(0.03, 0.06, 0.02), Vector3(0.06, 0.05, sz * 0.14), hair)     # sideburns
	var dome := _e(head, Vector3(0.172, 0.12, 0.168), Vector3(-0.01, 0.185, 0), cap)  # cap crown
	M.mi(head, M.cyl(0.172, 0.172, 0.045, 24), cap, Vector3(-0.01, 0.17, 0))           # cap band
	_e(head, Vector3(0.13, 0.012, 0.14), Vector3(0.16, 0.155, 0), cap, Vector3(0, 0, -8))     # brim, forward
	M.mi(head, M.sphere(0.02, 8, 6), cap, Vector3(-0.01, 0.302, 0))                    # button on top

	# ---------------------------------------------------------------- arms (shoulders at hip + 0.36)
	for side in [["L", 1.0], ["R", -1.0]]:
		var sz: float = side[1]
		var sh := M.node(torso, Vector3(0, 0.36, sz * 0.27), Vector3.ZERO, "Shoulder" + side[0])
		_e(sh, Vector3(0.085, 0.085, 0.08), Vector3(0, -0.01, 0), shirt)
		M.mi(sh, M.capsule(0.07, 0.3, 12), shirt, Vector3(0, -0.12, 0))                     # rolled sleeve
		var el := M.node(sh, Vector3(0, -0.26, 0), Vector3.ZERO, "Elbow" + side[0])
		M.mi(el, M.capsule(0.056, 0.24, 12), skin, Vector3(0, -0.1, 0))                     # forearm
		M.mi(el, M.cyl(0.066, 0.06, 0.06, 12), glove, Vector3(0, -0.18, 0))                 # glove cuff
		_e(el, Vector3(0.06, 0.075, 0.045), Vector3(0.012, -0.25, 0), glove)               # mitt
		_e(el, Vector3(0.02, 0.04, 0.02), Vector3(0.055, -0.22, sz * -0.02), glove, Vector3(0, 0, -30))  # thumb
		j["sh" + side[0]] = sh
		j["el" + side[0]] = el

	# ---------------------------------------------------------------- legs (hips at y = -0.05)
	for side in [["L", 1.0], ["R", -1.0]]:
		var sz: float = side[1]
		var hip := M.node(body, Vector3(0, -0.05, sz * 0.12), Vector3.ZERO, "Hip" + side[0])
		M.mi(hip, M.capsule(0.105, 0.42, 14), denim, Vector3(0, -0.16, 0))                  # thigh
		var kn := M.node(hip, Vector3(0, -0.34, 0), Vector3.ZERO, "Knee" + side[0])
		M.mi(kn, M.capsule(0.09, 0.33, 14), denim, Vector3(0, -0.11, 0))                    # shin
		M.mi(kn, M.cyl(0.097, 0.1, 0.05, 14), cuff, Vector3(0, -0.2, 0))                    # rolled cuff
		var ank := M.node(kn, Vector3(0, -0.26, 0), Vector3.ZERO, "Ankle" + side[0])
		# boot: shaft + rounded foot + sole; the sole is on the box bottom when the leg is straight
		M.mi(ank, M.cyl(0.082, 0.088, 0.14, 14), boot, Vector3(0, -0.015, 0))
		_e(ank, Vector3(0.15, 0.065, 0.085), Vector3(0.05, -0.035, 0), boot)
		M.mi(ank, M.box(Vector3(0.3, 0.03, 0.165), 0.012), sole, Vector3(0.055, -0.085, 0))
		j["hip" + side[0]] = hip
		j["kn" + side[0]] = kn
		j["ank" + side[0]] = ank
	return j


static func pose(j: Dictionary, p: Dictionary) -> void:
	var vx := float(p.get("vx", 0.0))
	var vy := float(p.get("vy", 0.0))
	var grounded := int(p.get("g", 1)) == 1
	var facing_right := int(p.get("fr", 1)) == 1
	var step := float(p.get("step", 0))
	var vmax := float(p.get("vmax", 7.0))
	var body: Node3D = j["body"]
	body.rotation = Vector3(0, 0.0 if facing_right else PI, 0)
	var hipL := 0.0
	var hipR := 0.0
	var knL := 0.0
	var knR := 0.0
	var shL := 0.0
	var shR := 0.0
	var shLx := 0.0
	var shRx := 0.0
	var elL := 0.25
	var elR := 0.25
	var lean := 0.0
	var head_tilt := 0.0
	if grounded:
		if absf(vx) > 0.15:
			var ph := step * 0.45
			var s := sin(ph)
			var amp := clampf(absf(vx) / vmax * 1.25, 0.3, 1.0)
			hipL = 0.8 * amp * s
			hipR = -0.8 * amp * s
			knL = -amp * (0.25 + 1.0 * maxf(0.0, -s))
			knR = -amp * (0.25 + 1.0 * maxf(0.0, s))
			shL = -0.75 * amp * s
			shR = 0.75 * amp * s
			elL = 0.5 + 0.6 * amp
			elR = elL
			lean = -0.14 * amp
	elif vy > 0.0:
		# take-off / rising: lead knee up, trailing leg extended, arms swung forward
		hipL = 1.0
		knL = -1.3
		hipR = -0.3
		knR = -0.35
		shL = -0.5
		shR = 1.1
		elL = 0.9
		elR = 0.6
		lean = -0.1
		head_tilt = 0.1
	else:
		# falling: legs reaching down, arms out for balance
		hipL = 0.35
		knL = -0.55
		hipR = -0.1
		knR = -0.3
		shLx = -0.9
		shRx = 0.9
		shL = 0.3
		shR = 0.3
		elL = 0.5
		elR = 0.5
		head_tilt = -0.1
	(j["torso"] as Node3D).rotation = Vector3(0, 0, lean)
	(j["head"] as Node3D).rotation = Vector3(0, 0, head_tilt - lean * 0.5)
	(j["hipL"] as Node3D).rotation = Vector3(0, 0, hipL)
	(j["hipR"] as Node3D).rotation = Vector3(0, 0, hipR)
	(j["knL"] as Node3D).rotation = Vector3(0, 0, knL)
	(j["knR"] as Node3D).rotation = Vector3(0, 0, knR)
	# keep the soles level with the ground
	(j["ankL"] as Node3D).rotation = Vector3(0, 0, -(hipL + knL) * (0.7 if grounded else 0.3))
	(j["ankR"] as Node3D).rotation = Vector3(0, 0, -(hipR + knR) * (0.7 if grounded else 0.3))
	(j["shL"] as Node3D).rotation = Vector3(shLx, 0, shL)
	(j["shR"] as Node3D).rotation = Vector3(shRx, 0, shR)
	(j["elL"] as Node3D).rotation = Vector3(0, 0, elL)
	(j["elR"] as Node3D).rotation = Vector3(0, 0, elR)
	# planted foot: lift the body so the lowest sole touches the box bottom while running
	var lowest := minf(_foot_y(hipL, knL), _foot_y(hipR, knR))
	body.position = Vector3(0, (FEET - lowest) if grounded else 0.0, 0)


static func _foot_y(hip: float, kn: float) -> float:
	## Height of the sole (body frame) for the given hip / knee angles.
	var knee_y := -0.05 - 0.34 * cos(hip)
	var ank_y := knee_y - 0.26 * cos(hip + kn)
	return ank_y - 0.1
