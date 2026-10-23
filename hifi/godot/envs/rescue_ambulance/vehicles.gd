extends RefCounted
## Models of the Rescue Ambulance world: the ambulance (fitted to the CarRacing hull), the city bus,
## striped roadblocks, debris piles and the casualty site. All built in body-local Godot frames:
## Box2D local (lx, ly) -> (lx, h, -ly). The ambulance faces -Z (Box2D hull +Y); obstacles keep
## GymBox's box axes (w along local X, h along local Z).

const WHEEL_R := 0.42
const WHEEL_W := 0.32


static func dark_glass(tint := Color(0.03, 0.04, 0.055)) -> StandardMaterial3D:
	## Opaque tinted glass (windows sit directly on the painted body, so no transparency).
	var m := M.mat(tint, 0.05, 0.25, {"clearcoat": 1.0, "clearcoat_roughness": 0.03})
	m.metallic_specular = 0.75
	return m


static func _prism_x(parent: Node3D, poly: PackedVector2Array, x0: float, x1: float, mat: Material) -> MeshInstance3D:
	## Side profile (u = forward distance, y) extruded across x0..x1 (car-local, forward = -Z).
	return M.mi(parent, M.prism(poly, x0, x1), mat, Vector3.ZERO, Vector3(0, 90, 0))


# ================================================================ ambulance

static func ambulance(parent: Node3D) -> Dictionary:
	var root := Node3D.new()
	root.name = "Ambulance"
	parent.add_child(root)
	var white := M.paint(Color(0.93, 0.93, 0.92), 0.8, 0.12)
	var red := M.paint(Color(0.75, 0.04, 0.04), 0.75, 0.15)
	var dark := M.mat(Color(0.07, 0.07, 0.075), 0.6, 0.2)
	var trim := M.mat(Color(0.2, 0.2, 0.22), 0.45, 0.5)
	var glass := dark_glass()
	var chrome := M.mat(Color(0.82, 0.82, 0.84), 0.18, 1.0)
	# Chassis between the wheels and the patient-compartment box (u = Box2D forward)
	M.mi(root, M.box(Vector3(1.66, 0.5, 4.6), 0.03), dark, Vector3(0, 0.64, 0.05))
	var box_u0 := -2.38
	var box_u1 := 0.52
	var bl := box_u1 - box_u0
	var bz := -(box_u0 + box_u1) * 0.5
	M.mi(root, M.box(Vector3(2.3, 1.6, bl), 0.07), white, Vector3(0, 1.7, bz))
	# Side skirts (wheel arches stay open) and rear skirt
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.08, 0.48, 2.12), 0.02), white, Vector3(sx * 1.1, 0.68, 0.0))
		M.mi(root, M.box(Vector3(0.08, 0.48, 0.24), 0.02), white, Vector3(sx * 1.1, 0.68, 2.26))
		M.mi(root, M.box(Vector3(0.1, 0.08, 2.14), 0.01), trim, Vector3(sx * 1.13, 0.46, 0.0))
	# Cab + bonnet profile, extruded across the cab width
	var cab := PackedVector2Array([Vector2(0.5, 0.9), Vector2(2.42, 0.9), Vector2(2.55, 1.0), Vector2(2.55, 1.12),
		Vector2(1.55, 1.28), Vector2(0.98, 2.12), Vector2(0.5, 2.12)])
	_prism_x(root, cab, -1.06, 1.06, white)
	# Red band 1.0 - 1.5 m around the box and the cab
	M.mi(root, M.box(Vector3(2.32, 0.5, bl + 0.02), 0.0), red, Vector3(0, 1.25, bz))
	# (cab: two thin side strips below the window line + a front strip under the bonnet edge)
	var band := PackedVector2Array([Vector2(0.5, 1.0), Vector2(2.555, 1.0), Vector2(2.555, 1.1), Vector2(1.6, 1.25),
		Vector2(1.47, 1.38), Vector2(0.5, 1.38)])
	_prism_x(root, band, 1.055, 1.07, red)
	_prism_x(root, band, -1.07, -1.055, red)
	M.mi(root, M.box(Vector3(2.12, 0.1, 0.01), 0.0), red, Vector3(0, 1.05, -2.553))
	# Windscreen (on the slope (1.55, 1.28) -> (0.98, 2.12)) and cab side windows
	var ws := PackedVector2Array([Vector2(1.53, 1.33), Vector2(1.565, 1.33), Vector2(1.0, 2.1), Vector2(0.965, 2.1)])
	_prism_x(root, ws, -0.98, 0.98, glass)
	var sw := PackedVector2Array([Vector2(0.62, 1.38), Vector2(1.46, 1.38), Vector2(1.03, 2.02), Vector2(0.62, 2.02)])
	_prism_x(root, sw, -1.075, 1.075, glass)
	# Bumpers, grille, lights
	M.mi(root, M.box(Vector3(2.2, 0.42, 0.26), 0.05), trim, Vector3(0, 0.66, -2.5))
	M.mi(root, M.box(Vector3(2.24, 0.34, 0.16), 0.04), trim, Vector3(0, 0.62, 2.44))
	M.mi(root, M.box(Vector3(0.9, 0.17, 0.03), 0.01), dark, Vector3(0, 1.02, -2.565))
	var head := M.emissive(Color(1.0, 0.97, 0.9), 2.2)
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.36, 0.12, 0.03), 0.01), head, Vector3(sx * 0.74, 1.05, -2.57))
		M.mi(root, M.box(Vector3(0.12, 0.08, 0.03), 0.01), M.emissive(Color(1.0, 0.55, 0.05), 1.5), Vector3(sx * 1.0, 0.82, -2.6))
	var brake := M.mat(Color(0.35, 0.02, 0.02), 0.3, 0.0, {"emission": Color(1.0, 0.04, 0.02), "emission_energy": 0.8})
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.14, 0.6, 0.03), 0.01), brake, Vector3(sx * 1.05, 1.35, 2.395))
		M.mi(root, M.box(Vector3(0.14, 0.16, 0.03), 0.01), M.emissive(Color(1.0, 0.5, 0.05), 1.2), Vector3(sx * 1.05, 1.82, 2.395))
	# Rear doors: seam, windows, step
	M.mi(root, M.box(Vector3(0.025, 1.5, 0.02), 0.0), dark, Vector3(0, 1.7, 2.392))
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.62, 0.42, 0.02), 0.01), glass, Vector3(sx * 0.45, 2.1, 2.395))
	M.mi(root, M.box(Vector3(1.4, 0.06, 0.3), 0.01), chrome, Vector3(0, 0.5, 2.55))
	# Side door seams + small window on the box, red crosses (sides, rear, roof)
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.02, 1.55, 0.025), 0.0), dark, Vector3(sx * 1.152, 1.7, -0.32))
		M.mi(root, M.box(Vector3(0.02, 0.5, 0.42), 0.01), glass, Vector3(sx * 1.155, 2.0, -0.08))
		var cx := Vector3(sx * 1.157, 2.0, 1.25)
		M.mi(root, M.box(Vector3(0.012, 0.66, 0.2), 0.0), red, cx)
		M.mi(root, M.box(Vector3(0.012, 0.2, 0.66), 0.0), red, cx)
		# mirrors (stay inside the Side view's 1.6 m section cut)
		M.mi(root, M.box(Vector3(0.18, 0.03, 0.03), 0.0), dark, Vector3(sx * 1.14, 1.62, -1.45))
		M.mi(root, M.box(Vector3(0.05, 0.26, 0.15), 0.02), dark, Vector3(sx * 1.25, 1.66, -1.45))
	M.mi(root, M.box(Vector3(0.36, 0.012, 1.6), 0.0), red, Vector3(0, 2.506, 0.7))
	M.mi(root, M.box(Vector3(1.6, 0.012, 0.36), 0.0), red, Vector3(0, 2.506, 0.7))
	M.mi(root, M.box(Vector3(0.55, 0.15, 0.7), 0.03), M.mat(Color(0.85, 0.85, 0.84), 0.4, 0.3), Vector3(0, 2.57, 1.7))
	# Light bars: box front-top edge and cab roof; two alternating blue groups + white centre
	var la := M.mat(Color(0.05, 0.12, 0.6), 0.2, 0.0, {"emission": Color(0.15, 0.35, 1.0), "emission_energy": 6.0})
	var lb := M.mat(Color(0.05, 0.12, 0.6), 0.2, 0.0, {"emission": Color(0.15, 0.35, 1.0), "emission_energy": 6.0})
	M.mi(root, M.box(Vector3(1.9, 0.07, 0.3), 0.02), trim, Vector3(0, 2.535, -0.36))
	M.mi(root, M.box(Vector3(0.8, 0.1, 0.26), 0.03), la, Vector3(-0.5, 2.62, -0.36))
	M.mi(root, M.box(Vector3(0.8, 0.1, 0.26), 0.03), lb, Vector3(0.5, 2.62, -0.36))
	M.mi(root, M.box(Vector3(0.18, 0.1, 0.26), 0.03), M.emissive(Color(1, 1, 1), 2.0), Vector3(0, 2.62, -0.36))
	M.mi(root, M.box(Vector3(1.2, 0.1, 0.2), 0.03), la, Vector3(0, 2.17, -0.72))
	for sx in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.05, 0.12, 0.22), 0.02), lb if sx > 0 else la, Vector3(sx * 1.16, 2.38, -0.4))
	var glow_a := M.light(root, Color(0.25, 0.4, 1.0), 0.0, 7.0, Vector3(-0.6, 2.9, -0.4))
	var glow_b := M.light(root, Color(0.25, 0.4, 1.0), 0.0, 7.0, Vector3(0.6, 2.9, -0.4))
	return {"root": root, "la": la, "lb": lb, "ga": glow_a, "gb": glow_b, "brake": brake}


static func ambulance_wheel(parent: Node3D, left: bool) -> Node3D:
	## One wheel node (registered separately, posed from its Box2D wheel body); returns the spin node.
	var holder := M.node(parent, Vector3.ZERO, Vector3(0, 180 if left else 0, 0), "Hub")
	var tyre := M.mat(Color(0.045, 0.045, 0.05), 0.88, 0.0, {"normal_noise": 0.5, "noise_scale": 4.0})
	var rim := M.mat(Color(0.78, 0.79, 0.8), 0.28, 0.9)
	return M.wheel(holder, WHEEL_R, WHEEL_W, tyre, rim, 5)


# ================================================================ bus

static func bus(parent: Node3D, w: float, h: float, height: float) -> Node3D:
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Bus")
	var k := height / 4.0
	var yellow := M.paint(Color(0.93, 0.66, 0.08), 0.7, 0.2)
	var dark := M.mat(Color(0.06, 0.06, 0.065), 0.55, 0.2)
	var trim := M.mat(Color(0.16, 0.16, 0.17), 0.45, 0.4)
	var glass := dark_glass(Color(0.04, 0.05, 0.065))
	var roofm := M.mat(Color(0.9, 0.9, 0.87), 0.5, 0.1)
	var hw := w * 0.5
	var hd := h * 0.5
	var ax := hw - 1.85
	# Upper body (above the wheel arches) up to the roof cap
	M.mi(root, M.box(Vector3(w, (3.85 - 1.05) * k, h), 0.14), yellow, Vector3(0, (1.05 + 3.85) * 0.5 * k, 0))
	# Lower body segments between the arches + rubbing strip, chassis between the wheels
	for seg in [[-hw, -ax - 0.68], [-ax + 0.68, ax - 0.68], [ax + 0.68, hw]]:
		var L: float = seg[1] - seg[0]
		var cxs: float = (seg[0] + seg[1]) * 0.5
		M.mi(root, M.box(Vector3(L, (1.1 - 0.42) * k, h), 0.06), yellow, Vector3(cxs, (0.42 + 1.1) * 0.5 * k, 0))
		M.mi(root, M.box(Vector3(L + 0.01, 0.16 * k, h + 0.02), 0.02), dark, Vector3(cxs, 0.52 * k, 0))
	M.mi(root, M.box(Vector3(w - 0.6, 0.6 * k, h - 0.7), 0.02), dark, Vector3(0, 0.75 * k, 0))
	# Window band 0.55 H .. 0.85 H (as GymBox), pillars, windscreen (+X end), rear window
	var gz0 := 2.2 * k
	var gz1 := 3.4 * k
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(w - 1.2, gz1 - gz0, 0.02), 0.0), glass, Vector3(0.1, (gz0 + gz1) * 0.5, sz * (hd + 0.005)))
		var n := int((w - 1.4) / 1.35)
		for i in n + 1:
			var x := -hw + 0.6 + i * (w - 1.2) / n
			M.mi(root, M.box(Vector3(0.09, gz1 - gz0 + 0.02, 0.03), 0.0), trim, Vector3(x + 0.1, (gz0 + gz1) * 0.5, sz * (hd + 0.01)))
		# door (front, both sides) and side indicator
		M.mi(root, M.box(Vector3(1.1, 2.0 * k, 0.02), 0.0), glass, Vector3(hw - 1.1, 1.65 * k, sz * (hd + 0.006)))
		M.mi(root, M.box(Vector3(0.02, 2.0 * k, 0.03), 0.0), trim, Vector3(hw - 1.1, 1.65 * k, sz * (hd + 0.01)))
	M.mi(root, M.box(Vector3(0.02, (3.45 - 1.3) * k, h - 0.3), 0.0), glass, Vector3(hw + 0.005, (1.3 + 3.45) * 0.5 * k, 0))
	M.mi(root, M.box(Vector3(0.02, (3.35 - 2.35) * k, h - 0.6), 0.0), glass, Vector3(-hw - 0.005, (2.35 + 3.35) * 0.5 * k, 0))
	M.mi(root, M.box(Vector3(0.04, 0.24 * k, h - 0.9), 0.01), M.emissive(Color(1.0, 0.6, 0.1), 2.5), Vector3(hw + 0.01, 3.62 * k, 0))
	M.mi(root, M.box(Vector3(w - 0.5, 0.15 * k, h - 0.25), 0.04), roofm, Vector3(0, 3.925 * k, 0))
	# Bumpers, lights, mirrors
	M.mi(root, M.box(Vector3(0.2, 0.4 * k, h + 0.04), 0.05), trim, Vector3(hw + 0.05, 0.62 * k, 0))
	M.mi(root, M.box(Vector3(0.2, 0.4 * k, h + 0.04), 0.05), trim, Vector3(-hw - 0.05, 0.62 * k, 0))
	for sz in [-1.0, 1.0]:
		M.mi(root, M.box(Vector3(0.03, 0.16, 0.38), 0.01), M.emissive(Color(1.0, 0.97, 0.9), 1.6), Vector3(hw + 0.02, 1.0 * k, sz * (hd - 0.35)))
		M.mi(root, M.box(Vector3(0.03, 0.3, 0.2), 0.01), M.emissive(Color(0.9, 0.05, 0.03), 1.2), Vector3(-hw - 0.02, 1.2 * k, sz * (hd - 0.3)))
		M.mi(root, M.box(Vector3(0.06, 0.4, 0.2), 0.02), dark, Vector3(hw + 0.15, 2.6 * k, sz * (hd + 0.05)))
	# Wheels in the arches
	var tyre := M.mat(Color(0.045, 0.045, 0.05), 0.88, 0.0, {"normal_noise": 0.5, "noise_scale": 4.0})
	var rim := M.mat(Color(0.7, 0.7, 0.72), 0.35, 0.8)
	for x in [-ax, ax]:
		for sz in [-1.0, 1.0]:
			var holder := M.node(root, Vector3(x, 0.5, sz * (hd - 0.2)), Vector3(0, 90 if sz > 0 else -90, 0))
			M.wheel(holder, 0.5, 0.36, tyre, rim, 8)
	return root


# ================================================================ roadblock

static func roadblock(parent: Node3D, w: float, h: float, height: float) -> Dictionary:
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Roadblock")
	var n := 3
	var seg := w / n
	var base_h := 0.82 * height / 1.2
	# Jersey barriers (procedural profile, concrete texture), one per third of the length
	var conc := Assets.pbr("concrete_wall_008", {"uv": 1.2, "tint": Color(0.92, 0.92, 0.9), "color": Color(0.7, 0.7, 0.68)})
	var dd := minf(h, 0.7) * 0.5
	var prof := PackedVector2Array([Vector2(-dd, 0.0), Vector2(dd, 0.0), Vector2(dd, 0.08 * base_h), Vector2(dd * 0.45, 0.4 * base_h),
		Vector2(dd * 0.26, base_h), Vector2(-dd * 0.26, base_h), Vector2(-dd * 0.45, 0.4 * base_h), Vector2(-dd, 0.08 * base_h)])
	for i in n:
		var x := -w * 0.5 + seg * (i + 0.5)
		M.mi(root, M.prism(prof, -(seg - 0.05) * 0.5, (seg - 0.05) * 0.5), conc, Vector3(x, 0, 0), Vector3(0, 90, 0))
	# Red / white striped board on posts (GymBox stripes, 0.75 m period)
	var red := M.mat(Color(0.8, 0.06, 0.04), 0.45, 0.0, {"clearcoat": 0.6})
	var white := M.mat(Color(0.95, 0.95, 0.93), 0.35, 0.0, {"clearcoat": 0.6})
	var post := M.mat(Color(0.25, 0.25, 0.27), 0.5, 0.7)
	var b0 := base_h + 0.04
	var b1 := height - 0.02
	var m_count := maxi(1, int(round(w / 0.75)))
	for s in m_count:
		var x0 := -w * 0.5 + w * s / m_count
		var x1 := -w * 0.5 + w * (s + 1) / m_count
		M.mi(root, M.box(Vector3(x1 - x0, b1 - b0, 0.05), 0.0), red if s % 2 == 0 else white, Vector3((x0 + x1) * 0.5, (b0 + b1) * 0.5, 0))
	for i in n + 1:
		var x := clampf(-w * 0.5 + seg * i, -w * 0.5 + 0.15, w * 0.5 - 0.15)
		M.mi(root, M.cyl(0.025, 0.025, b1 - base_h + 0.1, 8), post, Vector3(x, (base_h + b1) * 0.5, -0.05))
	# Amber blinkers on the board ends
	var amber := M.mat(Color(0.6, 0.35, 0.02), 0.2, 0.0, {"emission": Color(1.0, 0.6, 0.05), "emission_energy": 4.0})
	for sx in [-1.0, 1.0]:
		M.mi(root, M.cyl(0.07, 0.07, 0.05, 12), post, Vector3(sx * (w * 0.5 - 0.2), height + 0.025, 0))
		M.mi(root, M.sphere(0.075, 12, 6), amber, Vector3(sx * (w * 0.5 - 0.2), height + 0.08, 0)).scale = Vector3(1, 0.8, 1)
	return {"root": root, "amber": amber}


# ================================================================ debris

static func debris(parent: Node3D, idx: int, w: float, h: float, height: float) -> Node3D:
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Debris")
	var style := idx % 3
	var rng := RandomNumberGenerator.new()
	rng.seed = 555 + idx * 31
	if style == 0:
		# Rubble: broken concrete slabs and blocks piled to the box size, with rebar sticking out
		var conc := Assets.pbr("concrete_wall_008", {"uv": 1.0, "tint": Color(0.85, 0.84, 0.8), "color": Color(0.6, 0.6, 0.58)})
		var brick := Assets.pbr("red_brick", {"uv": 1.5, "tint": Color(0.9, 0.85, 0.8), "color": Color(0.5, 0.3, 0.25)})
		M.mi(root, M.box(Vector3(w * 0.9, height * 0.42, h * 0.9), 0.12), conc, Vector3(0, height * 0.18, 0), Vector3(0, rng.randf_range(-8, 8), rng.randf_range(-4, 4)))
		M.mi(root, M.box(Vector3(w * 0.75, height * 0.14, h * 0.62), 0.04), conc, Vector3(w * 0.06, height * 0.5, -h * 0.05),
			Vector3(rng.randf_range(-14, 14), rng.randf_range(0, 40), rng.randf_range(-12, 12)))
		M.mi(root, M.box(Vector3(w * 0.42, height * 0.36, h * 0.38), 0.05), brick, Vector3(-w * 0.18, height * 0.62, h * 0.12),
			Vector3(rng.randf_range(-12, 12), rng.randf_range(0, 90), rng.randf_range(-12, 12)))
		M.mi(root, M.box(Vector3(w * 0.36, height * 0.2, h * 0.4), 0.04), conc, Vector3(w * 0.22, height * 0.82, h * 0.05),
			Vector3(rng.randf_range(-18, 18), rng.randf_range(0, 90), rng.randf_range(-18, 18)))
		for i in 4:
			var s := rng.randf_range(0.18, 0.32)
			M.mi(root, M.box(Vector3(s * 1.4, s * 0.6, s), 0.03), conc if i % 2 == 0 else brick,
				Vector3(rng.randf_range(-w, w) * 0.45, s * 0.25, rng.randf_range(-h, h) * 0.45),
				Vector3(rng.randf_range(-15, 15), rng.randf_range(0, 180), rng.randf_range(-15, 15)))
		var rebar := M.mat(Color(0.3, 0.18, 0.1), 0.7, 0.6)
		for i in 3:
			M.mi(root, M.cyl(0.012, 0.012, height * 0.7, 5), rebar, Vector3(rng.randf_range(-w, w) * 0.25, height * 0.62, rng.randf_range(-h, h) * 0.25),
				Vector3(rng.randf_range(-35, 35), 0, rng.randf_range(-35, 35)))
	elif style == 1:
		# Crates: a large crate filling the footprint with a smaller one on top
		var h0 := height * 0.58
		# crate_02 is long along its Z: turn it to the footprint's long side
		var c0 := Assets.fitted_model("wooden_crate_02", Vector3(h, h0, w) if w > h else Vector3(w, h0, h), "stretch")
		var c1 := Assets.fitted_model("wooden_crate_01", Vector3(w * 0.72, height - h0, h * 0.66), "stretch")
		if c0 == null or c1 == null:
			if c0 != null:
				c0.free()
			if c1 != null:
				c1.free()
			M.mi(root, M.box(Vector3(w, height, h), 0.04), Assets.pbr("weathered_planks", {"uv": 1.0, "color": Color(0.45, 0.33, 0.2)}), Vector3(0, height * 0.5, 0))
		else:
			if w > h:
				c0.rotation_degrees = Vector3(0, 90, 0)
			root.add_child(c0)
			c1.position = Vector3(w * 0.08, h0, -h * 0.06)
			c1.rotation_degrees = Vector3(0, 9, 0)
			root.add_child(c1)
	else:
		# Tyre stacks on a 2 x 2 grid (procedural low-poly tyres)
		var rad := minf(w, h) * 0.25 * 0.96
		var tyre := M.mat(Color(0.05, 0.05, 0.055), 0.85, 0.0, {"normal_noise": 0.6, "noise_scale": 5.0})
		for gx in [-1.0, 1.0]:
			for gz in [-1.0, 1.0]:
				var px: float = gx * w * 0.25
				var pz: float = gz * h * 0.25
				var stack_h: float = height * (1.0 if (gx + gz) != 0.0 else rng.randf_range(0.6, 1.0))
				var count := maxi(1, int(round(stack_h / (rad * 0.62))))
				var th := stack_h / count
				var prof := PackedVector2Array([Vector2(rad * 0.55, 0.0), Vector2(rad * 0.9, 0.0), Vector2(rad, th * 0.25),
					Vector2(rad, th * 0.75), Vector2(rad * 0.9, th), Vector2(rad * 0.55, th), Vector2(rad * 0.5, th * 0.5), Vector2(rad * 0.55, 0.0)])
				var tm := M.lathe(prof, 18)
				for i in count:
					M.mi(root, tm, tyre, Vector3(px + rng.randf_range(-0.03, 0.03), th * i, pz + rng.randf_range(-0.03, 0.03)))
	return root


# ================================================================ casualty site

static func casualty_site(parent: Node3D, radius: float, curb: Vector2) -> Dictionary:
	## Red marker disc with a white cross, striped beacon pole (as GymBox) and a casualty lying on it.
	## `curb` = Box2D unit vector towards the nearest road edge.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Casualty")
	var red := M.mat(Color(0.62, 0.03, 0.03), 0.75, 0.0, {"normal_noise": 0.3, "noise_scale": 3.0})
	var white := M.mat(Color(0.92, 0.92, 0.9), 0.7, 0.0)
	M.mi(root, M.cyl(radius, radius, 0.012, 40), red, Vector3(0, 0.026, 0))
	M.mi(root, M.box(Vector3(0.5, 0.004, 2.0), 0.0), white, Vector3(0, 0.034, 0))
	M.mi(root, M.box(Vector3(2.0, 0.004, 0.5), 0.0), white, Vector3(0, 0.034, 0))
	# Beacon pole 3 m: red 0-1, white 1-1.6, red 1.6-3
	var pr := M.mat(Color(0.8, 0.05, 0.04), 0.35, 0.1, {"clearcoat": 0.5})
	var pw := M.mat(Color(0.95, 0.95, 0.94), 0.3, 0.1, {"clearcoat": 0.5})
	M.mi(root, M.cyl(0.3, 0.33, 0.08, 20), M.mat(Color(0.08, 0.08, 0.08), 0.85), Vector3(0, 0.04, 0))
	M.mi(root, M.cyl(0.15, 0.16, 1.0, 16), pr, Vector3(0, 0.5, 0))
	M.mi(root, M.cyl(0.15, 0.15, 0.6, 16), pw, Vector3(0, 1.3, 0))
	M.mi(root, M.cyl(0.14, 0.15, 1.3, 16), pr, Vector3(0, 2.25, 0))
	var bm := M.mat(Color(0.6, 0.05, 0.03), 0.2, 0.0, {"emission": Color(1.0, 0.1, 0.05), "emission_energy": 5.0})
	var beacon := M.mi(root, M.sphere(0.13, 16, 8), bm, Vector3(0, 3.0, 0))
	beacon.scale = Vector3(1, 1.2, 1)
	var bl := M.light(root, Color(1.0, 0.15, 0.08), 0.0, 6.0, Vector3(0, 3.1, 0))
	# The casualty lies along the curb, 0.6 m to the road side of the pole
	var cg := Vector3(curb.x, 0, -curb.y)
	var along := Vector3(-cg.z, 0, cg.x)
	var body := M.node(root, -cg * 0.82, Vector3.ZERO, "Person")
	body.basis = Basis(Vector3.UP, atan2(-along.z, along.x))
	_person(body, 1.0 if body.basis.z.dot(cg) < 0.0 else -1.0)
	# First-aid bag on the other side of the pole
	var bag := M.mi(root, M.box(Vector3(0.5, 0.28, 0.32), 0.05), M.mat(Color(0.85, 0.3, 0.05), 0.6), cg * 0.75 + Vector3(0, 0.17, 0))
	bag.rotation_degrees = Vector3(0, 25, 0)
	M.mi(bag, M.box(Vector3(0.16, 0.05, 0.33), 0.0), white, Vector3(0, 0.0, 0.0))
	return {"root": root, "beacon": bm, "light": bl}


static func _person(p: Node3D, out := -1.0) -> void:
	## Lying on the back along local +X (head at +X), on the ground; one arm stretched out to local
	## Z side `out` (away from the beacon pole).
	var jacket := Assets.pbr("knitted_fleece", {"uv": 6.0, "tint": Color(0.25, 0.45, 0.75), "color": Color(0.2, 0.35, 0.6), "triplanar": true})
	var jeans := Assets.pbr("denim_fabric", {"uv": 5.0, "tint": Color(0.55, 0.55, 0.6), "color": Color(0.15, 0.2, 0.35)})
	var skin := M.mat(Color(0.78, 0.58, 0.45), 0.6, 0.0)
	var hair := M.mat(Color(0.12, 0.08, 0.05), 0.8, 0.0)
	var shoe := M.mat(Color(0.08, 0.07, 0.07), 0.55, 0.0)
	M.mi(p, M.capsule(0.16, 0.66), jacket, Vector3(0.18, 0.16, 0), Vector3(0, 0, 90)).scale = Vector3(1, 1, 1.3)
	M.mi(p, M.cyl(0.05, 0.05, 0.1, 10), skin, Vector3(0.56, 0.13, 0), Vector3(0, 0, 90))
	M.mi(p, M.sphere(0.105, 16, 10), skin, Vector3(0.68, 0.12, 0))
	M.mi(p, M.sphere(0.11, 16, 10), hair, Vector3(0.7, 0.13, 0)).scale = Vector3(1, 0.85, 1)
	for sz in [-1.0, 1.0]:
		M.mi(p, M.capsule(0.08, 0.9), jeans, Vector3(-0.52, 0.09, sz * 0.11), Vector3(0, sz * 4.0, 90))
		M.mi(p, M.box(Vector3(0.26, 0.11, 0.11), 0.03), shoe, Vector3(-1.0, 0.07, sz * 0.13), Vector3(0, 0, 70))
	M.mi(p, M.capsule(0.055, 0.6), jacket, Vector3(0.18, 0.08, -out * 0.31), Vector3(0, 0, 90))
	M.mi(p, M.sphere(0.05, 10, 6), skin, Vector3(-0.15, 0.07, -out * 0.33))
	var arm := M.mi(p, M.capsule(0.055, 0.6), jacket, Vector3(0.28, 0.08, out * 0.42), Vector3(0, out * 50.0, 90))
	arm.name = "ArmOut"
	M.mi(p, M.sphere(0.05, 10, 6), skin, Vector3(0.48, 0.07, out * 0.66))
