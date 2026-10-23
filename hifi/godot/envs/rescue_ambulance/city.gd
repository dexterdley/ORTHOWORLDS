extends RefCounted
## Static city builders of the Rescue Ambulance world: roads, markings, paved blocks, buildings
## (merged per material, tinted per building through vertex colours), street lamps, outskirts and
## a distant skyline. Box2D (x, y) at height h -> Godot (x, h, -y).

const FAMILIES := ["rectangular_facade_tiles", "exterior_wall_cladding", "concrete_wall_008", "brick_wall_02", "red_brick"]
const FAMILY_UV := [0.3, 0.36, 0.3, 0.55, 0.55]
# mean sRGB albedo of each texture (measured): the material brightens it to ~0.78, vertex colours tint it
const TEX_MEAN := [Color(0.35, 0.34, 0.31), Color(0.48, 0.45, 0.42), Color(0.55, 0.52, 0.44), Color(0.64, 0.56, 0.46), Color(0.56, 0.40, 0.32)]
const TEX_TARGET := 0.74
const ROAD_Y := 0.015
const MARK_Y := 0.022
const WALK_Y := 0.08
const SIGN_COLORS := [Color(0.12, 0.25, 0.18), Color(0.1, 0.13, 0.28), Color(0.35, 0.08, 0.1),
	Color(0.08, 0.08, 0.09), Color(0.55, 0.38, 0.1), Color(0.18, 0.3, 0.4)]


# ================================================================ low-level mesh building

static func g(x: float, y: float, h: float) -> Vector3:
	return Vector3(x, h, -y)


static func gn(n: Vector2) -> Vector3:
	return Vector3(n.x, 0.0, -n.y)


static func tri(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, n: Vector3, col: Color) -> void:
	if (b - a).cross(c - a).dot(n) > 0.0:
		var t := b
		b = c
		c = t
	for v in [a, b, c]:
		st.set_color(col)
		st.set_normal(n)
		st.set_uv(_uv(v, n))
		st.add_vertex(v)


static func quad(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, n: Vector3, col: Color) -> void:
	tri(st, a, b, c, n, col)
	tri(st, a, c, d, n, col)


static func _uv(p: Vector3, n: Vector3) -> Vector2:
	var an := n.abs()
	if an.x >= an.y and an.x >= an.z:
		return Vector2(p.z, p.y)
	if an.y >= an.z:
		return Vector2(p.x, p.z)
	return Vector2(p.x, p.y)


static func hquad(st: SurfaceTool, x0: float, y0: float, x1: float, y1: float, h: float, col: Color, up := true) -> void:
	## Horizontal Box2D rect at height h.
	quad(st, g(x0, y0, h), g(x1, y0, h), g(x1, y1, h), g(x0, y1, h), Vector3.UP if up else Vector3.DOWN, col)


static func vquad(st: SurfaceTool, p0: Vector2, p1: Vector2, z0: float, z1: float, n2: Vector2, col: Color) -> void:
	## Vertical quad over the Box2D segment p0-p1, outward normal n2.
	quad(st, g(p0.x, p0.y, z0), g(p1.x, p1.y, z0), g(p1.x, p1.y, z1), g(p0.x, p0.y, z1), gn(n2), col)


static func slab(st: SurfaceTool, x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, col: Color, bottom := false) -> void:
	## Axis-aligned Box2D box from z0 to z1 (top + 4 sides, optional bottom).
	hquad(st, x0, y0, x1, y1, z1, col)
	if bottom:
		hquad(st, x0, y0, x1, y1, z0, col, false)
	vquad(st, Vector2(x0, y0), Vector2(x1, y0), z0, z1, Vector2(0, -1), col)
	vquad(st, Vector2(x1, y0), Vector2(x1, y1), z0, z1, Vector2(1, 0), col)
	vquad(st, Vector2(x1, y1), Vector2(x0, y1), z0, z1, Vector2(0, 1), col)
	vquad(st, Vector2(x0, y1), Vector2(x0, y0), z0, z1, Vector2(-1, 0), col)


static func facade_box(st: SurfaceTool, p0: Vector2, u: Vector2, n2: Vector2, s0: float, s1: float, z0: float, z1: float,
		d0: float, d1: float, col: Color) -> void:
	## Box on a facade: along u from s0 to s1, out along n2 from d0 to d1, heights z0..z1 (no back face).
	var a := p0 + u * s0
	var b := p0 + u * s1
	var fa := a + n2 * d1
	var fb := b + n2 * d1
	var ba := a + n2 * d0
	var bb := b + n2 * d0
	vquad(st, fa, fb, z0, z1, n2, col)
	quad(st, g(ba.x, ba.y, z1), g(bb.x, bb.y, z1), g(fb.x, fb.y, z1), g(fa.x, fa.y, z1), Vector3.UP, col)
	if z0 > 0.01:
		quad(st, g(ba.x, ba.y, z0), g(bb.x, bb.y, z0), g(fb.x, fb.y, z0), g(fa.x, fa.y, z0), Vector3.DOWN, col)
	vquad(st, ba, fa, z0, z1, -u, col)
	vquad(st, bb, fb, z0, z1, u, col)


static func commit(parent: Node3D, st: SurfaceTool, mat: Material, node_name := "") -> MeshInstance3D:
	var m := MeshInstance3D.new()
	if node_name != "":
		m.name = node_name
	m.mesh = st.commit()
	m.material_override = mat
	parent.add_child(m)
	return m


static func new_st() -> SurfaceTool:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	return st


# ================================================================ materials

static func family_of(c: Color) -> int:
	var brick := c.r > c.b * 1.35
	var k := int(round(c.r * 255.0 + c.g * 255.0))
	return (3 + k % 2) if brick else (k % 3)


static func tint_of(c: Color, fam: int) -> Color:
	## Vertex tint so that the brightened texture averages towards the GymBox building colour.
	var tm: Color = TEX_MEAN[fam]
	var b := TEX_TARGET / maxf(tm.r, maxf(tm.g, tm.b))
	var t := Color(c.r / (tm.r * b), c.g / (tm.g * b), c.b / (tm.b * b))
	t = Color.WHITE.lerp(t, 0.75)
	return Color(clampf(t.r, 0.0, 1.0), clampf(t.g, 0.0, 1.0), clampf(t.b, 0.0, 1.0))


static func vc(mat: BaseMaterial3D, boost := 1.0) -> BaseMaterial3D:
	## Copy of a (cached) material that multiplies its albedo by the (sRGB) vertex colour.
	var m: BaseMaterial3D = mat.duplicate()
	m.vertex_color_use_as_albedo = true
	m.vertex_color_is_srgb = true
	if boost != 1.0:
		m.albedo_color = Color(boost, boost, boost)
	return m


static func wall_material(fam: int) -> BaseMaterial3D:
	var tm: Color = TEX_MEAN[fam]
	var b := TEX_TARGET / maxf(tm.r, maxf(tm.g, tm.b))
	return vc(Assets.pbr(FAMILIES[fam], {"uv": FAMILY_UV[fam], "world": true, "color": Color(0.6, 0.6, 0.58)}), b)


# ================================================================ ground layers

static func roads(parent: Node3D, rects: Array, road_r: float, pitch: float) -> void:
	var st := new_st()
	for q in rects:
		hquad(st, q[0], q[1], q[2], q[3], ROAD_Y, Color.WHITE)
	var asphalt := Assets.pbr("asphalt_02", {"uv": 0.22, "world": true, "tint": Color(0.5, 0.5, 0.52), "color": Color(0.25, 0.25, 0.26)})
	commit(parent, st, asphalt, "Roads")


static func markings(parent: Node3D, dashes: Array, rects: Array, road_r: float) -> void:
	var st := new_st()
	var white := Color(0.92, 0.92, 0.9)
	for d in dashes:
		var x: float = d[0]
		var y: float = d[1]
		if int(d[2]) == 1:
			hquad(st, x - 1.5, y - 0.15, x + 1.5, y + 0.15, MARK_Y, white)
		else:
			hquad(st, x - 0.15, y - 1.5, x + 0.15, y + 1.5, MARK_Y, white)
	# Solid edge lines 0.45 m inside the curbs of every road segment (not across intersections)
	for q in rects:
		var w: float = q[2] - q[0]
		var h: float = q[3] - q[1]
		if absf(w - h) < 0.5:
			continue
		if w > h:
			for yy in [q[1] + 0.45, q[3] - 0.45]:
				hquad(st, q[0] + 0.6, yy - 0.075, q[2] - 0.6, yy + 0.075, MARK_Y, white)
			# stop lines of the right-hand lanes (eastbound south half at the east end, westbound north half at the west end)
			hquad(st, q[2] - 0.9, q[1] + 0.3, q[2] - 0.5, (q[1] + q[3]) * 0.5 - 0.15, MARK_Y, white)
			hquad(st, q[0] + 0.5, (q[1] + q[3]) * 0.5 + 0.15, q[0] + 0.9, q[3] - 0.3, MARK_Y, white)
		else:
			for xx in [q[0] + 0.45, q[2] - 0.45]:
				hquad(st, xx - 0.075, q[1] + 0.6, xx + 0.075, q[3] - 0.6, MARK_Y, white)
			hquad(st, (q[0] + q[2]) * 0.5 + 0.15, q[3] - 0.9, q[2] - 0.3, q[3] - 0.5, MARK_Y, white)
			hquad(st, q[0] + 0.3, q[1] + 0.5, (q[0] + q[2]) * 0.5 - 0.15, q[1] + 0.9, MARK_Y, white)
	var paint := M.mat(Color(0.9, 0.9, 0.88), 0.55, 0.0, {"noise": 1.0, "noise_scale": 0.35})
	paint.vertex_color_use_as_albedo = true
	var mi := commit(parent, st, paint, "Markings")
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


static func blocks(parent: Node3D, blks: Array) -> void:
	## One paved slab per block (sidewalk ring + alleys) with a concrete curb ring.
	var st := new_st()
	var curb := new_st()
	var cw := 0.22
	for q in blks:
		var x0: float = q[0]
		var y0: float = q[1]
		var x1: float = q[2]
		var y1: float = q[3]
		hquad(st, x0 + cw, y0 + cw, x1 - cw, y1 - cw, WALK_Y, Color.WHITE)
		var c := Color.WHITE
		for r in [[x0, y0, x1, y0 + cw], [x0, y1 - cw, x1, y1], [x0, y0 + cw, x0 + cw, y1 - cw], [x1 - cw, y0 + cw, x1, y1 - cw]]:
			slab(curb, r[0], r[1], r[2], r[3], 0.0, WALK_Y + 0.01, c)
	var pave := Assets.pbr("pavement_02", {"uv": 0.45, "world": true, "tint": Color(0.86, 0.96, 1.14), "color": Color(0.6, 0.6, 0.58)})
	commit(parent, st, pave, "Sidewalks")
	var conc := Assets.pbr("concrete_floor_02", {"uv": 0.6, "world": true, "tint": Color(1.0, 1.0, 0.98), "color": Color(0.7, 0.7, 0.68)})
	commit(parent, curb, conc, "Curbs")


# ================================================================ buildings

static func buildings(root: Node3D, blds: Array, blks: Array) -> void:
	## Every building of the episode, merged per material. Walls rise from 0 to the exact GymBox
	## height (parapet top); the flat roof is recessed 0.3 m; rooftop units stay below height + 0.25.
	var walls: Array = []
	for f in FAMILIES.size():
		walls.append(new_st())
	var glass := new_st()
	var frame := new_st()
	var trim := new_st()
	var roof_a := new_st()
	var roof_b := new_st()
	var units := new_st()
	var solar := new_st()
	for k in blds.size():
		var b: Array = blds[k]
		var x0: float = b[0]
		var y0: float = b[1]
		var x1: float = b[2]
		var y1: float = b[3]
		var h: float = b[4]
		var c := Color8(int(b[5][0]), int(b[5][1]), int(b[5][2]))
		var fam := family_of(c)
		var tint := tint_of(c, fam)
		var rng := RandomNumberGenerator.new()
		rng.seed = 7919 * (k + 1) + int(x0 * 13.0) + int(y0 * 7.0)
		var blk := _block_of(blks, x0, y0, x1, y1)
		var frame_col := Color(0.93, 0.93, 0.9) if fam >= 2 else Color(0.16, 0.17, 0.18)
		var sign_col: Color = SIGN_COLORS[rng.randi() % SIGN_COLORS.size()]
		var faces := [[Vector2(x0, y0), Vector2(x1, y0), Vector2(0, -1)], [Vector2(x1, y0), Vector2(x1, y1), Vector2(1, 0)],
			[Vector2(x1, y1), Vector2(x0, y1), Vector2(0, 1)], [Vector2(x0, y1), Vector2(x0, y0), Vector2(-1, 0)]]
		for fc in faces:
			var p0: Vector2 = fc[0]
			var p1: Vector2 = fc[1]
			var n2: Vector2 = fc[2]
			var L := p0.distance_to(p1)
			var u := (p1 - p0) / L
			vquad(walls[fam], p0, p1, 0.0, h, n2, tint)
			var street := _street_facing(blk, p0, p1, n2)
			# plinth + cornice band
			if street:
				facade_box(trim, p0, u, n2, -0.04, L + 0.04, 0.0, 0.45, 0.0, 0.05, Color(0.42, 0.41, 0.4))
			if h > 4.2:
				facade_box(trim, p0, u, n2, -0.08, L + 0.08, 3.3, 3.45, 0.0, 0.1, Color(0.82, 0.81, 0.78))
			_windows(glass, frame, trim, p0, u, n2, L, h, street, frame_col, sign_col, rng)
		# Roof: recessed slab, inner parapet faces, coping ring
		var ins := 0.25
		var rh := h - 0.3
		var roof_st: SurfaceTool = roof_a if rng.randf() < 0.55 else roof_b
		# roofs lean slightly towards the building colour (as GymBox's roof tops) with some variation
		var m := maxf(c.r, maxf(c.g, c.b))
		var rc := Color.WHITE.lerp(Color(c.r / m, c.g / m, c.b / m), 0.3) * (0.78 + 0.25 * rng.randf())
		hquad(roof_st, x0 + ins, y0 + ins, x1 - ins, y1 - ins, rh, rc)
		var ix0 := x0 + ins
		var iy0 := y0 + ins
		var ix1 := x1 - ins
		var iy1 := y1 - ins
		vquad(walls[fam], Vector2(ix0, iy0), Vector2(ix1, iy0), rh, h, Vector2(0, 1), tint * 0.9)
		vquad(walls[fam], Vector2(ix1, iy0), Vector2(ix1, iy1), rh, h, Vector2(-1, 0), tint * 0.9)
		vquad(walls[fam], Vector2(ix1, iy1), Vector2(ix0, iy1), rh, h, Vector2(0, -1), tint * 0.9)
		vquad(walls[fam], Vector2(ix0, iy1), Vector2(ix0, iy0), rh, h, Vector2(1, 0), tint * 0.9)
		var cc := Color(0.78, 0.77, 0.74)
		for r in [[x0 - 0.04, y0 - 0.04, x1 + 0.04, iy0], [x0 - 0.04, iy1, x1 + 0.04, y1 + 0.04],
				[x0 - 0.04, iy0, ix0, iy1], [ix1, iy0, x1 + 0.04, iy1]]:
			hquad(trim, r[0], r[1], r[2], r[3], h, cc)
		_rooftop(units, solar, glass, rng, ix0 + 0.4, iy0 + 0.4, ix1 - 0.4, iy1 - 0.4, rh)
	for f in FAMILIES.size():
		commit(root, walls[f], wall_material(f), "Walls_" + FAMILIES[f])
	var gm := M.mat(Color(0.32, 0.38, 0.44), 0.06, 0.55)
	gm.vertex_color_use_as_albedo = true
	gm.vertex_color_is_srgb = true
	gm.metallic_specular = 0.8
	commit(root, glass, gm, "Glass")
	var fm := M.mat(Color(1, 1, 1), 0.35, 0.4)
	fm.vertex_color_use_as_albedo = true
	fm.vertex_color_is_srgb = true
	commit(root, frame, fm, "Frames")
	var tm := vc(Assets.pbr("concrete_wall_008", {"uv": 0.6, "world": true, "color": Color(0.7, 0.7, 0.68)}), 1.35)
	commit(root, trim, tm, "Trim")
	commit(root, roof_a, vc(Assets.pbr("gravel", {"uv": 0.2, "world": true, "tint": Color(1.75, 2.15, 2.8), "color": Color(0.5, 0.5, 0.5)})), "RoofGravel")
	commit(root, roof_b, vc(Assets.pbr("concrete_floor_02", {"uv": 0.22, "world": true, "tint": Color(1.15, 1.24, 1.4), "color": Color(0.55, 0.55, 0.55)})), "RoofConcrete")
	var um := M.mat(Color(1, 1, 1), 0.45, 0.7, {"normal_noise": 0.4, "noise_scale": 2.0})
	um.vertex_color_use_as_albedo = true
	um.vertex_color_is_srgb = true
	commit(root, units, um, "RoofUnits")
	var sm := M.mat(Color(0.05, 0.08, 0.16), 0.12, 0.6, {"clearcoat": 1.0})
	sm.vertex_color_use_as_albedo = true
	commit(root, solar, sm, "Solar")


static func _block_of(blks: Array, x0: float, y0: float, x1: float, y1: float) -> Array:
	for q in blks:
		if x0 >= q[0] - 0.01 and x1 <= q[2] + 0.01 and y0 >= q[1] - 0.01 and y1 <= q[3] + 0.01:
			return q
	return [x0 - 2.0, y0 - 2.0, x1 + 2.0, y1 + 2.0]


static func _street_facing(blk: Array, p0: Vector2, p1: Vector2, n2: Vector2) -> bool:
	## True if the facade lies on the block's inner sidewalk edge (faces a road, not an alley).
	if n2.y < -0.5:
		return absf(p0.y - (blk[1] + 2.0)) < 0.05
	if n2.y > 0.5:
		return absf(p0.y - (blk[3] - 2.0)) < 0.05
	if n2.x < -0.5:
		return absf(p0.x - (blk[0] + 2.0)) < 0.05
	return absf(p0.x - (blk[2] - 2.0)) < 0.05


static func _window(glass: SurfaceTool, frame: SurfaceTool, p0: Vector2, u: Vector2, n2: Vector2, s: float, ww: float,
		z0: float, z1: float, fcol: Color, gcol: Color, mullions: int) -> void:
	var f := 0.07
	var g0 := p0 + u * (s - ww * 0.5) + n2 * 0.025
	var g1 := p0 + u * (s + ww * 0.5) + n2 * 0.025
	vquad(glass, g0, g1, z0, z1, n2, gcol)
	facade_box(frame, p0, u, n2, s - ww * 0.5 - f, s + ww * 0.5 + f, z1, z1 + f, 0.0, 0.07, fcol)
	facade_box(frame, p0, u, n2, s - ww * 0.5 - f - 0.04, s + ww * 0.5 + f + 0.04, z0 - f - 0.02, z0, 0.0, 0.11, fcol)
	facade_box(frame, p0, u, n2, s - ww * 0.5 - f, s - ww * 0.5, z0, z1, 0.0, 0.07, fcol)
	facade_box(frame, p0, u, n2, s + ww * 0.5, s + ww * 0.5 + f, z0, z1, 0.0, 0.07, fcol)
	for m in mullions:
		var sm := s - ww * 0.5 + ww * float(m + 1) / float(mullions + 1)
		facade_box(frame, p0, u, n2, sm - 0.03, sm + 0.03, z0, z1, 0.0, 0.05, fcol)


static func _windows(glass: SurfaceTool, frame: SurfaceTool, trim: SurfaceTool, p0: Vector2, u: Vector2, n2: Vector2,
		L: float, h: float, street: bool, fcol: Color, sign_col: Color, rng: RandomNumberGenerator) -> void:
	# Ground floor: shopfronts on streets (with a fascia band), small windows on alleys
	var spacing := 3.6 if street else 4.2
	var ww := 2.6 if street else 1.0
	var n := int(floor((L - 0.8) / spacing))
	if n >= 1:
		var off := (L - n * spacing) * 0.5
		for k in n:
			var s := off + spacing * (k + 0.5)
			var gcol := Color(0.6, 0.62, 0.66) * (0.75 + 0.5 * rng.randf())
			if street:
				_window(glass, frame, p0, u, n2, s, ww, 0.62, 2.75, fcol, gcol, 1)
			else:
				_window(glass, frame, p0, u, n2, s, ww, 1.25, 2.45, fcol, gcol, 0)
		if street and h > 4.2:
			facade_box(trim, p0, u, n2, off * 0.5, L - off * 0.5, 2.88, 3.25, 0.0, 0.09, sign_col)
	# Upper storeys every 3 m above the cornice
	var zb := 3.45
	while zb + 1.3 <= h - 0.35:
		var z0 := zb + 0.55
		var z1 := minf(zb + 2.15, h - 0.45)
		if z1 - z0 < 0.8:
			break
		var sp := 2.7
		var nu := int(floor((L - 0.6) / sp))
		if nu >= 1:
			var off2 := (L - nu * sp) * 0.5
			for k in nu:
				var gcol2 := Color(0.6, 0.62, 0.66) * (0.7 + 0.6 * rng.randf())
				_window(glass, frame, p0, u, n2, off2 + sp * (k + 0.5), 1.35, z0, z1, fcol, gcol2, 1 if rng.randf() < 0.5 else 0)
		zb += 3.0


static func _rooftop(units: SurfaceTool, solar: SurfaceTool, glass: SurfaceTool, rng: RandomNumberGenerator,
		x0: float, y0: float, x1: float, y1: float, z: float) -> void:
	## HVAC units, vents, skylights and sometimes a solar array; everything stays below z + 0.55.
	if x1 - x0 < 2.0 or y1 - y0 < 2.0:
		return
	var n := 2 + rng.randi() % 4
	for i in n:
		var w := rng.randf_range(0.9, 1.8)
		var d := rng.randf_range(0.8, 1.3)
		var hh := rng.randf_range(0.35, 0.5)
		var cx := rng.randf_range(x0 + w * 0.5, maxf(x0 + w * 0.5, x1 - w * 0.5))
		var cy := rng.randf_range(y0 + d * 0.5, maxf(y0 + d * 0.5, y1 - d * 0.5))
		var col := Color(0.78, 0.79, 0.8) * rng.randf_range(0.8, 1.05)
		slab(units, cx - w * 0.5, cy - d * 0.5, cx + w * 0.5, cy + d * 0.5, z, z + hh, col)
		# fan grilles on top
		var nf := 1 if w < 1.3 else 2
		for k in nf:
			var fx := cx - w * 0.5 + w * (k + 0.5) / nf
			var r := minf(w / nf, d) * 0.32
			hquad(units, fx - r, cy - r, fx + r, cy + r, z + hh + 0.01, Color(0.12, 0.12, 0.13))
		# duct to the roof
		slab(units, cx - 0.12, cy - d * 0.5 - 0.6, cx + 0.12, cy - d * 0.5, z, z + 0.22, col * 0.9)
	for i in 3 + rng.randi() % 4:
		var vx := rng.randf_range(x0, x1)
		var vy := rng.randf_range(y0, y1)
		slab(units, vx - 0.1, vy - 0.1, vx + 0.1, vy + 0.1, z, z + 0.3, Color(0.35, 0.36, 0.37))
	for i in rng.randi() % 3:
		var sw := rng.randf_range(1.0, 1.6)
		var sd := rng.randf_range(1.8, 3.0)
		var sx := rng.randf_range(x0 + sw, maxf(x0 + sw, x1 - sw))
		var sy := rng.randf_range(y0 + sd, maxf(y0 + sd, y1 - sd))
		slab(units, sx - sw * 0.5 - 0.08, sy - sd * 0.5 - 0.08, sx + sw * 0.5 + 0.08, sy + sd * 0.5 + 0.08, z, z + 0.18, Color(0.85, 0.85, 0.86))
		hquad(glass, sx - sw * 0.5, sy - sd * 0.5, sx + sw * 0.5, sy + sd * 0.5, z + 0.2, Color(0.75, 0.8, 0.85))
	if rng.randf() < 0.45 and (x1 - x0) > 5.0 and (y1 - y0) > 5.0:
		# Solar array: rows of tilted panels (top <= roof + 0.5)
		var rows := mini(int(floor((y1 - y0 - 1.0) / 1.6)), 4)
		var cols := mini(int(floor((x1 - x0 - 1.0) / 1.1)), 6)
		var sx0 := x1 - 0.5 - cols * 1.1
		var sy0 := y1 - 0.5 - rows * 1.6
		for rI in rows:
			for cI in cols:
				var px := sx0 + cI * 1.1
				var py := sy0 + rI * 1.6
				var a := g(px, py, z + 0.12)
				var b := g(px + 1.0, py, z + 0.12)
				var c := g(px + 1.0, py + 1.0, z + 0.48)
				var d := g(px, py + 1.0, z + 0.48)
				var nrm := (b - a).cross(d - a).normalized()
				if nrm.y < 0.0:
					nrm = -nrm
				quad(solar, a, b, c, d, nrm, Color(1, 1, 1))


# ================================================================ street furniture

static func lamp_transforms(blks: Array) -> Array:
	## Two lamps per block side on the sidewalk, 0.45 m from the curb, arm towards the road.
	var out: Array = []
	for q in blks:
		var x0: float = q[0]
		var y0: float = q[1]
		var x1: float = q[2]
		var y1: float = q[3]
		for t in [0.27, 0.73]:
			var x := lerpf(x0, x1, t)
			var y := lerpf(y0, y1, t)
			out.append([Vector2(x, y0 + 0.45), Vector2(0, -1)])
			out.append([Vector2(x, y1 - 0.45), Vector2(0, 1)])
			out.append([Vector2(x0 + 0.45, y), Vector2(-1, 0)])
			out.append([Vector2(x1 - 0.45, y), Vector2(1, 0)])
	return out


static func lamp_mesh(height: float) -> ArrayMesh:
	## Low-poly modern LED street light (pole, arm towards local +X, flat luminaire), two surfaces.
	var metal := new_st()
	var lens := new_st()
	var parts := [
		[M.cyl(0.13, 0.16, 0.6, 10), Transform3D(Basis(), Vector3(0, 0.3, 0)), metal],
		[M.cyl(0.055, 0.085, height - 0.6, 10), Transform3D(Basis(), Vector3(0, 0.6 + (height - 0.6) * 0.5, 0)), metal],
		[M.cyl(0.04, 0.05, 1.5, 8), Transform3D(Basis(Vector3.FORWARD, deg_to_rad(-80.0)), Vector3(0.72, height + 0.05, 0)), metal],
		[M.box(Vector3(0.7, 0.11, 0.3), 0.03), Transform3D(Basis(Vector3.FORWARD, deg_to_rad(-6.0)), Vector3(1.55, height + 0.12, 0)), metal],
		[M.box(Vector3(0.6, 0.02, 0.24), 0.0), Transform3D(Basis(Vector3.FORWARD, deg_to_rad(-6.0)), Vector3(1.55, height + 0.06, 0)), lens],
	]
	for pt in parts:
		(pt[2] as SurfaceTool).append_from(pt[0], 0, pt[1])
	var am := ArrayMesh.new()
	metal.commit(am)
	lens.commit(am)
	am.surface_set_material(0, M.mat(Color(0.17, 0.18, 0.19), 0.6, 0.5))
	am.surface_set_material(1, M.mat(Color(0.75, 0.75, 0.72), 0.3, 0.0, {"emission": Color(1.0, 0.95, 0.85), "emission_energy": 0.2}))
	return am


static func lamps(parent: Node3D, blks: Array, height := 5.4) -> void:
	var xforms: Array = []
	for it in lamp_transforms(blks):
		var p: Vector2 = it[0]
		var d: Vector2 = it[1]
		var yaw := atan2(-gn(d).z, gn(d).x)      # local +X (arm) towards the road
		xforms.append(Transform3D(Basis(Vector3.UP, yaw), Vector3(p.x, WALK_Y, -p.y)))
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = lamp_mesh(height)
	mm.instance_count = xforms.size()
	for i in xforms.size():
		mm.set_instance_transform(i, xforms[i])
	var mmi := MultiMeshInstance3D.new()
	mmi.name = "Lamps"
	mmi.multimesh = mm
	parent.add_child(mmi)


static func _multi(parent: Node3D, mesh: Mesh, xforms: Array, node_name: String) -> void:
	if xforms.is_empty():
		return
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = mesh
	mm.instance_count = xforms.size()
	for i in xforms.size():
		mm.set_instance_transform(i, xforms[i])
	var mmi := MultiMeshInstance3D.new()
	mmi.name = node_name
	mmi.multimesh = mm
	parent.add_child(mmi)


static func _parts_mesh(parts: Array) -> ArrayMesh:
	## parts: [[mesh, Transform3D, material index]], mats: one surface per material index.
	var sts := {}
	for pt in parts:
		var k: int = pt[2]
		if not sts.has(k):
			sts[k] = new_st()
		(sts[k] as SurfaceTool).append_from(pt[0], 0, pt[1])
	var am := ArrayMesh.new()
	var keys := sts.keys()
	keys.sort()
	for k in keys:
		(sts[k] as SurfaceTool).commit(am)
	return am


static func props(parent: Node3D, blks: Array) -> void:
	## Litter bins, fire hydrants and utility cabinets near block corners (deterministic).
	var rng := RandomNumberGenerator.new()
	rng.seed = 4242
	var bins: Array = []
	var hyd: Array = []
	var boxes: Array = []
	for q in blks:
		for corner in [[q[0] + 1.0, q[1] + 1.0], [q[2] - 1.0, q[3] - 1.0], [q[0] + 1.0, q[3] - 1.0], [q[2] - 1.0, q[1] + 1.0]]:
			var r := rng.randf()
			var xf := Transform3D(Basis(Vector3.UP, rng.randi() % 4 * PI * 0.5), Vector3(corner[0], WALK_Y, -corner[1]))
			if r < 0.16:
				boxes.append(xf)
			elif r < 0.4:
				bins.append(xf)
			elif r < 0.55:
				hyd.append(xf)
	var bin_mesh := _parts_mesh([
		[M.cyl(0.26, 0.24, 0.8, 14), Transform3D(Basis(), Vector3(0, 0.42, 0)), 0],
		[M.cyl(0.28, 0.28, 0.06, 14), Transform3D(Basis(), Vector3(0, 0.85, 0)), 1],
		[M.cyl(0.1, 0.1, 0.03, 10), Transform3D(Basis(), Vector3(0, 0.03, 0)), 1]])
	bin_mesh.surface_set_material(0, M.mat(Color(0.12, 0.25, 0.17), 0.45, 0.6))
	bin_mesh.surface_set_material(1, M.mat(Color(0.2, 0.2, 0.21), 0.5, 0.7))
	_multi(parent, bin_mesh, bins, "Bins")
	var hyd_mesh := _parts_mesh([
		[M.cyl(0.11, 0.13, 0.55, 12), Transform3D(Basis(), Vector3(0, 0.3, 0)), 0],
		[M.sphere(0.12, 12, 6), Transform3D(Basis(), Vector3(0, 0.6, 0)), 0],
		[M.cyl(0.04, 0.04, 0.38, 8), Transform3D(Basis(Vector3.FORWARD, PI * 0.5), Vector3(0, 0.42, 0)), 1],
		[M.cyl(0.16, 0.16, 0.05, 12), Transform3D(Basis(), Vector3(0, 0.05, 0)), 0]])
	hyd_mesh.surface_set_material(0, M.mat(Color(0.75, 0.08, 0.05), 0.4, 0.3, {"clearcoat": 0.5}))
	hyd_mesh.surface_set_material(1, M.mat(Color(0.75, 0.72, 0.6), 0.35, 0.8))
	_multi(parent, hyd_mesh, hyd, "Hydrants")
	var m := Assets.model("utility_box_01")
	if m != null and not boxes.is_empty():
		var bb := Assets.node_aabb(m)
		m.free()
		parent.add_child(Terrain.multimesh_model("utility_box_01", boxes, 1.15 / maxf(bb.size.y, 1e-3)))
	elif m != null:
		m.free()


# ================================================================ outskirts / skyline

static func _tree_variant(rng: RandomNumberGenerator) -> Array:
	## [trunk mesh, crown mesh] of one broadleaf tree: tapered trunk with two limbs and an irregular
	## crown of jittered low-poly blobs (~6.5-7.5 m tall).
	var trunk := new_st()
	trunk.append_from(M.cyl(0.13, 0.24, 3.6, 7), 0, Transform3D(Basis(), Vector3(0, 1.8, 0)))
	for k in 2:
		var a := rng.randf() * TAU
		var tilt := deg_to_rad(rng.randf_range(28.0, 42.0))
		var b := Basis(Vector3.UP, a) * Basis(Vector3.FORWARD, tilt)
		trunk.append_from(M.cyl(0.05, 0.1, 1.8, 6), 0, Transform3D(b, Vector3(0, 3.0, 0) + b * Vector3(0, 0.9, 0)))
	var crown := new_st()
	var n := 24 + rng.randi() % 6
	for i in n:
		var u := rng.randf_range(-1.0, 1.0)
		var th := rng.randf() * TAU
		var rr := sqrt(1.0 - u * u) * pow(rng.randf(), 0.35)
		var p := Vector3(cos(th) * rr * 2.2, 5.0 + u * 1.45, sin(th) * rr * 2.2)
		var s := rng.randf_range(0.55, 0.95) * (1.0 - 0.2 * absf(u))
		var sm := SphereMesh.new()
		sm.radius = s
		sm.height = s * rng.randf_range(1.5, 1.9)
		sm.radial_segments = 8
		sm.rings = 4
		crown.append_from(sm, 0, Transform3D(Basis(Vector3.UP, rng.randf() * TAU), p))
	return [trunk.commit(), crown.commit()]


static func trees(parent: Node3D, city_half: float) -> void:
	## Park trees on the outskirts (beyond the ring road), three crown variants, per-tree tint.
	var root := Node3D.new()
	root.name = "Trees"
	parent.add_child(root)
	var rng := RandomNumberGenerator.new()
	rng.seed = 991
	var variants: Array = []
	for v in 3:
		variants.append(_tree_variant(rng))
	var xfs: Array = [[], [], []]
	var cols: Array = [[], [], []]
	var tries := 0
	var count := 0
	while count < 170 and tries < 4000:
		tries += 1
		var x := rng.randf_range(-170.0, 170.0)
		var y := rng.randf_range(-170.0, 170.0)
		if maxf(absf(x), absf(y)) < city_half + 7.0:
			continue
		var sc := rng.randf_range(0.8, 1.1)
		var v := rng.randi() % 3
		xfs[v].append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(sc, sc * rng.randf_range(0.9, 1.08), sc)), Vector3(x, 0.0, -y)))
		cols[v].append(Color(rng.randf_range(0.8, 1.05), rng.randf_range(0.85, 1.1), rng.randf_range(0.7, 1.0)))
		count += 1
	var bark := Assets.pbr("rough_wood", {"uv": 1.5, "tint": Color(0.5, 0.42, 0.36), "color": Color(0.3, 0.22, 0.15)})
	var leaf := M.mat(Color(0.2, 0.33, 0.12), 0.85, 0.0, {"noise": 1.0, "normal_noise": 2.0, "noise_scale": 1.4})
	leaf.vertex_color_use_as_albedo = true
	leaf.vertex_color_is_srgb = true
	for v in 3:
		for part in 2:
			var mm := MultiMesh.new()
			mm.transform_format = MultiMesh.TRANSFORM_3D
			mm.use_colors = part == 1
			mm.mesh = variants[v][part]
			mm.instance_count = xfs[v].size()
			for i in xfs[v].size():
				mm.set_instance_transform(i, xfs[v][i])
				if part == 1:
					mm.set_instance_color(i, cols[v][i])
			var mmi := MultiMeshInstance3D.new()
			mmi.multimesh = mm
			mmi.material_override = bark if part == 0 else leaf
			root.add_child(mmi)


static func skyline(parent: Node3D) -> Node3D:
	## Distant towers (every view; the elevation far planes reach past radius 352 m): glass curtain
	## walls and banded concrete towers.
	var root := Node3D.new()
	root.name = "Skyline"
	parent.add_child(root)
	var rng := RandomNumberGenerator.new()
	rng.seed = 31337
	var walls: Array = [new_st(), new_st()]
	var curtain := new_st()
	var glass := new_st()
	var mull := new_st()
	for i in 46:
		var a := TAU * i / 46.0 + rng.randf_range(-0.05, 0.05)
		var rad := rng.randf_range(190.0, 330.0)
		var cx := cos(a) * rad
		var cy := sin(a) * rad
		var w := rng.randf_range(16.0, 30.0)
		var d := rng.randf_range(16.0, 30.0)
		var h := rng.randf_range(22.0, 85.0)
		var x0 := cx - w * 0.5
		var x1 := cx + w * 0.5
		var y0 := cy - d * 0.5
		var y1 := cy + d * 0.5
		var faces := [[Vector2(x0, y0), Vector2(x1, y0), Vector2(0, -1)], [Vector2(x1, y0), Vector2(x1, y1), Vector2(1, 0)],
			[Vector2(x1, y1), Vector2(x0, y1), Vector2(0, 1)], [Vector2(x0, y1), Vector2(x0, y0), Vector2(-1, 0)]]
		if rng.randf() < 0.45:
			# curtain wall: glass box + mullion grid
			var gc := Color(0.55, 0.65, 0.75) * rng.randf_range(0.6, 1.0)
			slab(curtain, x0, y0, x1, y1, 0.0, h, gc)
			for fc in faces:
				var p0: Vector2 = fc[0]
				var p1: Vector2 = fc[1]
				var n2: Vector2 = fc[2]
				var L := p0.distance_to(p1)
				var u := (p1 - p0) / L
				var z := 3.6
				while z < h:
					facade_box(mull, p0, u, n2, 0.0, L, z - 0.12, z, 0.0, 0.12, Color(0.8, 0.82, 0.84))
					z += 3.6
				var sx := 3.0
				while sx < L:
					facade_box(mull, p0, u, n2, sx - 0.08, sx + 0.08, 0.0, h, 0.0, 0.15, Color(0.8, 0.82, 0.84))
					sx += 3.0
		else:
			var fam := rng.randi() % 2
			var tint := Color(1, 1, 1) * rng.randf_range(0.75, 1.0)
			slab(walls[fam], x0, y0, x1, y1, 0.0, h, tint)
			var gc2 := Color(0.45, 0.52, 0.6) * rng.randf_range(0.6, 1.0)
			for fc in faces:
				var p0: Vector2 = fc[0]
				var p1: Vector2 = fc[1]
				var n2: Vector2 = fc[2]
				var u := (p1 - p0).normalized()
				var z := 4.0
				while z + 2.2 < h - 1.0:
					vquad(glass, p0 + n2 * 0.06 + u * 0.8, p1 + n2 * 0.06 - u * 0.8, z, z + 2.0, n2, gc2)
					z += 3.4
	for f in 2:
		commit(root, walls[f], vc(Assets.pbr(FAMILIES[f + 1], {"uv": 0.12, "world": true, "color": Color(0.6, 0.6, 0.6)}), 1.4))
	var gm := M.mat(Color(0.35, 0.42, 0.5), 0.06, 0.7)
	gm.vertex_color_use_as_albedo = true
	gm.vertex_color_is_srgb = true
	commit(root, glass, gm)
	commit(root, curtain, gm)
	var mm := M.mat(Color(1, 1, 1), 0.4, 0.6)
	mm.vertex_color_use_as_albedo = true
	mm.vertex_color_is_srgb = true
	commit(root, mull, mm)
	return root
