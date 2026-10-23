class_name M
extends RefCounted
## Procedural mesh and material helpers shared by every env world.
## Winding: Godot front faces are clockwise; _tri() orders each triangle from its intended normal,
## so callers only provide outward normals.
##
## Conventions: prism() extrudes a polygon in the XY plane along Z (side-view envs, Box2D local
## coords as-is); prism_up() extrudes a Box2D (x, y) polygon upwards, mapped to Godot (x, h, -y)
## (top-down envs). Every mesh carries UVs in metres (pair with triplanar or uv-scaled materials).


# ================================================================ materials

static func mat(color: Color, rough := 0.6, metal := 0.0, opts := {}) -> StandardMaterial3D:
	## Plain PBR material. opts: emission (Color), emission_energy, clearcoat, alpha (float),
	## cull (false = double-sided), rim, unshaded, noise (float: albedo noise strength)
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	m.metallic = metal
	if opts.has("emission"):
		m.emission_enabled = true
		m.emission = opts["emission"]
		m.emission_energy_multiplier = float(opts.get("emission_energy", 1.0))
	if opts.has("clearcoat"):
		m.clearcoat_enabled = true
		m.clearcoat = float(opts["clearcoat"])
		m.clearcoat_roughness = float(opts.get("clearcoat_roughness", 0.12))
	if opts.has("alpha"):
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.albedo_color.a = float(opts["alpha"])
	if opts.get("cull", true) == false:
		m.cull_mode = BaseMaterial3D.CULL_DISABLED
	if opts.has("rim"):
		m.rim_enabled = true
		m.rim = float(opts["rim"])
	if opts.get("unshaded", false):
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	if opts.has("noise"):
		m.albedo_texture = Assets.noise_texture(int(color.to_rgba32() % 997), 0.02, false)
		m.uv1_triplanar = true
		m.uv1_scale = Vector3.ONE * float(opts.get("noise_scale", 0.5))
		m.albedo_color = color
	if opts.has("normal_noise"):
		m.normal_enabled = true
		m.normal_texture = Assets.noise_texture(int(color.to_rgba32() % 991) + 7, 0.04, true)
		m.normal_scale = float(opts["normal_noise"])
		m.uv1_triplanar = true
		m.uv1_scale = Vector3.ONE * float(opts.get("noise_scale", 0.5))
	return m


static func paint(color: Color, gloss := 0.75, metal_flake := 0.35) -> StandardMaterial3D:
	## Car-paint style: base colour + clearcoat.
	return mat(color, 1.0 - gloss * 0.8, metal_flake, {"clearcoat": 1.0, "clearcoat_roughness": 0.08})


static func glass(tint := Color(0.08, 0.1, 0.12), alpha := 0.55) -> StandardMaterial3D:
	var m := mat(tint, 0.05, 0.1, {"alpha": alpha})
	m.metallic_specular = 0.9
	return m


static func emissive(color: Color, energy := 3.0) -> StandardMaterial3D:
	return mat(color, 0.4, 0.0, {"emission": color, "emission_energy": energy})


# ================================================================ nodes

static func mi(parent: Node, mesh: Mesh, material: Material = null, pos := Vector3.ZERO,
		rot_deg := Vector3.ZERO, scl := Vector3.ONE) -> MeshInstance3D:
	var m := MeshInstance3D.new()
	m.mesh = mesh
	if material != null:
		m.material_override = material
	m.position = pos
	m.rotation_degrees = rot_deg
	m.scale = scl
	parent.add_child(m)
	return m


static func node(parent: Node, pos := Vector3.ZERO, rot_deg := Vector3.ZERO, node_name := "") -> Node3D:
	var n := Node3D.new()
	if node_name != "":
		n.name = node_name
	n.position = pos
	n.rotation_degrees = rot_deg
	parent.add_child(n)
	return n


static func light(parent: Node, color: Color, energy: float, rng: float, pos := Vector3.ZERO, shadows := false) -> OmniLight3D:
	var l := OmniLight3D.new()
	l.light_color = color
	l.light_energy = energy
	l.omni_range = rng
	l.position = pos
	l.shadow_enabled = shadows
	parent.add_child(l)
	return l


# ================================================================ primitives

static func box(size: Vector3, bevel := 0.0) -> Mesh:
	## Box centred on the origin; bevel > 0 chamfers all edges (catches highlights).
	if bevel <= 0.0:
		var bm := BoxMesh.new()
		bm.size = size
		return bm
	var h := size * 0.5
	var b := minf(bevel, minf(h.x, minf(h.y, h.z)) * 0.95)
	var inn := h - Vector3(b, b, b)
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var corners := [[-1, -1], [1, -1], [1, 1], [-1, 1]]
	for a in 3:
		var u := (a + 1) % 3
		var v := (a + 2) % 3
		for s in [-1.0, 1.0]:
			var n := Vector3.ZERO
			n[a] = s
			var q: Array = []
			for k in corners:
				var p := Vector3.ZERO
				p[a] = s * h[a]
				p[u] = k[0] * inn[u]
				p[v] = k[1] * inn[v]
				q.append(p)
			_quad(st, q, n)
	for a in 3:
		for c in range(a + 1, 3):
			var w := 3 - a - c
			for sa in [-1.0, 1.0]:
				for sc in [-1.0, 1.0]:
					var n := Vector3.ZERO
					n[a] = sa
					n[c] = sc
					var q: Array = []
					for t in [[1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]:
						var p := Vector3.ZERO
						p[a] = sa * (h[a] if t[0] > 0.0 else inn[a])
						p[c] = sc * (h[c] if t[1] > 0.0 else inn[c])
						p[w] = t[2] * inn[w]
						q.append(p)
					_quad(st, q, n.normalized())
	for sx in [-1.0, 1.0]:
		for sy in [-1.0, 1.0]:
			for sz in [-1.0, 1.0]:
				var n := Vector3(sx, sy, sz).normalized()
				_tri(st, Vector3(sx * h.x, sy * inn.y, sz * inn.z), Vector3(sx * inn.x, sy * h.y, sz * inn.z),
					Vector3(sx * inn.x, sy * inn.y, sz * h.z), n)
	return st.commit()


static func cyl(r_top: float, r_bot: float, height: float, segs := 24) -> Mesh:
	var c := CylinderMesh.new()
	c.top_radius = r_top
	c.bottom_radius = r_bot
	c.height = height
	c.radial_segments = segs
	c.rings = 1
	return c


static func sphere(r: float, segs := 24, rings := 12) -> Mesh:
	var s := SphereMesh.new()
	s.radius = r
	s.height = 2.0 * r
	s.radial_segments = segs
	s.rings = rings
	return s


static func capsule(r: float, height: float, segs := 16) -> Mesh:
	var c := CapsuleMesh.new()
	c.radius = r
	c.height = maxf(height, 2.0 * r)
	c.radial_segments = segs
	c.rings = 4
	return c


static func plane(size: Vector2, subdiv := 0) -> Mesh:
	var p := PlaneMesh.new()
	p.size = size
	p.subdivide_width = subdiv
	p.subdivide_depth = subdiv
	return p


# ================================================================ extrusions

static func poly(arr) -> PackedVector2Array:
	var out := PackedVector2Array()
	for p in arr:
		out.append(Vector2(float(p[0]), float(p[1])))
	return out


static func signed_area(p: PackedVector2Array) -> float:
	var a := 0.0
	for i in p.size():
		var q := p[(i + 1) % p.size()]
		a += p[i].x * q.y - q.x * p[i].y
	return a * 0.5


static func prism(p: PackedVector2Array, z0: float, z1: float, caps := true) -> ArrayMesh:
	## Side-view extrusion: polygon in the XY plane, from z0 to z1 along +Z.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var ccw := signed_area(p) > 0.0
	var n := p.size()
	if caps:
		var idx := Geometry2D.triangulate_polygon(p)
		for i in range(0, idx.size(), 3):
			var a := p[idx[i]]
			var b := p[idx[i + 1]]
			var c := p[idx[i + 2]]
			_tri(st, Vector3(a.x, a.y, z1), Vector3(b.x, b.y, z1), Vector3(c.x, c.y, z1), Vector3(0, 0, 1))
			_tri(st, Vector3(a.x, a.y, z0), Vector3(b.x, b.y, z0), Vector3(c.x, c.y, z0), Vector3(0, 0, -1))
	var dist := 0.0
	for i in n:
		var a := p[i]
		var b := p[(i + 1) % n]
		var e := b - a
		if e.length() < 1e-7:
			continue
		var nn := Vector2(e.y, -e.x).normalized() * (1.0 if ccw else -1.0)
		var nrm := Vector3(nn.x, nn.y, 0.0)
		var l := e.length()
		_quad_uv(st, [Vector3(a.x, a.y, z0), Vector3(b.x, b.y, z0), Vector3(b.x, b.y, z1), Vector3(a.x, a.y, z1)], nrm,
			[Vector2(dist, z0), Vector2(dist + l, z0), Vector2(dist + l, z1), Vector2(dist, z1)])
		dist += l
	return st.commit()


static func prism_up(p: PackedVector2Array, h0: float, h1: float, caps := true) -> ArrayMesh:
	## Top-down extrusion: Box2D (x, y) polygon -> Godot (x, h, -y), from height h0 to h1.
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var ccw := signed_area(p) > 0.0
	var n := p.size()
	if caps:
		var idx := Geometry2D.triangulate_polygon(p)
		for i in range(0, idx.size(), 3):
			var a := p[idx[i]]
			var b := p[idx[i + 1]]
			var c := p[idx[i + 2]]
			_tri(st, Vector3(a.x, h1, -a.y), Vector3(b.x, h1, -b.y), Vector3(c.x, h1, -c.y), Vector3.UP)
			if h0 > -1e9:
				_tri(st, Vector3(a.x, h0, -a.y), Vector3(b.x, h0, -b.y), Vector3(c.x, h0, -c.y), Vector3.DOWN)
	var dist := 0.0
	for i in n:
		var a := p[i]
		var b := p[(i + 1) % n]
		var e := b - a
		if e.length() < 1e-7:
			continue
		var nn := Vector2(e.y, -e.x).normalized() * (1.0 if ccw else -1.0)
		var nrm := Vector3(nn.x, 0.0, -nn.y)
		var l := e.length()
		_quad_uv(st, [Vector3(a.x, h0, -a.y), Vector3(b.x, h0, -b.y), Vector3(b.x, h1, -b.y), Vector3(a.x, h1, -a.y)], nrm,
			[Vector2(dist, h0), Vector2(dist + l, h0), Vector2(dist + l, h1), Vector2(dist, h1)])
		dist += l
	return st.commit()


static func lathe(profile: PackedVector2Array, segs := 32, axis := "y") -> ArrayMesh:
	## Surface of revolution of (radius, height) profile points around +Y (or "x" / "z").
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var np := profile.size()
	var nrm2: Array = []
	for i in np:
		var a := profile[maxi(i - 1, 0)]
		var b := profile[mini(i + 1, np - 1)]
		var d := (b - a).normalized()
		nrm2.append(Vector2(d.y, -d.x))
	for s in segs:
		var t0 := TAU * s / segs
		var t1 := TAU * (s + 1) / segs
		for i in np - 1:
			var p0 := profile[i]
			var p1 := profile[i + 1]
			var q := [_rev(p0, t0, axis), _rev(p0, t1, axis), _rev(p1, t1, axis), _rev(p1, t0, axis)]
			var ns := [_revn(nrm2[i], t0, axis), _revn(nrm2[i], t1, axis), _revn(nrm2[i + 1], t1, axis), _revn(nrm2[i + 1], t0, axis)]
			var uv := [Vector2(t0 * 0.5, p0.y), Vector2(t1 * 0.5, p0.y), Vector2(t1 * 0.5, p1.y), Vector2(t0 * 0.5, p1.y)]
			var face_n: Vector3 = (ns[0] + ns[1] + ns[2] + ns[3]).normalized()
			if face_n.length() < 0.5:
				face_n = (q[1] - q[0]).cross(q[3] - q[0]).normalized()
			_tri_n(st, q[0], q[1], q[2], ns[0], ns[1], ns[2], uv[0], uv[1], uv[2], face_n)
			_tri_n(st, q[0], q[2], q[3], ns[0], ns[2], ns[3], uv[0], uv[2], uv[3], face_n)
	return st.commit()


static func _rev(p: Vector2, t: float, axis: String) -> Vector3:
	var v := Vector3(p.x * cos(t), p.y, p.x * sin(t))
	match axis:
		"x":
			return Vector3(v.y, v.x, v.z)
		"z":
			return Vector3(v.x, v.z, v.y)
	return v


static func _revn(n: Vector2, t: float, axis: String) -> Vector3:
	return _rev(n, t, axis).normalized()


static func tube(points: PackedVector3Array, radius: float, segs := 8, mesh: ArrayMesh = null) -> ArrayMesh:
	## Tube along a polyline. Pass an existing ArrayMesh to rebuild it in place (keeps mask twins valid).
	var out := mesh if mesh != null else ArrayMesh.new()
	out.clear_surfaces()
	if points.size() < 2:
		return out
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var rings: Array = []
	var prev_n := Vector3.UP
	for i in points.size():
		var tgt: Vector3
		if i == 0:
			tgt = points[1] - points[0]
		elif i == points.size() - 1:
			tgt = points[i] - points[i - 1]
		else:
			tgt = points[i + 1] - points[i - 1]
		tgt = tgt.normalized()
		var nx := prev_n - tgt * prev_n.dot(tgt)
		if nx.length() < 1e-4:
			nx = tgt.cross(Vector3.RIGHT if absf(tgt.x) < 0.9 else Vector3.FORWARD)
		nx = nx.normalized()
		prev_n = nx
		var bx := tgt.cross(nx)
		var ring: Array = []
		for s in segs:
			var a := TAU * s / segs
			ring.append(nx * cos(a) + bx * sin(a))
		rings.append(ring)
	var dist := 0.0
	for i in points.size() - 1:
		var l := points[i].distance_to(points[i + 1])
		for s in segs:
			var s1 := (s + 1) % segs
			var n00: Vector3 = rings[i][s]
			var n01: Vector3 = rings[i][s1]
			var n10: Vector3 = rings[i + 1][s]
			var n11: Vector3 = rings[i + 1][s1]
			var p00 := points[i] + n00 * radius
			var p01 := points[i] + n01 * radius
			var p10 := points[i + 1] + n10 * radius
			var p11 := points[i + 1] + n11 * radius
			var u0 := float(s) / segs
			var u1 := float(s + 1) / segs
			var fn := (n00 + n01 + n10 + n11).normalized()
			_tri_n(st, p00, p01, p11, n00, n01, n11, Vector2(u0, dist), Vector2(u1, dist), Vector2(u1, dist + l), fn)
			_tri_n(st, p00, p11, p10, n00, n11, n10, Vector2(u0, dist), Vector2(u1, dist + l), Vector2(u0, dist + l), fn)
		dist += l
	st.commit(out)
	return out


static func wheel(parent: Node, radius: float, width: float, tyre: Material, rim: Material, spokes := 6) -> Node3D:
	## Tyre + rim with the axle along X. Returns the spinning node (rotate it about X).
	var spin := Node3D.new()
	spin.name = "Spin"
	parent.add_child(spin)
	var hw := width * 0.5
	var r_in := radius * 0.62
	var prof := PackedVector2Array([Vector2(r_in, -hw), Vector2(radius * 0.93, -hw), Vector2(radius, -hw * 0.6),
		Vector2(radius, hw * 0.6), Vector2(radius * 0.93, hw), Vector2(r_in, hw)])
	mi(spin, lathe(prof, 28, "x"), tyre)
	mi(spin, cyl(r_in, r_in, width * 0.82, 24), rim, Vector3.ZERO, Vector3(0, 0, 90))
	mi(spin, cyl(radius * 0.16, radius * 0.16, width * 0.95, 12), rim, Vector3.ZERO, Vector3(0, 0, 90))
	for k in spokes:
		var a := TAU * k / spokes
		var sp := mi(spin, box(Vector3(width * 0.3, r_in * 0.9, radius * 0.1)), rim,
			Vector3(hw * 0.45, cos(a) * r_in * 0.45, sin(a) * r_in * 0.45))
		sp.rotation = Vector3(a, 0, 0)
	return spin


# ================================================================ SurfaceTool helpers (public)

static func tri(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, n: Vector3) -> void:
	## Triangle facing normal n (winding fixed automatically), UVs projected in metres.
	_tri(st, a, b, c, n)


static func quad(st: SurfaceTool, q: Array, n: Vector3) -> void:
	## Planar quad [p0, p1, p2, p3] facing n.
	_quad(st, q, n)


static func quad_uv(st: SurfaceTool, q: Array, n: Vector3, uv: Array) -> void:
	## Planar quad facing n with explicit UVs.
	_quad_uv(st, q, n, uv)


# ================================================================ low level

static func _tri(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, n: Vector3) -> void:
	_tri_n(st, a, b, c, n, n, n, _uv(a, n), _uv(b, n), _uv(c, n), n)


static func _tri_n(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, na: Vector3, nb: Vector3, nc: Vector3,
		ua: Vector2, ub: Vector2, uc: Vector2, face_n: Vector3) -> void:
	# Godot front faces are clockwise seen from the front: (b-a)x(c-a) must point away from the normal
	if (b - a).cross(c - a).dot(face_n) > 0.0:
		var t := b
		b = c
		c = t
		var tn := nb
		nb = nc
		nc = tn
		var tu := ub
		ub = uc
		uc = tu
	st.set_normal(na)
	st.set_uv(ua)
	st.add_vertex(a)
	st.set_normal(nb)
	st.set_uv(ub)
	st.add_vertex(b)
	st.set_normal(nc)
	st.set_uv(uc)
	st.add_vertex(c)


static func _quad(st: SurfaceTool, q: Array, n: Vector3) -> void:
	_tri(st, q[0], q[1], q[2], n)
	_tri(st, q[0], q[2], q[3], n)


static func _quad_uv(st: SurfaceTool, q: Array, n: Vector3, uv: Array) -> void:
	_tri_n(st, q[0], q[1], q[2], n, n, n, uv[0], uv[1], uv[2], n)
	_tri_n(st, q[0], q[2], q[3], n, n, n, uv[0], uv[2], uv[3], n)


static func _uv(p: Vector3, n: Vector3) -> Vector2:
	var an := n.abs()
	if an.x >= an.y and an.x >= an.z:
		return Vector2(p.z, p.y)
	if an.y >= an.z:
		return Vector2(p.x, p.z)
	return Vector2(p.x, p.y)
