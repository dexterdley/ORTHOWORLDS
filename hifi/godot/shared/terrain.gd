class_name Terrain
extends RefCounted
## Terrain and scenery builders shared by the env worlds.
##   side_strip()   side-view envs: a GymBox terrain profile y(x) extruded across z, with a
##                  displaced top surface outside the gameplay lane and soil cross-sections
##   ground_slab()  top-down envs: a ground box whose sides show a cross-section in the
##                  orthographic Rear/Side views (they look at the ground edge-on)
##   scatter()/multimesh_model()  seeded placement of Poly Haven models
##   tree_multimesh()  procedural trees (Poly Haven trees are 0.1-1 GB, too heavy to instance)


class SideProfile:
	## Terrain height for side_strip(): the Box2D profile y(x) inside |z| <= lane, displaced by
	## seeded noise of amplitude `rough` outside it (downwards in front, upwards behind).
	var xs: PackedFloat32Array
	var ys: PackedFloat32Array
	var lane := 1e9
	var rough := 0.0
	var front_down := true
	var noise := FastNoiseLite.new()

	func _init(px: PackedFloat32Array, py: PackedFloat32Array, opts := {}) -> void:
		xs = px
		ys = py
		lane = float(opts.get("lane", 1e9))
		rough = float(opts.get("rough", 0.0))
		front_down = bool(opts.get("front_down", true))
		noise.seed = int(opts.get("seed", 7))
		noise.frequency = float(opts.get("freq", 0.25))

	func base(x: float) -> float:
		if xs.is_empty():
			return 0.0
		if x <= xs[0]:
			return ys[0]
		var i := xs.bsearch(x)
		if i >= xs.size():
			return ys[-1]
		var t := (x - xs[i - 1]) / maxf(xs[i] - xs[i - 1], 1e-6)
		return lerpf(ys[i - 1], ys[i], t)

	func offset(x: float, z: float) -> float:
		var dist := absf(z) - lane
		if dist <= 0.0 or rough <= 0.0:
			return 0.0
		var h := clampf(dist / 1.5, 0.0, 1.0) * rough * (0.5 + 0.5 * noise.get_noise_2d(x, z))
		return -h if (z > 0.0 and front_down) else h

	func height(x: float, z: float) -> float:
		return base(x) + offset(x, z)


static func side_strip(xs: PackedFloat32Array, ys: PackedFloat32Array, z0: float, z1: float, bottom: float,
		top_mat: Material, section_mat: Material, opts := {}) -> Node3D:
	## xs/ys: terrain profile (Box2D metres, increasing x). The top surface spans z0..z1 with dz
	## spacing (SideProfile gives the heights; query it later via node.get_meta("profile")).
	## Front (z1) and back (z0) faces drop to `bottom` and show the section material.
	var root := Node3D.new()
	root.name = "Terrain"
	var prof := SideProfile.new(xs, ys, opts)
	root.set_meta("profile", prof)
	var dz := float(opts.get("dz", 0.5))
	var nz := maxi(1, int(ceil((z1 - z0) / dz)))
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var nx := xs.size()
	var front := PackedFloat32Array()
	var back := PackedFloat32Array()
	for i in nx:
		for j in nz + 1:
			var z := lerpf(z0, z1, float(j) / nz)
			var y := ys[i] + prof.offset(xs[i], z)
			if j == 0:
				back.append(y)
			if j == nz:
				front.append(y)
			st.set_uv(Vector2(xs[i], z))
			st.add_vertex(Vector3(xs[i], y, z))
	for i in nx - 1:
		for j in nz:
			var a := i * (nz + 1) + j
			var b := a + nz + 1
			# clockwise seen from above (+Y): Godot front faces
			st.add_index(a)
			st.add_index(b)
			st.add_index(b + 1)
			st.add_index(a)
			st.add_index(b + 1)
			st.add_index(a + 1)
	st.generate_normals()
	st.generate_tangents()
	var top := MeshInstance3D.new()
	top.name = "Top"
	top.mesh = st.commit()
	top.material_override = top_mat
	root.add_child(top)
	# Cross-sections (front z1, back z0) and end caps
	var sec := SurfaceTool.new()
	sec.begin(Mesh.PRIMITIVE_TRIANGLES)
	for face in [[z1, 1.0, front], [z0, -1.0, back]]:
		var zf: float = face[0]
		var n := Vector3(0, 0, face[1])
		var hy: PackedFloat32Array = face[2]
		for i in nx - 1:
			var q := [Vector3(xs[i], bottom, zf), Vector3(xs[i + 1], bottom, zf),
				Vector3(xs[i + 1], hy[i + 1], zf), Vector3(xs[i], hy[i], zf)]
			_quad(sec, q, n)
	for end in [[0, -1.0], [nx - 1, 1.0]]:
		var i: int = end[0]
		for j in nz:
			var za := lerpf(z0, z1, float(j) / nz)
			var zb := lerpf(z0, z1, float(j + 1) / nz)
			_quad(sec, [Vector3(xs[i], bottom, za), Vector3(xs[i], bottom, zb), Vector3(xs[i], ys[i] + prof.offset(xs[i], zb), zb),
				Vector3(xs[i], ys[i] + prof.offset(xs[i], za), za)], Vector3(end[1], 0, 0))
	var side := MeshInstance3D.new()
	side.name = "Section"
	side.mesh = sec.commit()
	side.material_override = section_mat
	root.add_child(side)
	return root


static func ground_slab(center: Vector2, size: Vector2, thickness: float, top_mat: Material,
		side_mat: Material, top_y := 0.0) -> Node3D:
	## Box2D-plane ground: centre/size in Box2D (x, y) metres, top surface at Godot y = top_y.
	var root := Node3D.new()
	root.name = "Ground"
	var top := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = size
	top.mesh = pm
	top.material_override = top_mat
	top.position = Vector3(center.x, top_y, -center.y)
	root.add_child(top)
	var body := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(size.x, thickness, size.y)
	body.mesh = bm
	body.material_override = side_mat
	body.position = Vector3(center.x, top_y - thickness * 0.5 - 0.01, -center.y)
	root.add_child(body)
	return root


static func scatter(parent: Node3D, names: Array, xforms: Array) -> void:
	## Instances Poly Haven models (cycling through names) at the given transforms.
	for i in xforms.size():
		var m: Node3D = Assets.model(names[i % names.size()])
		if m == null:
			continue
		m.transform = xforms[i]
		parent.add_child(m)


static func multimesh_model(name: String, xforms: Array, base_scale := 1.0) -> Node3D:
	## One MultiMeshInstance3D per mesh of a Poly Haven model, for many copies (grass, rocks).
	var root := Node3D.new()
	root.name = "MM_" + name
	var m := Assets.model(name)
	if m == null or xforms.is_empty():
		return root
	for mi in m.find_children("*", "MeshInstance3D", true, false) + ([m] if m is MeshInstance3D else []):
		if mi.mesh == null:
			continue
		var local := Assets._rel_xf(mi, m)
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = mi.mesh
		mm.instance_count = xforms.size()
		for i in xforms.size():
			var t: Transform3D = xforms[i]
			mm.set_instance_transform(i, t * Transform3D(Basis().scaled(Vector3.ONE * base_scale), Vector3.ZERO) * local)
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		if mi.material_override != null:
			mmi.material_override = mi.material_override
		root.add_child(mmi)
	m.free()
	return root


static func tree_multimesh(xforms: Array, kind := "broadleaf", seed := 1, leaf_mat: Material = null) -> Node3D:
	## Procedural trees (trunk + noisy foliage clusters) instanced with MultiMesh. Children: trunk
	## MultiMeshInstance3D, then crown. leaf_mat overrides the crown material (e.g. snowy pines).
	var root := Node3D.new()
	root.name = "Trees"
	if xforms.is_empty():
		return root
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var bark := Assets.pbr("rough_wood", {"uv": 1.5, "tint": Color(0.55, 0.45, 0.38), "color": Color(0.3, 0.22, 0.15)})
	var leaf: Material = leaf_mat if leaf_mat != null else M.mat(Color(0.18, 0.32, 0.12) if kind == "broadleaf" else Color(0.11, 0.22, 0.12), 0.85, 0.0,
		{"noise": 1.0, "normal_noise": 1.6, "noise_scale": 0.9})
	var trunk_mesh: Mesh
	var crown := SurfaceTool.new()
	crown.begin(Mesh.PRIMITIVE_TRIANGLES)
	if kind == "pine":
		trunk_mesh = M.cyl(0.12, 0.25, 4.0, 8)
		for k in 4:
			var h := 2.0 + k * 1.6
			var r := 2.0 - k * 0.42
			var cone := M.cyl(0.0, r, 2.6, 12)
			crown.append_from(cone, 0, Transform3D(Basis(Vector3.UP, rng.randf() * TAU), Vector3(0, h, 0)))
	else:
		trunk_mesh = M.cyl(0.14, 0.3, 4.0, 8)
		for k in 6:
			var a := rng.randf() * TAU
			var d := rng.randf_range(0.0, 1.3)
			var s := rng.randf_range(1.3, 2.0)
			var blob := SphereMesh.new()
			blob.radius = s
			blob.height = s * 1.7
			blob.radial_segments = 12
			blob.rings = 6
			crown.append_from(blob, 0, Transform3D(Basis(), Vector3(cos(a) * d, 4.2 + rng.randf_range(0.0, 1.6), sin(a) * d)))
	crown.generate_normals()
	var parts := [[trunk_mesh, bark, Transform3D(Basis(), Vector3(0, 2.0, 0))], [crown.commit(), leaf, Transform3D()]]
	for p in parts:
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = p[0]
		mm.instance_count = xforms.size()
		for i in xforms.size():
			mm.set_instance_transform(i, xforms[i] * p[2])
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		mmi.material_override = p[1]
		root.add_child(mmi)
	return root


static func _quad(st: SurfaceTool, q: Array, n: Vector3) -> void:
	M._quad_uv(st, q, n, [_suv(q[0], n), _suv(q[1], n), _suv(q[2], n), _suv(q[3], n)])


static func _suv(p: Vector3, n: Vector3) -> Vector2:
	return Vector2(p.z, p.y) if absf(n.x) > 0.5 else Vector2(p.x, p.y)
