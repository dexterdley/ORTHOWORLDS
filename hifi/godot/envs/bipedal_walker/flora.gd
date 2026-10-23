extends RefCounted
## Light procedural vegetation for the BipedalWalker family. The Poly Haven grass / shrub models are
## multi-plant sheets of 8-150k triangles that render as sparse twigs or black blobs here, too heavy
## to instance by the thousand with four views + shadow passes.
##   tuft_mesh() / blade_material()   grass tuft of thin curved vertex-coloured blades
##   leaf_texture() / foliage_material()  seeded leaf atlas on alpha-to-coverage cards
##   bush_mesh() / tree_meshes()      leaf-card shrubs and broadleaf trees (spherical card normals)
##   chunked() / plants()             MultiMesh instancing in x-chunks so each view culls
## Everything is seeded: same scene message, same vegetation.


static func tuft_mesh(seed: int, blades := 18, height := 0.32, base_col := Color(0.07, 0.11, 0.02), tip_col := Color(0.2, 0.25, 0.06)) -> ArrayMesh:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for b in blades:
		var a := rng.randf() * TAU
		var r := sqrt(rng.randf()) * 0.09
		var base := Vector3(cos(a) * r, -0.02, sin(a) * r)
		var h := height * rng.randf_range(0.45, 1.0)
		var lean_a := a + rng.randf_range(-0.6, 0.6)
		var lean := Vector3(cos(lean_a), 0, sin(lean_a)) * h * rng.randf_range(0.15, 0.55)
		var face_a := rng.randf() * TAU
		var side := Vector3(-sin(face_a), 0, cos(face_a)) * rng.randf_range(0.012, 0.022)
		var mid := base + Vector3(0, h * 0.55, 0) + lean * 0.35
		var tip := base + Vector3(0, h, 0) + lean
		var n := (Vector3.UP * 1.2 + lean.normalized() * 0.4).normalized()
		var shade := rng.randf_range(0.75, 1.15)
		var cb := base_col * shade
		var cm := base_col.lerp(tip_col, 0.5) * shade
		var ct := tip_col * shade
		cb.a = 1.0
		cm.a = 1.0
		ct.a = 1.0
		var v := [base - side, base + side, mid + side * 0.7, mid - side * 0.7, tip]
		var c := [cb, cb, cm, cm, ct]
		for tri in [[0, 1, 2], [0, 2, 3], [3, 2, 4]]:
			for k in tri:
				st.set_color(c[k])
				st.set_normal(n)
				st.add_vertex(v[k])
	return st.commit()


static func blade_material(rough := 0.8) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.vertex_color_use_as_albedo = true
	m.albedo_color = Color(1, 1, 1)
	m.roughness = rough
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.backlight_enabled = true
	m.backlight = Color(0.12, 0.16, 0.04)
	return m


static func scatter_xforms(rng: RandomNumberGenerator, n: int, x0: float, x1: float, z0: float, z1: float,
		height: Callable, s0 := 0.8, s1 := 1.4, sink := 0.0) -> Array:
	## n seeded transforms on the ground (height(x, z)), random yaw and uniform scale.
	var out: Array = []
	for i in n:
		var x := rng.randf_range(x0, x1)
		var z := rng.randf_range(z0, z1)
		var s := rng.randf_range(s0, s1)
		var b := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3.ONE * s)
		out.append(Transform3D(b, Vector3(x, float(height.call(x, z)) - sink * s, z)))
	return out


# ---------------------------------------------------------------- instancing

static func chunked(parent: Node3D, mesh: Mesh, xforms: Array, chunk := 12.0, vis_end := 0.0, mat: Material = null,
		shadows := false) -> Array:
	## MultiMeshes of `mesh` grouped in x-chunks so every view culls what it cannot see.
	## No mask twins (thin vegetation is not an occluder in the object-ID masks).
	var groups := {}
	for t in xforms:
		var k := int(floor((t as Transform3D).origin.x / chunk))
		if not groups.has(k):
			groups[k] = []
		groups[k].append(t)
	var out: Array = []
	var keys := groups.keys()
	keys.sort()
	for k in keys:
		var list: Array = groups[k]
		var mm := MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.mesh = mesh
		mm.instance_count = list.size()
		for i in list.size():
			mm.set_instance_transform(i, list[i])
		var mmi := MultiMeshInstance3D.new()
		mmi.multimesh = mm
		if mat != null:
			mmi.material_override = mat
		mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON if shadows else GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		if vis_end > 0.0:
			mmi.visibility_range_end = vis_end
			mmi.visibility_range_end_margin = vis_end * 0.1
			mmi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_DISABLED
		mmi.set_meta("no_mask", true)
		parent.add_child(mmi)
		out.append(mmi)
	return out


# ---------------------------------------------------------------- leaf-card foliage

static var _leaf_tex: ImageTexture
static var _foliage_mats := {}


static func leaf_texture() -> Texture2D:
	## Seeded 256 px atlas of leaves (RGBA, alpha cut-out) in grey-greens; plants tint it with
	## their vertex colour. Generated synchronously (deterministic).
	if _leaf_tex != null:
		return _leaf_tex
	var n := 256
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.3, 0.36, 0.22, 0.0))
	var rng := RandomNumberGenerator.new()
	rng.seed = 77
	for i in 230:
		var c := Vector2(rng.randf_range(14.0, n - 14.0), rng.randf_range(14.0, n - 14.0))
		var L := rng.randf_range(7.0, 13.0)
		var W := L * rng.randf_range(0.38, 0.55)
		var a := rng.randf() * TAU
		var ca := cos(a)
		var sa := sin(a)
		var tone := rng.randf_range(0.55, 1.0)
		var hue := rng.randf_range(-0.06, 0.06)
		for y in range(int(c.y - L) - 1, int(c.y + L) + 2):
			for x in range(int(c.x - L) - 1, int(c.x + L) + 2):
				var d := Vector2(x + 0.5, y + 0.5) - c
				var lx := d.x * ca + d.y * sa
				var ly := -d.x * sa + d.y * ca
				var e := (lx / L) * (lx / L) + (ly / W) * (ly / W)
				if e > 1.0:
					continue
				var shade := tone * (0.82 + 0.18 * (lx / L)) * (1.0 - 0.25 * absf(ly / W)) * (0.8 if absf(ly) < 0.7 else 1.0)
				img.set_pixel(x, y, Color(clampf((0.78 + hue) * shade, 0, 1), clampf(0.92 * shade, 0, 1), clampf((0.62 - hue) * shade, 0, 1), 1.0))
	img.generate_mipmaps()
	_leaf_tex = ImageTexture.create_from_image(img)
	return _leaf_tex


static func foliage_material(backlight := Color(0.22, 0.28, 0.08)) -> StandardMaterial3D:
	var key := backlight.to_html()
	if _foliage_mats.has(key):
		return _foliage_mats[key]
	var m := StandardMaterial3D.new()
	m.albedo_texture = leaf_texture()
	m.vertex_color_use_as_albedo = true
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	m.alpha_scissor_threshold = 0.45
	m.alpha_antialiasing_mode = BaseMaterial3D.ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.roughness = 0.78
	m.metallic_specular = 0.25
	m.backlight_enabled = true
	m.backlight = backlight
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	_foliage_mats[key] = m
	return m


static func _cards(st: SurfaceTool, rng: RandomNumberGenerator, n: int, center: Vector3, radius: Vector3, size: float,
		col: Color, light_dir := Vector3(0.3, 1.0, 0.2)) -> void:
	## n leaf cards in an ellipsoid; normals point away from the cluster centre (soft, rounded
	## lighting) and inner / lower cards are darker (self-shadowing).
	for i in n:
		var dir := Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1))
		if dir.length() > 1.0 or dir.length() < 0.05:
			dir = dir.normalized() * rng.randf_range(0.3, 1.0)
		var r := pow(rng.randf(), 0.35)
		var dn := dir.normalized()
		var p := center + Vector3(dn.x * radius.x, dn.y * radius.y, dn.z * radius.z) * r
		var nrm := ((p - center) / radius).normalized()
		nrm = (nrm + Vector3.UP * 0.35).normalized()
		var ao := clampf(0.45 + 0.55 * r, 0.3, 1.0) * clampf(0.75 + 0.35 * nrm.dot(light_dir.normalized()), 0.45, 1.1)
		var c := Color(col.r * ao, col.g * ao, col.b * ao, 1.0)
		var axis := Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1)).normalized()
		if axis.length() < 0.1:
			axis = Vector3.UP
		var b := Basis(axis, rng.randf() * TAU)
		var s := size * rng.randf_range(0.7, 1.25)
		var u := b.x * s
		var v := b.y * s
		var q := [p - u - v, p + u - v, p + u + v, p - u + v]
		var o := Vector2(0.5 * (i % 2), 0.5 * ((i / 2) % 2))
		var uv := [o + Vector2(0, 0.5), o + Vector2(0.5, 0.5), o + Vector2(0.5, 0), o]
		for tri in [[0, 1, 2], [0, 2, 3]]:
			for k in tri:
				st.set_color(c)
				st.set_normal(nrm)
				st.set_uv(uv[k])
				st.add_vertex(q[k])


static func bush_mesh(seed: int, height: float, col: Color) -> ArrayMesh:
	## Rounded shrub of 2-4 overlapping leaf clusters, base on y = 0.
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var k := 2 + seed % 3
	for i in k:
		var a := rng.randf() * TAU
		var d := rng.randf_range(0.0, 0.35) * height
		var r := height * rng.randf_range(0.42, 0.6)
		var c := Vector3(cos(a) * d, r * 0.85, sin(a) * d)
		_cards(st, rng, int(70 * r / 0.5 + 30), c, Vector3(r * 1.1, r * 0.85, r * 1.1), height * 0.11, col)
	return st.commit()


static func tree_meshes(seed: int, height: float, col: Color) -> Array:
	## [trunk + branches mesh, crown mesh] of a broadleaf tree, base on y = 0.
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var trunk := SurfaceTool.new()
	trunk.begin(Mesh.PRIMITIVE_TRIANGLES)
	var crown := SurfaceTool.new()
	crown.begin(Mesh.PRIMITIVE_TRIANGLES)
	var lean := Vector3(rng.randf_range(-0.3, 0.3), 0, rng.randf_range(-0.3, 0.3))
	var top := Vector3(0, height * 0.55, 0) + lean
	var pts := PackedVector3Array([Vector3(0, -0.2, 0), Vector3(0, height * 0.2, 0) + lean * 0.2, top])
	trunk.append_from(M.tube(pts, height * 0.035, 9), 0, Transform3D())
	var n_br := 4 + seed % 3
	var ends: Array = [top + Vector3(0, height * 0.12, 0)]
	for i in n_br:
		var a := TAU * i / n_br + rng.randf_range(-0.4, 0.4)
		var out := Vector3(cos(a), 0, sin(a)) * height * rng.randf_range(0.18, 0.3)
		var start := Vector3(0, height * rng.randf_range(0.38, 0.52), 0) + lean * 0.8
		var e := start + out + Vector3(0, height * rng.randf_range(0.15, 0.3), 0)
		trunk.append_from(M.tube(PackedVector3Array([start, start.lerp(e, 0.5) + Vector3(0, 0.1, 0), e]), height * 0.014, 6), 0, Transform3D())
		ends.append(e)
	for e in ends:
		var r := height * rng.randf_range(0.17, 0.24)
		_cards(crown, rng, int(110 * r / 1.0 + 40), e + Vector3(0, r * 0.25, 0), Vector3(r * 1.15, r * 0.8, r * 1.15), height * 0.06, col)
	return [trunk.commit(), crown.commit()]


static func plants(parent: Node3D, kind: String, xforms: Array, seed: int, height: float, col: Color,
		vis_end := 0.0, shadows := true) -> Array:
	## MultiMesh-instanced bushes or trees in 4 seeded variants (xforms carry position / yaw / scale).
	var bark := Assets.pbr("rough_wood", {"uv": 1.5, "tint": Color(0.42, 0.36, 0.3), "color": Color(0.28, 0.22, 0.16)})
	var fol := foliage_material()
	var groups := [[], [], [], []]
	var out: Array = []
	for i in xforms.size():
		groups[i % 4].append(xforms[i])
	for v in 4:
		if groups[v].is_empty():
			continue
		var c := col * Color(1.0 + 0.08 * (v - 1.5), 1.0, 1.0 - 0.06 * (v - 1.5))
		var parts: Array = [[bush_mesh(seed + v, height, c), fol]] if kind == "bush" else []
		if kind == "tree":
			var tm := tree_meshes(seed + v, height, c)
			parts = [[tm[0], bark], [tm[1], fol]]
		for pr in parts:
			out.append_array(chunked(parent, pr[0], groups[v], 25.0, vis_end, pr[1], shadows))
	return out
