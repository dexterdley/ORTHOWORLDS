class_name Assets
extends RefCounted
## Runtime loader for the CC0 Poly Haven assets fetched by hifi/tools/fetch_assets.py
## (res://assets/polyhaven, listed in index.json). The folder has a .gdignore, so nothing is
## imported by the editor: images load with Image.load_from_file, models with GLTFDocument.
## Every call degrades gracefully: a missing asset gives a procedural stand-in, never an error.

const INDEX := "res://assets/polyhaven/index.json"

static var _index := {}
static var _loaded := false
static var _tex := {}
static var _mats := {}
static var _models := {}
static var _hdri := {}
static var _noise := {}


static func index() -> Dictionary:
	if not _loaded:
		_loaded = true
		if FileAccess.file_exists(INDEX):
			var d = JSON.parse_string(FileAccess.get_file_as_string(INDEX))
			if typeof(d) == TYPE_DICTIONARY:
				_index = d
		if _index.is_empty():
			push_warning("hifi: no assets index (run hifi/tools/fetch_assets.py); using procedural fallbacks")
	return _index


static func clear_cache() -> void:
	_tex.clear()
	_mats.clear()
	for k in _models:
		if _models[k] != null:
			_models[k].free()
	_models.clear()
	_hdri.clear()
	_noise.clear()


static func has_texture(name: String) -> bool:
	return index().get("textures", {}).has(name)


static func has_model(name: String) -> bool:
	return index().get("models", {}).has(name)


static func model_tris(name: String) -> int:
	## Triangle count of a fetched model (index.json "model_tris"); -1 if unknown.
	return int(index().get("model_tris", {}).get(name, -1))


static func has_hdri(name: String) -> bool:
	return index().get("hdris", {}).has(name)


static func image_texture(path: String, mipmaps := true) -> Texture2D:
	var key := path + (":m" if mipmaps else "")
	if _tex.has(key):
		return _tex[key]
	var img := Image.load_from_file(ProjectSettings.globalize_path(path))
	if img == null or img.is_empty():
		_tex[key] = null
		return null
	if mipmaps and not img.has_mipmaps():
		img.generate_mipmaps()
	var t := ImageTexture.create_from_image(img)
	_tex[key] = t
	return t


static func hdri(name: String) -> Texture2D:
	if _hdri.has(name):
		return _hdri[name]
	var path: String = index().get("hdris", {}).get(name, "")
	var t: Texture2D = image_texture(path, false) if path != "" else null
	_hdri[name] = t
	return t


static func hdri_image(name: String) -> Image:
	var path: String = index().get("hdris", {}).get(name, "")
	return Image.load_from_file(ProjectSettings.globalize_path(path)) if path != "" else null


static func pbr(name: String, opts := {}) -> BaseMaterial3D:
	## PBR material from a Poly Haven texture set (diff + nor_gl + arm/rough).
	## opts: uv (float|Vector2: tiles per metre on triplanar, or UV scale), offset (float|Vector2), triplanar (bool, default true),
	##       tint (Color), color (fallback Color), roughness, metallic, normal (strength),
	##       world (bool: triplanar in world space), clearcoat (float), cull (bool double-sided)
	var key := name + JSON.stringify(opts)
	if _mats.has(key):
		return _mats[key]
	var maps: Dictionary = index().get("textures", {}).get(name, {})
	var m: BaseMaterial3D
	if maps.has("arm"):
		m = ORMMaterial3D.new()
		m.orm_texture = image_texture(maps["arm"])
	else:
		m = StandardMaterial3D.new()
		if maps.has("rough"):
			m.roughness_texture = image_texture(maps["rough"])
	var fallback: Color = opts.get("color", Color(0.5, 0.5, 0.5))
	if maps.has("diff"):
		m.albedo_texture = image_texture(maps["diff"])
		m.albedo_color = opts.get("tint", Color.WHITE)
	else:
		m.albedo_color = fallback
		m.albedo_texture = noise_texture(name.hash() % 1000, 0.05, false)
	if maps.has("nor_gl"):
		m.normal_enabled = true
		m.normal_texture = image_texture(maps["nor_gl"])
		m.normal_scale = float(opts.get("normal", 1.0))
	# With a texture the scalar multiplies it (ARM blue = metalness), otherwise it is the value
	var textured_rough := m is ORMMaterial3D or maps.has("rough")
	m.roughness = float(opts.get("roughness", 1.0 if textured_rough else 0.85))
	m.metallic = float(opts.get("metallic", 1.0 if m is ORMMaterial3D else 0.0))
	var uv = opts.get("uv", 1.0)
	# Vector2 uv: (u, v) scale; on triplanar the third axis follows u
	var uvv := Vector3(uv.x, uv.y, uv.x) if uv is Vector2 else Vector3(float(uv), float(uv), float(uv))
	if opts.get("triplanar", true):
		m.uv1_triplanar = true
		m.uv1_world_triplanar = bool(opts.get("world", false))
		m.uv1_triplanar_sharpness = 4.0
	m.uv1_scale = uvv
	if opts.has("offset"):
		var o = opts["offset"]
		m.uv1_offset = Vector3(o.x, o.y, 0.0) if o is Vector2 else Vector3(float(o), float(o), float(o))
	if opts.has("clearcoat"):
		m.clearcoat_enabled = true
		m.clearcoat = float(opts["clearcoat"])
		m.clearcoat_roughness = 0.15
	if opts.get("cull", false):
		m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	_mats[key] = m
	return m


static func model(name: String) -> Node3D:
	## A fresh instance of a Poly Haven model (origin at its own pivot), or null if unavailable.
	if not _models.has(name):
		var path: String = index().get("models", {}).get(name, "")
		var tmpl: Node3D = null
		if path != "":
			var doc := GLTFDocument.new()
			var st := GLTFState.new()
			if doc.append_from_file(ProjectSettings.globalize_path(path), st) == OK:
				tmpl = doc.generate_scene(st)
		_models[name] = tmpl
	var t: Node3D = _models[name]
	return t.duplicate() if t != null else null


static func fitted_model(name: String, target: Vector3, mode := "fit") -> Node3D:
	## Model scaled to the target size and re-centred so its footprint centre sits at the origin
	## with its base on y = 0. mode: "fit" (uniform, fits inside target), "height" (uniform by y),
	## "stretch" (non-uniform). Returns a wrapper Node3D, or null when the model is missing.
	var m := model(name)
	if m == null:
		return null
	var aabb := node_aabb(m)
	if aabb.size.length() < 1e-6:
		return null
	var s := Vector3.ONE
	match mode:
		"stretch":
			s = Vector3(target.x / maxf(aabb.size.x, 1e-6), target.y / maxf(aabb.size.y, 1e-6), target.z / maxf(aabb.size.z, 1e-6))
		"height":
			s = Vector3.ONE * (target.y / maxf(aabb.size.y, 1e-6))
		_:
			var k := INF
			for i in 3:
				if target[i] > 0.0:
					k = minf(k, target[i] / maxf(aabb.size[i], 1e-6))
			s = Vector3.ONE * (k if k < INF else 1.0)
	var wrap := Node3D.new()
	wrap.name = name
	wrap.add_child(m)
	m.scale = s
	var c := aabb.get_center() * s
	m.position = Vector3(-c.x, -aabb.position.y * s.y, -c.z)
	return wrap


static func node_aabb(n: Node3D) -> AABB:
	var out := AABB()
	var first := true
	var inv := n.global_transform.affine_inverse() if n.is_inside_tree() else Transform3D()
	for mi in n.find_children("*", "MeshInstance3D", true, false) + ([n] if n is MeshInstance3D else []):
		if mi.mesh == null or mi.has_meta("is_mask"):
			continue
		var xf: Transform3D = _rel_xf(mi, n) if not n.is_inside_tree() else inv * mi.global_transform
		var b: AABB = xf * mi.mesh.get_aabb()
		out = b if first else out.merge(b)
		first = false
	return out


static func _rel_xf(node: Node3D, root: Node3D) -> Transform3D:
	var xf := Transform3D()
	var cur: Node = node
	while cur != null and cur != root:
		if cur is Node3D:
			xf = cur.transform * xf
		cur = cur.get_parent()
	return xf


static func noise_texture(seed: int, freq: float, normal: bool, size := 256) -> Texture2D:
	## Seamless noise, generated synchronously (NoiseTexture2D builds in a thread, which would make
	## the first lockstep frames differ).
	var key := "%d:%f:%s:%d" % [seed, freq, normal, size]
	if _noise.has(key):
		return _noise[key]
	var n := FastNoiseLite.new()
	n.seed = seed
	n.frequency = freq
	n.fractal_octaves = 5
	var img := n.get_seamless_image(size, size, false, false, 0.1, true)
	img.convert(Image.FORMAT_RGBA8)
	if normal:
		img.bump_map_to_normal_map(6.0)
	img.generate_mipmaps()
	var t := ImageTexture.create_from_image(img)
	_noise[key] = t
	return t
