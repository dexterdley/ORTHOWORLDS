extends EnvWorld
## Snowball Fight: two kids in puffy winter clothes in a fenced snow field with snow-block forts
## (1.0 m) and ice pillars (2.5 m), snowy pine forest around, overcast winter sky.
## Python: hifi/exporters/snowball_fight.py. Top-down env: Box2D (x, y) at height h -> Godot
## (x, h, -y), facing -> yaw. Character local frame: +X = facing, -Z = Box2D left.
## Heights are GymBox's (occlusion experiments): coat 0-1.12, head 1.12-1.45, beanie 1.45-1.62,
## forts 1.0, pillars 2.5, fence 1.2, snowballs fly flat at 1.25.
## Moving ids: player, opponent. Static, hideable by the elevation cutaway: obs<i>, fence<i>.
## p.chars[i] = [hit flash 0..1, gait distance m, forward speed, rightward speed, steps since throw].
## Transient: ball (snowball, mask "snowball"), splat (snow puff).
## One consistent world: the forest, snowman, sled, fence and snowfall are drawn in every view whose
## frame contains them (the exporter's elevation far planes reach the far edge of the forest). Only
## the elevation cutaway in front of the player (state["vis"] + the near plane) removes anything.

const COAT_TOP := 1.12
const HAT_TOP := 1.62

const ICE_SHADER := """
shader_type spatial;
render_mode specular_schlick_ggx;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
varying vec3 lpos;
void vertex() {
	lpos = VERTEX;
}
void fragment() {
	float n = texture(noise_tex, vec2(lpos.x + lpos.z * 0.7, lpos.y * 0.45) * 0.55).r;
	float n2 = texture(noise_tex, vec2(lpos.z - lpos.x * 0.6, lpos.y) * 1.1 + 0.37).r;
	float fres = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 2.5);
	vec3 deep = vec3(0.26, 0.56, 0.84);
	vec3 frost = vec3(0.7, 0.86, 0.97);
	vec3 col = mix(deep, frost, clamp(smoothstep(0.3, 0.8, n) * 0.5 + fres * 0.35, 0.0, 1.0));
	float crack = (1.0 - smoothstep(0.0, 0.025, abs(n2 - 0.5))) * 0.6;
	col = mix(col, vec3(0.94, 0.98, 1.0), crack);
	ALBEDO = col;
	ROUGHNESS = clamp(0.07 + 0.25 * n2 + crack * 0.4, 0.0, 1.0);
	SPECULAR = 0.75;
	EMISSION = deep * 0.05;
}
"""

const SNOWBLOCK_SHADER := """
shader_type spatial;
uniform sampler2D alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform vec3 tint : source_color = vec3(0.95, 0.97, 1.0);
uniform vec2 brick = vec2(0.55, 0.26);
varying vec3 lpos;
varying vec3 lnrm;
varying vec3 wpos;
void vertex() {
	lpos = VERTEX;
	lnrm = NORMAL;
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 q = abs(lnrm.x) > 0.5 ? lpos.zy : (abs(lnrm.z) > 0.5 ? lpos.xy : lpos.xz);
	float row = floor(q.y / brick.y);
	vec2 b = vec2(q.x / brick.x + 0.5 * mod(row, 2.0), q.y / brick.y);
	vec2 g = abs(fract(b) - 0.5) * brick;
	float seam = abs(lnrm.y) > 0.5 ? 0.0 : 1.0 - smoothstep(0.0, 0.022, min(brick.x * 0.5 - g.x, brick.y * 0.5 - g.y));
	vec2 uv = (abs(lnrm.y) > 0.5 ? wpos.xz : vec2(wpos.x + wpos.z, wpos.y)) * 0.6;
	float n = texture(noise_tex, uv * 0.7).r;
	vec3 t = mix(vec3(0.95, 0.97, 1.0), texture(alb, uv).rgb, 0.45);
	ALBEDO = t * tint * (1.0 - 0.1 * seam) * (0.95 + 0.07 * n);
	NORMAL_MAP = texture(nor, uv).rgb;
	NORMAL_MAP_DEPTH = 0.6;
	ROUGHNESS = 0.82 - 0.2 * seam;
	SPECULAR = 0.35;
}
"""

const SNOW_GROUND_SHADER := """
shader_type spatial;
uniform sampler2D alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform float scale = 0.22;
uniform vec4 arena = vec4(0.3, 0.3, 29.7, 19.7);
varying vec3 wpos;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 p = wpos.xz;
	vec3 t = texture(alb, p * scale).rgb;
	float n = texture(noise_tex, p * 0.05).r;
	float n2 = texture(noise_tex, p * 0.29 + 0.5).r;
	vec2 b2 = vec2(wpos.x, -wpos.z);
	float inside = step(arena.x, b2.x) * step(b2.x, arena.z) * step(arena.y, b2.y) * step(b2.y, arena.w);
	vec3 base = vec3(0.94, 0.96, 1.0) * (0.965 + 0.05 * n);
	vec3 col = mix(base, t * vec3(0.96, 0.98, 1.03), mix(0.3, 0.5, inside));
	col *= 1.0 - inside * smoothstep(0.55, 0.8, n2 * 0.55 + n * 0.5) * 0.07;
	ALBEDO = col;
	NORMAL_MAP = texture(nor, p * scale).rgb;
	NORMAL_MAP_DEPTH = mix(0.5, 0.9, inside);
	ROUGHNESS = 0.78;
	SPECULAR = 0.45;
}
"""

const SNOWFALL_SHADER := """
shader_type spatial;
render_mode unshaded, blend_mix, depth_draw_never, cull_disabled, shadows_disabled, world_vertex_coords, fog_disabled;
global uniform float sim_time;
uniform vec3 box_min = vec3(-20.0, 0.0, -45.0);
uniform vec3 box_size = vec3(70.0, 14.0, 70.0);
uniform float fall = 0.85;
uniform float flake = 0.036;
void vertex() {
	vec3 c = VERTEX;
	float y = box_min.y + mod(c.y - sim_time * fall - box_min.y, box_size.y);
	vec3 p = vec3(c.x + 0.35 * sin(sim_time * 0.9 + c.y * 1.7 + c.z), y, c.z + 0.3 * cos(sim_time * 0.7 + c.x * 1.3));
	VERTEX = p + (INV_VIEW_MATRIX[0].xyz * UV.x + INV_VIEW_MATRIX[1].xyz * UV.y) * flake;
}
void fragment() {
	float d = length(UV);
	ALBEDO = vec3(1.0);
	ALPHA = clamp(1.0 - d, 0.0, 1.0) * 0.85;
}
"""

const PINE_SHADER := """
shader_type spatial;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
varying vec3 wn;
varying vec3 wp;
void vertex() {
	wn = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
	wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	float n = texture(noise_tex, wp.xz * 0.4 + wp.y * 0.13).r;
	float snow = smoothstep(0.15, 0.55, wn.y + (n - 0.5) * 0.6);
	vec3 needles = vec3(0.07, 0.15, 0.09) * (0.7 + 0.6 * n);
	ALBEDO = mix(needles, vec3(0.9, 0.93, 0.97), snow);
	ROUGHNESS = mix(0.9, 0.75, snow);
}
"""

var _chars: Array = []
var _scene_ok := false


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "snow_field_2_puresky", "sky_energy": 0.9, "sky_rotation": 0.0,
		"sun_from_hdri": false, "sun_elev": 38.0, "sun_azim": 205.0,
		"sun_energy": 1.3, "sun_color": Color(1.0, 0.95, 0.88),
		"ambient_energy": 1.0, "exposure": 0.98, "white": 5.0,
		"fog_density": 0.006, "fog_color": Color(0.86, 0.89, 0.94), "fog_aerial": 0.3, "fog_sky_affect": 0.0,
		"saturation": 1.06, "contrast": 1.08, "shadow_distance": 90.0,
		"card_ground": "snow_02", "card_ground_tint": Color(1.0, 1.02, 1.06), "card_ground_scale": 0.12,
		"card_eye": 1.4,
	})
	if lk.env_ortho != null:
		lk.env_ortho.fog_enabled = false


func build_scene(scene: Dictionary) -> void:
	_chars.clear()
	_build_ground(scene)
	_build_fence(scene)
	_build_cover(scene)
	_build_surroundings(scene)
	var looks := [
		{"coat": Color(0.16, 0.34, 0.82), "hat": Color(0.1, 0.2, 0.52), "scarf": Color(0.96, 0.76, 0.12), "pants": Color(0.12, 0.13, 0.18)},
		{"coat": Color(0.78, 0.15, 0.15), "hat": Color(0.48, 0.08, 0.09), "scarf": Color(0.13, 0.6, 0.3), "pants": Color(0.16, 0.14, 0.14)},
	]
	for i in 2:
		var c := _build_char(looks[i], i)
		_chars.append(c)
		register("player" if i == 0 else "opponent", c["root"], "player" if i == 0 else "opponent")
	_scene_ok = true


# ================================================================ helpers

static func _tex(name: String, map: String) -> Texture2D:
	var t: Dictionary = Assets.index().get("textures", {}).get(name, {})
	var path := str(t.get(map, ""))
	return Assets.image_texture(path) if path != "" else null


static func _shader_mat(code: String) -> ShaderMaterial:
	var m := ShaderMaterial.new()
	var sh := Shader.new()
	sh.code = code
	m.shader = sh
	return m


static func _fabric(name: String, color: Color, rough := 0.85) -> BaseMaterial3D:
	## Coloured cloth: the Poly Haven set's normal / ARM maps with a flat albedo colour.
	var m: BaseMaterial3D = Assets.pbr(name, {"uv": 6.0, "color": color}).duplicate()
	m.albedo_texture = null
	m.albedo_color = color
	m.roughness = rough
	m.metallic = 0.0
	return m


static func _limb(parent: Node3D, a: Vector3, b: Vector3, r: float, mat: Material) -> MeshInstance3D:
	## Capsule from a to b.
	var d := b - a
	var mi := M.mi(parent, M.capsule(r, d.length() + 2.0 * r, 12), mat, (a + b) * 0.5)
	var up := d.normalized()
	var side := up.cross(Vector3.FORWARD if absf(up.dot(Vector3.FORWARD)) < 0.9 else Vector3.RIGHT).normalized()
	mi.basis = Basis(side, up, side.cross(up))
	return mi


static func _snow_mat() -> BaseMaterial3D:
	return Assets.pbr("snow_02", {"uv": 0.35, "world": true, "tint": Color(0.93, 0.95, 0.99), "color": Color(0.92, 0.94, 0.97)})


# ================================================================ ground, fence, cover

func _build_ground(scene: Dictionary) -> void:
	var W: float = scene.get("W", 30.0)
	var H: float = scene.get("H", 20.0)
	var g := MeshInstance3D.new()
	g.name = "Ground"
	var pm := PlaneMesh.new()
	pm.size = Vector2(320, 320)
	g.mesh = pm
	var sm := _shader_mat(SNOW_GROUND_SHADER)
	sm.set_shader_parameter("alb", _tex("snow_02", "diff"))
	sm.set_shader_parameter("nor", _tex("snow_02", "nor_gl"))
	sm.set_shader_parameter("noise_tex", Assets.noise_texture(91, 0.03, false))
	sm.set_shader_parameter("arena", Vector4(0.3, 0.3, W - 0.3, H - 0.3))
	g.material_override = sm
	g.position = Vector3(W * 0.5, 0, -H * 0.5)
	g.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	content.add_child(g)


func _build_fence(scene: Dictionary) -> void:
	var fh: float = scene.get("fence_h", 1.2)
	var planks := Assets.pbr("weathered_planks", {"uv": 0.9, "tint": Color(1.05, 0.84, 0.64), "color": Color(0.5, 0.34, 0.2)})
	var wood := Assets.pbr("rough_wood", {"uv": 1.2, "tint": Color(0.85, 0.66, 0.48), "color": Color(0.45, 0.32, 0.2)})
	var snow := _snow_mat()
	var W: float = scene.get("W", 30.0)
	var H: float = scene.get("H", 20.0)
	var fences: Array = scene.get("fences", [])
	for fi in fences.size():
		var poly := M.poly(fences[fi])
		var mn := Vector2(INF, INF)
		var mx := Vector2(-INF, -INF)
		for p in poly:
			mn = Vector2(minf(mn.x, p.x), minf(mn.y, p.y))
			mx = Vector2(maxf(mx.x, p.x), maxf(mx.y, p.y))
		var c := (mn + mx) * 0.5
		var sz := mx - mn
		var along_x := sz.x > sz.y
		var length := maxf(sz.x, sz.y)
		var thick := minf(sz.x, sz.y)
		# inward direction (towards the arena centre), Box2D frame
		var inward := Vector2(0, 1.0 if c.y < H * 0.5 else -1.0) if along_x else Vector2(1.0 if c.x < W * 0.5 else -1.0, 0)
		var root := M.node(content, Vector3(c.x, 0, -c.y), Vector3.ZERO, "Fence")
		# board wall on the inner face, posts + rails on the outer half, wide cap board with snow
		var bi := inward * (thick * 0.5 - 0.035)
		var board_sz := Vector3(length, fh - 0.06, 0.07) if along_x else Vector3(0.07, fh - 0.06, length)
		M.mi(root, M.box(board_sz, 0.01), planks, Vector3(bi.x, (fh - 0.06) * 0.5, -bi.y))
		var bo := -inward * 0.06
		var n := int(round(length / 2.4))
		for k in n + 1:
			var t := -length * 0.5 + k * length / n
			var pp := (Vector2(t, 0) if along_x else Vector2(0, t)) + bo
			M.mi(root, M.box(Vector3(0.14, fh - 0.04, 0.14), 0.015), wood, Vector3(pp.x, (fh - 0.04) * 0.5, -pp.y))
		for hy in [0.3, 0.88]:
			var rs := Vector3(length, 0.09, 0.05) if along_x else Vector3(0.05, 0.09, length)
			var ro := -inward * (thick * 0.5 - 0.05)
			M.mi(root, M.box(rs, 0.01), wood, Vector3(ro.x, hy, -ro.y))
		var cap := Vector3(length + (thick if along_x else 0.0), 0.04, thick) if along_x else Vector3(thick, 0.04, length)
		M.mi(root, M.box(cap, 0.01), wood, Vector3(0, fh - 0.06, 0))
		var scap := Vector3(cap.x, 0.03, thick * 0.4) if along_x else Vector3(thick * 0.4, 0.03, cap.z)
		M.mi(root, M.box(scap, 0.012), snow, Vector3(0, fh - 0.025, 0))
		# snow drift along the inner foot
		var dr := inward * (thick * 0.5 + 0.06)
		var dsz := Vector3(length, 0.16, 0.22) if along_x else Vector3(0.22, 0.16, length)
		M.mi(root, M.box(dsz, 0.07), snow, Vector3(dr.x, 0.02, -dr.y))
		register("fence%d" % fi, root)   # hideable by the elevation cutaway; black occluder in masks


func _build_cover(scene: Dictionary) -> void:
	var ice := _shader_mat(ICE_SHADER)
	ice.set_shader_parameter("noise_tex", Assets.noise_texture(55, 0.06, false))
	var block := _shader_mat(SNOWBLOCK_SHADER)
	block.set_shader_parameter("alb", _tex("snow_02", "diff"))
	block.set_shader_parameter("nor", _tex("snow_02", "nor_gl"))
	block.set_shader_parameter("noise_tex", Assets.noise_texture(23, 0.05, false))
	var snow := _snow_mat()
	var cover: Array = scene.get("cover", [])
	for i in cover.size():
		var c: Dictionary = cover[i]
		var w := float(c["w"])
		var h := float(c["h"])
		var hgt := float(c.get("height", 1.0))
		var root := M.node(content, Vector3(float(c["x"]), 0, -float(c["y"])), Vector3.ZERO, "Cover%d" % i)
		root.rotation.y = float(c.get("a", 0.0))
		if str(c["kind"]) == "pillar":
			# three stacked, slightly irregular ice blocks
			var rng := RandomNumberGenerator.new()
			rng.seed = 300 + i
			var nb := 3
			var bh := hgt / nb
			for k in nb:
				var inset := rng.randf_range(0.0, 0.05)
				var mi := M.mi(root, M.box(Vector3(w - inset, bh - 0.01, h - inset), 0.05), ice,
					Vector3(rng.randf_range(-0.015, 0.015), bh * (k + 0.5), rng.randf_range(-0.015, 0.015)))
				mi.rotation.y = rng.randf_range(-0.03, 0.03)
			# drift at the foot
			M.mi(root, M.box(Vector3(w + 0.18, 0.1, h + 0.18), 0.048), snow, Vector3(0, 0.02, 0))
		else:
			# snow-brick wall with a rounded crest; drift around the base
			M.mi(root, M.box(Vector3(w, hgt, h), 0.1), block, Vector3(0, hgt * 0.5, 0))
			M.mi(root, M.box(Vector3(w + 0.2, 0.12, h + 0.2), 0.058), snow, Vector3(0, 0.02, 0))
		register("obs%d" % i, root, str(c["kind"]))


func _build_surroundings(scene: Dictionary) -> void:
	var W: float = scene.get("W", 30.0)
	var H: float = scene.get("H", 20.0)
	# snowy pine forest ring around the field (every view; the elevations show it at full size)
	var rng := RandomNumberGenerator.new()
	rng.seed = 808
	var xfs: Array = []
	var tries := 0
	while xfs.size() < 120 and tries < 2000:
		tries += 1
		var x := rng.randf_range(-45.0, W + 45.0)
		var y := rng.randf_range(-40.0, H + 40.0)
		if x > -7.0 and x < W + 7.0 and y > -7.0 and y < H + 7.0:
			continue
		var s := rng.randf_range(0.75, 1.5)
		xfs.append(Transform3D(Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s * rng.randf_range(0.9, 1.25), s)), Vector3(x, -0.1, -y)))
	var trees := Terrain.tree_multimesh(xfs, "pine", 17)
	var pine := _shader_mat(PINE_SHADER)
	pine.set_shader_parameter("noise_tex", Assets.noise_texture(66, 0.05, false))
	if trees.get_child_count() > 1:
		var crown: GeometryInstance3D = trees.get_child(1)
		crown.material_override = pine
	content.add_child(trees)
	_build_snowfall(scene)
	# a snowman and a sled just outside the fence
	var snow := _snow_mat()
	var sm := M.node(content, Vector3(-2.2, 0, -(H - 2.5)), Vector3(0, 30, 0), "Snowman")
	M.mi(sm, M.sphere(0.42, 20, 10), snow, Vector3(0, 0.36, 0))
	M.mi(sm, M.sphere(0.3, 20, 10), snow, Vector3(0, 0.95, 0))
	M.mi(sm, M.sphere(0.21, 18, 9), snow, Vector3(0, 1.4, 0))
	M.mi(sm, M.cyl(0.0, 0.035, 0.18, 8), M.mat(Color(0.95, 0.45, 0.08), 0.6), Vector3(0.26, 1.42, 0), Vector3(0, 0, -90))
	M.mi(sm, M.cyl(0.14, 0.14, 0.2, 14), M.mat(Color(0.06, 0.06, 0.07), 0.5), Vector3(0, 1.66, 0))
	M.mi(sm, M.cyl(0.21, 0.21, 0.02, 14), M.mat(Color(0.06, 0.06, 0.07), 0.5), Vector3(0, 1.56, 0))
	for k in 2:
		M.mi(sm, M.sphere(0.025, 8, 4), M.mat(Color(0.02, 0.02, 0.02), 0.4), Vector3(0.18, 1.47, -0.07 + k * 0.14))
	var sled := M.node(content, Vector3(W + 2.0, 0, -3.5), Vector3(0, -20, 0), "Sled")
	var red := M.mat(Color(0.6, 0.08, 0.06), 0.45, 0.1, {"clearcoat": 0.6})
	M.mi(sled, M.box(Vector3(1.1, 0.04, 0.42), 0.015), Assets.pbr("weathered_planks", {"uv": 2.0, "tint": Color(0.8, 0.62, 0.45)}), Vector3(0, 0.16, 0))
	for z in [-0.2, 0.2]:
		M.mi(sled, M.box(Vector3(1.2, 0.03, 0.03), 0.01), red, Vector3(0, 0.03, z))
		for x in [-0.35, 0.35]:
			M.mi(sled, M.box(Vector3(0.03, 0.13, 0.03)), red, Vector3(x, 0.1, z))


func _build_snowfall(scene: Dictionary) -> void:
	## Light deterministic snowfall (vertex shader driven by sim_time), every view.
	var W: float = scene.get("W", 30.0)
	var H: float = scene.get("H", 20.0)
	var bmin := Vector3(-20.0, 0.0, -(H + 25.0))
	var bsize := Vector3(W + 40.0, 14.0, H + 50.0)
	var rng := RandomNumberGenerator.new()
	rng.seed = 1209
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var corners := [Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, -1), Vector2(1, 1), Vector2(-1, 1)]
	for i in 5000:
		var c := bmin + Vector3(rng.randf() * bsize.x, rng.randf() * bsize.y, rng.randf() * bsize.z)
		for k in corners:
			st.set_uv(k)
			st.add_vertex(c)
	var mi := MeshInstance3D.new()
	mi.name = "Snowfall"
	mi.mesh = st.commit()
	var m := _shader_mat(SNOWFALL_SHADER)
	m.set_shader_parameter("box_min", bmin)
	m.set_shader_parameter("box_size", bsize)
	mi.material_override = m
	mi.custom_aabb = AABB(bmin - Vector3(2, 2, 2), bsize + Vector3(4, 4, 4))
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mi.set_meta("no_mask", true)
	content.add_child(mi)


# ================================================================ characters

func _build_char(lk: Dictionary, idx: int) -> Dictionary:
	var root := Node3D.new()
	root.name = "Kid%d" % idx
	content.add_child(root)
	var coat := _fabric("polar_fleece", lk["coat"], 0.62)
	coat.emission_enabled = true
	coat.emission = Color(1.0, 0.95, 0.6)
	coat.emission_energy_multiplier = 0.0
	var hat := _fabric("knitted_fleece", lk["hat"], 0.95)
	var scarf := _fabric("knitted_fleece", lk["scarf"], 0.95)
	var pants := _fabric("denim_fabric", lk["pants"], 0.8)
	var skin := M.mat(Color(0.96, 0.8, 0.68), 0.6, 0.0)
	var boot := M.mat(Color(0.16, 0.1, 0.07), 0.7, 0.0)
	var dark := M.mat(Color(0.03, 0.03, 0.035), 0.3, 0.0)
	var trim := M.mat(Color(0.93, 0.92, 0.9), 0.95, 0.0)

	# legs (hip pivots swing with the gait), boots
	var legs: Array = []
	for z in [-0.13, 0.13]:
		var hip := M.node(root, Vector3(0, 0.56, z))
		_limb(hip, Vector3(0, -0.02, 0), Vector3(0, -0.42, 0), 0.115, pants)
		M.mi(hip, M.box(Vector3(0.26, 0.13, 0.15), 0.05), boot, Vector3(0.04, -0.495, 0))
		legs.append(hip)
	# puffy quilted coat (lathe), max radius 0.43 inside the 0.45 m body circle
	var prof := PackedVector2Array()
	for k in 31:
		var y := lerpf(0.4, 1.06, k / 30.0)
		var base := 0.33 + 0.1 * sin(clampf((y - 0.4) / 0.56, 0.0, 1.0) * PI * 0.85)
		var quilt := 0.012 * absf(sin((y - 0.4) / 0.13 * PI))
		prof.append(Vector2(minf(base + quilt, 0.43), y))
	prof.append(Vector2(0.3, COAT_TOP - 0.03))
	prof.append(Vector2(0.16, COAT_TOP))
	prof.append(Vector2(0.0, COAT_TOP + 0.005))
	prof.insert(0, Vector2(0.0, 0.4))
	M.mi(root, M.lathe(prof, 28), coat)
	# zipper + hem trim
	M.mi(root, M.box(Vector3(0.02, 0.6, 0.025)), dark, Vector3(0.425, 0.74, 0))
	M.mi(root, M.cyl(0.345, 0.345, 0.05, 28), trim, Vector3(0, 0.425, 0))
	# scarf: collar ring + tail
	var ring := M.lathe(PackedVector2Array([Vector2(0.15, 1.06), Vector2(0.23, 1.07), Vector2(0.24, 1.12), Vector2(0.22, 1.16), Vector2(0.15, 1.17)]), 24)
	M.mi(root, ring, scarf)
	M.mi(root, M.box(Vector3(0.06, 0.34, 0.12), 0.025), scarf, Vector3(-0.2, 0.98, 0.14), Vector3(10, 0, -12))
	# head, face, beanie with cuff and pom-pom
	var head := M.mi(root, M.sphere(0.2, 24, 12), skin, Vector3(0.0, 1.29, 0))
	head.scale = Vector3(1.0, 0.96, 1.0)
	for z in [-0.07, 0.07]:
		M.mi(root, M.sphere(0.024, 10, 5), dark, Vector3(0.18, 1.315, z))
	M.mi(root, M.sphere(0.03, 10, 5), M.mat(Color(0.95, 0.55, 0.5), 0.6), Vector3(0.2, 1.27, 0))
	M.mi(root, M.box(Vector3(0.02, 0.018, 0.08), 0.008), M.mat(Color(0.45, 0.12, 0.12), 0.6), Vector3(0.185, 1.215, 0))
	for z in [-0.11, 0.11]:
		var ck := M.mi(root, M.sphere(0.035, 10, 5), M.mat(Color(0.98, 0.6, 0.58), 0.7), Vector3(0.16, 1.26, z))
		ck.scale = Vector3(0.4, 0.8, 1.0)
	var dome := PackedVector2Array([Vector2(0.215, 1.36), Vector2(0.215, 1.42), Vector2(0.205, 1.47), Vector2(0.18, 1.52),
		Vector2(0.13, 1.56), Vector2(0.06, 1.585), Vector2(0.0, 1.59)])
	M.mi(root, M.lathe(dome, 24), hat)
	M.mi(root, M.cyl(0.224, 0.224, 0.075, 24), hat, Vector3(0, 1.395, 0))
	M.mi(root, M.sphere(0.05, 12, 6), scarf, Vector3(0, HAT_TOP - 0.045, 0))
	# arms: shoulders -> mittens in front of the chest (the GymBox arm stub, x 0.25-0.85, z 0.82-1.0)
	var arms: Array = []
	for z in [-0.34, 0.34]:
		var sh := M.node(root, Vector3(0.02, 0.99, z))
		var hand := Vector3(0.74, -0.08, -z * 0.78)
		_limb(sh, Vector3.ZERO, hand * 0.92, 0.088, coat)
		M.mi(sh, M.sphere(0.075, 12, 6), scarf, hand)
		arms.append(sh)
	return {"root": root, "coat": coat, "legs": legs, "arms": arms}


# ================================================================ per step

func on_state(state: Dictionary, prev: Dictionary, alpha: float) -> void:
	if not _scene_ok:
		return
	var cur: Array = state.get("p", {}).get("chars", [])
	var old: Array = prev.get("p", {}).get("chars", cur)
	for i in mini(cur.size(), _chars.size()):
		var c: Array = cur[i]
		var o: Array = old[i] if i < old.size() else c
		var ch: Dictionary = _chars[i]
		var coat: BaseMaterial3D = ch["coat"]
		coat.emission_energy_multiplier = 1.6 * float(c[0])
		# walk cycle from the distance walked (deterministic), swing in the travel direction
		var gait := lerpf(float(o[1]), float(c[1]), alpha)
		var vf := float(c[2])
		var vr := float(c[3])
		var speed := sqrt(vf * vf + vr * vr)
		var amp := clampf(speed / 3.0, 0.0, 1.0) * 0.55
		var sw := sin(gait / 1.05 * TAU) * amp
		var dir := Vector2(vf, vr).normalized() if speed > 0.05 else Vector2(1, 0)
		var legs: Array = ch["legs"]
		for k in 2:
			var hip: Node3D = legs[k]
			var s := sw * (1.0 if k == 0 else -1.0)
			# rotate about the axis perpendicular to the travel direction (local frame)
			hip.rotation = Vector3(-s * dir.y, 0.0, -s * dir.x)
		# throw: right arm (Godot +Z side = Box2D right) winds up and snaps forward
		var age := int(c[4])
		var arms: Array = ch["arms"]
		var right: Node3D = arms[1]
		var left: Node3D = arms[0]
		var k_throw := 0.0
		if age < 10:
			k_throw = sin(float(age) / 10.0 * PI)
		right.rotation = Vector3(0.0, 0.25 * k_throw, 1.1 * k_throw)
		left.rotation = Vector3(0.0, 0.0, 0.08 * sw)


func spawn_dyn(kind: String, d: Dictionary) -> Node3D:
	var root := Node3D.new()
	match kind:
		"ball":
			var ball := M.mi(root, M.sphere(0.15, 16, 8), Assets.pbr("snow_02", {"uv": 3.0, "tint": Color(0.97, 0.98, 1.0), "color": Color(0.95, 0.96, 0.98)}))
			ball.name = "Ball"
			# faint powder trail behind the ball (effect: never masks / occludes)
			var trail := M.mi(root, M.capsule(0.09, 0.7, 8), M.mat(Color(0.95, 0.97, 1.0), 1.0, 0.0, {"alpha": 0.22, "unshaded": true}),
				Vector3(-0.38, 0, 0), Vector3(0, 0, 90))
			trail.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			trail.set_meta("no_mask", true)
		"splat":
			var pf := Vfx.puff(root, Color(0.95, 0.97, 1.0), int(str(d.get("id", "0")).hash()) % 97, 7)
			pf.name = "Puff"
		_:
			return null
	return root


func update_dyn(node: Node3D, a: Dictionary, b: Dictionary, alpha: float) -> void:
	if not b.has("t"):
		return
	var t := lerpf(float(a.get("t", b["t"])), float(b["t"]), alpha)
	var pf := node.get_node_or_null("Puff")
	if pf != null:
		Vfx.set_puff(pf, t, float(b.get("r", 0.6)), 0.95)


func chase_pose() -> Transform3D:
	if _chars.is_empty():
		return Transform3D(Basis(), Vector3(15, 8, 5))
	var r: Node3D = _chars[0]["root"]
	var fwd := r.global_transform.basis.x
	var p := r.global_position
	return Transform3D(Basis(), p - fwd * 4.5 + Vector3(0, 2.6, 0)).looking_at(p + fwd * 2.0 + Vector3(0, 1.0, 0), Vector3.UP)
