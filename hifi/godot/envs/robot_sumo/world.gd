extends EnvWorld
## Robot Sumo: two tall wheeled sumo robots on a raised clay dohyo (straw-bale tawara ring at the
## 5 m ring-out radius) in a dark indoor hall: polished wood floor, velvet curtains, stage lights,
## circus-arena HDRI above.
## Python: hifi/exporters/robot_sumo.py. Top-down env: Box2D (x, y) at height h -> Godot (x, h, -y),
## body angle -> yaw. Robot local frame: +X = Box2D local +x (right), -Z = local +y (forward, plow).
## Moving ids: player / opponent (body), *_plow, *_wheel0/1 (axle height 0.25).
## p.robots[i] = [wheel0 roll rad, wheel1 roll rad, centre distance / ring radius].

const RING_R := 5.0
const PLATFORM_R := 6.6
const FLOOR_Y := -0.3
const HALL_R := 16.0

const DOHYO_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform sampler2D sand_alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D sand_nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D clay_alb : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D clay_nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform float ring_r = 5.0;
varying vec3 wpos;
varying vec3 wnrm;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	wnrm = (MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz;
}
void fragment() {
	float r = length(wpos.xz);
	bool top = wnrm.y > 0.5;
	vec2 uv = top ? wpos.xz : vec2(atan(wpos.x, wpos.z) * 6.6, wpos.y);
	float n = texture(noise_tex, wpos.xz * 0.12).r;
	vec3 sand = texture(sand_alb, uv * 0.45).rgb * vec3(0.86, 0.77, 0.6);
	vec3 clay = texture(clay_alb, uv * 0.35).rgb * vec3(0.62, 0.47, 0.35);
	float inner = top ? 1.0 - smoothstep(ring_r - 0.14, ring_r - 0.1, r) : 0.0;
	// sand swept thinner towards the edge, scuffs in the middle
	vec3 col = mix(clay, sand * (0.92 + 0.12 * n), inner);
	// shikiri-sen: two white starting lines
	float lines = top ? (1.0 - smoothstep(0.06, 0.075, abs(abs(wpos.x) - 0.8))) * (1.0 - smoothstep(0.9, 0.93, abs(wpos.z))) : 0.0;
	col = mix(col, vec3(0.9, 0.89, 0.86), lines * 0.92);
	ALBEDO = col;
	NORMAL_MAP = mix(texture(clay_nor, uv * 0.35).rgb, texture(sand_nor, uv * 0.45).rgb, inner);
	ROUGHNESS = mix(0.9, 0.96, inner) - lines * 0.2;
	SPECULAR = 0.3;
}
"""

const STRAW_SHADER := """
shader_type spatial;
uniform sampler2D nor : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
varying vec3 wpos;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	float ang = atan(wpos.x, wpos.z);
	float s = ang * 5.0 / 0.55;
	float band = 1.0 - smoothstep(0.0, 0.06, abs(fract(s) - 0.5) - 0.4);
	float n = texture(noise_tex, vec2(ang * 30.0, wpos.y * 9.0)).r;
	vec3 straw = vec3(0.74, 0.62, 0.38) * (0.75 + 0.45 * n);
	ALBEDO = mix(straw, vec3(0.32, 0.24, 0.14), band * 0.75);
	NORMAL_MAP = texture(nor, vec2(ang * 12.0, wpos.y * 6.0)).rgb;
	ROUGHNESS = 0.92;
}
"""

const CURTAIN_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform vec3 velvet : source_color = vec3(0.11, 0.018, 0.028);
varying vec3 wpos;
void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	float ang = atan(wpos.x, wpos.z);
	float u = ang * 16.0 / 0.45;
	float fold = 0.5 + 0.5 * sin(u * 3.14159 + 0.6 * sin(u * 0.37 + wpos.y * 0.3));
	float n = texture(noise_tex, vec2(u * 0.05, wpos.y * 0.08)).r;
	float fall = smoothstep(-0.3, 1.2, wpos.y);
	ALBEDO = velvet * (0.45 + 0.75 * fold) * (0.85 + 0.3 * n) * (0.75 + 0.25 * fall);
	ROUGHNESS = 0.82;
	SPECULAR = 0.25;
	RIM = 0.4;
	RIM_TINT = 0.6;
}
"""

var _robots: Array = []
var _scene_ok := false


func setup_look(lk: Look) -> void:
	lk.configure({
		"sky": "circus_arena", "sky_energy": 0.55, "sky_rotation": 0.0,
		"sun_from_hdri": false, "sun_elev": 70.0, "sun_azim": 150.0,
		"sun_energy": 1.45, "sun_color": Color(1.0, 0.95, 0.86),
		"ambient_energy": 0.45, "exposure": 0.92, "white": 6.0,
		"saturation": 1.05, "contrast": 1.08, "shadow_distance": 60.0,
		"fog_density": 0.012, "fog_color": Color(0.16, 0.11, 0.09), "fog_aerial": 0.0, "fog_sky_affect": 0.0,
		# indoor hall: the curtain fills the ortho backdrop; below the floor edge a flat floor-brown band
		"horizon_y": FLOOR_Y, "ortho_bg": Color(0.36, 0.27, 0.19),
	})
	# stage haze only in the perspective views
	if lk.env_ortho != null:
		lk.env_ortho.fog_enabled = false


func build_scene(scene: Dictionary) -> void:
	_robots.clear()
	_build_hall()
	_build_dohyo(scene)
	var looks := [
		{"paint": Color(0.1, 0.45, 0.92), "accent": Color(0.02, 0.24, 0.6), "plow": Color(0.42, 0.76, 0.98), "eye": Color(0.2, 0.95, 1.0)},
		{"paint": Color(0.82, 0.14, 0.12), "accent": Color(0.55, 0.05, 0.05), "plow": Color(0.98, 0.42, 0.4), "eye": Color(1.0, 0.62, 0.1)},
	]
	var names := ["player", "opponent"]
	for i in 2:
		var rb := _build_robot(looks[i], i)
		_robots.append(rb)
		register(names[i], rb["body"], names[i])
		register(names[i] + "_plow", rb["plow"], names[i])
		for k in 2:
			register("%s_wheel%d" % [names[i], k], rb["wheels"][k], names[i])
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


static func _painted(color: Color, gloss := 0.7) -> BaseMaterial3D:
	## Painted steel: metal_plate normal / ARM detail under a flat colour with clearcoat.
	var m: BaseMaterial3D = Assets.pbr("blue_metal_plate", {"uv": 0.9, "color": color}).duplicate()
	m.albedo_texture = null
	m.albedo_color = color
	m.metallic = 0.15
	m.roughness = 1.0 - gloss * 0.6
	m.normal_scale = 0.6
	m.clearcoat_enabled = true
	m.clearcoat = 0.6
	m.clearcoat_roughness = 0.2
	return m


# ================================================================ hall and dohyo

func _build_hall() -> void:
	# polished wooden floor, slightly below the dohyo top
	var floor_mat: BaseMaterial3D = Assets.pbr("wood_floor", {"uv": 0.45, "world": true, "tint": Color(0.46, 0.34, 0.26), "color": Color(0.3, 0.22, 0.15)}).duplicate()
	floor_mat.roughness = 0.8
	var fl := M.mi(content, M.plane(Vector2(HALL_R * 2.3, HALL_R * 2.3)), floor_mat, Vector3(0, FLOOR_Y, 0))
	fl.name = "Floor"
	fl.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# velvet curtain wall all around (inner faces)
	var cur := _shader_mat(CURTAIN_SHADER)
	cur.set_shader_parameter("noise_tex", Assets.noise_texture(5, 0.05, false))
	var prof := PackedVector2Array([Vector2(HALL_R, FLOOR_Y + 9.5), Vector2(HALL_R, FLOOR_Y)])
	var cw := M.mi(content, M.lathe(prof, 128), cur)
	cw.name = "Curtain"
	cw.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	# low wooden barrier ring in front of the curtain with sponsor boards
	var board := _painted(Color(0.06, 0.06, 0.07), 0.8)
	var bprof := PackedVector2Array([Vector2(HALL_R - 1.6, FLOOR_Y + 0.9), Vector2(HALL_R - 1.6, FLOOR_Y), Vector2(HALL_R - 1.45, FLOOR_Y),
		Vector2(HALL_R - 1.45, FLOOR_Y + 0.9)])
	M.mi(content, M.lathe(bprof, 96), board)
	var stripe := M.emissive(Color(0.95, 0.75, 0.25), 0.9)
	var sp := PackedVector2Array([Vector2(HALL_R - 1.605, FLOOR_Y + 0.62), Vector2(HALL_R - 1.605, FLOOR_Y + 0.56)])
	M.mi(content, M.lathe(sp, 96), stripe)
	# overhead lighting rig (perspective views only: it would cover the top view)
	var rig := M.node(content, Vector3.ZERO, Vector3.ZERO, "Rig")
	var truss := M.mat(Color(0.12, 0.12, 0.13), 0.4, 0.8)
	var ring := M.lathe(PackedVector2Array([Vector2(7.2, 9.3), Vector2(7.5, 9.3), Vector2(7.5, 9.0), Vector2(7.2, 9.0), Vector2(7.2, 9.3)]), 64)
	M.mi(rig, ring, truss)
	var lens := M.emissive(Color(1.0, 0.95, 0.85), 6.0)
	for k in 12:
		var a := TAU * k / 12.0
		var p := Vector3(cos(a) * 7.35, 8.75, sin(a) * 7.35)
		var can := M.node(rig, p)
		can.look_at_from_position(p, Vector3(0, 0, 0), Vector3.UP)
		M.mi(can, M.cyl(0.18, 0.22, 0.45, 14), truss, Vector3.ZERO, Vector3(90, 0, 0))
		M.mi(can, M.cyl(0.16, 0.16, 0.02, 14), lens, Vector3(0, 0, -0.235), Vector3(90, 0, 0))
	set_views(rig, ["fpv", "chase"])
	# stage spotlights (no shadows: the key light casts them)
	for k in 4:
		var a := TAU * (k + 0.5) / 4.0
		var sl := SpotLight3D.new()
		sl.position = Vector3(cos(a) * 7.35, 8.75, sin(a) * 7.35)
		content.add_child(sl)
		sl.look_at(Vector3(0, 0, 0), Vector3.UP)
		sl.light_color = Color(1.0, 0.93, 0.82) if k % 2 == 0 else Color(0.85, 0.9, 1.0)
		sl.light_energy = 2.2
		sl.spot_range = 22.0
		sl.spot_angle = 26.0
		sl.spot_attenuation = 0.6
		sl.shadow_enabled = false


func _build_dohyo(scene: Dictionary) -> void:
	var ring_r: float = scene.get("ring", RING_R)
	var dm := _shader_mat(DOHYO_SHADER)
	dm.set_shader_parameter("sand_alb", _tex("sand_01", "diff"))
	dm.set_shader_parameter("sand_nor", _tex("sand_01", "nor_gl"))
	dm.set_shader_parameter("clay_alb", _tex("clay_plaster", "diff"))
	dm.set_shader_parameter("clay_nor", _tex("clay_plaster", "nor_gl"))
	dm.set_shader_parameter("noise_tex", Assets.noise_texture(3, 0.04, false))
	dm.set_shader_parameter("ring_r", ring_r)
	var plat := M.mi(content, M.cyl(PLATFORM_R, PLATFORM_R - FLOOR_Y * 1.5, -FLOOR_Y, 96), dm, Vector3(0, FLOOR_Y * 0.5, 0))
	plat.name = "Dohyo"
	# tawara: straw-bale ring on the ring-out line, half buried (top 6 cm above the clay)
	var sm := _shader_mat(STRAW_SHADER)
	sm.set_shader_parameter("nor", _tex("knitted_fleece", "nor_gl"))
	sm.set_shader_parameter("noise_tex", Assets.noise_texture(8, 0.08, false))
	var prof := PackedVector2Array()
	for k in 13:
		var t := PI * k / 12.0
		prof.append(Vector2(ring_r + 0.11 * cos(t), -0.05 + 0.11 * sin(t)))
	var taw := M.mi(content, M.lathe(prof, 160), sm)
	taw.name = "Tawara"


# ================================================================ robots

func _build_robot(lk: Dictionary, idx: int) -> Dictionary:
	var paint := _painted(lk["paint"])
	var accent := _painted(lk["accent"], 0.5)
	var plow_paint := _painted(lk["plow"], 0.55)
	var steel := Assets.pbr("metal_plate", {"uv": 2.0, "tint": Color(0.62, 0.62, 0.64), "color": Color(0.4, 0.4, 0.42)})
	var dark := M.mat(Color(0.06, 0.06, 0.07), 0.5, 0.6)
	var rubber := M.mat(Color(0.035, 0.035, 0.035), 0.82, 0.0, {"normal_noise": 0.8, "noise_scale": 6.0})
	var rim := M.mat(Color(0.7, 0.7, 0.72), 0.3, 0.95)
	var glass := M.glass(Color(0.02, 0.03, 0.04), 0.9)
	var eye := M.emissive(lk["eye"], 3.5)
	var grate := Assets.pbr("metal_grate_rusty", {"uv": 5.0, "tint": Color(0.22, 0.22, 0.23), "color": Color(0.1, 0.1, 0.1)})
	var white := M.mat(Color(0.88, 0.88, 0.86), 0.5, 0.0)

	# ---- body (0.8 wide x 1.0 long x 1.5 tall)
	var body := Node3D.new()
	body.name = "Robot%d" % idx
	content.add_child(body)
	M.mi(body, M.box(Vector3(0.7, 0.34, 0.96), 0.04), dark, Vector3(0, 0.21, 0))           # chassis tub
	M.mi(body, M.box(Vector3(0.8, 0.96, 1.0), 0.07), paint, Vector3(0, 0.86, 0))            # armour shell
	M.mi(body, M.box(Vector3(0.82, 0.08, 1.0), 0.02), accent, Vector3(0, 0.98, 0))         # side stripe band
	M.mi(body, M.box(Vector3(0.7, 0.16, 0.86), 0.05), _painted(Color(0.16, 0.17, 0.19), 0.6), Vector3(0, 1.42, 0))  # sensor head
	# front (-Z): visor with LED eyes, number plate
	M.mi(body, M.box(Vector3(0.6, 0.12, 0.02), 0.01), glass, Vector3(0, 1.42, -0.432))
	for x in [-0.15, 0.15]:
		M.mi(body, M.box(Vector3(0.14, 0.035, 0.012)), eye, Vector3(x, 1.425, -0.437))
	M.mi(body, M.box(Vector3(0.36, 0.26, 0.012), 0.01), white, Vector3(0, 0.74, -0.503))
	M.mi(body, M.box(Vector3(0.3, 0.04, 0.012)), accent, Vector3(0, 0.74, -0.51))
	# plow hinge bracket (body-local (0, 0.6) -> Godot z = -0.6)
	M.mi(body, M.box(Vector3(0.16, 0.1, 0.14), 0.02), steel, Vector3(0, 0.3, -0.54))
	M.mi(body, M.cyl(0.045, 0.045, 0.42, 12), steel, Vector3(0, 0.32, -0.6))
	# back (+Z): cooling vents, tail light, battery hatch
	M.mi(body, M.box(Vector3(0.5, 0.3, 0.012), 0.005), grate, Vector3(0, 1.08, 0.503))
	M.mi(body, M.box(Vector3(0.44, 0.22, 0.03), 0.02), accent, Vector3(0, 0.62, 0.505))
	M.mi(body, M.box(Vector3(0.3, 0.05, 0.012)), M.mat(Color(0.95, 0.75, 0.1), 0.5), Vector3(0, 0.62, 0.522))
	for x in [-0.3, 0.3]:
		M.mi(body, M.box(Vector3(0.08, 0.05, 0.015)), M.emissive(Color(1.0, 0.08, 0.04), 2.0), Vector3(x, 1.26, 0.505))
	# sides: bolted access panels
	for sx in [-1.0, 1.0]:
		M.mi(body, M.box(Vector3(0.012, 0.36, 0.62), 0.008), accent, Vector3(sx * 0.402, 0.68, 0))
		for zz in [-0.27, 0.27]:
			for yy in [0.53, 0.83]:
				M.mi(body, M.cyl(0.018, 0.018, 0.01, 6), steel, Vector3(sx * 0.41, yy, zz), Vector3(0, 0, 90))
	# top: antenna + beacon
	M.mi(body, M.cyl(0.006, 0.008, 0.32, 6), dark, Vector3(0.22, 1.66, 0.3))
	M.mi(body, M.sphere(0.045, 12, 6), M.emissive(lk["eye"], 2.0), Vector3(-0.18, 1.52, 0.25))

	# ---- wheels (separate Box2D bodies): node at the axle, Spin rotates about X
	var wheels: Array = []
	for k in 2:
		var w := Node3D.new()
		w.name = "Wheel%d" % k
		content.add_child(w)
		var spin := M.wheel(w, 0.25, 0.15, rubber, rim, 5)
		spin.name = "Spin"
		# side-wall lettering band + tread blocks so the roll reads
		for t in 12:
			var a := TAU * t / 12.0
			M.mi(spin, M.box(Vector3(0.12, 0.024, 0.05)), rubber, Vector3(0, cos(a) * 0.243, sin(a) * 0.243), Vector3(rad_to_deg(a), 0, 0))
		wheels.append(w)

	# ---- plow (own Box2D body, 1.2 x 0.3, 0.8 tall): curved scoop blade facing local +y
	var plow := Node3D.new()
	plow.name = "Plow%d" % idx
	content.add_child(plow)
	var sc := PackedVector2Array([Vector2(0.15, 0.0), Vector2(0.15, 0.03), Vector2(0.1, 0.12), Vector2(0.055, 0.25),
		Vector2(0.02, 0.4), Vector2(-0.015, 0.55), Vector2(-0.04, 0.68), Vector2(-0.05, 0.8), Vector2(-0.15, 0.8),
		Vector2(-0.15, 0.04), Vector2(-0.12, 0.0)])
	# prism: profile (forward, height) extruded over the width; rotated so mesh +X -> forward (-Z)
	M.mi(plow, M.prism(sc, -0.57, 0.57), plow_paint, Vector3.ZERO, Vector3(0, 90, 0))
	for sz in [-1.0, 1.0]:
		M.mi(plow, M.prism(sc, -0.03, 0.03), steel, Vector3(sz * 0.57, 0, 0), Vector3(0, 90, 0))
	M.mi(plow, M.box(Vector3(1.2, 0.025, 0.07), 0.008), rim, Vector3(0, 0.012, -0.12))      # hardened edge
	M.mi(plow, M.box(Vector3(1.14, 0.05, 0.05), 0.01), steel, Vector3(0, 0.78, 0.12))      # top rail
	M.mi(plow, M.cyl(0.06, 0.06, 0.36, 12), steel, Vector3(0, 0.3, 0.09))                   # hinge knuckle
	for x in [-0.4, 0.0, 0.4]:
		M.mi(plow, M.box(Vector3(0.04, 0.6, 0.06), 0.01), steel, Vector3(x, 0.38, 0.12))    # back ribs
	return {"body": body, "plow": plow, "wheels": wheels}


# ================================================================ per step

func on_state(state: Dictionary, prev: Dictionary, alpha: float) -> void:
	if not _scene_ok:
		return
	var cur: Array = state.get("p", {}).get("robots", [])
	var old: Array = prev.get("p", {}).get("robots", cur)
	for i in mini(cur.size(), _robots.size()):
		var c: Array = cur[i]
		var o: Array = old[i] if i < old.size() else c
		var wheels: Array = _robots[i]["wheels"]
		for k in 2:
			var w: Node3D = wheels[k]
			var spin: Node3D = w.get_node("Spin")
			spin.rotation.x = -lerpf(float(o[k]), float(c[k]), alpha)


func chase_pose() -> Transform3D:
	if _robots.is_empty():
		return Transform3D(Basis(), Vector3(0, 6, 9))
	var b: Node3D = _robots[0]["body"]
	var fwd := -b.global_transform.basis.z
	var p := b.global_position
	return Transform3D(Basis(), p - fwd * 4.0 + Vector3(0, 2.8, 0)).looking_at(p + fwd * 2.0 + Vector3(0, 0.7, 0), Vector3.UP)
