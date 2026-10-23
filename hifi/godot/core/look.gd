class_name Look
extends Node3D
## Sky, sun, fog and post-processing of an env world, plus the per-camera Environment variants:
##   env        perspective views (FPV, chase)
##   env_ortho  orthographic views: no volumetric fog / SSR / SSIL (unreliable with ortho cameras)
##   env_mask   object-ID mask views: black background, linear tonemap, no effects
## Lockstep data rendering keeps everything deterministic: no SDFGI, no TAA, no auto-exposure,
## no temporal reprojection. World scripts call configure() from setup_look().

const DEFAULTS := {
	"sky": "procedural",          # Poly Haven HDRI name | "procedural" | "space"
	"sky_energy": 1.0,
	"sky_rotation": 0.0,          # degrees about +Y
	"sky_top": Color(0.22, 0.42, 0.78),
	"sky_horizon": Color(0.68, 0.76, 0.86),
	"ground_color": Color(0.32, 0.30, 0.26),
	"sun_elev": 45.0,             # degrees above the horizon
	"sun_azim": 135.0,            # degrees; 0 = north (-Z), 90 = east (+X)
	"sun_from_hdri": true,        # aim the sun at the HDRI's brightest pixel when it has a sun
	"sun_energy": 1.4,
	"sun_color": Color(1.0, 0.96, 0.9),
	"shadow_distance": 220.0,
	"ambient_energy": 1.0,
	"ambient_color": null,        # Color: constant ambient instead of sky light (e.g. space)
	"exposure": 1.0,
	"white": 6.0,
	"ssao": true,
	"ssil": true,
	"ssr": false,
	"glow": true,
	"fog_density": 0.0,
	"fog_color": Color(0.72, 0.78, 0.86),
	"fog_aerial": 0.4,
	"fog_sky_affect": 0.15,
	"fog_height": 0.0,            # > 0: height fog below this altitude
	"fog_height_density": 0.0,
	"volumetric": 0.0,            # volumetric fog density (perspective views only)
	"vol_albedo": Color(0.9, 0.9, 0.9),
	"vol_emission": Color(0, 0, 0),
	"vol_anisotropy": 0.3,
	"vol_length": 120.0,
	"saturation": 1.05,
	"contrast": 1.04,
	"brightness": 1.0,
	"ortho_bg": null,             # Color (sRGB, like any Godot Color): flat background for the ortho views
	"ortho_fog": false,           # depth fog in ortho views (their camera distances are arbitrary, so off)
	"horizon_y": 0.0,             # world height of the ground the ortho backdrop's horizon sits on
	"card_ground": "",            # Poly Haven texture painted below the horizon in ortho views ("" = sky colours)
	"card_ground_tint": Color(1, 1, 1),
	"card_ground_scale": 0.15,    # texture repeats per metre on the backdrop's virtual ground
	"card_eye": 2.0,              # virtual eye height of the backdrop's ground plane (perspective strength)
	"stars": 0.0,                 # "space" sky: star density multiplier
	"planet": [],                 # "space" sky: [dir_x, dir_y, dir_z, angular_radius_deg]
}

var cfg := {}
var env: Environment
var env_ortho: Environment
var env_mask: Environment
var sun: DirectionalLight3D
var world_env: WorldEnvironment
var sun_dir := Vector3(0, 1, 0)     # unit vector towards the sun


func configure(overrides: Dictionary) -> void:
	cfg = DEFAULTS.duplicate()
	cfg.merge(overrides, true)
	for c in get_children():
		c.free()
	env = Environment.new()
	_setup_sky()
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	if cfg["ambient_color"] != null:
		env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		env.ambient_light_color = cfg["ambient_color"]
	env.ambient_light_energy = float(cfg["ambient_energy"])
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_exposure = float(cfg["exposure"])
	env.tonemap_white = float(cfg["white"])
	env.ssao_enabled = bool(cfg["ssao"])
	env.ssao_radius = 1.2
	env.ssao_intensity = 1.6
	env.ssao_power = 1.4
	env.ssao_detail = 0.6
	env.ssil_enabled = bool(cfg["ssil"])
	env.ssil_radius = 4.0
	env.ssil_intensity = 0.7
	env.ssr_enabled = bool(cfg["ssr"])
	env.glow_enabled = bool(cfg["glow"])
	env.glow_intensity = 0.35
	env.glow_bloom = 0.04
	env.glow_hdr_threshold = 1.1
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SOFTLIGHT
	if float(cfg["fog_density"]) > 0.0 or float(cfg["fog_height_density"]) > 0.0:
		env.fog_enabled = true
		env.fog_light_color = cfg["fog_color"]
		env.fog_density = float(cfg["fog_density"])
		env.fog_aerial_perspective = float(cfg["fog_aerial"])
		env.fog_sky_affect = float(cfg["fog_sky_affect"])
		if float(cfg["fog_height_density"]) > 0.0:
			env.fog_height = float(cfg["fog_height"])
			env.fog_height_density = float(cfg["fog_height_density"])
	if float(cfg["volumetric"]) > 0.0:
		env.volumetric_fog_enabled = true
		env.volumetric_fog_density = float(cfg["volumetric"])
		env.volumetric_fog_albedo = cfg["vol_albedo"]
		env.volumetric_fog_emission = cfg["vol_emission"]
		env.volumetric_fog_anisotropy = float(cfg["vol_anisotropy"])
		env.volumetric_fog_length = float(cfg["vol_length"])
		env.volumetric_fog_temporal_reprojection_enabled = false
	env.adjustment_enabled = true
	env.adjustment_saturation = float(cfg["saturation"])
	env.adjustment_contrast = float(cfg["contrast"])
	env.adjustment_brightness = float(cfg["brightness"])

	env_ortho = env.duplicate()
	env_ortho.volumetric_fog_enabled = false
	env_ortho.ssr_enabled = false
	env_ortho.ssil_enabled = false
	env_ortho.fog_aerial_perspective = 0.0
	if not bool(cfg["ortho_fog"]) and float(cfg["fog_height_density"]) <= 0.0:
		env_ortho.fog_enabled = false
	if cfg["ortho_bg"] != null:
		env_ortho.background_mode = Environment.BG_COLOR
		env_ortho.background_color = cfg["ortho_bg"]

	env_mask = Environment.new()
	env_mask.background_mode = Environment.BG_COLOR
	env_mask.background_color = Color(0, 0, 0)
	env_mask.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	env_mask.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	env_mask.tonemap_mode = Environment.TONE_MAPPER_LINEAR

	world_env = WorldEnvironment.new()
	world_env.environment = env
	add_child(world_env)

	sun = DirectionalLight3D.new()
	sun.name = "Sun"
	sun.light_energy = float(cfg["sun_energy"])
	sun.light_color = cfg["sun_color"]
	sun.shadow_enabled = true
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	sun.directional_shadow_max_distance = float(cfg["shadow_distance"])
	sun.shadow_blur = 1.2
	sun.shadow_bias = 0.04
	sun.shadow_normal_bias = 1.2
	add_child(sun)
	set_sun_direction(_initial_sun_dir())


func set_sun_direction(d: Vector3) -> void:
	sun_dir = d.normalized()
	var up := Vector3.UP if absf(sun_dir.y) < 0.99 else Vector3.FORWARD
	sun.basis = Basis.looking_at(-sun_dir, up)


static func dir_from_angles(elev_deg: float, azim_deg: float) -> Vector3:
	var e := deg_to_rad(elev_deg)
	var a := deg_to_rad(azim_deg)
	return Vector3(cos(e) * sin(a), sin(e), -cos(e) * cos(a))


func _initial_sun_dir() -> Vector3:
	var d := dir_from_angles(float(cfg["sun_elev"]), float(cfg["sun_azim"]))
	var sky: String = cfg["sky"]
	if bool(cfg["sun_from_hdri"]) and Assets.has_hdri(sky):
		var found := _hdri_sun(sky)
		if found != Vector3.ZERO:
			d = found
	return d


func _hdri_sun(name: String) -> Vector3:
	## Direction of the brightest region of the HDRI's upper hemisphere (rotated with the sky),
	## or ZERO if it has no distinct sun (overcast / indoor).
	var img := Assets.hdri_image(name)
	if img == null:
		return Vector3.ZERO
	img.convert(Image.FORMAT_RGBF)
	img.resize(128, 64, Image.INTERPOLATE_BILINEAR)
	var best := -1.0
	var best_uv := Vector2.ZERO
	var total := 0.0
	var n := 0
	for y in 32:
		for x in 128:
			var c := img.get_pixel(x, y)
			var l := c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722
			total += l
			n += 1
			if l > best:
				best = l
				best_uv = Vector2((x + 0.5) / 128.0, (y + 0.5) / 64.0)
	if best < 8.0 * total / maxf(n, 1):
		return Vector3.ZERO
	var phi := best_uv.x * TAU
	var theta := best_uv.y * PI
	# Godot panorama: u = atan2(x, -z) / 2pi, v = acos(y) / pi
	var d := Vector3(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi))
	d = Basis(Vector3.UP, deg_to_rad(float(cfg["sky_rotation"]))) * d
	if d.y < 0.08:
		d.y = 0.08
	return d.normalized()


func _setup_sky() -> void:
	var sky := Sky.new()
	var kind: String = cfg["sky"]
	if Assets.has_hdri(kind):
		var pm := PanoramaSkyMaterial.new()
		pm.panorama = Assets.hdri(kind)
		pm.energy_multiplier = float(cfg["sky_energy"])
		sky.sky_material = pm
		sky.radiance_size = Sky.RADIANCE_SIZE_256
	elif kind == "space":
		var sm := ShaderMaterial.new()
		sm.shader = _space_shader()
		sm.set_shader_parameter("star_density", float(cfg["stars"]) if float(cfg["stars"]) > 0.0 else 1.0)
		var pl: Array = cfg["planet"]
		if pl.size() == 4:
			sm.set_shader_parameter("planet_dir", Vector3(pl[0], pl[1], pl[2]).normalized())
			sm.set_shader_parameter("planet_radius", deg_to_rad(float(pl[3])))
		sky.sky_material = sm
		sky.radiance_size = Sky.RADIANCE_SIZE_64
	else:
		var ps := ProceduralSkyMaterial.new()
		ps.sky_top_color = cfg["sky_top"]
		ps.sky_horizon_color = cfg["sky_horizon"]
		ps.ground_horizon_color = cfg["sky_horizon"].lerp(cfg["ground_color"], 0.5)
		ps.ground_bottom_color = cfg["ground_color"]
		ps.sun_angle_max = 20.0
		ps.energy_multiplier = float(cfg["sky_energy"])
		sky.sky_material = ps
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.sky_rotation = Vector3(0, deg_to_rad(float(cfg["sky_rotation"])), 0)


func card_material() -> ShaderMaterial:
	## Sky backdrop for the orthographic views. Godot renders a sky through an ortho camera as
	## streaks (every pixel looks the same way), so each ortho camera carries a far quad
	## (view_rig.gd) that paints a world-anchored backdrop: rows above the world horizon height
	## ("horizon_y") sample the sky at an elevation that grows with height, rows below it show a
	## receding ground plane ("card_ground" texture) fading into haze at the horizon line.
	var m := ShaderMaterial.new()
	var sh := Shader.new()
	sh.code = """
shader_type spatial;
render_mode unshaded, fog_disabled, cull_disabled, shadows_disabled, ambient_light_disabled;
uniform sampler2D panorama : source_color, filter_linear, repeat_enable;
uniform sampler2D ground_tex : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform int mode = 0;                  // 0 panorama, 1 gradient, 2 flat
uniform bool has_ground = false;
uniform vec3 top_color : source_color = vec3(0.22, 0.42, 0.78);
uniform vec3 horizon_color : source_color = vec3(0.68, 0.76, 0.86);
uniform vec3 ground_color : source_color = vec3(0.3, 0.3, 0.28);
uniform vec3 ground_tint : source_color = vec3(1.0);
uniform float ground_scale = 0.15;
uniform float ground_light = 1.0;
uniform float energy = 1.0;
uniform float sky_rot = 0.0;
uniform float horizon_y = 0.0;
uniform float eye_height = 2.0;
vec3 sky(vec3 d) {
	float c = cos(sky_rot), s = sin(sky_rot);
	d = vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
	if (mode == 0) {
		vec2 uv = vec2(atan(d.x, -d.z), acos(clamp(d.y, -1.0, 1.0)));
		if (uv.x < 0.0) uv.x += 6.2831853;
		uv /= vec2(6.2831853, 3.1415927);
		return texture(panorama, uv).rgb * energy;
	} else if (mode == 1) {
		return (d.y > 0.0 ? mix(horizon_color, top_color, pow(clamp(d.y, 0.0, 1.0), 0.6))
			: mix(horizon_color, ground_color, pow(clamp(-d.y, 0.0, 1.0), 0.4))) * energy;
	}
	return top_color;
}
void fragment() {
	vec3 right = normalize(INV_VIEW_MATRIX[0].xyz);
	vec3 up = normalize(INV_VIEW_MATRIX[1].xyz);
	vec3 fwd = -normalize(INV_VIEW_MATRIX[2].xyz);
	vec3 cam = INV_VIEW_MATRIX[3].xyz;
	float size = 2.0 / PROJECTION_MATRIX[1][1];
	float aspect = VIEWPORT_SIZE.x / VIEWPORT_SIZE.y;
	vec2 ndc = SCREEN_UV * 2.0 - 1.0;
	vec3 P = cam + ndc.x * size * aspect * 0.5 * right + ndc.y * size * 0.5 * up;  // SCREEN_UV.y grows upwards here
	vec3 fh = vec3(fwd.x, 0.0, fwd.z);
	vec3 col;
	if (mode == 2 || length(fh) < 0.05) {
		col = sky(normalize(fwd));
	} else {
		fh = normalize(fh);
		vec3 rh = normalize(cross(fh, vec3(0.0, 1.0, 0.0)));
		float D = size * 1.2;
		float lat = dot(P - cam, rh);
		float h = P.y - horizon_y;
		float elev = atan(h, D);
		float az = atan(lat, D);
		vec3 d = normalize(fh * cos(az) * cos(elev) + rh * sin(az) * cos(elev) + vec3(0.0, sin(elev), 0.0));
		if (h >= 0.0 || !has_ground) {
			col = sky(d);
		} else {
			float g = eye_height * D / max(-h, 1e-3);
			vec2 gp = cam.xz + fh.xz * g + rh.xz * lat * (g / D);
			vec3 gc = texture(ground_tex, gp * ground_scale).rgb * ground_tint * ground_light;
			vec3 hz = sky(normalize(fh * cos(az) + rh * sin(az) + vec3(0.0, 0.02, 0.0)));
			col = mix(gc, hz, clamp(1.0 - exp(-g / (D * 6.0)), 0.0, 0.85));
		}
	}
	ALBEDO = col;
}
"""
	m.shader = sh
	var kind: String = cfg["sky"]
	m.set_shader_parameter("energy", float(cfg["sky_energy"]))
	m.set_shader_parameter("sky_rot", deg_to_rad(float(cfg["sky_rotation"])))
	m.set_shader_parameter("horizon_y", float(cfg["horizon_y"]))
	m.set_shader_parameter("eye_height", float(cfg["card_eye"]))
	if cfg["ortho_bg"] != null:
		m.set_shader_parameter("mode", 2)
		m.set_shader_parameter("top_color", cfg["ortho_bg"])
	elif Assets.has_hdri(kind):
		m.set_shader_parameter("mode", 0)
		m.set_shader_parameter("panorama", Assets.hdri(kind))
	elif kind == "space":
		m.set_shader_parameter("mode", 2)
		m.set_shader_parameter("top_color", Color(0.004, 0.004, 0.006))
	else:
		m.set_shader_parameter("mode", 1)
		m.set_shader_parameter("top_color", cfg["sky_top"])
		m.set_shader_parameter("horizon_color", cfg["sky_horizon"])
		m.set_shader_parameter("ground_color", cfg["ground_color"])
	var gname: String = cfg["card_ground"]
	if gname != "" and Assets.has_texture(gname):
		m.set_shader_parameter("has_ground", true)
		m.set_shader_parameter("ground_tex", Assets.image_texture(Assets.index()["textures"][gname]["diff"]))
		m.set_shader_parameter("ground_tint", cfg["card_ground_tint"])
		m.set_shader_parameter("ground_scale", float(cfg["card_ground_scale"]))
		var lightk := clampf(sin(deg_to_rad(maxf(rad_to_deg(asin(clampf(sun_dir.y, -1.0, 1.0))), 5.0))) * float(cfg["sun_energy"]) * 0.75 + 0.3 * float(cfg["ambient_energy"]), 0.15, 2.0)
		m.set_shader_parameter("ground_light", lightk)
	return m


static func _space_shader() -> Shader:
	var s := Shader.new()
	s.code = """
shader_type sky;
uniform float star_density = 1.0;
uniform vec3 planet_dir = vec3(0.4, 0.55, -0.73);
uniform float planet_radius = 0.0;
float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
void sky() {
	vec3 d = normalize(EYEDIR);
	vec3 col = vec3(0.0);
	// Stars: hashed cells on three scales
	for (int i = 0; i < 3; i++) {
		float sc = 180.0 * pow(2.2, float(i));
		vec3 cell = floor(d * sc);
		float h = hash(cell);
		float thr = 1.0 - 0.0012 * star_density;
		if (h > thr) {
			vec3 c = (cell + 0.5) / sc;
			float dist = length(d - normalize(c)) * sc;
			float b = smoothstep(0.6, 0.0, dist) * pow((h - thr) / (1.0 - thr), 2.0);
			col += b * mix(vec3(0.75, 0.82, 1.0), vec3(1.0, 0.9, 0.75), hash(cell + 3.1)) * 2.5;
		}
	}
	if (planet_radius > 0.0) {
		float a = acos(clamp(dot(d, normalize(planet_dir)), -1.0, 1.0));
		if (a < planet_radius) {
			vec3 n = normalize(d - normalize(planet_dir) * cos(a));
			float lit = clamp(dot(n, normalize(LIGHT0_DIRECTION)) * 0.9 + 0.25, 0.0, 1.0);
			float cloud = hash(floor(d * 300.0)) * 0.25;
			vec3 ocean = vec3(0.06, 0.18, 0.45);
			vec3 land = vec3(0.25, 0.32, 0.18);
			float m = step(0.62, hash(floor(d * 60.0)));
			col = (mix(ocean, land, m) + cloud) * lit * 1.4;
		} else if (a < planet_radius * 1.06) {
			col += vec3(0.2, 0.4, 0.9) * (1.0 - (a - planet_radius) / (planet_radius * 0.06)) * 0.6;
		}
	}
	COLOR = col;
}
"""
	return s
