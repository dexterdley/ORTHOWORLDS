class_name Vfx
extends RefCounted
## Deterministic visual effects. Every effect is a pure function of the phase Python sends
## (age / life in [0, 1]) and the global `sim_time` shader parameter, so a lockstep frame depends
## only on the state it renders (no particle systems, no wall-clock time).
##
##   var ex := Vfx.explosion(parent);  Vfx.set_explosion(ex, t, radius)
##   var pf := Vfx.puff(parent, color, seed);  Vfx.set_puff(pf, t, radius)
##   var fl := Vfx.flame(parent, length, radius, color);  Vfx.set_flame(fl, power)   (flame along -Y)
##   var gl := Vfx.glow_capsule(parent, length, radius, color, energy)                (tracer / bolt)
## Effect meshes carry the "no_mask" meta: they never occlude or appear in object-ID masks.

static var _fire_shader: Shader
static var _smoke_shader: Shader
static var _noise: Texture2D


static func clear_cache() -> void:
	_noise = null
	_fire_shader = null
	_smoke_shader = null


static func noise() -> Texture2D:
	if _noise == null:
		_noise = Assets.noise_texture(4242, 0.03, false, 256)
	return _noise


static func fire_material() -> ShaderMaterial:
	if _fire_shader == null:
		_fire_shader = Shader.new()
		_fire_shader.code = """
shader_type spatial;
render_mode blend_add, unshaded, depth_draw_never, cull_back, shadows_disabled;
global uniform float sim_time;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
instance uniform vec4 fx_color : source_color = vec4(1.0, 0.55, 0.15, 1.0);
instance uniform float fx_energy = 4.0;
instance uniform float fx_alpha = 1.0;
void fragment() {
	float f = clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	float n = texture(noise_tex, UV * vec2(2.0, 1.5) + vec2(sim_time * 0.35, -sim_time * 0.9)).r;
	float a = clamp(pow(f, 1.6) * (0.45 + 0.9 * n) * fx_alpha, 0.0, 1.0);
	ALBEDO = mix(fx_color.rgb, vec3(1.0, 0.95, 0.8), pow(f, 6.0)) * fx_energy;
	ALPHA = a;
}
"""
	var m := ShaderMaterial.new()
	m.shader = _fire_shader
	m.set_shader_parameter("noise_tex", noise())
	return m


static func smoke_material() -> ShaderMaterial:
	if _smoke_shader == null:
		_smoke_shader = Shader.new()
		_smoke_shader.code = """
shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_back, diffuse_lambert_wrap, specular_disabled, shadows_disabled;
global uniform float sim_time;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
instance uniform vec4 fx_color : source_color = vec4(0.35, 0.33, 0.31, 1.0);
instance uniform float fx_alpha = 0.8;
void fragment() {
	float f = clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	float n = texture(noise_tex, UV * vec2(2.5, 2.0) + vec2(sim_time * 0.04, sim_time * 0.07)).r;
	ALBEDO = fx_color.rgb * (0.75 + 0.5 * n);
	ALPHA = clamp(pow(f, 1.3) * (0.35 + 1.1 * n) * fx_alpha, 0.0, 1.0);
}
"""
	var m := ShaderMaterial.new()
	m.shader = _smoke_shader
	m.set_shader_parameter("noise_tex", noise())
	return m


static func _fx_mesh(parent: Node3D, mesh: Mesh, mat: Material, node_name: String) -> MeshInstance3D:
	var mi := M.mi(parent, mesh, mat)
	mi.name = node_name
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mi.set_meta("no_mask", true)
	return mi


# ================================================================ explosion

static func explosion(parent: Node3D) -> Node3D:
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Explosion")
	var core := _fx_mesh(root, M.sphere(1.0, 20, 10), fire_material(), "Core")
	core.set_instance_shader_parameter("fx_color", Color(1.0, 0.75, 0.3))
	var fire := _fx_mesh(root, M.sphere(1.0, 24, 12), fire_material(), "Fire")
	fire.set_instance_shader_parameter("fx_color", Color(1.0, 0.38, 0.08))
	_fx_mesh(root, M.sphere(1.0, 24, 12), smoke_material(), "Smoke")
	var l := M.light(root, Color(1.0, 0.6, 0.25), 0.0, 6.0)
	l.name = "Flash"
	return root


static func set_explosion(root: Node3D, t: float, radius: float, smoke_color := Color(0.22, 0.2, 0.19)) -> void:
	## t = age / life in [0, 1]; radius = full radius in metres.
	t = clampf(t, 0.0, 1.0)
	var ease := 1.0 - pow(1.0 - t, 2.0)
	var core: MeshInstance3D = root.get_node("Core")
	var fire: MeshInstance3D = root.get_node("Fire")
	var smoke: MeshInstance3D = root.get_node("Smoke")
	var flash: OmniLight3D = root.get_node("Flash")
	core.scale = Vector3.ONE * maxf(radius * (0.25 + 0.35 * ease), 0.01)
	core.set_instance_shader_parameter("fx_energy", 9.0 * pow(1.0 - t, 2.0))
	fire.scale = Vector3.ONE * maxf(radius * (0.45 + 0.55 * ease), 0.01)
	fire.set_instance_shader_parameter("fx_energy", 5.0 * pow(1.0 - t, 1.5))
	fire.set_instance_shader_parameter("fx_alpha", 1.0 - t * 0.6)
	smoke.scale = Vector3.ONE * maxf(radius * (0.6 + 0.7 * ease), 0.01)
	smoke.position = Vector3(0, radius * 0.4 * ease, 0)
	smoke.set_instance_shader_parameter("fx_color", smoke_color)
	smoke.set_instance_shader_parameter("fx_alpha", 0.85 * sin(PI * minf(t * 1.4, 1.0)))
	flash.light_energy = 6.0 * pow(1.0 - t, 3.0)
	flash.omni_range = radius * 5.0
	root.visible = t < 1.0


# ================================================================ puff (dust, snow, smoke)

static func puff(parent: Node3D, color: Color, seed := 0, blobs := 5) -> Node3D:
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Puff")
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	for i in blobs:
		var b := _fx_mesh(root, M.sphere(1.0, 16, 8), smoke_material(), "B%d" % i)
		b.set_instance_shader_parameter("fx_color", color)
		b.set_meta("dir", Vector3(rng.randf_range(-1, 1), rng.randf_range(0.2, 1.0), rng.randf_range(-1, 1)).normalized())
		b.set_meta("size", rng.randf_range(0.55, 1.0))
	return root


static func set_puff(root: Node3D, t: float, radius: float, alpha := 0.8) -> void:
	t = clampf(t, 0.0, 1.0)
	var ease := 1.0 - pow(1.0 - t, 2.0)
	for b in root.get_children():
		if not b is MeshInstance3D:
			continue
		var d: Vector3 = b.get_meta("dir")
		var s: float = b.get_meta("size")
		b.position = d * radius * 0.7 * ease
		b.scale = Vector3.ONE * maxf(radius * s * (0.35 + 0.65 * ease), 0.01)
		b.set_instance_shader_parameter("fx_alpha", alpha * (1.0 - t))
	root.visible = t < 1.0


# ================================================================ flame / exhaust

static func flame(parent: Node3D, length: float, radius: float, color := Color(1.0, 0.55, 0.2)) -> Node3D:
	## Exhaust plume pointing along local -Y from the node origin.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Flame")
	var outer := _fx_mesh(root, M.cyl(radius, radius * 0.25, 1.0, 16), fire_material(), "Outer")
	outer.set_instance_shader_parameter("fx_color", color)
	var inner := _fx_mesh(root, M.cyl(radius * 0.55, radius * 0.1, 1.0, 12), fire_material(), "Inner")
	inner.set_instance_shader_parameter("fx_color", Color(0.75, 0.85, 1.0))
	var l := M.light(root, color, 0.0, length * 4.0, Vector3(0, -length * 0.3, 0))
	l.name = "Light"
	root.set_meta("length", length)
	set_flame(root, 0.0)
	return root


static func set_flame(root: Node3D, power: float) -> void:
	power = clampf(power, 0.0, 1.0)
	var length: float = root.get_meta("length")
	root.visible = power > 0.01
	for nm in ["Outer", "Inner"]:
		var m: MeshInstance3D = root.get_node(nm)
		var k := length * (0.4 + 0.6 * power) * (1.0 if nm == "Outer" else 0.6)
		m.scale = Vector3(1.0, k, 1.0)
		m.position = Vector3(0, -k * 0.5, 0)
		m.set_instance_shader_parameter("fx_energy", (3.0 if nm == "Outer" else 6.0) * power)
	var l: OmniLight3D = root.get_node("Light")
	l.light_energy = 3.0 * power


# ================================================================ glowing capsule (tracers, bolts)

static func glow_capsule(parent: Node3D, length: float, radius: float, color: Color, energy := 6.0) -> Node3D:
	## Capsule along local +X, centred on the origin, with a small light.
	var root := M.node(parent, Vector3.ZERO, Vector3.ZERO, "Glow")
	var core := M.mi(root, M.capsule(radius, length, 10), M.emissive(color.lerp(Color.WHITE, 0.5), energy),
		Vector3.ZERO, Vector3(0, 0, 90))
	core.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var halo := _fx_mesh(root, M.capsule(radius * 2.6, length * 1.25, 12), fire_material(), "Halo")
	halo.rotation_degrees = Vector3(0, 0, 90)
	halo.set_instance_shader_parameter("fx_color", color)
	halo.set_instance_shader_parameter("fx_energy", energy * 0.5)
	M.light(root, color, energy * 0.25, maxf(length * 3.0, 2.0))
	return root
