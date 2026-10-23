extends RefCounted
## Toxic gas body: a box (x from the West end to the hard front at gas_x) rendered as a ray-marched
## volume. Back faces are rasterised with the depth test off; the shader rebuilds the view ray
## (perspective or orthographic), intersects it with the box, clips it at the opaque scene depth and
## integrates a noisy, height-faded density. Works the same in the ortho and perspective views (no
## FogVolume needed; the ortho Environment disables volumetric fog). Deterministic: the only
## animation input is the global `sim_time`.
## Section cuts: in the Rear / Side ortho views the ray starts at the car (cut_rear_z / cut_side_x),
## like the walls on the viewer's side of the car that state["vis"] hides.

static var _shader: Shader


static func material() -> ShaderMaterial:
	if _shader == null:
		_shader = Shader.new()
		_shader.code = CODE
	var m := ShaderMaterial.new()
	m.shader = _shader
	m.set_shader_parameter("noise_tex", Assets.noise_texture(913, 0.018, false, 256))
	return m


const CODE := """
shader_type spatial;
render_mode blend_mix, depth_draw_never, depth_test_disabled, cull_front, unshaded, shadows_disabled, fog_disabled;

global uniform float sim_time;
uniform sampler2D depth_tex : hint_depth_texture, filter_nearest, repeat_disable;
uniform sampler2D noise_tex : repeat_enable, filter_linear_mipmap;
uniform vec3 box_min = vec3(0.0);
uniform vec3 box_max = vec3(1.0);
uniform float cut_rear_z = 100000.0;
uniform float cut_side_x = 100000.0;
uniform vec3 gas_lit : source_color = vec3(0.42, 0.74, 0.1);
uniform vec3 gas_deep : source_color = vec3(0.2, 0.4, 0.04);
uniform float density = 0.62;   // ~3.35 m seen from Top -> alpha ~0.55-0.6
uniform float max_alpha = 0.8;

float gas_density(vec3 p) {
	float top = box_max.y;
	float hf = 1.0 - smoothstep(top - 1.1, top, p.y);
	vec2 q1 = p.xz * 0.024 + vec2(sim_time * 0.012, sim_time * 0.005);
	vec2 q2 = vec2(p.x * 0.055 - sim_time * 0.02, p.z * 0.045 + p.y * 0.09);
	vec2 q3 = vec2(p.z * 0.13 + sim_time * 0.03, p.y * 0.16 - p.x * 0.1);
	float n = textureLod(noise_tex, q1, 1.0).r * 0.5 + textureLod(noise_tex, q2, 1.0).r * 0.33 + textureLod(noise_tex, q3, 1.0).r * 0.17;
	// a denser rolling bank right behind the front
	float bank = 1.0 + 1.4 * exp(-max(box_max.x - p.x, 0.0) * 2.2);
	return hf * bank * clamp(0.55 + 1.2 * (n - 0.5), 0.3, 1.4);
}

void fragment() {
	bool ortho = PROJECTION_MATRIX[3][3] > 0.5;
	vec3 ro_v = ortho ? vec3(VERTEX.xy, 0.0) : vec3(0.0);
	vec3 rd_v = ortho ? vec3(0.0, 0.0, -1.0) : normalize(VERTEX);
	vec3 ro = (INV_VIEW_MATRIX * vec4(ro_v, 1.0)).xyz;
	vec3 rd = normalize((INV_VIEW_MATRIX * vec4(rd_v, 0.0)).xyz);
	float z = texture(depth_tex, SCREEN_UV).r;
	vec4 sp = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, z, 1.0);
	sp.xyz /= sp.w;
	float t_scene = ortho ? -sp.z : length(sp.xyz);
	vec3 bmin = box_min;
	vec3 bmax = box_max;
	if (ortho && abs(rd.z) > 0.99) { bmax.z = min(bmax.z, cut_rear_z); }
	if (ortho && abs(rd.x) > 0.99) { bmax.x = min(bmax.x, cut_side_x); }
	vec3 srd = vec3(abs(rd.x) < 1e-6 ? 1e-6 : rd.x, abs(rd.y) < 1e-6 ? 1e-6 : rd.y, abs(rd.z) < 1e-6 ? 1e-6 : rd.z);
	vec3 inv = 1.0 / srd;
	vec3 ta = (bmin - ro) * inv;
	vec3 tb = (bmax - ro) * inv;
	vec3 t0 = min(ta, tb);
	vec3 t1 = max(ta, tb);
	float tn = max(max(t0.x, t0.y), max(t0.z, 0.0));
	float tf = min(min(t1.x, t1.y), min(t1.z, t_scene));
	if (tf <= tn + 1e-4) {
		discard;
	}
	const int N = 16;
	float dt = (tf - tn) / float(N);
	float tau = 0.0;
	float lit = 0.0;
	for (int i = 0; i < N; i++) {
		vec3 p = ro + rd * (tn + (float(i) + 0.5) * dt);
		float d = gas_density(p) * density * dt;
		float tr = exp(-tau);
		float hl = 0.55 + 0.45 * smoothstep(0.0, box_max.y, p.y);
		lit += (1.0 - exp(-d)) * tr * hl;
		tau += d;
	}
	float a = 1.0 - exp(-tau);
	vec3 col = mix(gas_deep, gas_lit, clamp(lit / max(a, 1e-4), 0.0, 1.0));
	// billows on the entry surface (the hard front, the top) so the body never reads as a flat slab
	vec3 pe = ro + rd * tn;
	float b1 = textureLod(noise_tex, vec2(pe.z * 0.12 + pe.x * 0.05, pe.y * 0.16) + vec2(sim_time * 0.02, -sim_time * 0.04), 1.0).r;
	float b2 = textureLod(noise_tex, pe.xz * 0.03 + vec2(-sim_time * 0.01, sim_time * 0.005), 1.0).r;
	float bil = mix(b1, b2, abs(rd.y));
	float amp = mix(1.0, 0.45, abs(rd.y));
	col *= 1.0 + amp * 0.6 * (bil - 0.5);
	a = clamp(a * (1.0 + amp * 0.4 * (bil - 0.5)), 0.0, 1.0);
	// brighter, denser band right at the hard front (reads as a crisp edge in Top, also at 96x96)
	float fb = exp(-max(box_max.x - pe.x, 0.0) * 4.0);
	col = mix(col, gas_lit * 1.2, 0.45 * fb);
	a = max(a, 0.72 * fb);
	ALBEDO = col;
	ALPHA = min(a, max_alpha);
}
"""
