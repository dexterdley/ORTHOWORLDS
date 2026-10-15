extends Node3D
## Procedural open-wheel race car driven by streamed GymBox car state.
## The layout follows CarRacing's Box2D hull (front wing, nose, sidepods, rear wing) with
## wheels at the physics wheel positions. Front is -Z, as in every Godot scene.
## Box2D supplies pose, velocity, wheel steer/spin and skid flags; body roll/pitch,
## brake light, tyre smoke and skid marks are cosmetic and never feed back into physics.

const WHEEL_RADIUS := 0.54
const WHEEL_WIDTH := 0.5
# Box2D WHEELPOS (x, y) * 0.02 -> Godot (x, r, -y): FL, FR, RL, RR
const WHEEL_POS := [
	Vector3(-1.1, WHEEL_RADIUS, -1.6), Vector3(1.1, WHEEL_RADIUS, -1.6),
	Vector3(-1.1, WHEEL_RADIUS, 1.64), Vector3(1.1, WHEEL_RADIUS, 1.64),
]
const SMOKE_COLOR := Color(0.92, 0.92, 0.92)
const DUST_COLOR := Color(0.55, 0.45, 0.28)

var yaw := 0.0
var speed := 0.0

var _body: Node3D
var _wheel_steer: Array[Node3D] = []
var _wheel_spin: Array[Node3D] = []
var _smoke: Array[GPUParticles3D] = []
var _last_mark: Array[Vector3] = []
var _brake_mat: StandardMaterial3D
var _skids: Node
var _vel_prev := Vector3.ZERO
var _acc := Vector3.ZERO
var _roll := 0.0
var _roll_v := 0.0
var _pitch := 0.0
var _pitch_v := 0.0
var _has_prev := false


func setup(paint_color: Color, skids: Node) -> void:
	_skids = skids
	var paint := _mat(paint_color, 0.5, 0.28)
	paint.clearcoat_enabled = true
	paint.clearcoat = 1.0
	paint.clearcoat_roughness = 0.08
	var carbon := _mat(Color(0.05, 0.05, 0.06), 0.3, 0.35)
	var white := _mat(Color(0.95, 0.95, 0.95), 0.1, 0.35)
	var helmet := _mat(Color(1.0, 0.85, 0.1), 0.2, 0.25)
	_brake_mat = _mat(Color(0.3, 0.0, 0.0), 0.0, 0.4)
	_brake_mat.emission_enabled = true
	_brake_mat.emission = Color(1.0, 0.05, 0.02)

	_body = Node3D.new()
	add_child(_body)
	_part(_body, _box(Vector3(1.9, 0.08, 4.2)), carbon, Vector3(0, 0.2, 0.1))              # floor
	_part(_body, _cyl(0.16, 0.36, 2.0), paint, Vector3(0, 0.45, -1.5), Vector3(-90, 0, 0))  # nose
	_part(_body, _box(Vector3(2.5, 0.06, 0.45)), carbon, Vector3(0, 0.22, -2.45))          # front wing
	for sx in [-1.0, 1.0]:
		_part(_body, _box(Vector3(0.05, 0.3, 0.5)), paint, Vector3(sx * 1.25, 0.3, -2.45))
	_part(_body, _box(Vector3(1.7, 0.5, 1.6)), paint, Vector3(0, 0.47, 0.3))                # sidepods
	_part(_body, _box(Vector3(1.72, 0.08, 1.2)), white, Vector3(0, 0.7, 0.35))              # livery stripe
	_part(_body, _box(Vector3(0.7, 0.25, 0.9)), paint, Vector3(0, 0.78, -0.3))              # cockpit rim
	var head := SphereMesh.new()
	head.radius = 0.2
	head.height = 0.4
	_part(_body, head, helmet, Vector3(0, 0.92, -0.2))
	_part(_body, _box(Vector3(0.6, 0.55, 1.8)), paint, Vector3(0, 0.85, 0.9))               # engine cover
	_part(_body, _box(Vector3(0.5, 0.35, 0.7)), carbon, Vector3(0, 0.45, 1.95))             # gearbox
	_part(_body, _box(Vector3(0.08, 0.6, 0.15)), carbon, Vector3(0, 0.95, 2.1))             # wing pylon
	_part(_body, _box(Vector3(1.9, 0.06, 0.55)), carbon, Vector3(0, 1.25, 2.15))            # rear wing
	_part(_body, _box(Vector3(1.9, 0.05, 0.3)), paint, Vector3(0, 1.38, 2.32))              # flap
	for sx in [-1.0, 1.0]:
		_part(_body, _box(Vector3(0.05, 0.75, 0.75)), paint, Vector3(sx * 0.95, 1.05, 2.15))
	_part(_body, _box(Vector3(0.28, 0.12, 0.05)), _brake_mat, Vector3(0, 0.72, 2.44))       # rain light

	var tire := _mat(Color(0.06, 0.06, 0.06), 0.0, 0.8)
	var rim := _mat(Color(0.75, 0.75, 0.78), 0.9, 0.25)
	var spoke := _mat(Color(0.12, 0.12, 0.13), 0.6, 0.4)
	for i in 4:
		var pos: Vector3 = WHEEL_POS[i]
		var steer := Node3D.new()
		steer.position = pos
		add_child(steer)
		var spin := Node3D.new()
		steer.add_child(spin)
		_part(spin, _cyl(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 28), tire, Vector3.ZERO, Vector3(0, 0, 90))
		_part(spin, _cyl(0.34, 0.34, WHEEL_WIDTH + 0.02, 20), rim, Vector3.ZERO, Vector3(0, 0, 90))
		_part(spin, _box(Vector3(WHEEL_WIDTH + 0.04, 0.08, 0.62)), spoke, Vector3.ZERO)
		_part(spin, _box(Vector3(WHEEL_WIDTH + 0.04, 0.62, 0.08)), spoke, Vector3.ZERO)
		# suspension arm from the body to the hub
		var arm_x := signf(pos.x) * 0.55
		_part(self, _box(Vector3(0.6, 0.05, 0.08)), carbon, Vector3(arm_x, 0.5, pos.z))
		_wheel_steer.append(steer)
		_wheel_spin.append(spin)
		var smoke := _make_smoke()
		smoke.position = Vector3(pos.x, 0.15, pos.z)
		add_child(smoke)
		_smoke.append(smoke)
		_last_mark.append(Vector3.INF)


func apply_state(a: Dictionary, b: Dictionary, t: float, delta: float) -> void:
	## Interpolates between the previous (a) and latest (b) physics snapshots.
	position = Vector3(lerpf(a["x"], b["x"], t), 0.0, -lerpf(a["y"], b["y"], t))
	yaw = lerp_angle(a["a"], b["a"], t)
	rotation = Vector3(0.0, yaw, 0.0)
	var vel := Vector3(lerpf(a["vx"], b["vx"], t), 0.0, -lerpf(a["vy"], b["vy"], t))
	speed = vel.length()
	_update_body_motion(vel, delta)

	var wa: Array = a["wheels"]
	var wb: Array = b["wheels"]
	for i in 4:
		_wheel_steer[i].rotation.y = lerpf(wa[i][0], wb[i][0], t)
		_wheel_spin[i].rotation.x = -lerpf(wa[i][1], wb[i][1], t)
		_update_effects(i, wb[i][2] > 0, wb[i][3] > 0)
	_brake_mat.emission_energy_multiplier = 8.0 if float(b["brake"]) > 0.05 else 0.8


func _update_body_motion(vel: Vector3, delta: float) -> void:
	if delta <= 0.0:
		return
	if _has_prev:
		var acc := (vel - _vel_prev) / delta
		_acc = _acc.lerp(acc, clampf(delta * 8.0, 0.0, 1.0))
	_has_prev = true
	_vel_prev = vel
	var local := Basis(Vector3.UP, yaw).inverse() * _acc
	# Lateral accel rolls the body to the outside of the turn; braking (+Z accel) dips the nose.
	var roll_target := clampf(local.x * 0.0006, -0.07, 0.07)
	var pitch_target := clampf(-local.z * 0.0012, -0.05, 0.05)
	# Damped spring in fixed sub-steps so a frame hitch (e.g. shader compile) can't blow it up
	var remaining := minf(delta, 0.1)
	while remaining > 0.0:
		var h := minf(remaining, 1.0 / 120.0)
		_roll_v += ((roll_target - _roll) * 160.0 - _roll_v * 16.0) * h
		_roll += _roll_v * h
		_pitch_v += ((pitch_target - _pitch) * 160.0 - _pitch_v * 16.0) * h
		_pitch += _pitch_v * h
		remaining -= h
	_body.rotation = Vector3(_pitch, 0.0, _roll)


func _update_effects(i: int, skid: bool, grass: bool) -> void:
	var kick := skid or (grass and speed > 6.0)
	_smoke[i].emitting = kick
	if kick:
		(_smoke[i].process_material as ParticleProcessMaterial).color = DUST_COLOR if grass else SMOKE_COLOR
	if skid:
		var wp := _wheel_steer[i].global_position
		if wp.distance_to(_last_mark[i]) > 0.4:
			_skids.add_mark(wp, yaw + _wheel_steer[i].rotation.y, grass)
			_last_mark[i] = wp
	else:
		_last_mark[i] = Vector3.INF


func _make_smoke() -> GPUParticles3D:
	var p := GPUParticles3D.new()
	p.amount = 48
	p.lifetime = 1.5
	p.local_coords = false
	p.emitting = false
	p.visibility_aabb = AABB(Vector3(-30, -5, -30), Vector3(60, 20, 60))

	var pm := ParticleProcessMaterial.new()
	pm.direction = Vector3(0, 1, 0)
	pm.spread = 40.0
	pm.initial_velocity_min = 0.5
	pm.initial_velocity_max = 1.8
	pm.gravity = Vector3(0, 0.5, 0)
	pm.damping_min = 0.5
	pm.damping_max = 1.2
	pm.scale_min = 0.8
	pm.scale_max = 1.6
	var scale_curve := Curve.new()
	scale_curve.max_value = 3.0
	scale_curve.add_point(Vector2(0.0, 0.6))
	scale_curve.add_point(Vector2(1.0, 3.0))
	var sct := CurveTexture.new()
	sct.curve = scale_curve
	pm.scale_curve = sct
	var fade := Gradient.new()
	fade.set_color(0, Color(1, 1, 1, 0.45))
	fade.set_color(1, Color(1, 1, 1, 0.0))
	var fade_tex := GradientTexture1D.new()
	fade_tex.gradient = fade
	pm.color_ramp = fade_tex
	p.process_material = pm

	var puff := Gradient.new()
	puff.set_color(0, Color(1, 1, 1, 1))
	puff.set_color(1, Color(1, 1, 1, 0))
	var puff_tex := GradientTexture2D.new()
	puff_tex.gradient = puff
	puff_tex.fill = GradientTexture2D.FILL_RADIAL
	puff_tex.fill_from = Vector2(0.5, 0.5)
	puff_tex.fill_to = Vector2(1.0, 0.5)
	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
	mat.vertex_color_use_as_albedo = true
	mat.albedo_texture = puff_tex
	var quad := QuadMesh.new()
	quad.size = Vector2(1.0, 1.0)
	quad.material = mat
	p.draw_pass_1 = quad
	return p


func _part(parent: Node3D, mesh: Mesh, mat: Material, pos: Vector3, rot_deg := Vector3.ZERO) -> void:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	mi.rotation_degrees = rot_deg
	parent.add_child(mi)


static func _box(size: Vector3) -> BoxMesh:
	var m := BoxMesh.new()
	m.size = size
	return m


static func _cyl(r_top: float, r_bottom: float, h: float, segments := 20) -> CylinderMesh:
	var m := CylinderMesh.new()
	m.top_radius = r_top
	m.bottom_radius = r_bottom
	m.height = h
	m.radial_segments = segments
	return m


static func _mat(color: Color, metallic: float, roughness: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.metallic = metallic
	m.roughness = roughness
	return m
