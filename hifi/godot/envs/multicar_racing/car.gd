extends Node3D
## Open-wheel race car laid out on CarRacing's Box2D hull (front wing, nose, sidepods, rear wing),
## wheels at the physics wheel positions. Front is -Z (Box2D local +y). Python sends the pose;
## per step apply() sets wheel steer/spin (Box2D joint angle / wheel phase), brake light and the
## cosmetic body roll/pitch (computed in Python from the acceleration, so it is deterministic).

const WHEEL_RADIUS := 0.54
const WHEEL_WIDTH := 0.5
# Box2D WHEELPOS (x, y) * 0.02 -> Godot (x, r, -y): FL, FR, RL, RR
const WHEEL_POS := [
	Vector3(-1.1, WHEEL_RADIUS, -1.6), Vector3(1.1, WHEEL_RADIUS, -1.6),
	Vector3(-1.1, WHEEL_RADIUS, 1.64), Vector3(1.1, WHEEL_RADIUS, 1.64),
]

var body: Node3D
var _steer: Array[Node3D] = []
var _spin: Array[Node3D] = []
var _smoke: Array[Node3D] = []
var _brake_mat: StandardMaterial3D


func setup(paint_color: Color) -> void:
	var paint := M.paint(paint_color, 0.8, 0.45)
	var carbon := M.mat(Color(0.04, 0.04, 0.05), 0.32, 0.35, {"normal_noise": 0.6, "noise_scale": 6.0})
	var white := M.paint(Color(0.92, 0.92, 0.92), 0.7, 0.1)
	var helmet := M.paint(Color(1.0, 0.82, 0.1), 0.9, 0.2)
	var visor := M.glass(Color(0.02, 0.02, 0.03), 0.9)
	_brake_mat = M.mat(Color(0.3, 0.0, 0.0), 0.4, 0.0, {"emission": Color(1.0, 0.05, 0.02), "emission_energy": 0.8})
	body = Node3D.new()
	add_child(body)
	M.mi(body, M.box(Vector3(1.9, 0.08, 4.2), 0.02), carbon, Vector3(0, 0.2, 0.1))                      # floor
	M.mi(body, M.lathe(PackedVector2Array([Vector2(0.36, 0.0), Vector2(0.3, 0.9), Vector2(0.2, 1.7), Vector2(0.12, 2.05), Vector2(0.0, 2.1)]), 20, "z"),
		paint, Vector3(0, 0.45, -0.5), Vector3(0, 180, 0))                                              # nose cone
	M.mi(body, M.box(Vector3(2.5, 0.06, 0.45), 0.02), carbon, Vector3(0, 0.22, -2.45))                  # front wing
	M.mi(body, M.box(Vector3(2.3, 0.04, 0.22), 0.01), paint, Vector3(0, 0.3, -2.35), Vector3(-12, 0, 0)) # flap
	for sx in [-1.0, 1.0]:
		M.mi(body, M.box(Vector3(0.05, 0.3, 0.5), 0.01), paint, Vector3(sx * 1.25, 0.3, -2.45))
	M.mi(body, M.box(Vector3(1.7, 0.5, 1.6), 0.12), paint, Vector3(0, 0.47, 0.3))                       # sidepods
	M.mi(body, M.box(Vector3(1.72, 0.08, 1.2), 0.02), white, Vector3(0, 0.7, 0.35))                     # livery stripe
	M.mi(body, M.box(Vector3(0.7, 0.25, 0.9), 0.08), paint, Vector3(0, 0.78, -0.3))                     # cockpit rim
	M.mi(body, M.sphere(0.2), helmet, Vector3(0, 0.92, -0.2))
	M.mi(body, M.box(Vector3(0.34, 0.1, 0.05), 0.02), visor, Vector3(0, 0.95, -0.4))
	M.mi(body, M.box(Vector3(0.6, 0.55, 1.8), 0.15), paint, Vector3(0, 0.85, 0.9))                      # engine cover
	M.mi(body, M.box(Vector3(0.5, 0.35, 0.7), 0.04), carbon, Vector3(0, 0.45, 1.95))                    # gearbox
	M.mi(body, M.box(Vector3(0.08, 0.6, 0.15), 0.01), carbon, Vector3(0, 0.95, 2.1))                    # wing pylon
	M.mi(body, M.box(Vector3(1.9, 0.06, 0.55), 0.02), carbon, Vector3(0, 1.25, 2.15))                   # rear wing
	M.mi(body, M.box(Vector3(1.9, 0.05, 0.3), 0.01), paint, Vector3(0, 1.38, 2.32), Vector3(15, 0, 0))  # flap
	for sx in [-1.0, 1.0]:
		M.mi(body, M.box(Vector3(0.05, 0.75, 0.75), 0.01), paint, Vector3(sx * 0.95, 1.05, 2.15))
	M.mi(body, M.box(Vector3(0.28, 0.12, 0.05), 0.01), _brake_mat, Vector3(0, 0.72, 2.44))              # rain light
	var tyre := M.mat(Color(0.05, 0.05, 0.05), 0.85, 0.0, {"normal_noise": 0.5, "noise_scale": 8.0})
	var rim := M.mat(Color(0.72, 0.72, 0.75), 0.25, 0.95)
	for i in 4:
		var pos: Vector3 = WHEEL_POS[i]
		var steer := M.node(self, pos)
		var holder := M.node(steer)
		var spin := M.wheel(holder, WHEEL_RADIUS, WHEEL_WIDTH, tyre, rim, 6)
		if pos.x > 0.0:
			spin.rotation_degrees = Vector3(0, 180, 0)  # rim face outwards on the right side
		# suspension wishbone from the body to the hub
		M.mi(self, M.box(Vector3(0.6, 0.05, 0.08), 0.01), carbon, Vector3(signf(pos.x) * 0.55, 0.5, pos.z))
		M.mi(self, M.box(Vector3(0.6, 0.05, 0.08), 0.01), carbon, Vector3(signf(pos.x) * 0.55, 0.32, pos.z + 0.12))
		_steer.append(steer)
		_spin.append(holder)
		var puff := Vfx.puff(self, Color(0.9, 0.9, 0.9), 31 + i, 4)
		puff.position = Vector3(pos.x, 0.2, pos.z + 0.4)
		_smoke.append(puff)


func apply(c: Dictionary, sim_t: float) -> void:
	var w: Array = c.get("w", [])
	for i in mini(w.size(), 4):
		_steer[i].rotation.y = float(w[i][0])
		_spin[i].rotation.x = -float(w[i][1])
		var skid := int(w[i][2]) == 1
		var grass := int(w[i][3]) == 1
		var speed := float(c.get("speed", 0.0))
		var kick := skid or (grass and speed > 6.0)
		if kick:
			# phase from sim time: deterministic, staggered per wheel
			var ph := fposmod(sim_t * 2.2 + i * 0.27, 1.0)
			_smoke[i].get_child(0).set_instance_shader_parameter("fx_color", Color(0.55, 0.45, 0.3) if grass else Color(0.9, 0.9, 0.9))
			Vfx.set_puff(_smoke[i], ph, 1.2 + 0.4 * ph, 0.5)
		else:
			_smoke[i].visible = false
	_brake_mat.emission_energy_multiplier = 7.0 if float(c.get("brake", 0.0)) > 0.05 else 0.8
	body.rotation = Vector3(float(c.get("pitch", 0.0)), 0.0, float(c.get("roll", 0.0)))
