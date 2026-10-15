extends MultiMeshInstance3D
## Ring buffer of flat tyre-mark quads shared by all cars.

const MAX_MARKS := 8000
const ROAD_MARK := Color(0.03, 0.03, 0.03, 0.55)
const GRASS_MARK := Color(0.3, 0.24, 0.1, 0.7)

var _next := 0


func _init() -> void:
	var mesh := PlaneMesh.new()
	mesh.size = Vector2(0.45, 0.5)
	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.vertex_color_use_as_albedo = true
	mat.roughness = 0.9
	mesh.material = mat

	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_colors = true
	multimesh.mesh = mesh
	multimesh.instance_count = MAX_MARKS
	multimesh.visible_instance_count = 0
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF


func add_mark(pos: Vector3, yaw: float, grass: bool) -> void:
	var xf := Transform3D(Basis(Vector3.UP, yaw), Vector3(pos.x, 0.035, pos.z))
	multimesh.set_instance_transform(_next, xf)
	multimesh.set_instance_color(_next, GRASS_MARK if grass else ROAD_MARK)
	_next = (_next + 1) % MAX_MARKS
	multimesh.visible_instance_count = mini(multimesh.visible_instance_count + 1, MAX_MARKS)


func clear() -> void:
	_next = 0
	multimesh.visible_instance_count = 0
