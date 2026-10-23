extends Node
## TCP client for hifi/bridge.py (protocol v2: newline-delimited JSON, plus raw frame bytes that
## follow a "frames" header). Reconnects every second until the Python side is up.
## Address: user args --host=127.0.0.1 --port=9876
## --exit-on-disconnect (passed by hifi/renderer.py): quit when the sim goes away instead of
## waiting to reconnect, so a killed Python process never leaves Godot running.

signal message(msg: Dictionary)
signal connection_changed(connected: bool)

var host := "127.0.0.1"
var port := 9876

var _tcp := StreamPeerTCP.new()
var _buf := PackedByteArray()
var _connected := false
var _retry := 0.0
var exit_on_disconnect := false


func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			port = int(arg.substr(7))
		elif arg.begins_with("--host="):
			host = arg.substr(7)
		elif arg == "--exit-on-disconnect":
			exit_on_disconnect = true


func is_connected_to_sim() -> bool:
	return _connected


func send(msg: Dictionary) -> void:
	if _connected:
		_tcp.put_data((JSON.stringify(msg) + "\n").to_utf8_buffer())


func send_with_payload(msg: Dictionary, chunks: Array) -> void:
	## Header line followed by raw bytes (the header announces the byte counts).
	if not _connected:
		return
	_tcp.put_data((JSON.stringify(msg) + "\n").to_utf8_buffer())
	for c in chunks:
		_tcp.put_data(c)


func _process(delta: float) -> void:
	_tcp.poll()
	var status := _tcp.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _connected:
			_connected = true
			_tcp.set_no_delay(true)
			_buf.clear()
			connection_changed.emit(true)
		_read_messages()
	elif status != StreamPeerTCP.STATUS_CONNECTING:
		if _connected:
			_connected = false
			connection_changed.emit(false)
			if exit_on_disconnect:
				get_tree().quit()
				return
		_retry -= delta
		if _retry <= 0.0:
			_retry = 1.0
			_tcp = StreamPeerTCP.new()
			_tcp.connect_to_host(host, port)


func _read_messages() -> void:
	var n := _tcp.get_available_bytes()
	if n > 0:
		var res: Array = _tcp.get_data(n)
		if res[0] == OK:
			_buf.append_array(res[1])
	var start := 0
	var msgs: Array = []
	while true:
		var idx := _buf.find(10, start)  # '\n'
		if idx < 0:
			break
		var msg = JSON.parse_string(_buf.slice(start, idx).get_string_from_utf8())
		start = idx + 1
		if typeof(msg) == TYPE_DICTIONARY:
			msgs.append(msg)
	if start > 0:
		_buf = _buf.slice(start)
	for m in msgs:
		message.emit(m)
