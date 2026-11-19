"""Wire protocol between Python (server) and Godot (client, GymBridge.cs).

Every message: [u8 type][u32 payload length][payload], little-endian.
Keep in sync with GymBridge.cs.
"""

import socket
import struct

# Bump when the messages change; GymBridge.cs sends its own in HELLO.
PROTOCOL_VERSION = 2

# G->P  JSON {protocol_version, n_envs, obs_size, action_type ("box" or "discrete"), action_size,
#             agent_parameters, physics_ticks_per_second}
HELLO = 1
RESET = 2  # P->G  i64[n] seeds (-1 = no seed)
RESET_RESULT = 3  # G->P  f32[n*obs]
STEP = 4  # P->G  f32[n*act] (Box) or i32[n] (Discrete)
STEP_RESULT = 5  # G->P  f32[n*obs], f32[n] reward, u8[n] term, u8[n] trunc, f32[k*obs] final obs
CLOSE = 6  # P->G  empty

_HEADER = struct.Struct("<BI")


def send_msg(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    sock.sendall(_HEADER.pack(msg_type, len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:])
        if r == 0:
            raise ConnectionError("Godot closed the connection")
        got += r
    return buf


def recv_msg(sock: socket.socket, expected_type: int) -> bytearray:
    msg_type, length = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if msg_type != expected_type:
        raise ConnectionError(f"Expected message type {expected_type}, got {msg_type}")
    return _recv_exact(sock, length)
