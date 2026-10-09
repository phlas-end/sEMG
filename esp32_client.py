"""TCP client for the existing ESP32 window-inference protocol."""

import socket
import struct

import numpy as np


def recv_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError(f"Incomplete ESP32 response: {len(result)}/{size} bytes")
        result.extend(chunk)
    return bytes(result)


def send_sample(server_ip, server_port, sample, timeout=10.0):
    sample = np.asarray(sample, dtype='<f4')
    if sample.shape != (200, 8):
        raise ValueError(f"Expected a (200, 8) window, got {sample.shape}")
    if not np.isfinite(sample).all():
        raise ValueError("Input contains NaN or infinity")
    with socket.create_connection((server_ip, server_port), timeout=timeout) as sock:
        sock.sendall(sample.tobytes(order='C'))
        return struct.unpack('<i', recv_exact(sock, 4))[0]
