import socket
import struct
import threading
import unittest

import numpy as np

from esp32_client import recv_exact, send_sample


class ProtocolTests(unittest.TestCase):
    def test_fragmented_reply_and_little_endian_payload(self):
        server = socket.socket()
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        errors = []

        def serve():
            try:
                with server, server.accept()[0] as client:
                    payload = recv_exact(client, 6400)
                    self.assertEqual(payload[:4], struct.pack('<f', 0.5))
                    for byte in struct.pack('<i', 5):
                        client.sendall(bytes([byte]))
            except Exception as exc:
                errors.append(exc)

        host, port = server.getsockname()
        thread = threading.Thread(target=serve)
        thread.start()
        self.assertEqual(send_sample(host, port, np.full((200, 8), 0.5)), 5)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_truncated_reply_is_not_a_prediction(self):
        left, right = socket.socketpair()
        with left, right:
            right.sendall(b'\x02')
            right.shutdown(socket.SHUT_WR)
            with self.assertRaises(ConnectionError):
                recv_exact(left, 4)

    def test_invalid_windows_fail_before_connecting(self):
        for sample in [np.zeros((8, 200)), np.full((200, 8), np.nan)]:
            with self.assertRaises(ValueError):
                send_sample('127.0.0.1', 1, sample)


if __name__ == '__main__':
    unittest.main()
