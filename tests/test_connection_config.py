import json
import tempfile
import unittest
from pathlib import Path

from connection_config import check_wifi_config, resolve_endpoint, serial_port


class ConnectionConfigTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'local_connection.json'

    def write_settings(self, **values):
        self.path.write_text(json.dumps(values), encoding='utf-8')

    def test_cli_overrides_local_settings_and_model_defaults(self):
        self.write_settings(server_ip='board.local', server_port=4444)
        defaults = {'server_ip': 'saved-board.local', 'server_port': 5555}
        self.assertEqual(resolve_endpoint(defaults=defaults, settings_path=self.path), ('board.local', 4444))
        self.assertEqual(resolve_endpoint('other-board.local', 3333, defaults, self.path), ('other-board.local', 3333))

    def test_missing_host_and_placeholder_fail_with_configuration_hint(self):
        for host in ('', 'BOARD_IP', 'YOUR_BOARD_IP'):
            self.write_settings(server_ip=host)
            with self.assertRaisesRegex(ValueError, 'local_connection.json'):
                resolve_endpoint(settings_path=self.path)

    def test_invalid_ports_fail_instead_of_silently_falling_back(self):
        for port in (0, 65536, True, 1.5, 'bad'):
            self.write_settings(server_ip='board.local', server_port=port)
            with self.assertRaises(ValueError):
                resolve_endpoint(settings_path=self.path)

    def test_serial_port_is_explicit_and_validated(self):
        self.write_settings(serial_port='com42')
        self.assertEqual(serial_port(self.path), 'COM42')
        self.write_settings(serial_port='')
        with self.assertRaisesRegex(ValueError, 'serial_port'):
            serial_port(self.path)

    def test_wifi_requires_real_values_without_disclosing_them(self):
        header = Path(self.folder.name) / 'network_config.h'
        for password in ('', 'YOUR_WIFI_PASSWORD'):
            header.write_text('#define EMG_WIFI_SSID "test-network"\n#define EMG_WIFI_PASSWORD "{}"\n'.format(password), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'EMG_WIFI_PASSWORD'):
                check_wifi_config(header)
        header.write_text('#define EMG_WIFI_SSID "test-network"\n#define EMG_WIFI_PASSWORD "test-password-only"\n', encoding='utf-8')
        check_wifi_config(header)

    def test_malformed_local_settings_are_not_ignored(self):
        self.path.write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            resolve_endpoint('board.local', settings_path=self.path)


if __name__ == '__main__':
    unittest.main()
