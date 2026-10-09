"""Read machine-specific connection settings without publishing them."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_settings(path=None):
    path = Path(path) if path is not None else ROOT / 'local_connection.json'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig') as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError('local_connection.json must contain a JSON object')
    return settings


def validate_host(host):
    if not isinstance(host, str) or not host.strip() or host.upper().startswith(('YOUR_', 'BOARD_IP')):
        raise ValueError('Set server_ip in local_connection.json or pass --server-ip; run scripts\\configure.cmd first')
    if re.search(r'\s|[/\\]', host):
        raise ValueError('server_ip must be a board IP address or hostname, without a URL or spaces')
    return host.strip()


def validate_port(port):
    if isinstance(port, bool):
        raise ValueError('server_port must be an integer from 1 to 65535')
    if isinstance(port, str):
        if not port.isdigit():
            raise ValueError('server_port must be an integer from 1 to 65535')
        port = int(port)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError('server_port must be an integer from 1 to 65535')
    return port


def resolve_endpoint(server_ip=None, server_port=None, defaults=None, settings_path=None):
    settings = load_settings(settings_path)
    defaults = defaults or {}
    host = server_ip if server_ip is not None else settings.get('server_ip') or defaults.get('server_ip')
    if server_port is None:
        server_port = settings.get('server_port', defaults.get('server_port', 3333))
    return validate_host(host), validate_port(server_port)


def serial_port(settings_path=None):
    value = load_settings(settings_path).get('serial_port', '')
    if not isinstance(value, str) or not re.fullmatch(r'COM[1-9]\d*', value.upper()):
        raise ValueError('Set serial_port to the actual COM port in local_connection.json, or pass PORT to scripts\\esp32.cmd')
    return value.upper()


def check_wifi_config(path):
    path = Path(path)
    if not path.exists():
        raise ValueError('Wi-Fi configuration is missing; run scripts\\configure.cmd and fill network_config.h')
    content = path.read_text(encoding='utf-8-sig')
    for key in ('EMG_WIFI_SSID', 'EMG_WIFI_PASSWORD'):
        match = re.search(r'^\s*#define\s+' + key + r'\s+"((?:\\.|[^"\\])*)"', content, re.MULTILINE)
        if match is None or not match.group(1).strip() or match.group(1).startswith('YOUR_'):
            raise ValueError('Fill {} in the ignored network_config.h before building or flashing'.format(key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--get', choices=['serial_port', 'server_ip', 'server_port'])
    group.add_argument('--check-wifi', type=Path)
    args = parser.parse_args()
    try:
        if args.check_wifi:
            check_wifi_config(args.check_wifi)
        elif args.get == 'serial_port':
            print(serial_port())
        elif args.get == 'server_ip':
            print(validate_host(load_settings().get('server_ip')))
        else:
            print(validate_port(load_settings().get('server_port', 3333)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
