import os
import sys
import time

import numpy as np


def resolve_path(base_dir, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_myo_module(cfg):
    base_dir = os.path.dirname(os.path.abspath(cfg.get("_config_path", "config.yaml")))
    module_root = resolve_path(base_dir, cfg.get("myo", {}).get("module_root"))
    if module_root:
        if module_root not in sys.path:
            sys.path.insert(0, module_root)

    import myo

    dll_root = resolve_path(base_dir, cfg.get("myo", {}).get("dll_root"))
    sdk_path = resolve_path(base_dir, cfg.get("myo", {}).get("sdk_path"))
    dll_name = "myo64.dll" if sys.maxsize > 2 ** 32 else "myo32.dll"

    if dll_root:
        dll_path = os.path.join(dll_root, dll_name)
        if os.path.exists(dll_path):
            myo.init(lib_name=dll_path)
        elif sdk_path:
            myo.init(sdk_path=sdk_path)
        else:
            myo.init()
    elif sdk_path:
        myo.init(sdk_path=sdk_path)
    else:
        myo.init()
    return myo


class MyoEMGCollector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg["data"]["fs"]
        self.channels = cfg["data"]["channel"]
        self.myo = load_myo_module(cfg)
        self.listener = self._build_listener()

    def _build_listener(self):
        outer = self
        myo = self.myo

        class Listener(myo.DeviceListener):
            def __init__(self):
                self.hub = myo.Hub()
                self.frames = []
                self.collecting = False
                self.connected = False

            def on_connected(self, event):
                self.connected = True
                print("Myo connected")
                event.device.stream_emg(True)

            def on_emg(self, event):
                if self.collecting:
                    self.frames.append(event.emg[: outer.channels])

            def on_event(self, event):
                super().on_event(event)

        return Listener()

    @property
    def hub(self):
        return self.listener.hub

    def wait_for_connection(self, timeout_seconds=10.0):
        start = time.time()
        while time.time() - start < timeout_seconds:
            self.hub.run(self.listener.on_event, 10)
            if self.listener.connected:
                return True
            time.sleep(0.02)
        return False

    def collect_duration(self, seconds):
        self.listener.frames = []
        self.listener.collecting = True
        start = time.time()
        while time.time() - start < seconds:
            self.hub.run(self.listener.on_event, 10)
        self.listener.collecting = False
        return np.asarray(self.listener.frames, dtype=np.float32)
