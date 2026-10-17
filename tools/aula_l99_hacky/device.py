"""Linux hidraw discovery and I/O for the AULA L99 keyboard's vendor HID channel."""
from __future__ import annotations

import fcntl
import os
import select
import time
from dataclasses import dataclass
from pathlib import Path

CABLE_VID = 0x0C45
CABLE_PID = 0x800A
DONGLE_VID = 0x05AC
DONGLE_PID = 0x024F
VENDOR_INTERFACE = 3  # confirmed by prior art (F75_Initializer) for both cable and dongle


@dataclass(frozen=True)
class HidrawDevice:
    path: str
    vendor_id: int | None
    product_id: int | None
    interface_number: int | None
    name: str | None

    @property
    def is_cable(self) -> bool:
        return self.vendor_id == CABLE_VID and self.product_id == CABLE_PID

    @property
    def is_dongle(self) -> bool:
        return self.vendor_id == DONGLE_VID and self.product_id == DONGLE_PID


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _walk_up_for_file(start: Path, filename: str) -> str | None:
    for candidate in (start, *start.parents):
        value = _read_text(candidate / filename)
        if value:
            return value
    return None


def _parse_hid_id(uevent: str | None) -> tuple[int | None, int | None]:
    if not uevent:
        return (None, None)
    for line in uevent.splitlines():
        if line.startswith("HID_ID="):
            _, value = line.split("=", 1)
            parts = value.split(":")
            if len(parts) == 3:
                try:
                    return (int(parts[1], 16), int(parts[2], 16))
                except ValueError:
                    pass
    return (None, None)


def enumerate_hidraw() -> list[HidrawDevice]:
    devices: list[HidrawDevice] = []
    for sys_node in sorted(Path("/sys/class/hidraw").glob("hidraw*")):
        device_dir = sys_node / "device"
        uevent = _read_text(device_dir / "uevent")
        vendor_id, product_id = _parse_hid_id(uevent)

        interface_raw = _walk_up_for_file(device_dir.resolve(), "bInterfaceNumber")
        interface_number = int(interface_raw, 16) if interface_raw else None

        name = None
        if uevent:
            for line in uevent.splitlines():
                if line.startswith("HID_NAME="):
                    _, name = line.split("=", 1)
                    break

        devices.append(
            HidrawDevice(
                path=str(Path("/dev") / sys_node.name),
                vendor_id=vendor_id,
                product_id=product_id,
                interface_number=interface_number,
                name=name,
            )
        )
    return devices


def find_l99(interface: int = VENDOR_INTERFACE) -> HidrawDevice:
    """Prefer the wired keyboard, fall back to the 2.4G dongle."""
    devices = enumerate_hidraw()
    for vid, pid in ((CABLE_VID, CABLE_PID), (DONGLE_VID, DONGLE_PID)):
        for device in devices:
            if device.vendor_id == vid and device.product_id == pid and device.interface_number == interface:
                return device
    raise FileNotFoundError(
        f"no AULA L99 vendor HID interface found (looked for interface {interface} "
        f"on {CABLE_VID:04x}:{CABLE_PID:04x} or {DONGLE_VID:04x}:{DONGLE_PID:04x})"
    )


_IOC_SIZE_MAX = 0x3FFF  # the size field is 14 bits wide


def _ioctl_code(direction: int, ioc_type: int, nr: int, size: int) -> int:
    # Anything wider than the size field would silently carry into the
    # direction bits above it and build a different ioctl entirely. Latent
    # today (callers pass PACKET_SIZE + 1), so this documents the constraint
    # as much as it enforces it.
    if not 0 <= size <= _IOC_SIZE_MAX:
        raise ValueError(f"ioctl size must be 0..{_IOC_SIZE_MAX}, got {size}")
    return (direction << 30) | (ioc_type << 8) | nr | (size << 16)


class HidrawTransport:
    """Raw read/write plus HID feature-report get/set via /dev/hidrawN."""

    def __init__(self, path: str, timeout_seconds: float = 1.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def __enter__(self) -> "HidrawTransport":
        self._fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
        return self

    def __exit__(self, *_exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def write(self, payload: bytes) -> int:
        assert self._fd is not None
        return os.write(self._fd, payload)

    def read_report(self, max_length: int = 64) -> bytes:
        assert self._fd is not None
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for a report from {self.path}")
            readable, _, _ = select.select([self._fd], [], [], remaining)
            if readable:
                report = os.read(self._fd, max_length)
                if report:
                    return report

    def drain(self, max_reads: int = 32) -> list[bytes]:
        assert self._fd is not None
        drained = []
        for _ in range(max_reads):
            readable, _, _ = select.select([self._fd], [], [], 0)
            if not readable:
                break
            report = os.read(self._fd, 64)
            if not report:
                break
            drained.append(report)
        return drained

    def set_feature(self, report: bytes) -> None:
        """HIDIOCSFEATURE"""
        assert self._fd is not None
        request = _ioctl_code(3, ord("H"), 0x06, len(report))
        fcntl.ioctl(self._fd, request, bytearray(report), True)

    def get_feature(self, report_id: int, size: int) -> bytes:
        """HIDIOCGFEATURE"""
        assert self._fd is not None
        request = _ioctl_code(3, ord("H"), 0x07, size)
        buffer = bytearray(size)
        buffer[0] = report_id
        fcntl.ioctl(self._fd, request, buffer, True)
        return bytes(buffer)


def query_keyboard_battery(dev: HidrawDevice | None = None) -> int | None:
    """Query live battery percentage from the AULA L99 keyboard over 2.4GHz dongle."""
    from . import protocol
    if dev is None:
        dev = find_l99()
    if not dev:
        return None
    try:
        with HidrawTransport(dev.path, timeout_seconds=1.2) as transport:
            transport.drain()
            pkt = protocol.build_dongle_battery_query()
            transport.write(bytes([protocol.REPORT_ID]) + pkt)
            reply = transport.read_report(64)
            return protocol.parse_dongle_battery_reply(reply)
    except Exception:
        return None


def set_keyboard_lighting(
    mode: int,
    rgb: tuple[int, int, int] = (255, 0, 0),
    direction: int = 0,
    brightness: int = 5,
    speed: int = 3,
    dev: HidrawDevice | None = None,
) -> bool:
    """Set RGB lighting effect for AULA L99 across both 2.4GHz dongle and cable."""
    from . import protocol
    if dev is None:
        dev = find_l99()
    if not dev:
        return False
    try:
        with HidrawTransport(dev.path, timeout_seconds=1.5) as transport:
            transport.drain()
            if dev.is_dongle:
                pkt = protocol.build_dongle_lighting_packet(mode, rgb, direction, brightness, speed)
                transport.write(bytes([protocol.REPORT_ID]) + pkt)
                reply = transport.read_report(64)
                return bool(reply and reply[0] == 0x05)
            else:
                txs = protocol.build_effect_transfer(mode, rgb=rgb, speed=speed, brightness=brightness)
                for tx in txs:
                    transport.set_feature(bytes([protocol.REPORT_ID]) + tx.outgoing)
                    time.sleep(protocol.PACKET_GAP_SECONDS)
                return True
    except Exception:
        return False


def set_keyboard_settings(
    sleep_level: int | None = None,
    response_level: int | None = None,
    dev: HidrawDevice | None = None,
) -> bool:
    """Configure sleep timeout (0-3) and/or key debounce response level (1-5)."""
    from . import protocol
    if dev is None:
        dev = find_l99()
    if not dev:
        return False
    try:
        with HidrawTransport(dev.path, timeout_seconds=1.5) as transport:
            transport.drain()
            if dev.is_dongle:
                if sleep_level is not None:
                    pkt = protocol.build_dongle_sleep_time_packet(sleep_level)
                    transport.write(bytes([protocol.REPORT_ID]) + pkt)
                    transport.read_report(64)
                if response_level is not None:
                    pkt = protocol.build_dongle_key_response_packet(response_level)
                    transport.write(bytes([protocol.REPORT_ID]) + pkt)
                    transport.read_report(64)
                return True
            else:
                s_val = sleep_level if sleep_level is not None else 2
                r_val = response_level if response_level is not None else 1
                txs = protocol.build_settings_transfer(s_val, r_val)
                for tx in txs:
                    transport.set_feature(bytes([protocol.REPORT_ID]) + tx.outgoing)
                    time.sleep(protocol.PACKET_GAP_SECONDS)
                return True
    except Exception:
        return False


def set_keyboard_shortcut_locks(
    win_lock: bool = False,
    alt_tab_lock: bool = False,
    alt_f4_lock: bool = False,
    dev: HidrawDevice | None = None,
) -> bool:
    """Toggle gaming shortcut locks (Win key, Alt+Tab, Alt+F4)."""
    from . import protocol
    if dev is None:
        dev = find_l99()
    if not dev or not dev.is_dongle:
        return False
    try:
        with HidrawTransport(dev.path, timeout_seconds=1.5) as transport:
            transport.drain()
            pkt = protocol.build_dongle_shortcut_lock_packet(win_lock, alt_tab_lock, alt_f4_lock)
            transport.write(bytes([protocol.REPORT_ID]) + pkt)
            reply = transport.read_report(64)
            return bool(reply and reply[0] == 0x07)
    except Exception:
        return False
