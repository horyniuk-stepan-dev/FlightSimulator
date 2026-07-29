"""
Polled keyboard state — replaces the pynput global hook.

Why this exists
---------------
``pynput.keyboard.Listener`` installs a ``WH_KEYBOARD_LL`` low-level hook.
On Windows that hook sits between raw input and the message queues of *all*
windows, including the OpenCV HighGUI preview window.  While a key is held,
the flood of hook callbacks starves the preview window's repaint messages and
the picture appears to freeze even though the simulation loop keeps running at
a full 30 FPS.  (Verified: ``--mode auto``, which never constructs
``ManualControl``, renders perfectly smoothly under identical load.)

``GetAsyncKeyState`` is a *polled* API — no hook, no callbacks, no interference
with any message queue.  We read it once per frame from the main loop.

Bonus: virtual-key codes are layout-independent on Windows (the physical W key
reports ``VK_W`` even under the Ukrainian layout, which is why Ctrl+C keeps
working there).  That removes the need for the ``('w', 'ц')`` alias lists.

On non-Windows platforms this falls back to the old pynput listener so the
module stays importable and the simulator still runs.
"""

import sys

# Virtual-key codes (winuser.h).  Layout-independent.
_VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
    "shift": 0x10,
    "ctrl": 0x11,
    "esc": 0x1B,
    "q": 0x51,
    "e": 0x45,
}

_HELD = 0x8000  # high bit of GetAsyncKeyState = key is currently down


class _WindowsKeyState:
    """Polled key state via user32.GetAsyncKeyState. No hooks installed."""

    def __init__(self, require_foreground: bool = False, window_title: str = ""):
        import ctypes

        self._user32 = ctypes.windll.user32
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        self._ctypes = ctypes
        self._require_foreground = require_foreground
        self._window_title = window_title

    def _foreground_ok(self) -> bool:
        if not self._require_foreground:
            return True
        ctypes = self._ctypes
        hwnd = self._user32.GetForegroundWindow()
        if not hwnd:
            return False
        buf = ctypes.create_unicode_buffer(512)
        self._user32.GetWindowTextW(hwnd, buf, 512)
        return self._window_title in buf.value

    def pressed(self) -> set:
        """Return the set of currently held key names."""
        if not self._foreground_ok():
            return set()
        get = self._user32.GetAsyncKeyState
        return {name for name, vk in _VK.items() if get(vk) & _HELD}

    def stop(self) -> None:
        pass


class _PynputKeyState:
    """Fallback for non-Windows: the original global listener."""

    def __init__(self, require_foreground: bool = False, window_title: str = ""):
        import threading

        from pynput import keyboard

        self._keyboard = keyboard
        self._lock = threading.Lock()
        self._keys = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _name(key):
        try:
            return key.char.lower()
        except AttributeError:
            return key.name

    def _on_press(self, key):
        with self._lock:
            self._keys.add(self._name(key))

    def _on_release(self, key):
        with self._lock:
            self._keys.discard(self._name(key))

    def pressed(self) -> set:
        with self._lock:
            return set(self._keys)

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass


def make_key_state(require_foreground: bool = False, window_title: str = ""):
    """Build the best available key-state reader for this platform."""
    if sys.platform == "win32":
        try:
            return _WindowsKeyState(require_foreground, window_title)
        except Exception:  # pragma: no cover - ctypes/user32 unavailable
            pass
    return _PynputKeyState(require_foreground, window_title)
