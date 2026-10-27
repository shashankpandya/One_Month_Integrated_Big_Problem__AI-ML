"""Keep long batch runs at full speed on Windows 11 (background processes are
otherwise put in 'efficiency mode' and run at about a quarter of a core)."""
import ctypes
import sys


def boost():
    if sys.platform != "win32":
        return
    try:
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        h = k.GetCurrentProcess()
        k.SetPriorityClass(h, 0x00008000)                       # ABOVE_NORMAL_PRIORITY_CLASS

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        s = PowerThrottling(1, 1, 0)                            # control EXECUTION_SPEED, state off
        k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))   # ProcessPowerThrottling
    except Exception:
        pass
