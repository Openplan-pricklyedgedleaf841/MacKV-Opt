from mackv_opt import profiler


def test_hardware_profile_collects_macos_environment(monkeypatch):
    monkeypatch.setattr("mackv_opt.profiler.platform.system", lambda: "Darwin")
    monkeypatch.setattr("mackv_opt.profiler.platform.machine", lambda: "arm64")
    monkeypatch.setattr("mackv_opt.profiler.platform.release", lambda: "25.0.0")
    monkeypatch.setattr("mackv_opt.profiler._detect_chip", lambda system: "Apple M3 Pro")
    monkeypatch.setattr("mackv_opt.profiler._detect_total_memory", lambda system: 36 * 1024**3)
    monkeypatch.setattr("mackv_opt.profiler._detect_available_memory", lambda system, total: 24 * 1024**3)
    monkeypatch.setattr("mackv_opt.profiler._detect_pressure", lambda system: "normal")

    def fake_stdout(command, **kwargs):
        joined = " ".join(command)
        if joined == "sw_vers -productVersion":
            return "15.5"
        if joined == "sw_vers -buildVersion":
            return "24F74"
        if joined == "pmset -g batt":
            return "Now drawing from 'AC Power'"
        if joined == "pmset -g custom":
            return "lowpowermode 0"
        if joined == "pmset -g therm":
            return "Thermal Warning Level: 0"
        return ""

    monkeypatch.setattr("mackv_opt.profiler._command_stdout", fake_stdout)

    hardware = profiler.get_hardware_profile()

    assert hardware.os_version == "macOS 15.5 (24F74)"
    assert hardware.kernel_version == "25.0.0"
    assert hardware.power_source == "ac"
    assert hardware.power_mode == "normal"
    assert hardware.thermal_state == "0"


def test_hardware_profile_uses_unknown_environment_off_macos(monkeypatch):
    monkeypatch.setattr("mackv_opt.profiler.platform.system", lambda: "Windows")
    monkeypatch.setattr("mackv_opt.profiler.platform.machine", lambda: "AMD64")
    monkeypatch.setattr("mackv_opt.profiler.platform.release", lambda: "10")
    monkeypatch.setattr("mackv_opt.profiler.platform.processor", lambda: "x86")
    monkeypatch.setattr("mackv_opt.profiler._detect_total_memory", lambda system: 16 * 1024**3)
    monkeypatch.setattr("mackv_opt.profiler._detect_available_memory", lambda system, total: 12 * 1024**3)
    monkeypatch.setattr("mackv_opt.profiler._detect_pressure", lambda system: "unknown")

    hardware = profiler.get_hardware_profile()

    assert hardware.platform == "Windows"
    assert hardware.power_source == "unknown"
    assert hardware.power_mode == "unknown"
    assert hardware.thermal_state == "unknown"
