"""Native binaries the app shells out to must be reported at startup.

Both are looked up with shutil.which at call time and degrade *silently*:
pandoc missing → DOCX export drops to the python engine and loses OMML math;
soffice missing → .doc/.docx conversion fails. Neither announces itself, so a
container (or a host whose conda PATH changed) can lose a feature unnoticed.
"""

from utils.native_deps import (
    NATIVE_DEPS,
    log_native_dependency_warnings,
    missing_native_dependencies,
)


class TestRegistry:
    def test_covers_both_shelled_out_binaries(self):
        assert {dep.name for dep in NATIVE_DEPS} == {"pandoc", "soffice"}

    def test_every_dep_states_what_breaks(self):
        """A warning that doesn't say what degrades is noise."""
        for dep in NATIVE_DEPS:
            assert dep.impact.strip()


class TestDetection:
    def test_reports_missing_binary(self, monkeypatch):
        monkeypatch.setattr("utils.native_deps.shutil.which", lambda _: None)
        missing = missing_native_dependencies()
        assert {dep.name for dep in missing} == {"pandoc", "soffice"}

    def test_reports_nothing_when_all_present(self, monkeypatch):
        monkeypatch.setattr("utils.native_deps.shutil.which", lambda name: f"/usr/bin/{name}")
        assert missing_native_dependencies() == []

    def test_soffice_honours_configured_path(self, monkeypatch):
        """LIBREOFFICE_PATH may point at a binary that isn't called
        'soffice' — probing the literal name would false-alarm."""
        from config.settings import settings

        seen = []

        def _which(name):
            seen.append(name)
            return "/opt/lo/program/soffice.bin"

        monkeypatch.setattr(settings, "libreoffice_path", "/opt/lo/program/soffice.bin")
        monkeypatch.setattr("utils.native_deps.shutil.which", _which)

        assert missing_native_dependencies() == []
        assert "/opt/lo/program/soffice.bin" in seen


class TestLogging:
    def test_warns_once_per_missing_dep(self, monkeypatch):
        monkeypatch.setattr("utils.native_deps.shutil.which", lambda _: None)
        messages = log_native_dependency_warnings()
        assert len(messages) == 2
        joined = " ".join(messages)
        assert "pandoc" in joined and "soffice" in joined

    def test_silent_when_nothing_missing(self, monkeypatch):
        monkeypatch.setattr("utils.native_deps.shutil.which", lambda name: f"/usr/bin/{name}")
        assert log_native_dependency_warnings() == []
