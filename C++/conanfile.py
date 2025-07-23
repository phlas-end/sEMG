from conan import ConanFile

class YourProjectConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = [
        "cnpy/cci.20180601",
        "zlib/1.3.1",
        "fmt/11.2.0"
    ]
    generators = "CMakeDeps", "CMakeToolchain"

    def layout(self):
        build_type = str(self.settings.build_type) if self.settings.build_type else "default"
        build_type = build_type.lower()

        self.folders.build = f"build-{build_type}"
        self.folders.generators = self.folders.build
