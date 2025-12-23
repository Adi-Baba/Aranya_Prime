from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import os
import sys
import shutil
import platform

class CustomBuildPy(build_py):
    def run(self):
        # 1. Run the native build script
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, "scripts", "build.py")
        
        print(f"Running native build script: {script_path}")
        try:
            # Capture output so we can print it if something goes wrong
            result = subprocess.run(
                [sys.executable, script_path], 
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write("ERROR: Native build failed.\n")
            sys.stderr.write(f"STDOUT:\n{e.stdout}\n")
            sys.stderr.write(f"STDERR:\n{e.stderr}\n")
            raise
            
        # 2. Copy the compiled library to the package directory
        system = platform.system()
        if system == "Windows":
            lib_ext = ".dll"
        elif system == "Darwin":
            lib_ext = ".dylib"
        else:
            lib_ext = ".so"
            
        lib_name = f"aranya_prime{lib_ext}"
        src_lib = os.path.join(project_root, "bin", lib_name)
        dst_lib = os.path.join(project_root, "aranya_prime", lib_name)
        
        if os.path.exists(src_lib):
            print(f"Copying native library from {src_lib} to {dst_lib}")
            shutil.copy2(src_lib, dst_lib)
        else:
            print(f"ERROR: Native library {src_lib} was not found after build.")
            print("Build script may have failed or produced output in unexpected location.")
            sys.exit(1)

        # 3. Proceed with standard packaging
        super().run()

setup(
    packages=find_packages(),
    cmdclass={
        'build_py': CustomBuildPy,
    },
    include_package_data=True,
)
