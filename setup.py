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
        
        try:
            # Stream output directly to console
            # We explicitly flush to ensure pip captures it before any crash
            print(f"Running native build script: {script_path}", flush=True)
            
            subprocess.check_call(
                [sys.executable, script_path], 
                cwd=project_root,
            )
        except subprocess.CalledProcessError as e:
            # If the script failed, it should have printed why. 
            # We add a final persistent error message.
            sys.stderr.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            sys.stderr.write("ERROR: Aranya Prime Native Compilation Failed.\n")
            sys.stderr.write("See messages above for details (missing g++ or gfortran?).\n")
            sys.stderr.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            sys.stderr.flush()
            sys.exit(1)
            
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
