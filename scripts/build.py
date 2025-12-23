import os
import subprocess
import glob
import sys

# Configuration
import platform

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
OBJ_DIR = os.path.join(PROJECT_ROOT, "obj")

# OS Detection
SYSTEM = platform.system()
if SYSTEM == "Windows":
    LIB_EXT = ".dll"
    SHARED_FLAG = "-shared"
elif SYSTEM == "Darwin":
    LIB_EXT = ".dylib"
    SHARED_FLAG = "-dynamiclib"
else: # Linux
    LIB_EXT = ".so"
    SHARED_FLAG = "-shared"

DLL_PATH = os.path.join(BIN_DIR, f"aranya_prime{LIB_EXT}")

# Compiler Commands
CC = "g++"
FC = "gfortran"
VC = "v"
CFLAGS = ["-c", "-O3", "-mavx2", "-mfma", "-fopenmp", "-fPIC"]
FFLAGS = ["-c", "-O3", "-mavx2", "-fopenmp", "-fPIC"]
# V compilation is trikcy for linking. We compile V to C first.
VFLAGS = ["-shared", "-prod", "-cc", "gcc", "-cflags", "-O3 -mavx2 -fopenmp"]

def ensure_dirs():
    os.makedirs(BIN_DIR, exist_ok=True)
    os.makedirs(OBJ_DIR, exist_ok=True)

def compile_cpp(src_file):
    obj_file = os.path.join(OBJ_DIR, os.path.basename(src_file).replace(".cpp", ".o"))
    print(f"[CPP] {os.path.basename(src_file)}")
    cmd = [CC] + CFLAGS + [src_file, "-o", obj_file]
    subprocess.check_call(cmd)
    return obj_file

def compile_fortran(src_file):
    obj_file = os.path.join(OBJ_DIR, os.path.basename(src_file).replace(".f90", ".o"))
    print(f"[F90] {os.path.basename(src_file)}")
    cmd = [FC] + FFLAGS + [src_file, "-o", obj_file]
    subprocess.check_call(cmd)
    return obj_file

def compile_v(src_file):
    # For V, we produce a C file then compile it, or let V output an object.
    # V's -o flag with .o acts weird sometimes depending on version.
    # We will compile V to C, then compile C to O.
    c_file = os.path.join(OBJ_DIR, os.path.basename(src_file).replace(".v", ".c"))
    obj_file = os.path.join(OBJ_DIR, os.path.basename(src_file).replace(".v", ".o"))
    print(f"[V]   {os.path.basename(src_file)}")
    
    # 1. V -> C
    subprocess.check_call([VC, "-o", c_file, src_file])
    
    # 2. C -> O
    # V generated C often needs specific linkage, but for simple kernels typical settings work.
    subprocess.check_call([CC, "-c", "-O3", "-fPIC", "-w", c_file, "-o", obj_file])
    return obj_file

def link(objects):
    print(f"[LINK] Creating {os.path.basename(DLL_PATH)}")
    
    # Base Link Command
    cmd = [CC, SHARED_FLAG, "-fopenmp", "-o", DLL_PATH] + objects
    
    # OS Specific Link Flags
    if SYSTEM == "Windows":
        # Windows often needs static linking for simple distribution
        # But we try dynamic first.
        pass
    else:
        # Linux/Mac often need -lm
        cmd.append("-lm")

    # Attempt to link.
    try:
        subprocess.check_call(cmd)
        print("Build Successful.")
    except subprocess.CalledProcessError:
        print("Link failed. Trying with static-libgcc...")
        subprocess.check_call(cmd + ["-static-libgcc", "-static-libstdc++"])

def main():
    ensure_dirs()
    objects = []
    
    # 1. C++
    for f in glob.glob(os.path.join(SRC_DIR, "*.cpp")):
        objects.append(compile_cpp(f))
        
    # 2. Fortran
    for f in glob.glob(os.path.join(SRC_DIR, "*.f90")):
        objects.append(compile_fortran(f))
        
    # 3. V
    for f in glob.glob(os.path.join(SRC_DIR, "*.v")):
        try:
            objects.append(compile_v(f))
        except Exception as e:
            print(f"Skipping V file {f} due to error: {e}")

    # 4. Link
    if objects:
        link(objects)
    else:
        print("ERROR: No source files found! Build failed.")
        print(f"Checked SRC_DIR: {SRC_DIR}")
        sys.exit(1)

if __name__ == "__main__":
    main()
