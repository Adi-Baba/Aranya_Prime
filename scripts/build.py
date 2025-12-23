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
# Compiler Commands
CC = os.environ.get("CC", "g++")
FC = os.environ.get("FC", "gfortran")
VC = os.environ.get("VC", "v")
# Flags
# Start with safe, generic optimization flags.
# We DO NOT hardcode -mavx2 or -mfma because they break ARM64 (Mac M1/M2) builds.
CFLAGS = ["-c", "-O3", "-fPIC"]
FFLAGS = ["-c", "-O3", "-fPIC"]

# V compilation is trikcy for linking. We compile V to C first.
# -gc none: Don't use Boehm GC (removes gc.h dependency). We manage memory manually (or leak, it's a kernel).
VFLAGS = ["-shared", "-prod", "-cc", "gcc", "-gc", "none", "-cflags", "-O3 -fPIC"]

def check_flag(compiler, flag):
    """Checks if a compiler supports a specific flag."""
    try:
        # Create a dummy file
        with open("test_flag.c", "w") as f:
            f.write("int main() { return 0; }")
        
        # Try to compile with the flag
        subprocess.check_call(
            [compiler, flag, "test_flag.c", "-o", "test_flag.o"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        os.remove("test_flag.c")
        if os.path.exists("test_flag.o"):
            os.remove("test_flag.o")
        return True
    except (OSError, subprocess.CalledProcessError):
        if os.path.exists("test_flag.c"): os.remove("test_flag.c")
        return False

# Detect OpenMP support dynamically
HAS_OPENMP = False
if check_flag(CC, "-fopenmp"):
    CFLAGS.append("-fopenmp")
    FFLAGS.append("-fopenmp")
    VFLAGS[-1] += " -fopenmp" # Update V flags string
    SHARED_FLAG_EXTRAS = ["-fopenmp"]
    HAS_OPENMP = True
    print("OpenMP support detected. Enabled.")
else:
    # Mac Clang might need -Xpreprocessor -fopenmp, but linking is hard without libomp.
    # We will fallback to NO OpenMP for robustness.
    SHARED_FLAG_EXTRAS = []
    print("OpenMP NOT detected. Building single-threaded.")

def ensure_dirs():
    os.makedirs(BIN_DIR, exist_ok=True)
    os.makedirs(OBJ_DIR, exist_ok=True)

def check_compiler(cmd):
    try:
        subprocess.check_call([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

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
    # We pass -gc none to avoid generating code that depends on gc.h
    # We pass -prod for optimized C generation.
    subprocess.check_call([VC, "-gc", "none", "-prod", "-o", c_file, src_file])
    
    # 2. C -> O
    # V generated C often needs specific linkage, but for simple kernels typical settings work.
    # CRITICAL: Use gcc (C compiler) not CC (g++) because V generates C code that violates C++ strictness.
    # We assume 'gcc' is in PATH if 'g++' is.
    subprocess.check_call(["gcc", "-c", "-O3", "-fPIC", "-w", c_file, "-o", obj_file])
    return obj_file

def link(objects):
    print(f"[LINK] Creating {os.path.basename(DLL_PATH)}")
    
    # Base Link Command
    cmd = [CC, SHARED_FLAG] + SHARED_FLAG_EXTRAS + ["-o", DLL_PATH] + objects
    
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
    
    print(f"Build Script Running...")
    print(f"Platform: {SYSTEM}")
    print(f"Source Directory: {SRC_DIR}")

    # Check Compilers
    # We use flush=True to ensure these prints appear in PIP logs immediately.
    
    if not check_compiler(CC):
        sys.stderr.write(f"ERROR: C++ compiler '{CC}' not found.\n")
        sys.stderr.write("Aranya Prime REQUIRES a C++ compiler (g++).\n")
        sys.stderr.flush()
        sys.exit(1)
        
    skip_fortran = False
    if not check_compiler(FC):
        sys.stderr.write(f"WARNING: Fortran compiler '{FC}' not found.\n")
        sys.stderr.write("Skipping Fortran kernels. Some features (BLAS) will be slower or unavailable.\n")
        sys.stderr.flush()
        skip_fortran = True

    objects = []
    
    # 1. C++
    cpp_files = glob.glob(os.path.join(SRC_DIR, "*.cpp"))
    print(f"Found {len(cpp_files)} C++ files: {[os.path.basename(f) for f in cpp_files]}")
    for f in cpp_files:
        objects.append(compile_cpp(f))
        
    # 2. Fortran
    if not skip_fortran:
        f90_files = glob.glob(os.path.join(SRC_DIR, "*.f90"))
        print(f"Found {len(f90_files)} Fortran files: {[os.path.basename(f) for f in f90_files]}", flush=True)
        for f in f90_files:
            objects.append(compile_fortran(f))
            
        # LINKING FIX:
        # When mixing Clang (C++) and GFortran, we must manually link the Fortran runtime.
        # Clang doesn't know about it.
        if f90_files and SYSTEM != "Windows":
             SHARED_FLAG_EXTRAS.append("-lgfortran")
        
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
        sys.stderr.write("ERROR: No source files found! Build failed.\n")
        sys.stderr.write(f"Checked SRC_DIR: {SRC_DIR}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
