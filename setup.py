from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import shutil
import glob
import sys
import subprocess


# ==============================================================================
# 1. Environment & Compiler Detection Logic
# ==============================================================================
def is_legacy_gcc():
    """
    Detect if the system GCC version is too old (< 5.0).
    Legacy GCC (e.g., 4.8.5 on CentOS 7) does not support the full C11 standard
    required to build modern Pandas (2.x) and Numpy (2.x) from source.
    """
    # This issue is specific to Linux; Windows/MacOS usually have modern toolchains.
    if not sys.platform.startswith('linux'):
        return False

    # Check for manual override via environment variable (e.g., LISA_LEGACY=1)
    if os.environ.get("LISA_LEGACY") == "1":
        print("[Setup] Legacy mode forced via environment variable.")
        return True

    try:
        # Try to invoke the compiler to get the version.
        # 'cc' is the system default; 'gcc' is specific.
        commands_to_try = [['cc', '-dumpversion'], ['gcc', '-dumpversion']]

        version_str = None
        for cmd in commands_to_try:
            try:
                # capture_output requires Python 3.7+, using check_output for 3.x compatibility
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                version_str = output.decode('utf-8').strip()
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if version_str:
            # Parse major version (e.g., "4.8.5" -> 4, "11.2.0" -> 11)
            major_ver = int(version_str.split('.')[0])
            print(f"[Setup] Detected Compiler Version: {version_str}")

            if major_ver < 5:
                print(f"[Setup] Warning: Compiler is old (< 5). Enforcing legacy dependencies to avoid build errors.")
                return True

    except Exception as e:
        print(f"[Setup] Warning: Could not detect GCC version ({e}). Assuming modern environment.")

    return False


def get_requirements():
    """
    Dynamically generate the list of requirements based on the compiler version.
    """
    # Base requirements needed for all environments
    reqs = [
        "matplotlib>=3.5.0,<4.0.0",
        "numba>=0.56.0",
    ]

    if is_legacy_gcc():
        # --- Legacy Environment (e.g., Cluster with GCC 4.8) ---
        # Cap versions to avoid C11 compilation errors
        reqs.extend([
            "numpy>=1.21.0,<2.0.0",
            "scipy>=1.7.0,<1.13.0",  # Scipy 1.13+ also requires newer compilers
            "pandas>=1.4.0,<2.0.0",
        ])
    else:
        # --- Modern Environment (e.g., Colab, Local PC) ---
        # Allow newer versions so pip can find binary wheels (much faster install).
        # We still set the minimum version to ensure API compatibility.
        reqs.extend([
            "numpy>=1.21.0,<2.0.0",  # <--- 加上 <2.0.0
            "scipy>=1.7.0",  # Scipy 如果不报错可以不封顶，但保险起见 <1.14 比较稳
            "pandas>=1.4.0",
        ])

    return reqs


# ==============================================================================
# 2. Cache Cleaning Logic
# ==============================================================================
def clean_numba_cache():
    """
    Recursively delete all __pycache__ folders and Numba cache files (*.nbc, *.nbi).
    This prevents path mismatch errors when the package is moved or reinstalled.
    """
    print("\n[Setup] Cleaning up old Numba caches and __pycache__...")

    # Get the directory where setup.py is located
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Remove __pycache__ directories
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_path)
                print(f"  - Removed directory: {cache_path}")
            except Exception as e:
                print(f"  ! Failed to remove {cache_path}: {e}")

    # 2. Remove Numba compiled cache files (*.nbc, *.nbi)
    # These often cause issues across different environments or paths
    extensions = ['*.nbc', '*.nbi']
    for ext in extensions:
        for file_path in glob.glob(os.path.join(root_dir, 'LISAeccentric', '**', ext), recursive=True):
            try:
                os.remove(file_path)
                print(f"  - Removed cache file: {file_path}")
            except Exception as e:
                print(f"  ! Failed to remove {file_path}: {e}")

    print("[Setup] Cleanup complete.\n")


# ==============================================================================
# 3. Custom Command Classes
# ==============================================================================
class CustomInstall(install):
    """Override standard install to clean cache first."""

    def run(self):
        clean_numba_cache()
        install.run(self)


class CustomDevelop(develop):
    """Override editable install (pip install -e .) to clean cache first."""

    def run(self):
        clean_numba_cache()
        develop.run(self)


# ==============================================================================
# 4. Setup Configuration
# ==============================================================================

# Read README.md for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A toolkit for eccentric Binary Black Hole population and waveform analysis for LISA."

setup(
    name="LISAeccentric",
    version="0.1.0",
    description="Toolbox for Eccentric BBH Populations and LISA Waveforms",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Zeyuan",
    author_email="your_email@example.com",  # Replace with your email

    # Automatically find all packages (directories with __init__.py)
    packages=find_packages(),

    # Crucial: Must be False because we use __file__ to load data paths in the code
    zip_safe=False,

    # Register custom commands for cache cleaning
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },

    # Dependencies: Dynamically determined based on environment compiler capabilities
    install_requires=get_requirements(),

    # Python version requirement
    python_requires=">=3.9",

    # Data Inclusion Strategy
    include_package_data=True,
    package_data={
        # Explicitly include data files within sub-packages
        'LISAeccentric': [
            'GN_modeling/data/*.npy',
            'GC_modeling/data/*.csv',
            'Field_modeling/data/*.npy',
            'Waveform_modeling/*.npz',  # Includes the acceleration table
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)