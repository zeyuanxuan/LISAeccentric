from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import shutil
import glob
import sys

# ==============================================================================
# 1. Cache Cleaning Logic
# ==============================================================================
def clean_numba_cache():
    """
    Recursively delete all __pycache__ folders and Numba cache files (*.nbc, *.nbi).
    This prevents path mismatch errors when the package is moved or reinstalled.
    Uses .format() instead of f-strings to ensure setup.py parses on older Python versions.
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
                print("  - Removed directory: {}".format(cache_path))
            except Exception as e:
                print("  ! Failed to remove {}: {}".format(cache_path, e))

    # 2. Remove Numba compiled cache files (*.nbc, *.nbi)
    # These often cause issues across different environments or paths
    extensions = ['*.nbc', '*.nbi']
    for ext in extensions:
        # Use glob to find files recursively
        # Note: recursive=True in glob requires Python 3.5+
        if sys.version_info >= (3, 5):
            files_to_remove = glob.glob(os.path.join(root_dir, 'LISAeccentric', '**', ext), recursive=True)
        else:
            # Fallback for very old python (just in case)
            files_to_remove = []
            for root, dirs, files in os.walk(os.path.join(root_dir, 'LISAeccentric')):
                for file in files:
                    if file.endswith(ext[1:]):
                         files_to_remove.append(os.path.join(root, file))

        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print("  - Removed cache file: {}".format(file_path))
            except Exception as e:
                print("  ! Failed to remove {}: {}".format(file_path, e))

    print("[Setup] Cleanup complete.\n")


# ==============================================================================
# 2. Custom Command Classes
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
# 3. Setup Configuration
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
    author_email="your_email@example.com",  # Remember to update this if needed

    # Automatically find all packages (directories with __init__.py)
    packages=find_packages(),

    # Crucial: Must be False because we use __file__ to load data paths in the code
    zip_safe=False,

    # Register custom commands for cache cleaning
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },

    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "numba>=0.55.0",
        # KEY CHANGE: Install backport if Python is older than 3.7
        'dataclasses>=0.6;python_version<"3.7"',
    ],

    # Python version requirement
    # KEY CHANGE: Lowered to 3.6 to support older clusters
    python_requires=">=3.6",

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)