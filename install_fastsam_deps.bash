#!/bin/bash

##
## Download and install the dependencies of FastSAM (Assume a virtual python environment is already created by install_puma_deps.bash)
##

# Upgrade pip
pip install --upgrade pip

# Install empy, box, quaternion, termcolor, scikit-image, defusedxml, PySide2, and wheel
pip install empy
pip install python-box
pip install numpy-quaternion
pip install termcolor
pip install -U scikit-image
pip install defusedxml
pip install PySide2
pip install wheel

##
## Download and install motlee and clipper in the submodules directory
##

# Go to the submodules directory
cd submodules

# Clone the FastSAM repository and install it
cd FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install -e .
cd ..

# Install motlee
cd motlee
pip install -e .
cd ..

# Install clipper
cd clipper
cd build
cmake ..
make
python3 -m pip install bindings/python
cd ../../..

# Create a symlink of FastSAM to the scripts directory
# Note that "ln -s FastSAM ../../primer/primer/scripts/FastSAM/" creates a broken link (https://mokacoding.com/blog/symliks-in-git/)
cd primer/scripts
if [ -d "FastSAM" ]; then
    echo "FastSAM does exist so remove it and recreate a symlink"
    rm -rf FastSAM
fi
ln -s ../../../primer/submodules/FastSAM/ .
