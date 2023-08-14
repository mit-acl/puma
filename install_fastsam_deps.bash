# Create an installations direcotry
mkdir -p ~/installations && cd $_

# Create a venv direcotry
mkdir -p venvs && cd $_

# Create a virtual environment
python3 -m venv fastsam_venv

# Activate the virtual environment and write the activation command to bashrc
source fastsam_venv/bin/activate
printf '\nalias activate_fastsam_venv="source ~/installations/venvs/fastsam_venv/bin/activate"' >> ~/.bashrc

# Upgrade pip and install wheel
pip install --upgrade pip
pip install wheel

# Install empy, box, quaternion, termcolor, scikit-image, defusedxml, and PySide2
pip install empy
pip install python-box
pip install numpy-quaternion
pip install termcolor
pip install -U scikit-image
pip install defusedxml
pip install PySide2

# Move back into the installations directory
cd ..

# Clone the FastSAM repository and install it
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install -e .
cd ..

# Install motlee
git clone https://gitlab.com/mit-acl/dmot/motlee.git
cd motlee
pip install -e .
cd ..

# Install clipper
git clone https://github.com/mit-acl/clipper.git
cd clipper
mkdir build && cd $_
cmake ..
make
pip install bindings/python