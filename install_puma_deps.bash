# Check for proper ROS Setup
if output=$(rosversion -d); then
    if [[ "$output" != "noetic" ]]; then
        echo "ROS Noetic not installed! Please install the full version of ROS Noetic before preceding (http://wiki.ros.org/noetic/Installation/Ubuntu)."
    fi
else
    echo "No ROS installed! Please install the full version of ROS Noetic before preceding (http://wiki.ros.org/noetic/Installation/Ubuntu)."
fi

# Install IPOPT (optionally with HSL solvers)
# See: https://github.com/ami-iit/ami-commons/blob/master/doc/casadi-ipopt-hsl.md
sudo apt-get update
sudo apt-get install build-essential gfortran liblapack-dev libmetis-dev libopenblas-dev

# Make installations dir and clone coinbrew
mkdir -p  ~/installations/coin_ipopt && cd ~/installations/coin_ipopt
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew

# Fetch and build Ipopt
./coinbrew fetch Ipopt --no-prompt
mkdir -p install
./coinbrew build Ipopt --prefix=install --test --no-prompt --verbosity=3

# Check if an HSL zip file was provided
if [[ -h $1 ]]; then 
    echo "No HSL provided, installing IPOPT with MUMPS."
else
    echo "HSL Provided, installing!"

    # Make the HSL directory and unszip the archive there
    mkdir -p ~/installations/coin_ipopt/ThirdParty/HSL/
    tar -xzvf $1 -C ~/installations/coin_ipopt/ThirdParty/HSL/

    # Rebuild Ipopt
    ./coinbrew build Ipopt --prefix=install --test --no-prompt --verbosity=3
fi

# Install Ipopt
./coinbrew install Ipopt --no-prompt

# Link the HSL file it it exists
if [[ -h $1 ]]; then 
    ln -s libcoinhsl.so libhsl.so
fi

# Add all the paths to .bashrc and source it
printf '\n# Ipopt' >> ~/.bashrc
printf 'export IPOPT_DIR=~/robot-code/CoinIpopt/install' >> ~/.bashrc
printf 'export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${IPOPT_DIR}/lib/pkgconfig' >> ~/.bashrc 
printf 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${IPOPT_DIR}/lib' >> ~/.bashrc 
printf 'export PATH=${PATH}:${IPOPT_DIR}/lib' >> ~/.bashrc 

# Install Swig
# Remove any non-compatible swig versions and replace it with the matlab-customdoc branch
sudo apt-get remove swig swig3.0 swig4.0 #If you don't do this, the compilation of casadi may fail with the error "swig error : Unrecognized option -matlab"
mkdir ~/installations && cd ~/installations
git clone https://github.com/jaeandersson/swig
cd swig
git checkout -b matlab-customdoc origin/matlab-customdoc  

# Configure, build, and install swig
sh autogen.sh
sudo apt-get install gcc-7 g++-7 bison byacc
./configure CXX=g++-7 CC=gcc-7            
make
sudo make install

# Install Casadi
# Clone casadi
mkdir -p  ~/installations/casadi && cd ~/installations/casadi
git clone https://github.com/casadi/casadi
cd casadi

# Checkout an older branch
git checkout 3.5.5

# Remove any old build directories and clean, then make the new build directory
rm -rf build
make clean
mkdir build && cd build

# Build and install Casadi
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON ..
cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON ..
make -j20
sudo make install

# Setup the python enviroment
sudo apt-get install python3-venv

# Create a venvs directory
mkdir -p  ~/installations/venvs && cd ~/installations/venvs

# Create the venv, add an activation command, and activate it
python3 -m venv ./puma_venv
printf '\n# PUMA venv' >> ~/.bashrc
printf '\nalias activate_puma_venv="source ~/installations/venvs_python/puma_venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc
activate_puma_venv

# Install PUMA
# Setup git lfs
sudo apt-get install git-lfs ccache 

# Setup the workspace folders
mkdir -p ~/code && cd ~/code
mkdir -p ws && cd ws && mkdir -p src && cd src

# Clone the repo
git clone https://github.com/mit-acl/puma.git
cd puma

# Install git lfs
git lfs install
git submodule init && git submodule update

# Install the python dependancies and library
cd panther_compression/imitation
pip install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest
pip install -e .

# Install the ROS dependencies
sudo apt-get install python3-catkin-tools #To use catkin build
sudo apt-get install ros-"${ROS_DISTRO}"-rviz-visual-tools ros-"${ROS_DISTRO}"-pybind11-catkin ros-"${ROS_DISTRO}"-tf2-sensor-msgs ros-"${ROS_DISTRO}"-jsk-rviz-plugins

# Go back into the workspace and buidl puma
cd ~/code/ws
catkin build

# Insert commands to add PUMA developments files to path and source it
printf '\n# PUMA' >> ~/.bashrc
printf '\nsource ~/code/ws/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS
printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find primer)/../panther_compression"' >> ~/.bashrc 
source ~/.bashrc