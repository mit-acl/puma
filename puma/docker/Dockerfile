FROM ros:noetic

LABEL Maintainer="Kota Kondo <kkondo@mit.edu>" \
      Description="Ubuntu 20.04, ROS noetic for PUMA"

ENV ROS_DISTRO=noetic
ENV DEBIAN_FRONTEND=noninteractive

# Usuals
##########################################
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
      apt-get upgrade -y && \
      apt-get install -q -y --no-install-recommends git tmux vim apt-utils && \
      apt-get install -q -y --no-install-recommends wget && \
      apt-get autoremove -y && \
      apt-get clean

# Making directories and setting up the environment
##########################################
WORKDIR /home
RUN mkdir installations
RUN mkdir code

# Install IPOPT (optionally with HSL solvers)
# See: https://github.com/ami-iit/ami-commons/blob/master/doc/casadi-ipopt-hsl.md
##########################################
RUN apt-get update
RUN apt-get install build-essential gfortran liblapack-dev libmetis-dev libopenblas-dev -y

# Make installations dir and clone coinbrew
##########################################
RUN mkdir -p  /home/installations/coin_ipopt
WORKDIR /home/installations/coin_ipopt
RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
RUN chmod u+x coinbrew

# Fetch and build Ipopt
##########################################
RUN ./coinbrew fetch Ipopt --no-prompt
RUN mkdir install
RUN ./coinbrew build Ipopt --prefix=install --test --no-prompt --verbosity=3
RUN apt-get install autoconf automake libtool -y

# Obtain an archive with HSL source code
##########################################
ADD coinhsl-2015.06.23 /home/installations/coinhsl-2015.06.23
WORKDIR /home/installations/coinhsl-2015.06.23
RUN wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz
RUN tar -xvf metis-4.0.3.tar.gz
RUN autoreconf -fi
RUN ./configure LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O3 -fopenmp" FCFLAGS="-g -O3 -fopenmp" CFLAGS="-g -O3 -fopenmp"
RUN sudo make install
WORKDIR /usr/local/lib
RUN ln -s libcoinhsl.so libhsl.so
RUN echo "export LD_LIBRARY_PATH='\${LD_LIBRARY_PATH}:/usr/local/lib'" >> ~/.bashrc

# Add all the paths to .bashrc and source it
##########################################
RUN echo '# Ipopt' >> ~/.bashrc
RUN echo 'export IPOPT_DIR=/home/installations/coin_ipopt/install' >> ~/.bashrc
RUN echo 'export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${IPOPT_DIR}/lib/pkgconfig' >> ~/.bashrc 
RUN echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${IPOPT_DIR}/lib' >> ~/.bashrc 
RUN echo 'export PATH=${PATH}:${IPOPT_DIR}/lib' >> ~/.bashrc 
RUN echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc

# Install Swig
# Remove any non-compatible swig versions and replace it with the matlab-customdoc branch
# If you don't do this, the compilation of casadi may fail with the error "swig error : Unrecognized option -matlab"
##########################################
RUN sudo apt-get remove swig swig3.0 swig4.0 
RUN mkdir -p /home/installations
WORKDIR /home/installations
RUN git clone https://github.com/jaeandersson/swig
WORKDIR /home/installations/swig
RUN git checkout -b matlab-customdoc origin/matlab-customdoc  

# Configure, build, and install swig
##########################################
RUN sudo apt-get install autotools-dev
RUN sudo apt-get install automake -y
RUN sh autogen.sh
RUN sudo apt-get install gcc-7 g++-7 bison byacc -y
RUN ./configure CXX=g++-7 CC=gcc-7            
RUN make
RUN sudo make install

# Install Casadi
##########################################
RUN mkdir -p  /home/installations/casadi
WORKDIR /home/installations/casadi
RUN git clone https://github.com/casadi/casadi
WORKDIR /home/installations/casadi/casadi

# Checkout an older branch
##########################################
RUN git checkout 3.5.5

# Remove any old build directories and clean, then make the new build directory
##########################################
RUN rm -rf build
RUN mkdir build 
WORKDIR /home/installations/casadi/casadi/build

# Build and install Casadi
##########################################
RUN . ~/.bashrc && cmake . -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT:BOOL=ON -DWITH_MATLAB=OFF -DWITH_PYTHON=ON -DWITH_DEEPBIND=ON -DWITH_HSL:BOOL=ON ..
RUN make -j20
RUN sudo make install

# Create a venvs 
##########################################
RUN sudo apt-get install python3-venv -y
RUN mkdir -p /home/installations/venvs
WORKDIR /home/installations/venvs
RUN python3 -m venv ./puma_venv
RUN printf '\n# PUMA venv' >> ~/.bashrc
RUN printf '\nalias activate_puma_venv="source /home/installations/venvs/puma_venv/bin/activate"' >> ~/.bashrc
RUN . /home/installations/venvs/puma_venv/bin/activate

# Install PUMA
##########################################
RUN sudo apt-get install git-lfs ccache -y
RUN mkdir -p /home/code/ws/src
WORKDIR /home/code/ws/src
# ARG DUMMY=unknown
# RUN DUMMY=${DUMMY} git clone https://github.com/mit-acl/puma.git
RUN git clone https://github.com/mit-acl/puma.git
ENV SCRIPT_DIR /home/code/ws/src/puma
WORKDIR $SCRIPT_DIR
RUN git lfs install
RUN git submodule init
RUN git submodule update
WORKDIR panther_compression/imitation
RUN sudo apt-get install python3-pip -y
RUN pip install numpy Cython wheel seals rospkg defusedxml empy pyquaternion pytest
RUN pip install -e .
RUN sudo apt-get install python3-catkin-tools \
    ros-${ROS_DISTRO}-rviz-visual-tools \
    ros-${ROS_DISTRO}-pybind11-catkin \
    ros-${ROS_DISTRO}-tf2-sensor-msgs \
    ros-${ROS_DISTRO}-rqt-gui \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-gazebo-ros -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install ros-${ROS_DISTRO}-jsk-rviz-plugins ros-${ROS_DISTRO}-pcl-ros -y

# INSTALL CGAL v4.14.2
##########################################
RUN apt-get update && apt-get install python-is-python3 libgmp3-dev libmpfr-dev -y
RUN mkdir -p /home/kota/installations/cgal
WORKDIR /home/kota/installations/cgal
RUN wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.14.2/CGAL-4.14.2.tar.xz
RUN tar -xf CGAL-4.14.2.tar.xz
WORKDIR CGAL-4.14.2/
RUN cmake . -DCMAKE_BUILD_TYPE=Release
RUN make install

# Go back into the workspace and build puma
##########################################
WORKDIR $SCRIPT_DIR/../..
RUN . ~/.bashrc
RUN . /opt/ros/noetic/setup.sh && catkin build

# Insert commands to add PUMA developments files to path and source it
##########################################
RUN printf '\n# PUMA' >> ~/.bashrc
RUN printf '\nsource /home/code/ws/devel/setup.bash' >> ~/.bashrc #Remember to change PATH_TO_YOUR_WS
RUN printf '\nexport PYTHONPATH="${PYTHONPATH}:$(rospack find puma)/../panther_compression"' >> ~/.bashrc 
RUN . ~/.bashrc
RUN apt-get install -y tmuxp ros-${ROS_DISTRO}-xacro psmisc

### for frame alignment
##########################################
RUN pip install --upgrade pip
RUN pip install python-box numpy-quaternion termcolor -U scikit-image defusedxml PySide2 wheel
RUN pip install empy

## Download and install motlee and clipper in the submodules directory
WORKDIR /home/code/ws/src/puma/submodules

# Clone the FastSAM repository and install it
##########################################
WORKDIR /home/code/ws/src/puma/submodules/FastSAM
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install -e .

# Install motlee
##########################################
WORKDIR /home/code/ws/src/puma/submodules/motlee
RUN pip install --ignore-installed PyYAML
RUN pip install -e .

# Install clipper
##########################################
WORKDIR /home/code/ws/src/puma/submodules/clipper/build
RUN cmake ..
RUN make
RUN python3 -m pip install bindings/python
WORKDIR /home/code/ws/src/puma

# Create a symlink of FastSAM to the scripts directory
# Note that "ln -s FastSAM ../../puma/puma/scripts/FastSAM/" creates a broken link (https://mokacoding.com/blog/symliks-in-git/)
##########################################
WORKDIR /home/code/ws/src/puma/puma/scripts
RUN ln -s ../../../puma/submodules/FastSAM/ .

# Need to run this to make Gazebo publish images
##########################################
RUN apt-get install ros-${ROS_DISTRO}-gazebo-* -y

# Set up Entry Point
##########################################
COPY ./puma.sh /
RUN chmod +x /puma.sh
ENTRYPOINT ["/puma.sh"]