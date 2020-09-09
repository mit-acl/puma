#!/bin/bash
# Author: Jesus Tordesillas Torres

source ~/.bashrc
sudo apt-get update

#INSTALL NLOPT v2.6.2
##########################################
cd ~/
wget https://github.com/stevengj/nlopt/archive/v2.6.2.tar.gz
tar -zxvf v2.6.2.tar.gz 
cd nlopt-2.6.2/
cmake . && make && sudo make install

#INSTALL CGAL v4.14.2
##########################################
sudo apt-get install libgmp3-dev libmpfr-dev -y
wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.14.2/CGAL-4.14.2.tar.xz
tar -xf CGAL-4.14.2.tar.xz
cd CGAL-4.14.2/
cmake . -DCMAKE_BUILD_TYPE=Release
sudo make install

#INSTALL python-catkin-tools (to be able to use catkin build)
##########################################
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get install python-catkin-tools -y

#CLONE SUBMODULES, INSTALL DEPENDENCIES AND COMPILE
##########################################
cd ~/ws/src/mader && git submodule init && git submodule update && cd ../../
rosdep install --from-paths src --ignore-src -r -y
catkin config -DCMAKE_BUILD_TYPE=Release
catkin build #GLPK will be installed when the `separator` package is compiled (see its CMakeList.txt)
echo "source ~/ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
