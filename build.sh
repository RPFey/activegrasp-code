#!/bin/bash
# get workspace folder
ws=$(pwd)
task=$1

# TORCH_CUDA_ARCH is empty, set default to 8.0
if [ -z "$TORCH_CUDA_ARCH" ]; then
    export TORCH_CUDA_ARCH="8.0;8.6;8.9"
fi
echo "TORCH_CUDA_ARCH: ${TORCH_CUDA_ARCH}"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH

# remove the dot in TORCH_CUDA_ARCH
CMAKE_CUDA_ARCH=$(echo $TORCH_CUDA_ARCH | tr -d '.')
echo "CMAKE_CUDA_ARCH: ${CMAKE_CUDA_ARCH}"
export CUDA_HOME=${CONDA_PREFIX}

# install dependencies
# build-essential checkinstall zlib1g-dev libssl-dev gcc-9 g++-9 -> nvblox
# sudo apt update && sudo apt install -y build-essential checkinstall zlib1g-dev libssl-dev
# source /opt/ros/noetic/setup.bash

# check the python version is >= 3.11; SAM2 needs >=3.11
if [[ $(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 11))') == "False" ]]; then
    echo "[WARN] Please install python 3.11"
fi

# get the current python interpreter path
python3_path=$(which python3)

# get python major version, for example 3.11
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

# setup the python3.11 venv
function python_setup {
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3-pip python3.11 python3.11-dev python3.11-venv

    python3.11 -m venv $ws/../grasp
    # install pytorch
    source $ws/../grasp/bin/activate
    python3.11 -m pip install --upgrade pip
    python3.11 -m pip install rospkg==1.5.1 defusedxml==0.6.0
    python3.11 -m pip install numpy==1.26.3 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
}

# install ros kortex driver
function kortex_setup {
    source /opt/ros/noetic/setup.bash 

    # install moveit pakcages
    sudo apt install -y python3-pip ros-noetic-moveit-planners-ompl ros-noetic-moveit-setup-assistant \
            ros-noetic-moveit-msgs ros-noetic-moveit-simple-controller-manager \
            ros-noetic-moveit-ros-planning-interface ros-noetic-ros-controllers

    sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev \
	    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-base

    sudo apt-get install -y ros-noetic-rgbd-launch libglib2.0-dev ros-noetic-depth-image-proc ros-noetic-image-proc

    PYTHON_BUILD_BIN=/usr/bin/python3.8
    $PYTHON_BUILD_BIN -m pip install conan==1.59
    rm -rf ~/.conan
    conan config set general.revisions_enabled=1
    conan profile new default --detect > /dev/null
    conan profile update settings.compiler.libcxx=libstdc++11 default

    cd $ws/../
    mkdir -p kortex_ws/src
    cd kortex_ws/src
    git clone https://github.com/Kinovarobotics/ros_kortex.git
    git clone https://github.com/Kinovarobotics/ros_kortex_vision.git
    cd ../
    catkin_make -DPYTHON_EXECUTABLE=$PYTHON_BUILD_BIN
}

# define function: copy ros files
function copy_ros_files {
    # copy ros files
    sudo cp dump/roslogging.py /opt/ros/noetic/lib/python3/dist-packages/rosgraph/roslogging.py
    cp dump/CMakeLists.txt src/orocos_kinematics_dynamics/python_orocos_kdl/CMakeLists.txt
    python3 -m pip install pyquaternion powerline_shell open3d scikit-image pynput gdown rospkg defusedxml
    # python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # setup python empy
    python3 -m pip install empy==3.3.2 && \
    emfile=$(python3 -c "import em; print(em.__file__)") 
    emdir=$(dirname ${emfile}) # this is the site-package dir
    ln -s ${emfile} ${emdir}/em

    # update lsb_release, the default is legacy
    sudo cp dump/lsb_release /usr/bin/lsb_release
    sudo cp dump/lsb_release.py ${emdir}/lsb_release.py
}

function install_gsplat {
    # gsplat trainer 
    cd ${ws}
    CPATH=${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3:${CONDA_PREFIX}/include/suitesparse/ python3 -m pip install -r src/gaussian_splatting/requirements.txt

    # gsplat
    cd ${ws}/../

    # if gsplat exists, remove it
    if [ -d "gsplat" ]; then
        rm -rf gsplat
    fi

    git clone --recursive https://github.com/nerfstudio-project/gsplat.git && cd gsplat
    git checkout v1.4.0
    MAX_JOBS=4 python3 -m pip install -e .
}

function install_romatch {
    # RoMatch
    cd ${ws}/../

    # if RoMa exists, remove it
    if [ -d "RoMa" ]; then
        rm -rf RoMa
    fi

    git clone https://github.com/RPFey/RoMa.git
    cd RoMa && python3 -m pip install -e .
}

function realsense {
    sudo apt-get install -y --no-install-recommends --allow-unauthenticated \
                libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
    cd ${ws}/../

    # if librealsense exists, remove it
    if [ -d "librealsense" ]; then
        rm -rf librealsense
    fi

    git clone https://github.com/IntelRealSense/librealsense.git && cd librealsense
    ./scripts/setup_udev_rules.sh # setup udev rule
    git checkout v2.55.1 # we use this version
    mkdir build && cd build
    cmake -DPYBIND11_PYTHON_VERSION=${python_version} \
            -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
            -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
            -DCMAKE_INSTALL_PREFIX=/opt/librealsense -DBUILD_EXAMPLES=ON \
            -DBUILD_PYTHON_BINDINGS:bool=true  \
            -DPYTHON_EXECUTABLE=${python3_path}  \
            -DCMAKE_BUILD_TYPE=Release ../
    make -j8
}

function diff_eig {
    # Diff Eig
    cd ${ws}/../

    # if diff-eig exists, remove it
    if [ -d "diff-eig" ]; then
        rm -rf diff-eig
    fi

    GIT_SSL_NO_VERIFY=1 git clone --recursive -b grasp git@github.com:JiangWenPL/diff-eig.git
    cd diff-eig
    git submodule init && git submodule update
    python3 -m pip install -e .
}

function curobo {
    # Set up CuRobo
    PKGS_PATH=${ws}/../pkgs
    if [ -d ${PKGS_PATH} ]; then
        rm -r ${PKGS_PATH}
    fi

    cd ${ws}/../
    
    # if curobo exists, remove it
    if [ -d "curobo" ]; then
        rm -rf curobo
    fi
    git clone https://github.com/NVlabs/curobo.git 
    cd curobo && git checkout ebb71702f
    python3 -m pip install -e . --no-build-isolation
    # echo "export CUROBO_TORCH_CUDA_GRAPH_RESET=1" >> ~/.bashrc

    mkdir -p ${PKGS_PATH}

    if [ $(python3 -c 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)') == "True" ]; then
        sudo apt-get install libgoogle-glog-dev libgtest-dev libsqlite3-dev curl tcl libbenchmark-dev

        cd ${PKGS_PATH}
        git clone https://github.com/valtsblukis/nvblox.git && cd nvblox/nvblox && mkdir build && \
            cmake .. \
            -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
            && make -j32 && \
            sudo make install
    else
        echo "CXX_ABI is False"
        # CXX11_ABI = Fasle
        export TORCH_CXX11=0

        # I don't think this is necessary, any cmake >= 3.27 should be fine
        # cd ${PKGS_PATH}
        # cd ${PKGS_PATH} && wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz && \
        #     tar -xvzf cmake-3.27.1.tar.gz && \
        #     cd cmake-3.27.1 && ./bootstrap && \
        #     make -j8 &&
        #     export PATH=${PKGS_PATH}/cmake-3.27.1/bin:$PATH # export to System Path

        cd ${PKGS_PATH}
        git clone https://github.com/sqlite/sqlite.git -b version-3.39.4 && \
            cd ${PKGS_PATH}/sqlite && CFLAGS=-fPIC ./configure --prefix=${PKGS_PATH}/sqlite/install/ && \
            make -j8 && make install

        cd ${PKGS_PATH}
        git clone https://github.com/google/glog.git -b v0.6.0 && \
            cd glog  && \
            mkdir build && cd build && \
            cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DCMAKE_INSTALL_PREFIX=${PKGS_PATH}/glog/install/ \
            -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11} \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5\
            && make -j8 && make install

        export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${PKGS_PATH}/glog/install/lib/pkgconfig
        cd ${PKGS_PATH}
        git clone https://github.com/gflags/gflags.git -b v2.2.2 && \
            cd gflags &&  \
            mkdir build && cd build && \
            cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DCMAKE_INSTALL_PREFIX=${PKGS_PATH}/gflags/install/ \
            -DGFLAGS_BUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11} \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            && make -j8 && make install

        cd ${PKGS_PATH}
        # Maybe I can use CPATH instead. to include conda header files
        export CPATH=${CONDA_PREFIX}/include
        git clone https://github.com/RPFey/nvblox.git && cd nvblox && git checkout a02151f38
        cd ${PKGS_PATH}/nvblox/nvblox && \
            mkdir build && cd build && \
            cmake ..  -DBUILD_REDISTRIBUTABLE=ON \
            -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCH} \
            -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -I${CONDA_PREFIX}/include"\
            -DCMAKE_INSTALL_PREFIX=${PKGS_PATH}/nvblox/nvblox/install -DPRE_CXX11_ABI_LINKABLE=ON \
            -DSQLITE3_BASE_PATH="${PKGS_PATH}/sqlite/install/" -DGLOG_BASE_PATH="${PKGS_PATH}/glog/install/" \
            -DGFLAGS_BASE_PATH="${PKGS_PATH}/gflags/install/" -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 && \
            make -j32 && make install
    fi 

    cd ${PKGS_PATH}
    export CPATH=$CPATH:${PKGS_PATH}/nvblox/nvblox/install/include
    export LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib
    git clone https://github.com/RPFey/nvblox_torch.git
    cd nvblox_torch && git checkout 14029c6a6
    mkdir -p src/nvblox_torch/bin
    cd src/nvblox_torch/cpp && mkdir -p build && cd build \
        && cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');${CONDA_PREFIX};${PKGS_PATH}/nvblox/nvblox/install/share/nvblox/cmake" \
            -DGLOG_BASE_PATH="${PKGS_PATH}/glog/install/" .. \
        && make -j32 && cd ../../../../ && cp -r src/nvblox_torch/cpp/build/*.so src/nvblox_torch/bin/ && \
        rm -rf src/nvblox_torch/cpp/build
    python3 -m pip install -e .

    # # Set LD Library Path
    # echo "export LD_LIBRARY_PATH=${PKGS_PATH}/nvblox/nvblox/install/lib:${PKGS_PATH}/glog/install/lib:${PKGS_PATH}/gflags/install/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc
}

function contact_graspnet_setup {
    # Set up Contact Grasp Net
    cd ${ws}/../

    # if contact_graspnet_pytorch exists, remove it
    if [ -d "contact_graspnet_pytorch" ]; then
        rm -rf contact_graspnet_pytorch
    fi

    git clone --recursive https://github.com/RPFey/contact_graspnet_pytorch.git
    cd contact_graspnet_pytorch
    python3 -m pip install -e .
}

function se3diff_setup {
    # Setup Se3diff
    cd ${ws}/../

    # if grasp_diffusion exists, remove it
    if [ -d "grasp_diffusion" ]; then
        rm -rf grasp_diffusion
    fi

    git clone https://github.com/RPFey/calibrate_grasp_diffusion.git grasp_diffusion
    cd grasp_diffusion
    MAX_JOBS=4 python3 -m pip install -r requirements.txt
    python3 -m pip install -e .

    # get model weights for se3diff
    echo "Download Model Weights"
    mkdir data && cd data
    git clone https://huggingface.co/camusean/grasp_diffusion models

    cd models
    gdown --id 1cLJNzMmuLqqMtqRfoETnaeehcJhR2JG9
    unzip multiobject_scene_graspdif_dual.zip
}

function vgn_setup {
    # VGN weights
    cd /tmp
    gdown --id 1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ
    unzip data.zip
    mv data/models/vgn_conv.pth ${ws}/src/gaussian_splatting/weights
}

function ActiveNGF_setup {
    # Dependency for ActiveNGF
    # Graspnet API
    cd /tmp
    git clone https://github.com/graspnet/graspnetAPI.git && cd graspnetAPI
    # some dependency is outdated, need to modify setup.py
    sed -i 's/sklearn/scikit-learn/g' setup.py
    sed -i 's/numpy==1.23.4/numpy==1.26.4/g' setup.py
    sed -i 's/transforms3d==0.3.1/transforms3d==0.4.2/g' setup.py
    python3 -m pip install Cython
    python3 -m pip install .

    # MinKowski Engine
    cd /tmp
    # apt install build-essential python3-dev libopenblas-dev
    python3 -m pip install ninja

    git clone https://github.com/NVIDIA/MinkowskiEngine.git && cd MinkowskiEngine
    sed -i '30i #include <thrust/execution_policy.h> \n' src/convolution_kernel.cuh
    sed -i '33i #include <thrust/unique.h> \n#include <thrust/remove.h> \n' src/coordinate_map_gpu.cu
    sed -i '29i #include <thrust/execution_policy.h> \n#include <thrust/reduce.h> \n#include <thrust/sort.h>' src/spmm.cu
    sed -i '30i #include <thrust/execution_policy.h> \n' src/3rdparty/concurrent_unordered_map.cuh
    # setuptools for MinkowskiEngine
    # python3 -m pip install setuptools==59.8.0
    # Specify the blas option can avoid downgrading setuptools < 60.0
    CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=4 python3 setup.py install --blas=mkl

    # Install PointNet2 for ActiveNGF
    cd ${ws}/src/gaussian_splatting/gaussian_splatting_py/pointnet2
    python3 setup.py install
}

function graspnet {
    # grasp network setup
    contact_graspnet_setup
    se3diff_setup
    vgn_setup
    ActiveNGF_setup

    # get episode data [ !!Deprecated!! after we ensure reproducibility ]
    cd ${ws}/../
    git clone https://github.com/RPFey/grasp_episode.git

    # Model Weights
    cd ${ws}/src/gaussian_splatting/weights

    # Weights for Contact Grasp Net
    gdown --id 1RfdpEM2y0x98rV28d7B2Dg8LLFKnBkfL
    
    # weights for ACE NBV
    gdown --id 10gXW5bXQ1adFRyfM22r8UcEYPLaFfm-V
    tar -zxvf ACE.tar.gz

    # Weights for ActiveNGF
    mkdir ckpts && cd ckpts
    gdown --id 1OswUcXVJv_LAgyyNt_KjfOPhIE4LEyk7
}

function bullet {
    python3 -m pip install pybullet

    # simulation code 
    cd ${ws}/../
    if [ -d "franka_pybullet_ros" ]; then
        rm -rf franka_pybullet_ros
    fi

    git clone https://github.com/RPFey/franka_pybullet_ros.git && cd franka_pybullet_ros
    git checkout e437d4f && python -m pip install -e .
    
    # uncomment this if you have LD library issue
    # echo "export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7" >> ~/.bashrc
    
    # setup xvfb
    # echo "export DISPLAY=:1.0" >> ~/.bashrc
}

function foundationpose {
    # Set up Foundation Pose
    cd ${ws}/src/gaussian_splatting

    CPATH=${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3:${CONDA_PREFIX}/include/suitesparse/ python3 -m pip install -r requirements.txt
    python3 -m pip install --upgrade psutil
    python3 -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
    MAX_JOBS=4 python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  
    cd ${ws}/src/gaussian_splatting/gaussian_splatting_py/FoundationPose/mycpp
    rm -rf build && mkdir -p build && cd build && \
    CMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 cmake -DPYTHON_EXECUTABLE:FILEPATH=${python3_path} .. && \
    make -j4    
}

# Build catkin
function catkin_build {
    cd ${ws}
    sudo apt-get install -y python3-catkin-tools

    sudo rm /usr/bin/python3 && sudo ln -s ${python3_path} /usr/bin/python3
    ROS_PYTHON_VERSION=${python_version} catkin_make_isolated --cmake-args -DPYTHON_EXECUTABLE=${python3_path}
    # link 3.8 again
    sudo rm /usr/bin/python3 && sudo ln -s /usr/bin/python3.8 /usr/bin/python3
}

# build sam2
function sam2 {
    cd ${ws}/../

    # if SAM2 dir exists, remove it
    if [ -d "Grounded-SAM-2" ]; then
        rm -rf Grounded-SAM-2
    fi

    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    cd Grounded-SAM-2 && git checkout dd4c514

    # setuptools > 75.9 may cause error for SAM2 installation
    python3 -m pip install setuptools==75.2.0
    
    # install dependencies
    python3 -m pip install gdown # make sure the following gdown command works
    python3 -m pip install -v --no-build-isolation ".[notebook]"

    # use own weight
    cd checkpoints

    SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
    wget --no-check-certificate $sam2p1_hiera_l_url

    cp sam2.1_hiera_large.pt ${ws}/src/gaussian_splatting/weights
}

function server {
    # on a headless server
    curobo
    # foundationpose
    sam2
    install_gsplat
    install_romatch
    bullet
    graspnet
    diff_eig

    # local install two packages
    cd ${ws}
    python3 -m pip install -e src/gaussian_splatting
    python3 -m pip install -e src/kinova_control
}

# if task == "ros"
if [ $task == "ros" ]; then
    copy_ros_files
elif [ $task == "gsplat" ]; then
    install_gsplat
elif [ $task == "romatch" ]; then
    install_romatch
elif [ $task == "catkin" ]; then
    catkin_build
elif [ $task == "realsense" ]; then
    realsense
elif [ $task == "curobo" ]; then
    curobo
elif [ $task == "graspnet" ]; then
    graspnet
elif [ $task == "foundationpose" ]; then
    foundationpose
elif [ $task == "diff_eig" ]; then
    diff_eig    
elif [ $task == "sam2" ]; then
    sam2
elif [ $task == "bullet" ]; then
    bullet
elif [ $task == "server" ]; then
    server
elif [ $task == "kortex" ]; then
    kortex_setup
elif [ $task == "python" ]; then
    python_setup
elif [ $task == "all" ]; then
    copy_ros_files
    sam2
    install_gsplat
    diff_eig
    foundationpose
    curobo
    install_romatch
    realsense
    graspnet
    catkin_build
    bullet
else   
    echo "Please specify the task: ros, gsplat, romatch, catkin, realsense, curobo, graspnet, foundationpose, diff_eig, sam2, bullet, all"
fi

cd ${ws}
