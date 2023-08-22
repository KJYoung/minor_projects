FROM atsushisaito/docker-ubuntu-sweb:jammy

# focal(20.04)에서는 아래 링크에서 정상 설치 안됨. jammy(22.04)에서는 OK.
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

RUN locale && \
    sudo apt update && \
    sudo apt install locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN locale && \
    sudo apt install software-properties-common && \
    add-apt-repository universe && \
    sudo apt update && sudo apt install curl -y && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && sudo apt upgrade

RUN sudo apt install -y ros-humble-desktop-full ros-dev-tools
RUN echo 'source /opt/ros/humble/setup.bash' > ~/.bashrc

# 근데 그냥 다음과 같이 하면 되더라..
# docker run -p 6080:80 --security-opt seccomp=unconfined --shm-size=512m -v /Users/junyoungkim/Downloads/:/hostOSdir tiryoh/ros2-desktop-vnc:humble