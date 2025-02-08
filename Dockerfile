FROM registry.gitlab.inria.fr/starpu/starpu-docker/starpu:1.4.7 AS starpu-base

LABEL maintainer "Aleksandr Mikhalev <al.mikhalev@skoltech.ru>"

LABEL org.opencontainers.image.source="https://github.com/muxas/starpupy_tutorial"

RUN sudo apt install -y python3-dev python3-pip

RUN sudo pip3 install joblib numpy cloudpickle

WORKDIR /home/starpu/starpu.git/build

RUN ../configure --enable-maxcpus=20 --prefix=/usr/local --disable-mpi --enable-starpupy

RUN make -j 10

RUN sudo mkdir -p /usr/local/lib/python3.8/site-packages

RUN sudo make install

WORKDIR /home/starpu/starpu.git/starpupy/examples

RUN echo "source /usr/local/bin/starpu_env" >> ~/.bashrc

SHELL ["bash"]
