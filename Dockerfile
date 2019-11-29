FROM ubuntu:latest
WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y python3 python3-pip python3-tk libopenmpi-dev wget libsm6 libxext6 libxrender-dev

RUN cd /root \
  && mkdir drl_fluid_film_python3/
COPY ./drl_fluid_film_python3 drl_fluid_film_python3/

# Let's install the dependencies of the projects, they are in `setup.py`
RUN cd drl_fluid_film_python3/gym-film \
  && python3 -m pip install -U setuptools \
  && python3 -m pip install -e ./

# We will now get the boost library, which we will install in /usr/local/
RUN cd /usr/local \
  && wget -4 https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.bz2 \
  && tar --bzip2 -xf boost_1_67_0.tar.bz2

# Build it
RUN cd /usr/local/boost_1_67_0 \
  && ./bootstrap.sh --with-libraries=python --with-python=python3.6 \
  && ./b2

# You need to let the linker know the path to the library `libboost_numpy36.so.1.67.0`
RUN echo "export LD_LIBRARY_PATH=/usr/local/boost_1_67_0/bin.v2/libs/python/build/gcc-7.4.0/release/threading-multi:$LD_LIBRARY_PATH" >> /root/.bashrc

# Now we need to make sure the python/cpp bridge in our simulation solver is well built
RUN cd /root/drl_fluid_film_python3/gym-film/gym_film/envs/simulation_solver \
  && make clean \
  && make