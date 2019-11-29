CC = g++
CFLAGS = -std=c++11 -Ofast -march=native -flto
LIBRARIES=\
-L/usr/local/boost_1_67_0/stage/lib -lpython3.6m -lboost_numpy36

INCLUDES=\
-I/usr/include/python3.6\
-I/usr/local/boost_1_67_0\
-I/usr/local/boost_1_67_0/stage/lib\
-I/usr/local/boost_1_67_0/bin.v2/libs/python/build/gcc-7.4.0/release/threading-multi

OBJ_FILES = simulation_cpp.o TVD3.o advect_in_time.o

cpp.so : $(OBJ_FILES)
	$(CC) -shared -o $@ $(OBJ_FILES) $(LIBRARIES) $(CFLAGS)

%.o: %.cpp
	$(CC) -fPIC -o $@ -c $< $(INCLUDES) $(CFLAGS)

clean :
	rm -f simulation_cpp.o advect_in_time.o TVD3.o cpp.so
