cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
include(CTest)
include_directories(./include)
add_library(expspmm STATIC
include/data.h
include/dbg.h
include/valid.h
include/util.h
include/args.hxx
src/data.cu
src/util.cu
src/valid.cu
src/spmm_ref.cu
src/spmm_opt.cu
src/spmm_cusparse.cu
  )

set(CMAKE_CUDA_ARCHITECTURES 60)
find_package(CUDA REQUIRED)
string(REPLACE "/usr/lib/x86_64-linux-gnu/librt.a" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")


# FIND_PACKAGE(OpenMP)
# IF(OPENMP_FOUND)
#   SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# ENDIF()
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(expspmm PUBLIC cxx_std_11)

include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)


# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( expspmm 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -O3 --use_fast_math -lcurand  ")

# if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
#     message(STATUS "hah")
# endif()

add_subdirectory(third_party/googletest)


enable_testing()
include(GoogleTest)



add_subdirectory(test)



