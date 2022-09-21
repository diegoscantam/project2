cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENTIONS OFF)

# Define a "project", providing a description and a programming language
project(project2
	VERSION 1.0
	DESCRIPTION "Project 2 of Computational Physics course"
	LANGUAGES CXX
)
enable_testing()

# --- LIBRARY ---
add_library(trace src/useful.cpp)

#find_package(OpenMP REQUIRED)
#target_link_libraries(trace PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
target_link_libraries(trace PUBLIC ${ARMADILLO_LIBRARIES})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# This is needed if we keep .h files in the "include" directory
target_include_directories (trace PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
	${ARMADILLO_LIBRARIES}
	)

# --- EXECUTABLES ---
add_executable(chrono
	timing.cpp
)
target_link_libraries(chrono PUBLIC trace)
target_compile_features(chrono PUBLIC cxx_std_11)

add_executable(p2
	problem2.cpp
)
target_link_libraries(p2 PUBLIC trace)
target_compile_features(p2 PUBLIC cxx_std_11)

add_executable(p3
	problem3.cpp
)
target_link_libraries(p3 PUBLIC trace)
target_compile_features(p3 PUBLIC cxx_std_11)

add_executable(p5
	problem5.cpp
)
target_link_libraries(p5 PUBLIC trace)
target_compile_features(p5 PUBLIC cxx_std_11)


# --- TESTS ---
add_executable(tridiag-test
	test/tridiag_mat.cpp
	)
target_link_libraries(tridiag-test PUBLIC trace)
add_test(NAME tridiag-test
	COMMAND tridiag-test
	)

add_executable(jacobi-test
	test/jacobi.cpp
	)
target_link_libraries(jacobi-test PUBLIC trace)
add_test(NAME jacobi-test
	COMMAND jacobi-test
	)
