# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.24.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.24.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build"

# Include any dependencies generated for this target.
include CMakeFiles/tridiag-test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tridiag-test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tridiag-test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tridiag-test.dir/flags.make

CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o: CMakeFiles/tridiag-test.dir/flags.make
CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o: /Users/martapiscitelli/Desktop/Marta/uni/V\ Semestre/FYS\ 3150/project2/test/tridiag_mat.cpp
CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o: CMakeFiles/tridiag-test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o -MF CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o.d -o CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o -c "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/test/tridiag_mat.cpp"

CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/test/tridiag_mat.cpp" > CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.i

CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/test/tridiag_mat.cpp" -o CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.s

# Object files for target tridiag-test
tridiag__test_OBJECTS = \
"CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o"

# External object files for target tridiag-test
tridiag__test_EXTERNAL_OBJECTS =

tridiag-test: CMakeFiles/tridiag-test.dir/test/tridiag_mat.cpp.o
tridiag-test: CMakeFiles/tridiag-test.dir/build.make
tridiag-test: libtrace.a
tridiag-test: /opt/homebrew/lib/libarmadillo.dylib
tridiag-test: CMakeFiles/tridiag-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tridiag-test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tridiag-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tridiag-test.dir/build: tridiag-test
.PHONY : CMakeFiles/tridiag-test.dir/build

CMakeFiles/tridiag-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tridiag-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tridiag-test.dir/clean

CMakeFiles/tridiag-test.dir/depend:
	cd "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2" "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2" "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build" "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build" "/Users/martapiscitelli/Desktop/Marta/uni/V Semestre/FYS 3150/project2/build/CMakeFiles/tridiag-test.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/tridiag-test.dir/depend

