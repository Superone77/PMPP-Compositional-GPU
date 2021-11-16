# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build

# Include any dependencies generated for this target.
include CMakeFiles/app.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/app.dir/flags.make

CMakeFiles/app.dir/main.cc.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/main.cc.o: ../main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/app.dir/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/app.dir/main.cc.o -c /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/main.cc

CMakeFiles/app.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/app.dir/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/main.cc > CMakeFiles/app.dir/main.cc.i

CMakeFiles/app.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/app.dir/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/main.cc -o CMakeFiles/app.dir/main.cc.s

CMakeFiles/app.dir/kernel/nop_kernel.cu.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/kernel/nop_kernel.cu.o: ../kernel/nop_kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/app.dir/kernel/nop_kernel.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/kernel/nop_kernel.cu -o CMakeFiles/app.dir/kernel/nop_kernel.cu.o

CMakeFiles/app.dir/kernel/nop_kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/app.dir/kernel/nop_kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/app.dir/kernel/nop_kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/app.dir/kernel/nop_kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target app
app_OBJECTS = \
"CMakeFiles/app.dir/main.cc.o" \
"CMakeFiles/app.dir/kernel/nop_kernel.cu.o"

# External object files for target app
app_EXTERNAL_OBJECTS =

app: CMakeFiles/app.dir/main.cc.o
app: CMakeFiles/app.dir/kernel/nop_kernel.cu.o
app: CMakeFiles/app.dir/build.make
app: /usr/local/cuda/lib64/libcudart_static.a
app: /usr/lib/x86_64-linux-gnu/librt.so
app: CMakeFiles/app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/app.dir/build: app

.PHONY : CMakeFiles/app.dir/build

CMakeFiles/app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/app.dir/clean

CMakeFiles/app.dir/depend:
	cd /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build /gris/gris-f/homelv/wayang/pmpp/pmpp-yang-he/gpu_task/build/CMakeFiles/app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/app.dir/depend

