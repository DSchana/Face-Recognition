# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix

# Include any dependencies generated for this target.
include CMakeFiles/FaceRecognition.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceRecognition.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceRecognition.dir/flags.make

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o: CMakeFiles/FaceRecognition.dir/flags.make
CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o: /Users/DSchana/Documents/GitHub/Face-Recognotion/Face\ Recognition/Face\ Recognition/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/DSchana/Documents/GitHub/Face-Recognotion/Unix/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o -c "/Users/DSchana/Documents/GitHub/Face-Recognotion/Face Recognition/Face Recognition/main.cpp"

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/DSchana/Documents/GitHub/Face-Recognotion/Face Recognition/Face Recognition/main.cpp" > CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.i

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/DSchana/Documents/GitHub/Face-Recognotion/Face Recognition/Face Recognition/main.cpp" -o CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.s

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.requires:

.PHONY : CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.requires

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.provides: CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/FaceRecognition.dir/build.make CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.provides.build
.PHONY : CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.provides

CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.provides.build: CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o


# Object files for target FaceRecognition
FaceRecognition_OBJECTS = \
"CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o"

# External object files for target FaceRecognition
FaceRecognition_EXTERNAL_OBJECTS =

FaceRecognition: CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o
FaceRecognition: CMakeFiles/FaceRecognition.dir/build.make
FaceRecognition: /usr/local/lib/libopencv_videostab.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_superres.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_stitching.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_shape.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_photo.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_objdetect.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_calib3d.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_features2d.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_ml.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_highgui.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_videoio.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_imgcodecs.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_flann.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_video.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_imgproc.3.1.0.dylib
FaceRecognition: /usr/local/lib/libopencv_core.3.1.0.dylib
FaceRecognition: CMakeFiles/FaceRecognition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/DSchana/Documents/GitHub/Face-Recognotion/Unix/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FaceRecognition"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceRecognition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceRecognition.dir/build: FaceRecognition

.PHONY : CMakeFiles/FaceRecognition.dir/build

CMakeFiles/FaceRecognition.dir/requires: CMakeFiles/FaceRecognition.dir/Users/DSchana/Documents/GitHub/Face-Recognotion/Face_Recognition/Face_Recognition/main.cpp.o.requires

.PHONY : CMakeFiles/FaceRecognition.dir/requires

CMakeFiles/FaceRecognition.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceRecognition.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceRecognition.dir/clean

CMakeFiles/FaceRecognition.dir/depend:
	cd /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix /Users/DSchana/Documents/GitHub/Face-Recognotion/Unix/CMakeFiles/FaceRecognition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceRecognition.dir/depend

