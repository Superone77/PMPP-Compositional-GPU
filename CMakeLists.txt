cmake_minimum_required(VERSION 3.0)
project(app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package (Threads)

#add_compile_definitions(_ENABLE_ATOMIC_ALIGNMENT_FIX)

file(GLOB_RECURSE TOP_INCS "source/*.hpp" "source/*.h")
file(GLOB_RECURSE ALG_INCS "source/algorithms/*.hpp" "source/algorithms/*.h" "source/algorithms/*.cpp" "source/algorithms/*.c")
file(GLOB_RECURSE INT_INCS "source/interfaces/*.hpp" "source/interfaces/*.h" "source/interfaces/*.cpp" "source/interfaces/*.c")
file(GLOB_RECURSE PAT_INCS "source/pattern/*.hpp" "source/pattern/*.h" "source/pattern/*.cpp" "source/pattern/*.c")
file(GLOB_RECURSE HEL_INCS "source/helper/*.hpp" "source/helper/*.h" "source/helper/*.cpp" "source/helper/*.c")
file(GLOB_RECURSE TST_INCS "source/tests/*.hpp" "source/tests/*.h" "source/tests/*.cpp" "source/tests/*.c")

source_group("algorithms" FILES ${ALG_INCS})
source_group("interfaces" FILES ${INT_INCS})
source_group("pattern" FILES ${PAT_INCS})
source_group("helper" FILES ${HEL_INCS})
source_group("tests" FILES ${TST_INCS})

add_executable(app
	source/main.cpp
	${TOP_INCS}
	${ALG_INCS}
	${INT_INCS}
	${PAT_INCS}
	${HEL_INCS}
	${TST_INCS}
)

target_link_libraries (app ${CMAKE_THREAD_LIBS_INIT})

if (UNIX)
	target_link_libraries (app stdc++fs)
endif ()
