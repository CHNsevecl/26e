cmake_minimum_required(VERSION 3.10)
project(ps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 找 OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 添加可执行文件
add_executable(ps ${SRC_FILES})

# 链接 OpenCV
target_link_libraries(ps ${OpenCV_LIBS})

# 显示 OpenCV 信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

include(CTest)
if(BUILD_TESTING)
    add_executable(puzzle_solver_test tests/puzzle_solver_test.cpp src/puzzle_solver.cpp)
    target_link_libraries(puzzle_solver_test ${OpenCV_LIBS})
    add_test(NAME puzzle_solver_test COMMAND puzzle_solver_test)
endif()
