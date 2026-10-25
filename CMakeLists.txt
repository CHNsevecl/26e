cmake_minimum_required(VERSION 3.10)
project(ps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Debug keeps symbols while optimizing the time-budgeted vision search.
option(PUZZLE_OPTIMIZE_DEBUG "Optimize vision search in Debug builds" ON)
if(PUZZLE_OPTIMIZE_DEBUG AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options("$<$<CONFIG:Debug>:-O2>")
endif()

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
    add_executable(machine_result_test tests/machine_result_test.cpp src/machine_result.cpp)
    target_link_libraries(machine_result_test ${OpenCV_LIBS})
    add_test(NAME machine_result_test COMMAND machine_result_test)
    add_executable(a4_detection_test tests/a4_detection_test.cpp src/a4_detection.cpp)
    target_link_libraries(a4_detection_test ${OpenCV_LIBS})
    add_test(NAME a4_detection_test COMMAND a4_detection_test)
    add_executable(card_puzzle_test tests/card_puzzle_test.cpp src/card_puzzle.cpp src/puzzle_solver.cpp src/machine_result.cpp)
    target_link_libraries(card_puzzle_test ${OpenCV_LIBS})
    target_compile_definitions(card_puzzle_test PRIVATE CARD_PHOTO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/card_three_uneven_light.jpeg")
    target_compile_definitions(card_puzzle_test PRIVATE CARD_FOUR_PHOTO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/card_four_heart_j.jpeg")
    target_compile_definitions(card_puzzle_test PRIVATE CARD_ROTATED_PHOTO_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/card_four_heart_j_rotated.jpeg")
    add_test(NAME card_puzzle_test COMMAND card_puzzle_test)
endif()
