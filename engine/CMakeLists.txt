cmake_minimum_required(VERSION 3.20)
project(oryx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ORYX_BUILD_TESTS "Build unit tests" ON)
option(ORYX_BUILD_BENCH "Build benchmarks" ON)
option(ORYX_BUILD_PYTHON "Build the Python module (normally via pip install ./engine)" OFF)
set(ORYX_SANITIZE "" CACHE STRING "Sanitizer to enable: address, thread, or undefined")

if(ORYX_SANITIZE)
  add_compile_options(-fsanitize=${ORYX_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${ORYX_SANITIZE})
endif()

# WebAssembly: fixed-width SIMD is supported by every browser the demo targets, so the whole
# build uses it (there is no runtime feature detection in WASM). Threads map to Web Workers
# sharing one SharedArrayBuffer, which requires a cross-origin isolated page.
option(ORYX_WASM_THREADS "WebAssembly build uses threads (needs a cross-origin isolated page)" ON)
if(EMSCRIPTEN)
  add_compile_options(-msimd128 -fwasm-exceptions)
  add_link_options(-fwasm-exceptions -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB)
  if(ORYX_WASM_THREADS)
    add_compile_options(-pthread)
    add_link_options(-pthread -Wno-pthreads-mem-growth)
  endif()
endif()

find_package(Threads REQUIRED)

add_library(oryx_engine
  src/distance.cpp
  src/distance_scalar.cpp
  src/hnsw.cpp
  src/brute_force.cpp
  src/dataset.cpp
)
target_include_directories(oryx_engine PUBLIC include PRIVATE src)
target_link_libraries(oryx_engine PUBLIC Threads::Threads)
target_compile_options(oryx_engine PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(oryx_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SIMD kernels get ISA flags per file; the rest of the library stays baseline so the binary
# runs on any CPU of the architecture and picks the best kernel at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_sources(oryx_engine PRIVATE src/distance_avx2.cpp src/distance_avx512.cpp)
  set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  target_sources(oryx_engine PRIVATE src/distance_neon.cpp)
elseif(EMSCRIPTEN)
  target_sources(oryx_engine PRIVATE src/distance_wasm.cpp)
endif()

if(EMSCRIPTEN AND NOT ORYX_WASM_THREADS)
  set(ORYX_BUILD_TESTS OFF)  # the concurrency tests need threads
endif()

if(ORYX_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(oryx_tests tests/distance_test.cpp tests/hnsw_test.cpp tests/dataset_test.cpp)
  target_link_libraries(oryx_tests PRIVATE oryx_engine GTest::gtest_main)
  if(EMSCRIPTEN)
    # Runs under Node (CMAKE_CROSSCOMPILING_EMULATOR); workers are pre-spawned for the
    # concurrency tests and the real filesystem backs the save/load tests.
    target_link_options(oryx_tests PRIVATE -sPTHREAD_POOL_SIZE=16 -sNODERAWFS=1 -sENVIRONMENT=node)
  endif()
  include(GoogleTest)
  gtest_discover_tests(oryx_tests DISCOVERY_TIMEOUT 60)
endif()

if(ORYX_BUILD_BENCH)
  add_executable(bench_distance bench/bench_distance.cpp)
  target_link_libraries(bench_distance PRIVATE oryx_engine)
  add_executable(bench_recall bench/bench_recall.cpp)
  target_link_libraries(bench_recall PRIVATE oryx_engine)
endif()

if(EMSCRIPTEN)
  # ES module for the browser demo: `import createOryx from "./oryx.js"`.
  add_executable(oryx_web wasm/oryx_wasm.cpp)
  target_link_libraries(oryx_web PRIVATE oryx_engine)
  set_target_properties(oryx_web PROPERTIES OUTPUT_NAME oryx SUFFIX .js)
  target_link_options(oryx_web PRIVATE
    -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createOryx -sENVIRONMENT=web,worker -sSTACK_SIZE=1MB)
  if(ORYX_WASM_THREADS)
    # Pre-spawn the workers: the module's own thread blocks while a build runs, so it cannot
    # create new ones on demand.
    target_link_options(oryx_web PRIVATE "-sPTHREAD_POOL_SIZE=Math.min(navigator.hardwareConcurrency,16)")
  endif()
endif()

if(ORYX_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(oryx_python python/oryx_py.cpp)
  set_target_properties(oryx_python PROPERTIES OUTPUT_NAME oryx)
  target_link_libraries(oryx_python PRIVATE oryx_engine)
  install(TARGETS oryx_python DESTINATION .)
endif()
