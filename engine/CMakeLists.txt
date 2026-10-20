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

# Fuzzing (Clang + libFuzzer): every parser of untrusted bytes, under AddressSanitizer and UBSan.
# The engine is instrumented for coverage; only the fuzz targets link libFuzzer's main().
option(ORYX_BUILD_FUZZ "Build the libFuzzer targets (Clang only)" OFF)
if(ORYX_BUILD_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "ORYX_BUILD_FUZZ needs Clang (libFuzzer)")
  endif()
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
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
  src/recall.cpp
  src/database.cpp
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
  add_executable(oryx_tests tests/distance_test.cpp tests/hnsw_test.cpp tests/recall_test.cpp tests/dataset_test.cpp tests/database_test.cpp)
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
  add_executable(bench_filter bench/bench_filter.cpp)
  target_link_libraries(bench_filter PRIVATE oryx_engine)
  add_executable(bench_churn bench/bench_churn.cpp)
  target_link_libraries(bench_churn PRIVATE oryx_engine)
  add_executable(bench_guard bench/bench_guard.cpp)
  target_link_libraries(bench_guard PRIVATE oryx_engine)
endif()

if(ORYX_BUILD_FUZZ)
  foreach(target snapshot graph wal)
    add_executable(fuzz_${target} fuzz/fuzz_${target}.cpp)
    target_link_libraries(fuzz_${target} PRIVATE oryx_engine)
    target_link_options(fuzz_${target} PRIVATE -fsanitize=fuzzer)
  endforeach()
  add_executable(fuzz_seeds fuzz/make_seeds.cpp)
  target_link_libraries(fuzz_seeds PRIVATE oryx_engine)
endif()

# Writes the web demo's prebuilt graph (see tools/export_graph.cpp and web/scripts/build-graph.sh).
if(NOT EMSCRIPTEN)
  add_executable(oryx_export_graph tools/export_graph.cpp)
  set_target_properties(oryx_export_graph PROPERTIES OUTPUT_NAME oryx-export-graph)
  target_link_libraries(oryx_export_graph PRIVATE oryx_engine)
endif()

option(ORYX_BUILD_NODE "Build the cluster data node (needs gRPC C++ and protobuf)" ON)
if(ORYX_BUILD_NODE AND NOT EMSCRIPTEN)
  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    pkg_check_modules(GRPCPP IMPORTED_TARGET grpc++)
    pkg_check_modules(PROTOBUF IMPORTED_TARGET protobuf)
  endif()
  find_program(PROTOC protoc)
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  if(GRPCPP_FOUND AND PROTOBUF_FOUND AND PROTOC AND GRPC_CPP_PLUGIN)
    set(PROTO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
    set(PROTO_OUT ${CMAKE_CURRENT_BINARY_DIR}/proto)
    file(MAKE_DIRECTORY ${PROTO_OUT})
    set(PROTO_SRCS ${PROTO_OUT}/oryx/v1/oryx.pb.cc ${PROTO_OUT}/oryx/v1/oryx.grpc.pb.cc)
    add_custom_command(
      OUTPUT ${PROTO_SRCS}
      COMMAND ${PROTOC} -I ${PROTO_ROOT} --cpp_out=${PROTO_OUT} --grpc_out=${PROTO_OUT}
              --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} oryx/v1/oryx.proto
      DEPENDS ${PROTO_ROOT}/oryx/v1/oryx.proto
      COMMENT "Generating gRPC code for oryx.proto")
    add_executable(oryx_node server/node_main.cpp ${PROTO_SRCS})
    set_target_properties(oryx_node PROPERTIES OUTPUT_NAME oryx-node)
    target_include_directories(oryx_node PRIVATE ${PROTO_OUT})
    target_link_libraries(oryx_node PRIVATE oryx_engine PkgConfig::GRPCPP PkgConfig::PROTOBUF)
    install(TARGETS oryx_node RUNTIME DESTINATION bin)
  else()
    message(STATUS "gRPC C++ not found: skipping oryx-node")
  endif()
endif()

if(EMSCRIPTEN)
  # ES module for the browser demo: `import createOryx from "./oryx.js"`.
  add_executable(oryx_web wasm/oryx_wasm.cpp)
  target_link_libraries(oryx_web PRIVATE oryx_engine)
  set_target_properties(oryx_web PROPERTIES OUTPUT_NAME oryx SUFFIX .js)
  target_link_options(oryx_web PRIVATE
    -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createOryx -sENVIRONMENT=web,worker -sSTACK_SIZE=1MB
    # No eval / new Function in the generated bindings, so the site's Content Security Policy can
    # forbid it (only WebAssembly compilation is allowed).
    -sDYNAMIC_EXECUTION=0)
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
