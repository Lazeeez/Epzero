cmake_minimum_required(VERSION 3.20)
project(epzero VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EPZERO_BUILD_TESTS "Build unit tests" ON)
option(EPZERO_BUILD_BENCH "Build benchmarks" ON)
option(EPZERO_BUILD_PYTHON "Build the Python module (normally via pip install ./engine)" OFF)
set(EPZERO_SANITIZE "" CACHE STRING "Sanitizer to enable: address, thread, or undefined")

if(EPZERO_SANITIZE)
  add_compile_options(-fsanitize=${EPZERO_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${EPZERO_SANITIZE})
endif()

# Fuzzing (Clang + libFuzzer): every parser of untrusted bytes, under AddressSanitizer and UBSan.
# The engine is instrumented for coverage; only the fuzz targets link libFuzzer's main().
option(EPZERO_BUILD_FUZZ "Build the libFuzzer targets (Clang only)" OFF)
if(EPZERO_BUILD_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "EPZERO_BUILD_FUZZ needs Clang (libFuzzer)")
  endif()
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# WebAssembly: fixed-width SIMD is supported by every browser the demo targets, so the whole
# build uses it (there is no runtime feature detection in WASM). Threads map to Web Workers
# sharing one SharedArrayBuffer, which requires a cross-origin isolated page.
option(EPZERO_WASM_THREADS "WebAssembly build uses threads (needs a cross-origin isolated page)" ON)
if(EMSCRIPTEN)
  add_compile_options(-msimd128 -fwasm-exceptions)
  add_link_options(-fwasm-exceptions -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB)
  if(EPZERO_WASM_THREADS)
    add_compile_options(-pthread)
    add_link_options(-pthread -Wno-pthreads-mem-growth)
  endif()
endif()

find_package(Threads REQUIRED)

add_library(epzero_engine
  src/distance.cpp
  src/distance_scalar.cpp
  src/hnsw.cpp
  src/recall.cpp
  src/database.cpp
  src/text.cpp
  src/brute_force.cpp
  src/dataset.cpp
)
target_include_directories(epzero_engine PUBLIC include PRIVATE src)
target_link_libraries(epzero_engine PUBLIC Threads::Threads)
target_compile_options(epzero_engine PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(epzero_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SIMD kernels get ISA flags per file; the rest of the library stays baseline so the binary
# runs on any CPU of the architecture and picks the best kernel at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_sources(epzero_engine PRIVATE src/distance_avx2.cpp src/distance_avx512.cpp)
  set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mpopcnt")
  set_source_files_properties(src/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
  # VNNI integer kernels, where the toolchain can build them: the compiler must know the
  # instructions (GCC 11+, Clang 12+ for AVX-VNNI) and so must the assembler it hands them to
  # (binutils 2.36+ for AVX-VNNI; RHEL 9's 2.35 doesn't), so each check compiles and links a use.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mavxvnni")
  check_cxx_source_compiles([[
    #include <immintrin.h>
    int main(int argc, char**) {
      __m256i a = _mm256_set1_epi8(static_cast<char>(argc));
      a = _mm256_dpbusd_avx_epi32(a, a, a);
      return _mm256_extract_epi32(a, 0);
    }]] EPZERO_TOOLCHAIN_AVXVNNI)
  set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vnni")
  check_cxx_source_compiles([[
    #include <immintrin.h>
    int main(int argc, char**) {
      __m512i a = _mm512_set1_epi8(static_cast<char>(argc));
      a = _mm512_dpbusd_epi32(a, a, a);
      return _mm512_reduce_add_epi32(a);
    }]] EPZERO_TOOLCHAIN_AVX512VNNI)
  unset(CMAKE_REQUIRED_FLAGS)
  if(EPZERO_TOOLCHAIN_AVXVNNI)
    target_sources(epzero_engine PRIVATE src/distance_avxvnni.cpp)
    set_source_files_properties(src/distance_avxvnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
    set_property(SOURCE src/distance.cpp APPEND PROPERTY COMPILE_DEFINITIONS EPZERO_HAVE_AVXVNNI)
  endif()
  if(EPZERO_TOOLCHAIN_AVX512VNNI)
    target_sources(epzero_engine PRIVATE src/distance_avx512vnni.cpp)
    set_source_files_properties(src/distance_avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")
    set_property(SOURCE src/distance.cpp APPEND PROPERTY COMPILE_DEFINITIONS EPZERO_HAVE_AVX512VNNI)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  target_sources(epzero_engine PRIVATE src/distance_neon.cpp)
elseif(EMSCRIPTEN)
  target_sources(epzero_engine PRIVATE src/distance_wasm.cpp)
endif()

if(EMSCRIPTEN AND NOT EPZERO_WASM_THREADS)
  set(EPZERO_BUILD_TESTS OFF)  # the concurrency tests need threads
endif()

if(EPZERO_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(epzero_tests tests/distance_test.cpp tests/hnsw_test.cpp tests/recall_test.cpp tests/dataset_test.cpp tests/database_test.cpp tests/text_test.cpp)
  target_link_libraries(epzero_tests PRIVATE epzero_engine GTest::gtest_main)
  if(EMSCRIPTEN)
    # Runs under Node (CMAKE_CROSSCOMPILING_EMULATOR); workers are pre-spawned for the
    # concurrency tests and the real filesystem backs the save/load tests.
    target_link_options(epzero_tests PRIVATE -sPTHREAD_POOL_SIZE=16 -sNODERAWFS=1 -sENVIRONMENT=node)
  endif()
  include(GoogleTest)
  gtest_discover_tests(epzero_tests DISCOVERY_TIMEOUT 60)
endif()

if(EPZERO_BUILD_BENCH)
  add_executable(bench_distance bench/bench_distance.cpp)
  target_link_libraries(bench_distance PRIVATE epzero_engine)
  add_executable(bench_recall bench/bench_recall.cpp)
  target_link_libraries(bench_recall PRIVATE epzero_engine)
  add_executable(bench_filter bench/bench_filter.cpp)
  target_link_libraries(bench_filter PRIVATE epzero_engine)
  add_executable(bench_churn bench/bench_churn.cpp)
  target_link_libraries(bench_churn PRIVATE epzero_engine)
  add_executable(bench_guard bench/bench_guard.cpp)
  target_link_libraries(bench_guard PRIVATE epzero_engine)
  add_executable(bench_adaptive bench/bench_adaptive.cpp)
  target_link_libraries(bench_adaptive PRIVATE epzero_engine)
endif()

if(EPZERO_BUILD_FUZZ)
  foreach(target snapshot graph wal text)
    add_executable(fuzz_${target} fuzz/fuzz_${target}.cpp)
    target_link_libraries(fuzz_${target} PRIVATE epzero_engine)
    target_link_options(fuzz_${target} PRIVATE -fsanitize=fuzzer)
  endforeach()
  add_executable(fuzz_seeds fuzz/make_seeds.cpp)
  target_link_libraries(fuzz_seeds PRIVATE epzero_engine)
endif()

# Writes the web demo's prebuilt graph (see tools/export_graph.cpp and web/scripts/build-graph.sh).
if(NOT EMSCRIPTEN)
  add_executable(epzero_export_graph tools/export_graph.cpp)
  set_target_properties(epzero_export_graph PROPERTIES OUTPUT_NAME epzero-export-graph)
  target_link_libraries(epzero_export_graph PRIVATE epzero_engine)
endif()

option(EPZERO_BUILD_NODE "Build the cluster data node (needs gRPC C++ and protobuf)" ON)
if(EPZERO_BUILD_NODE AND NOT EMSCRIPTEN)
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
    set(PROTO_SRCS ${PROTO_OUT}/epzero/v1/epzero.pb.cc ${PROTO_OUT}/epzero/v1/epzero.grpc.pb.cc)
    add_custom_command(
      OUTPUT ${PROTO_SRCS}
      COMMAND ${PROTOC} -I ${PROTO_ROOT} --cpp_out=${PROTO_OUT} --grpc_out=${PROTO_OUT}
              --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} epzero/v1/epzero.proto
      DEPENDS ${PROTO_ROOT}/epzero/v1/epzero.proto
      COMMENT "Generating gRPC code for epzero.proto")
    add_executable(epzero_node server/node_main.cpp ${PROTO_SRCS})
    set_target_properties(epzero_node PROPERTIES OUTPUT_NAME epzero-node)
    target_include_directories(epzero_node PRIVATE ${PROTO_OUT})
    target_link_libraries(epzero_node PRIVATE epzero_engine PkgConfig::GRPCPP PkgConfig::PROTOBUF)
    install(TARGETS epzero_node RUNTIME DESTINATION bin)
  else()
    message(STATUS "gRPC C++ not found: skipping epzero-node")
  endif()
endif()

if(EMSCRIPTEN)
  # ES module for the browser demo: `import createEpzero from "./epzero.js"`.
  add_executable(epzero_web wasm/epzero_wasm.cpp)
  target_link_libraries(epzero_web PRIVATE epzero_engine)
  set_target_properties(epzero_web PROPERTIES OUTPUT_NAME epzero SUFFIX .js)
  target_link_options(epzero_web PRIVATE
    -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEpzero -sENVIRONMENT=web,worker -sSTACK_SIZE=1MB
    # No eval / new Function in the generated bindings, so the site's Content Security Policy can
    # forbid it (only WebAssembly compilation is allowed).
    -sDYNAMIC_EXECUTION=0)
  if(EPZERO_WASM_THREADS)
    # Pre-spawn the workers: the module's own thread blocks while a build runs, so it cannot
    # create new ones on demand.
    target_link_options(epzero_web PRIVATE "-sPTHREAD_POOL_SIZE=Math.min(navigator.hardwareConcurrency,16)")
  endif()
endif()

if(EPZERO_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(epzero_python python/epzero_py.cpp)
  # The compiled core of the Python package epzero (engine/python/epzero), installed inside it.
  set_target_properties(epzero_python PROPERTIES OUTPUT_NAME _core)
  target_link_libraries(epzero_python PRIVATE epzero_engine)
  install(TARGETS epzero_python DESTINATION epzero)
endif()
