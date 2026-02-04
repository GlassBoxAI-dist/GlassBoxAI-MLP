cmake_minimum_required(VERSION 3.14)
project(facaded_mlp_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Rust library
set(RUST_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/release")
set(RUST_LIB_NAME "facaded_mlp_cuda")

# Check for library
find_library(FACADED_MLP_LIB
    NAMES ${RUST_LIB_NAME}
    PATHS ${RUST_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT FACADED_MLP_LIB)
    message(WARNING "Rust library not found. Build with: cargo build --release --features julia")
    set(FACADED_MLP_LIB "${RUST_LIB_DIR}/lib${RUST_LIB_NAME}.so")
endif()

# Header-only library target
add_library(facaded_mlp INTERFACE)
target_include_directories(facaded_mlp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(facaded_mlp INTERFACE ${FACADED_MLP_LIB})

# Examples
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_executable(xor_example examples/xor_example.cpp)
    target_link_libraries(xor_example PRIVATE facaded_mlp)
    
    # Set rpath for finding the library at runtime
    set_target_properties(xor_example PROPERTIES
        BUILD_RPATH "${RUST_LIB_DIR}"
        INSTALL_RPATH "${RUST_LIB_DIR}"
    )
endif()

# Install
include(GNUInstallDirs)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS facaded_mlp
    EXPORT facaded_mlp-targets
)

install(EXPORT facaded_mlp-targets
    FILE facaded_mlp-config.cmake
    NAMESPACE facaded::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/facaded_mlp
)
