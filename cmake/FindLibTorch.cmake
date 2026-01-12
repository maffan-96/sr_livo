# CMake module to find LibTorch (PyTorch C++ API)
#
# Usage:
#   find_package(LibTorch REQUIRED)
#
# Sets:
#   LIBTORCH_FOUND
#   LIBTORCH_INCLUDE_DIRS
#   LIBTORCH_LIBRARIES
#   LIBTORCH_CXX_FLAGS

set(LIBTORCH_FOUND FALSE)

# Try to find LibTorch
if(NOT DEFINED LIBTORCH_ROOT)
    set(LIBTORCH_ROOT "/usr/local/libtorch" CACHE PATH "LibTorch root directory")
endif()

# Find Torch cmake config
find_package(Torch QUIET
    PATHS ${LIBTORCH_ROOT}
    NO_DEFAULT_PATH
)

if(Torch_FOUND)
    set(LIBTORCH_FOUND TRUE)
    set(LIBTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    set(LIBTORCH_LIBRARIES ${TORCH_LIBRARIES})
    set(LIBTORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")

    message(STATUS "Found LibTorch: ${TORCH_INSTALL_PREFIX}")
    message(STATUS "  Include dirs: ${LIBTORCH_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${LIBTORCH_LIBRARIES}")
    message(STATUS "  CXX flags: ${LIBTORCH_CXX_FLAGS}")

    # Check for CUDA support
    if(TORCH_CUDA_AVAILABLE)
        message(STATUS "  CUDA support: YES")
    else()
        message(STATUS "  CUDA support: NO")
    endif()
else()
    message(WARNING "LibTorch not found. Set LIBTORCH_ROOT to LibTorch installation directory.")
    message(WARNING "Download from: https://pytorch.org/get-started/locally/")
endif()

# Set standard find_package variables
if(LIBTORCH_FOUND)
    if(NOT TARGET LibTorch::LibTorch)
        add_library(LibTorch::LibTorch INTERFACE IMPORTED)
        target_include_directories(LibTorch::LibTorch INTERFACE ${LIBTORCH_INCLUDE_DIRS})
        target_link_libraries(LibTorch::LibTorch INTERFACE ${LIBTORCH_LIBRARIES})
        target_compile_options(LibTorch::LibTorch INTERFACE ${LIBTORCH_CXX_FLAGS})
    endif()
endif()
