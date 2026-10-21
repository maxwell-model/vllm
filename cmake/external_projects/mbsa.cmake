# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Install MBSA and its MXFP6 extensions into vllm.third_party.mbsa.
include(FetchContent)

# If MBSA_SRC_DIR is set, MBSA is installed from that directory instead of
# downloading. This is useful for local MBSA development.
if(DEFINED ENV{MBSA_SRC_DIR})
  set(MBSA_SRC_DIR $ENV{MBSA_SRC_DIR})
endif()

if(MBSA_SRC_DIR)
  FetchContent_Declare(
    mbsa
    SOURCE_DIR ${MBSA_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
    mbsa
    GIT_REPOSITORY https://github.com/maxwell-model/MBSA.git
    GIT_TAG 07ebee24d561190437bae8740fabeea0f506d0b7
    GIT_PROGRESS TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
endif()

FetchContent_GetProperties(mbsa)
if(NOT mbsa_POPULATED)
  FetchContent_Populate(mbsa)
endif()
message(STATUS "MBSA is available at ${mbsa_SOURCE_DIR}")

add_custom_target(mbsa)

install(DIRECTORY "${mbsa_SOURCE_DIR}/python/mbsa/"
  DESTINATION vllm/third_party/mbsa
  COMPONENT mbsa
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".git*" EXCLUDE)

install(FILES
  "${mbsa_SOURCE_DIR}/LICENSE"
  "${mbsa_SOURCE_DIR}/NOTICE"
  DESTINATION vllm/third_party/mbsa
  COMPONENT mbsa)

# MXFP6 requires architecture-specific tcgen05 code. CUDA 13 maps SM103 to
# SM100 in CUDA_ARCHS, so build both variants when either is requested.
set(MBSA_MXFP6_ARCHS)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND MBSA_MXFP6_ARCHS "10.0a")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
  list(APPEND MBSA_MXFP6_ARCHS "10.3a")
endif()
set(_mbsa_sm100_requested FALSE)
foreach(_arch IN LISTS CUDA_ARCHS)
  if(_arch MATCHES "^10\\.[03]")
    set(_mbsa_sm100_requested TRUE)
  endif()
endforeach()

if(_mbsa_sm100_requested AND MBSA_MXFP6_ARCHS)
  message(STATUS "MBSA CUDA extensions: MXFP6 ${MBSA_MXFP6_ARCHS}")
  set(_mbsa_py "${mbsa_SOURCE_DIR}/python/mbsa")
  set(_mbsa_mxfp6_csrc "${_mbsa_py}/mxfp6/csrc")

  if(DEFINED deepgemm_SOURCE_DIR AND
     EXISTS "${deepgemm_SOURCE_DIR}/third-party/cutlass/include/cute")
    set(_mbsa_cutlass_include "${deepgemm_SOURCE_DIR}/third-party/cutlass/include")
  else()
    set(_mbsa_cutlass_include "${CUTLASS_INCLUDE_DIR}")
  endif()
  message(STATUS "MBSA CUTLASS headers: ${_mbsa_cutlass_include}")

  # pybind11 bindings use libtorch_python, as cpp_extension links them.
  find_library(MBSA_TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

  run_python(_mbsa_torch_nvcc_flags
    "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
    "Failed to read torch's extension nvcc flags")
  set(_mbsa_prefill_src "${_mbsa_mxfp6_csrc}/fp6_ext.cu")
  set(_mbsa_decode_src "${_mbsa_mxfp6_csrc}/fp6_dg_ext.cu")
  set_gencode_flags_for_srcs(
    SRCS "${_mbsa_prefill_src};${_mbsa_decode_src}"
    CUDA_ARCHS "${MBSA_MXFP6_ARCHS}")
  set(_mbsa_targets)
  foreach(_mbsa_ext IN ITEMS
      "mbsa_mxfp6_prefill|${_mbsa_prefill_src}|mxfp6"
      "mbsa_mxfp6_decode|${_mbsa_decode_src}|mxfp6")
    string(REPLACE "|" ";" _mbsa_ext "${_mbsa_ext}")
    list(GET _mbsa_ext 0 _mbsa_name)
    list(GET _mbsa_ext 1 _mbsa_src)
    list(GET _mbsa_ext 2 _mbsa_subdir)
    set(_mbsa_flags ${_mbsa_torch_nvcc_flags} "--expt-relaxed-constexpr")
    set(_mbsa_includes "${_mbsa_mxfp6_csrc}" "${_mbsa_cutlass_include}")
    define_extension_target(
      ${_mbsa_name}
      DESTINATION vllm/third_party/mbsa/${_mbsa_subdir}
      LANGUAGE CUDA
      SOURCES ${_mbsa_src}
      COMPILE_FLAGS ${_mbsa_flags}
      ARCHITECTURES ${VLLM_GPU_ARCHES}
      INCLUDE_DIRECTORIES ${_mbsa_includes}
      LIBRARIES ${MBSA_TORCH_PYTHON_LIBRARY}
      WITH_SOABI)
    target_compile_options(${_mbsa_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
    target_compile_definitions(${_mbsa_name} PRIVATE
      TORCH_API_INCLUDE_EXTENSION_H)
    install(TARGETS ${_mbsa_name}
      LIBRARY DESTINATION vllm/third_party/mbsa/${_mbsa_subdir}
      COMPONENT mbsa)
    list(APPEND _mbsa_targets ${_mbsa_name})
  endforeach()
  add_dependencies(mbsa ${_mbsa_targets})
else()
  message(STATUS "MBSA CUDA extensions will not be built: no SM100/SM103 "
    "target in ${CUDA_ARCHS}; MBSA will JIT-compile them if used.")
endif()
