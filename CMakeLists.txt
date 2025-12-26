set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.18)

project(flash_attn LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_JOB_POOLS cuda=6)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=medium")

# == Find dependencies ==
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# == Setup CUDA ==
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set up ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

if(NOT DEFINED FLASH_ATTN_CUDA_ARCHS)
    set(TARGET_CUDA_ARCHITECTURES "90")
else()
    set(TARGET_CUDA_ARCHITECTURES "${FLASH_ATTN_CUDA_ARCHS}")
endif()
message(STATUS "Building for CUDA Architectures: ${TARGET_CUDA_ARCHITECTURES}")

set(FEATURE_FLAGS
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
)

# Automatically disable architecture-specific kernels based on build targets
list(FIND TARGET_CUDA_ARCHITECTURES "80" HAS_SM80)
list(FIND TARGET_CUDA_ARCHITECTURES "86" HAS_SM86)
list(FIND TARGET_CUDA_ARCHITECTURES "89" HAS_SM89)
list(FIND TARGET_CUDA_ARCHITECTURES "90" HAS_SM90)
list(FIND TARGET_CUDA_ARCHITECTURES "100" HAS_SM100)

# Disable SM8x if only SM90+ architectures are being built
if(HAS_SM80 EQUAL -1 AND HAS_SM86 EQUAL -1 AND HAS_SM89 EQUAL -1)
    set(FLASHATTENTION_DISABLE_SM8x_AUTO ON)
    message(STATUS "SM8x architectures not in build targets - automatically disabling SM8x kernels")
else()
    set(FLASHATTENTION_DISABLE_SM8x_AUTO OFF)
endif()

# Disable SM90+ if only SM8x architectures are being built
if(HAS_SM90 EQUAL -1 AND HAS_SM100 EQUAL -1)
    set(FLASHATTENTION_DISABLE_SM90_AUTO ON)
    message(STATUS "SM90+ architectures not in build targets - automatically disabling SM90 kernels")
else()
    set(FLASHATTENTION_DISABLE_SM90_AUTO OFF)
endif()

# Core features
option(FLASHATTENTION_DISABLE_BACKWARD "Disable backward pass kernels" OFF)
option(FLASHATTENTION_DISABLE_SPLIT "Disable split-K kernels" OFF)
option(FLASHATTENTION_DISABLE_PAGEDKV "Disable paged KV cache support" OFF)
option(FLASHATTENTION_DISABLE_APPENDKV "Disable append KV support" OFF)
option(FLASHATTENTION_DISABLE_LOCAL "Disable local/sliding window attention" OFF)
option(FLASHATTENTION_DISABLE_SOFTCAP "Disable softcap attention" OFF)
option(FLASHATTENTION_DISABLE_PACKGQA "Disable PackGQA optimization" OFF)

# Data types
option(FLASHATTENTION_DISABLE_FP16 "Disable FP16 kernels" OFF)
option(FLASHATTENTION_DISABLE_FP8 "Disable FP8 kernels" OFF)

# Architecture support
option(FLASHATTENTION_DISABLE_SM8x "Disable SM80/SM86/SM89 kernels" ${FLASHATTENTION_DISABLE_SM8x_AUTO})
option(FLASHATTENTION_DISABLE_SM90 "Disable SM90+ kernels" ${FLASHATTENTION_DISABLE_SM90_AUTO})

# Variable length sequences
option(FLASHATTENTION_DISABLE_VARLEN "Disable variable-length sequence support" OFF)

# Advanced features
option(FLASHATTENTION_DISABLE_CLUSTER "Disable cluster mode" OFF)
option(FLASHATTENTION_ENABLE_VCOLMAJOR "Enable V column-major layout" OFF)

# Head dimensions
option(FLASHATTENTION_DISABLE_HDIM64 "Disable head dimension 64" OFF)
option(FLASHATTENTION_DISABLE_HDIM96 "Disable head dimension 96" OFF)
option(FLASHATTENTION_DISABLE_HDIM128 "Disable head dimension 128" OFF)
option(FLASHATTENTION_DISABLE_HDIM192 "Disable head dimension 192" OFF)
option(FLASHATTENTION_DISABLE_HDIM256 "Disable head dimension 256" OFF)

# Different Q/KV head dimensions
option(FLASHATTENTION_DISABLE_HDIMDIFF64 "Disable different Q/KV head dims for 64" OFF)
option(FLASHATTENTION_DISABLE_HDIMDIFF192 "Disable different Q/KV head dims for 192" OFF)

if(FLASHATTENTION_DISABLE_BACKWARD)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_BACKWARD)
    message(STATUS "Backward pass kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_SPLIT)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_SPLIT)
    message(STATUS "Split-K kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_PAGEDKV)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_PAGEDKV)
    message(STATUS "Paged KV cache: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_SOFTCAP)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_SOFTCAP)
    message(STATUS "Softcap attention: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_FP16)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_FP16)
    message(STATUS "FP16 kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_FP8)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_FP8)
    message(STATUS "FP8 kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_SM8x)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_SM8x)
    message(STATUS "SM8x kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_SM90)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_SM90)
    message(STATUS "SM90 kernels: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_APPENDKV)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_APPENDKV)
    message(STATUS "Append KV support: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_LOCAL)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_LOCAL)
    message(STATUS "Local/sliding window attention: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_PACKGQA)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_PACKGQA)
    message(STATUS "PackGQA optimization: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_VARLEN)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_VARLEN)
    message(STATUS "Variable-length sequence support: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_CLUSTER)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_CLUSTER)
    message(STATUS "Cluster mode: DISABLED")
endif()

if(FLASHATTENTION_ENABLE_VCOLMAJOR)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_ENABLE_VCOLMAJOR)
    message(STATUS "V column-major layout: ENABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIM64)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIM64)
    message(STATUS "Head dimension 64: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIM96)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIM96)
    message(STATUS "Head dimension 96: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIM128)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIM128)
    message(STATUS "Head dimension 128: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIM192)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIM192)
    message(STATUS "Head dimension 192: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIM256)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIM256)
    message(STATUS "Head dimension 256: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIMDIFF64)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIMDIFF64)
    message(STATUS "Different Q/KV head dims for 64: DISABLED")
endif()

if(FLASHATTENTION_DISABLE_HDIMDIFF192)
    list(APPEND FEATURE_FLAGS -DFLASHATTENTION_DISABLE_HDIMDIFF192)
    message(STATUS "Different Q/KV head dims for 192: DISABLED")
endif()

set(COMMON_CUDA_FLAGS
    -O3
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -DNDEBUG
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DCUTLASS_ENABLE_GDC_FOR_SM90
    -DCUTLASS_DEBUG_TRACE_LEVEL=0
    --resource-usage
    -lineinfo
    -Xcompiler=-fPIC
    -Xcompiler=-mcmodel=medium
    ${FEATURE_FLAGS}
)

set(FLASH_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

file(GLOB CC_SOURCES "csrc/flash_attn/*.cpp" "csrc/flash_attn/src/*.cpp")
file(GLOB GENERIC_CUDA_SOURCES "csrc/flash_attn/*.cu" "csrc/flash_attn/src/*.cu")

# Filter out split-K combine kernel if split-K is disabled
if(FLASHATTENTION_DISABLE_SPLIT)
    list(FILTER GENERIC_CUDA_SOURCES EXCLUDE REGEX ".*flash_fwd_combine\\.cu$")
endif()

# Create object libraries for each architecture to isolate compiler flags
set(FLASH_OBJECTS "")

list(FIND TARGET_CUDA_ARCHITECTURES "80" IS_SM80_ENABLED)
if(NOT IS_SM80_ENABLED EQUAL -1)
    message(STATUS "Configuring SM80 build.")

    # Get forward and backward sources separately since they use different SOFTCAP logic
    # SM80 Forward uses SOFTCAP_ALL, SM80 Backward uses SOFTCAP
    file(GLOB SM80_FWD_SOURCES "csrc/flash_attn/src/instantiations/flash_fwd_*_sm80.cu")
    file(GLOB SM80_BWD_SOURCES "csrc/flash_attn/src/instantiations/flash_bwd_*_sm80.cu")

    # Handle SOFTCAP filtering based on setup.py logic from the FA repo:
    # - SM80 Forward uses SOFTCAP_ALL: ["_softcapall"] when enabled, [""] when disabled
    # - SM80 Backward uses SOFTCAP: ["", "_softcap"] when enabled, [""] when disabled
    if(FLASHATTENTION_DISABLE_SOFTCAP)
        # Keep base only (exclude both _softcap and _softcapall variants)
        list(FILTER SM80_FWD_SOURCES EXCLUDE REGEX ".*_softcap.*_sm80\\.cu$")
        list(FILTER SM80_BWD_SOURCES EXCLUDE REGEX ".*_softcap.*_sm80\\.cu$")
    else()
        # Forward: keep _softcapall only (exclude base and _softcap)
        list(FILTER SM80_FWD_SOURCES INCLUDE REGEX ".*_softcapall_sm80\\.cu$")
        # Backward: keep base and _softcap (exclude _softcapall)
        list(FILTER SM80_BWD_SOURCES EXCLUDE REGEX ".*_softcapall_sm80\\.cu$")
    endif()

    # Combine forward and backward sources
    set(SM80_SOURCES ${SM80_FWD_SOURCES} ${SM80_BWD_SOURCES})

    # Apply feature flag filters to exclude disabled kernel files
    if(FLASHATTENTION_DISABLE_BACKWARD)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*flash_bwd_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_SPLIT)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_split_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_PAGEDKV)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_paged_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_FP16)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_fp16_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_APPENDKV)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_appendkv_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_LOCAL)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_local_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_PACKGQA)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_packgqa_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_VARLEN)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_varlen_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_CLUSTER)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*_cluster_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM64)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*hdim64_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM96)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*hdim96_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM128)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*hdim128_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM192)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*hdim192_.*_sm80\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM256)
        list(FILTER SM80_SOURCES EXCLUDE REGEX ".*hdim256_.*_sm80\\.cu$")
    endif()

    # FP8 is not supported on SM80, so no need to filter
    # HDIMDIFF is not used on SM80 in the current implementation
    add_library(flash_sm80_objects OBJECT ${SM80_SOURCES})
    target_include_directories(flash_sm80_objects PRIVATE ${FLASH_INCLUDE_DIRS})
    target_compile_options(flash_sm80_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-code=arch=compute_80,code=sm_80
        ${COMMON_CUDA_FLAGS}
    >)
    list(APPEND FLASH_OBJECTS $<TARGET_OBJECTS:flash_sm80_objects>)
endif()

list(FIND TARGET_CUDA_ARCHITECTURES "90" IS_SM90_ENABLED)
if(NOT IS_SM90_ENABLED EQUAL -1)
    message(STATUS "Configuring SM90 build.")

    # Get forward and backward sources separately since they use different SOFTCAP logic
    # SM90 Forward uses SOFTCAP, SM90 Backward uses SOFTCAP_ALL (opposite of SM80)
    file(GLOB SM90_FWD_SOURCES "csrc/flash_attn/src/instantiations/flash_fwd_*_sm90.cu")
    file(GLOB SM90_BWD_SOURCES "csrc/flash_attn/src/instantiations/flash_bwd_*_sm90.cu")

    # Handle SOFTCAP filtering based on setup.py logic:
    # - SM90 Forward uses SOFTCAP: ["", "_softcap"] when enabled, [""] when disabled
    # - SM90 Backward uses SOFTCAP_ALL: ["_softcapall"] when enabled, [""] when disabled
    if(FLASHATTENTION_DISABLE_SOFTCAP)
        # Keep base only (exclude both _softcap and _softcapall variants)
        list(FILTER SM90_FWD_SOURCES EXCLUDE REGEX ".*_softcap.*_sm90\\.cu$")
        list(FILTER SM90_BWD_SOURCES EXCLUDE REGEX ".*_softcap.*_sm90\\.cu$")
    else()
        # Forward: keep base and _softcap (exclude _softcapall)
        list(FILTER SM90_FWD_SOURCES EXCLUDE REGEX ".*_softcapall_sm90\\.cu$")
        # Backward: keep _softcapall only (exclude base and _softcap)
        list(FILTER SM90_BWD_SOURCES INCLUDE REGEX ".*_softcapall_sm90\\.cu$")
    endif()

    # Combine forward and backward sources
    set(SM90_SOURCES ${SM90_FWD_SOURCES} ${SM90_BWD_SOURCES})

    # Apply feature flag filters to exclude disabled kernel files
    if(FLASHATTENTION_DISABLE_BACKWARD)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*flash_bwd_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_SPLIT)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_split_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_PAGEDKV)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_paged_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_FP16)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_fp16_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_FP8)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_e4m3_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_APPENDKV)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_appendkv_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_LOCAL)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_local_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_PACKGQA)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_packgqa_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_VARLEN)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_varlen_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_CLUSTER)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*_cluster_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM64)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim64_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM96)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim96_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM128)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim128_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM192)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim192_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIM256)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim256_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIMDIFF64)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim64_256_.*_sm90\\.cu$")
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim64_512_.*_sm90\\.cu$")
    endif()

    if(FLASHATTENTION_DISABLE_HDIMDIFF192)
        list(FILTER SM90_SOURCES EXCLUDE REGEX ".*hdim192_128_.*_sm90\\.cu$")
    endif()

    add_library(flash_sm90_objects OBJECT ${SM90_SOURCES})
    # MODIFIED: Add include directories to the object library
    target_include_directories(flash_sm90_objects PRIVATE ${FLASH_INCLUDE_DIRS})
    target_compile_options(flash_sm90_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-code=arch=compute_90a,code=sm_90a
        ${COMMON_CUDA_FLAGS}
    >)
    list(APPEND FLASH_OBJECTS $<TARGET_OBJECTS:flash_sm90_objects>)
endif()

pybind11_add_module(flash_api
    ${CC_SOURCES}
    ${GENERIC_CUDA_SOURCES}
    ${FLASH_OBJECTS}
)

set_property(TARGET flash_api PROPERTY JOB_POOL_COMPILE cuda)

set(GENCODE_FLAGS "")
foreach(ARCH ${TARGET_CUDA_ARCHITECTURES})
    if(ARCH STREQUAL "80")
        list(APPEND GENCODE_FLAGS "--generate-code=arch=compute_80,code=sm_80")
    elseif(ARCH STREQUAL "86")
        list(APPEND GENCODE_FLAGS "--generate-code=arch=compute_86,code=sm_86")
    elseif(ARCH STREQUAL "89")
        list(APPEND GENCODE_FLAGS "--generate-code=arch=compute_89,code=sm_89")
    elseif(ARCH STREQUAL "90")
        list(APPEND GENCODE_FLAGS "--generate-code=arch=compute_90a,code=sm_90a")
    elseif(ARCH STREQUAL "100")
        list(APPEND GENCODE_FLAGS "--generate-code=arch=compute_100a,code=sm_100a")
    else()
        message(WARNING "Unknown CUDA architecture: ${ARCH}")
    endif()
endforeach()

target_compile_options(flash_api PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC ${FEATURE_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:
        ${GENCODE_FLAGS}
        ${COMMON_CUDA_FLAGS}
    >
)
target_include_directories(flash_api PRIVATE ${FLASH_INCLUDE_DIRS})
target_link_libraries(flash_api PRIVATE CUDA::cudart)
target_link_options(flash_api PRIVATE)

install(TARGETS flash_api
        DESTINATION ${SKBUILD_PLATLIB_DIR}/flash_attn3_jax_lib
)
