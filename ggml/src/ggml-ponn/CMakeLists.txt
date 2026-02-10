set(TARGET_NAME ggml-ponn)

list(APPEND GGML_CDEF_PUBLIC GGML_USE_PONN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-unused-parameter -Wno-unused-variable ")

file(GLOB   GGML_HEADERS_PONN "*.hpp")
list(APPEND GGML_HEADERS_PONN "../../include/ggml-ponn.h")

file(GLOB   GGML_SOURCES_PONN "*.cpp")

set(PONN_EXTRA_LIBS ${GGML_EXTRA_LIBS} ariseponn_wrap)
list(APPEND GGML_CDEF_PUBLIC GGML_USE_PONN)

if(GGML_PONN_CHECK)
    list(APPEND GGML_CDEF_PUBLIC GGML_PONN_CHECK)
endif()
if(GGML_PONN_PROFILING)
    add_compile_definitions(GGML_PONN_PROFILING)
endif()
if(GGML_PONN_PROFILING_PRECISION)
    add_compile_definitions(GGML_PONN_PROFILING_PRECISION)
endif()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ponn/CMakeLists.txt")
    add_subdirectory(ponn)
endif()

ggml_add_backend_library(ggml-ponn ${GGML_SOURCES_PONN})
target_link_libraries(ggml-ponn PRIVATE ${PONN_EXTRA_LIBS})
target_include_directories(ggml-ponn PRIVATE ${PONN_INCLUDE_DIRS})

#ggml_add_backend_library(${TARGET_NAME}
#                         ggml-ponn.cpp
#                         ../../include/ggml-ponn.h)
#target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})
#target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
