#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# The label_image_odt example for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/examples/label_image_odt"
  TFLITE_LABEL_IMAGE_SRCS
  FILTER "_test\\.cc$"
)
list(APPEND TFLITE_LABEL_IMAGE_SRCS
  ${TF_SOURCE_DIR}/core/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_LABEL_IMAGE_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_LABEL_IMAGE_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

add_executable(label_image_odt
  EXCLUDE_FROM_ALL
  ${TFLITE_LABEL_IMAGE_SRCS}
)
target_compile_options(label_image_odt
  PRIVATE
    ${TFLITE_LABEL_IMAGE_CC_OPTIONS}
)

find_library(TENSORFLOW_FLEX_LIB "libtensorflowlite_flex.so" ${CMAKE_BINARY_DIR})
find_library(TENSORFLOW_LIB "libtensorflowlite_bazel.so" ${CMAKE_BINARY_DIR})
message(STATUS "Flex library : ${TENSORFLOW_FLEX_LIB}")
target_link_libraries(label_image_odt
  #${TENSORFLOW_FLEX_LIB}
  #${TENSORFLOW_LIB}
  tensorflow-lite
  "-lstdc++fs"
)
target_include_directories(label_image_odt PUBLIC $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>)
