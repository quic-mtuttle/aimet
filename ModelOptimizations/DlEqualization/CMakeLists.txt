#==============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its contributors 
#     may be used to endorse or promote products derived from this software 
#     without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  
#  SPDX-License-Identifier: BSD-3-Clause
#  
#  @@-COPYRIGHT-END-@@
#==============================================================================

add_library(MoDlEqualization STATIC
      src/BatchNormFoldForPython.cpp
      src/BatchNormFold.cpp
      src/HighBiasFold.cpp
      src/CrossLayerScaling.cpp
      src/CrossLayerScalingForPython.cpp
      src/HighBiasFoldForPython.cpp
      src/TensorOperations.cpp
      src/TensorOperations.h
      src/ScaleFactorCalculator.cpp
      src/ScaleFactorCalculator.h
      src/BiasCorrection.cpp
      src/BiasCorrectionForPython.cpp
      include/DlEqualization/def.h
      include/DlEqualization/BatchNormFold.h
      include/DlEqualization/HighBiasFoldForPython.h
      include/DlEqualization/BatchNormFoldForPython.h
      include/DlEqualization/HighBiasFold.h
      include/DlEqualization/CrossLayerScaling.h
      include/DlEqualization/CrossLayerScalingForPython.h
      include/DlEqualization/BiasCorrection.h
      include/DlEqualization/BiasCorrectionForPython.h
      test/python/test_cross_layer_scaling.py )

target_include_directories(MoDlEqualization
      PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      )

target_link_libraries(MoDlEqualization
      PUBLIC PYBIND11
      )

target_compile_options(MoDlEqualization
        PRIVATE
            -DUSE_OPENCV
        )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlEqualization
        DESTINATION ${AIMET_INSTALL_DIR}/lib/x86_64-linux-gnu/include)

add_subdirectory(test)