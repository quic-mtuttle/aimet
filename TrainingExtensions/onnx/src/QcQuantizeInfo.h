//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#pragma once

#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <DlQuantization/TensorQuantizer.h>
#include <string>


struct QcQuantizeInfo
{
    void set_tensor_quantizer(std::vector<uint64_t>& addr)
    {
        tensorQuantizerRef = std::vector<DlQuantization::TensorQuantizer*>();
        for(uint64_t i : addr){
            tensorQuantizerRef.push_back(reinterpret_cast<DlQuantization::TensorQuantizer*>(i));
        }
    }
    std::vector<DlQuantization::TensorQuantizer*> get_tensor_quantizer()
    {
        return tensorQuantizerRef;
    }

    std::vector<DlQuantization::TensorQuantizer*> tensorQuantizerRef;
    std::vector<DlQuantization::TfEncoding*> encoding;
    DlQuantization::TensorQuantizerOpMode opMode;
    bool useSymmetricEncoding;
    bool enabled;
    bool isIntDataType;
    int channelAxis;
    std::string name;
};
