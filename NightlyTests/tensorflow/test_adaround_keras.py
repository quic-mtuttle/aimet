# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================
""" Keras AdaRound Nightly Tests """
import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pytest
pytestmark = pytest.mark.skip("Disable tests that requires eager execution")
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.adaround_weight import Adaround, AdaroundParameters

@pytest.mark.cuda
def test_adaround_mobilenet_only_weights():
    """ test end to end adaround with only weight quantized """

    def dummy_forward_pass(model: tf.keras.Model):
        """ Dummy forward pass """
        input_data = np.random.rand(1, 224, 224, 3)
        return model(input_data)

    AimetLogger.set_level_for_all_areas(logging.DEBUG)

    mobilenet_model = MobileNet(weights=None, input_shape=(224, 224, 3))
    dataset_size = 128
    batch_size = 64
    possible_batches = dataset_size // batch_size
    input_data = np.random.rand(dataset_size, 224, 224, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=1,
                                default_reg_param=0.01, default_beta_range=(20, 2), default_warm_start=0.2)

    adarounded_model = Adaround.apply_adaround(mobilenet_model, params, path='./',
                                               filename_prefix='mobilenet', default_param_bw=4,
                                               default_quant_scheme=QuantScheme.post_training_tf_enhanced)

    orig_output = dummy_forward_pass(mobilenet_model)
    adarounded_output = dummy_forward_pass(adarounded_model)
    assert orig_output.shape == adarounded_output.shape

    # Test exported encodings JSON file
    with open('./mobilenet.encodings') as json_file:
        encoding_data = json.load(json_file)
        print(encoding_data)

    assert isinstance(encoding_data["conv1/kernel:0"], list)

    # Delete encodings JSON file
    if os.path.exists("./mobilenet.encodings"):
        os.remove("./mobilenet.encodings")

def test_adaround_followed_by_quantsim():
    """ test end to end adaround with weight 4 bits and output activations 8 bits quantized """

    def dummy_forward_pass(model: tf.keras.Model, _):
        """ Dummy forward pass """
        input_data = np.random.rand(32, 16, 16, 3)
        return model(input_data)

    np.random.seed(1)
    AimetLogger.set_level_for_all_areas(logging.DEBUG)

    model = keras_model()
    dataset_size = 32
    batch_size = 16
    possible_batches = dataset_size // batch_size
    input_data = np.random.rand(dataset_size, 16, 16, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)

    # W4A8
    param_bw = 4
    output_bw = 8
    quant_scheme = QuantScheme.post_training_tf_enhanced

    adarounded_model = Adaround.apply_adaround(model, params, path='./', filename_prefix='dummy',
                                               default_param_bw=param_bw, default_quant_scheme=quant_scheme)

    # Read exported param encodings JSON file
    with open('./dummy.encodings') as json_file:
        encoding_data = json.load(json_file)

    encoding = encoding_data["conv2d/kernel:0"][0]
    before_min, before_max, before_delta, before_offset = encoding.get('min'), encoding.get('max'), \
                                                          encoding.get('scale'), encoding.get('offset')

    print(before_min, before_max, before_delta, before_offset)

    # Create QuantSim using adarounded_model, set and freeze parameter encodings and then invoke compute_encodings
    sim = QuantizationSimModel(adarounded_model, quant_scheme, default_output_bw=output_bw, default_param_bw=param_bw)

    sim.set_and_freeze_param_encodings(encoding_path='./dummy.encodings')
    sim.compute_encodings(dummy_forward_pass, None)

    conv_encoding = sim.model.layers[0].param_quantizers[0].encoding
    after_min, after_max, after_delta, after_offset = conv_encoding.min, conv_encoding.max, conv_encoding.delta,\
                                                      conv_encoding.offset

    print(after_min, after_max, after_delta, after_offset)

    assert before_min == after_min
    assert before_max == after_max
    assert np.allclose(before_delta, after_delta, atol=1e-4)
    assert before_offset == after_offset

    # Delete encodings file
    if os.path.exists("./dummy.encodings"):
        os.remove("./dummy.encodings")
