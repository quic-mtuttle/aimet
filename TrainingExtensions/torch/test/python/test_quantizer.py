# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import copy
import logging
import json as json
import os
import unittest.mock
from packaging import version
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
import yaml
from torchvision import models

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_common.utils import AimetLogger
from aimet_torch import transformer_utils, onnx_utils
from aimet_torch import utils, elementwise_ops
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.examples.test_models import TwoLayerBidirectionalLSTMModel, SingleLayerRNNModel, \
    ModelWithTwoInputs, SimpleConditional, RoiModel, InputOutputDictModel
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeStandalone, \
    StaticGridQuantWrapper, QcQuantizeOpMode, LearnedGridQuantWrapper
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.quantsim import QuantizationSimModel, check_accumulator_overflow, load_encodings_to_sim
from aimet_torch.quantsim_straight_through_grad import compute_dloss_by_dx

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    model.eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    with torch.no_grad():
        model(*dummy_input)


def dummy_forward_pass(model, args):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn((32, 1, 28, 28)))
    return output


class SmallMnistNoDropoutWithPassThrough(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropoutWithPassThrough, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pt1 = torch.nn.Identity()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pt2 = torch.nn.Identity()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.pt1(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu2(self.pt2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(self.conv2_drop(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnistNoDropout(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class SoftMaxAvgPoolModel(torch.nn.Module):
    def __init__(self):
        super(SoftMaxAvgPoolModel, self).__init__()
        self.sfmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(3)

    def forward(self, inp):
        x = self.sfmax(inp)
        return self.avgpool(x)


class ModelWithStandaloneOps(nn.Module):
    def __init__(self):
        super(ModelWithStandaloneOps, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.myquant = QcQuantizeStandalone(activation_bw=8, round_mode=MAP_ROUND_MODE_TO_PYMO['nearest'],
                                            quant_scheme=QuantScheme.post_training_tf_enhanced,
                                            is_symmetric=False, data_type=QuantizationDataType.int)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = self.relu1(self.maxpool1(self.conv1(inputs[0])))
        x = self.conv2(x)
        x = self.myquant(x)
        x = self.relu2(self.maxpool2(self.conv2_drop(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithTwoInputsOneToAdd(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputsOneToAdd, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(10, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.add = elementwise_ops.Add()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x1 = self.relu1_b(self.maxpool1_b(self.conv1_b(x1)))

        x = self.add(x1, x2)

        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class PreluModel(nn.Module):
    def __init__(self):
        super(PreluModel, self).__init__()
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(x)


class PixelShuffleModel(nn.Module):
    def __init__(self):
        super(PixelShuffleModel, self).__init__()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(x)


class FakeMultiOutputOp(torch.autograd.Function):
    """
    This function helps create a custom onnx op to simulate a 5 output tensor onnx node
    Note: the forward pass has some tensor computation to prevent torch onnx export from removing onnx node.
    """

    @staticmethod
    def symbolic(g, inp):
        """
        Magic method that helps with exporting a custom ONNX node
        """
        return g.op('aimet_torch::FakeMultiOutputOp', inp, outputs=5)

    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        return x * 2, x * 4, x * 8, x * 16, x * 32

    @staticmethod
    def backward(ctx, _grad):  # pylint: disable=arguments-differ
        raise NotImplementedError()


class ModuleWith5Output(torch.nn.Module):
    def forward(self, x):
        return FakeMultiOutputOp.apply(x)


class ModelWith5Output(torch.nn.Module):
    def __init__(self):
        super(ModelWith5Output, self).__init__()
        self.cust = ModuleWith5Output()

    def forward(self, x):
        return self.cust(x)


class ModuleListModel(nn.Module):
    def __init__(self):
        super(ModuleListModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
        self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                          nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                          StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8,
                                                                 activation_bw=8,
                                                                 round_mode='nearest',
                                                                 quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                 data_type=QuantizationDataType.int),
                                          nn.ModuleList([nn.MaxPool2d(2), nn.PReLU()])])

    def forward(self, *inputs):
        return self.layers[2](inputs[0])


class TestQuantizationSimStaticGrad:
    def test_is_leaf_module_positive(self):
        """With an actual leaf module"""
        conv1 = nn.Conv2d(1, 10, 5)
        assert utils.is_leaf_module(conv1)

    # -------------------------------------------
    def test_is_leaf_module_negative(self):
        """With a non-leaf module"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        net = Net()
        model = net.to(torch.device('cpu'))

        assert not utils.is_leaf_module(model)

    # -------------------------------------------------------------
    def test_is_quantizable_module_positive(self):
        """With a quantizable module"""
        conv1 = nn.Conv2d(1, 10, 5)
        assert QuantizationSimModel._is_quantizable_module(conv1)

    # -------------------------------------------------------------
    def test_is_quantizable_module_negative(self):
        """With a non-quantizable module"""
        conv1 = StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8, round_mode='nearest',
                                       quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       data_type=QuantizationDataType.int)
        assert not QuantizationSimModel._is_quantizable_module(conv1)

    # ------------------------------------------------------------
    def verify_quantization_wrappers(self, original_model, quantized_model,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced):
        """Test utility to determine if quantization wrappers were added correctly"""

        # All leaf modules in the original model
        orig_modules = [(name, module) for name, module in original_model.named_modules()
                        if len(list(module.modules())) == 1]

        # All QcQuantized modules in the quantized model
        quant_modules = [(name, module) for name, module in quantized_model.named_modules()
                         if isinstance(module, QcQuantizeWrapper)]

        for i, orig_mod_tuple in enumerate(orig_modules):
            quant_mod_tuple = quant_modules[i]

            # If the original model has any QcQuantizeWrapper node, skip the checks
            if '_module_to_wrap' in orig_mod_tuple[0]:
                continue

            # Checks --------
            # Modules should be in the same order as before
            assert orig_mod_tuple[0] == quant_mod_tuple[0], "Quantized model has a incorrectly named module"

            if quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]:
                # For every leaf module in the first list, there is a corresponding QcQuantized model in the second list
                assert str(type(quant_mod_tuple[1]).__name__) == 'StaticGridQuantWrapper'

            # Each QcQuantized model has 1 child, that is the same type as the corresponding module in the original list
            assert len(list(quant_mod_tuple[1].modules())) == 2
            child = list(quant_mod_tuple[1].modules())[1]
            logger.debug("{} -> {}".format(type(child), type(orig_mod_tuple[1])))
            assert type(child) == type(orig_mod_tuple[1])

    # --------------------------------------------------------
    def test_add_quantization_wrappers_one_deep(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = Net()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # ------------------------------------------------------
    def test_add_quantization_wrappers_with_preexisting_quantization_layers(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                    round_mode='stochastic',
                                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                    data_type=QuantizationDataType.int)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12), in_place=True)

        # Add wrappers again, expect to be a nop
        sim._add_quantization_wrappers(model, num_inout_tensors={}, default_data_type=QuantizationDataType.int)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_two_deep(self):
        """With a one-deep model"""

        class SubNet(nn.Module):
            def __init__(self):
                super(SubNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = SubNet()
                self.fc1 = nn.Linear(320, 50)
                self.SubNet2 = SubNet()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequentials(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        print(model)
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))
        print(sim.model)
        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequential_two_deep(self):
        """With a one-deep model"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Sequential(
                        nn.Conv2d(1, 10, 5),
                        nn.ReLU(),
                        nn.Conv2d(1, 10, 5)
                    ),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist(self):
        """With a one-deep model using ModuleList"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                             StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                                    round_mode='nearest',
                                                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                    data_type=QuantizationDataType.int)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_two_deep(self):
        """With a two-deep model using ModuleList"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
                self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                                  nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                                  StaticGridQuantWrapper(nn.Conv2d(1, 10, 5), weight_bw=8,
                                                                         activation_bw=8,
                                                                         round_mode='nearest',
                                                                         quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                                         data_type=QuantizationDataType.int)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        print(model)
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12))
        print(sim.model)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_with_layers_to_ignore(self):
        """With a two-deep model using ModuleList and layers_to_ignore"""

        model = ModuleListModel()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 12, 12))
        layers_to_exclude = [sim.model.layers_deep[1], sim.model.layers_deep[3], sim.model.layers_deep[5]]
        sim.exclude_layers_from_quantization(layers_to_exclude)
        print(sim.model)

        assert isinstance(sim.model.layers[0]._module_to_wrap, nn.Linear)
        assert isinstance(sim.model.layers[1]._module_to_wrap, nn.Linear)
        assert isinstance(sim.model.layers[2]._module_to_wrap, nn.Conv2d)

        assert isinstance(sim.model.layers_deep[0][0]._module_to_wrap, nn.BatchNorm2d)
        assert isinstance(sim.model.layers_deep[0][1]._module_to_wrap, nn.ReLU)

        # layer ignored, so no QcQuantizeWrapper wrapper
        assert isinstance(sim.model.layers_deep[1], nn.Linear)
        assert isinstance(sim.model.layers_deep[2]._module_to_wrap, nn.Linear)

        # layer ignored, so no QcQuantizeWrapper wrapper
        assert isinstance(sim.model.layers_deep[3], nn.Conv2d)

        # non leaf layer specified, check that all submodules had wrappers removed
        assert isinstance(sim.model.layers_deep[5][0], nn.MaxPool2d)
        assert isinstance(sim.model.layers_deep[5][1], nn.PReLU)

        assert len(sim._excluded_layer_names) == 4
        assert 'layers_deep.1' in sim._excluded_layer_names
        assert 'layers_deep.3' in sim._excluded_layer_names
        assert 'layers_deep.5.0' in sim._excluded_layer_names
        assert 'layers_deep.5.1' in sim._excluded_layer_names

        sim.export('./data/', 'modulelist_with_layers_to_ignore', dummy_input=torch.rand(1, 3, 12, 12))
        with open("./data/modulelist_with_layers_to_ignore.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert 'layers_deep.1' in encodings['excluded_layers']
        assert 'layers_deep.3' in encodings['excluded_layers']
        assert 'layers_deep.5.0' in encodings['excluded_layers']
        assert 'layers_deep.5.1' in encodings['excluded_layers']
        assert len(encodings['excluded_layers']) == 4

    # -------------------------------------------
    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)

        sim.model.conv1_a.param_quantizers['weight'].use_symmetric_encodings = True
        sim.model.conv1_a.param_quantizers['weight'].use_strict_symmetric = True

        # Quantize
        sim.compute_encodings(forward_pass, None)
        model(*dummy_input)

        # save encodings
        sim.export('./data/', 'two_input_model', dummy_input)

    # -------------------------------------------
    def test_model_with_two_inputs_fp16(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                   default_data_type=QuantizationDataType.float)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # save encodings
        sim.export('./data', 'two_input_model_fp16', dummy_input)
        encoding_file_path_pytorch = os.path.join('./data', 'two_input_model_fp16' + '_torch' + '.encodings')
        load_encodings_to_sim(sim, encoding_file_path_pytorch)

        layer = sim.model.conv1_a
        if isinstance(layer, QcQuantizeWrapper):
            for input_quantizer in layer.input_quantizers:
                if input_quantizer.enabled:
                    assert input_quantizer.encoding is None
                    assert input_quantizer.data_type is QuantizationDataType.float
            for output_quantizer in layer.output_quantizers:
                if output_quantizer.enabled:
                    assert output_quantizer.encoding is None
                    assert output_quantizer.data_type is QuantizationDataType.float
            for name in layer.param_quantizers:
                if layer.param_quantizers[name].enabled:
                    assert layer.param_quantizers[name].encoding is None
                    assert layer.param_quantizers[name].data_type is QuantizationDataType.float

    # -------------------------------------------
    def test_model_with_two_inputs_one_to_add(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        model = ModelWithTwoInputsOneToAdd()

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        assert 2 == len(sim.model.add.input_quantizers)
        assert not sim.model.add.input_quantizers[0].enabled
        assert not sim.model.add.input_quantizers[1].enabled

        sim.model.add.input_quantizers[1].enabled = True

        # Quantize
        sim.compute_encodings(forward_pass, None)
        print(sim)

        # save encodings
        sim.export('./data/', 'two_input_model_one_with_add', dummy_input)
        onnx_model = onnx_model = onnx.load('./data/two_input_model_one_with_add.onnx')
        for node in onnx_model.graph.node:
            if node.name == 'add':
                break
        assert 2 == len(node.input)
        model_input_tensor = node.input[1]

        with open("./data/two_input_model_one_with_add.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert model_input_tensor in encodings['activation_encodings']
        activation_enc = encodings['activation_encodings'][model_input_tensor]
        assert isinstance(activation_enc[0]['offset'], int)
        param_enc = encodings["param_encodings"]["conv1_a.weight"]
        assert isinstance(param_enc[0]['offset'], int)

    def test_export_unified_encoding_format(self):
        """ test export functionality on ResNet18 """

        resnet18 = models.resnet18()
        resnet18.eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet18, dummy_input=dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.export('./data/', 'resnet18', dummy_input)
        with open('./data/resnet18.encodings') as json_file:
            encoding_data = json.load(json_file)
            print(encoding_data)

        activation_keys = list(encoding_data["activation_encodings"].keys())

        assert len(activation_keys) == 24
        assert isinstance(encoding_data["activation_encodings"][activation_keys[0]], list)

        param_keys = list(encoding_data["param_encodings"].keys())
        assert "conv1.weight" in param_keys
        assert isinstance(encoding_data["param_encodings"]["conv1.weight"], list)

    def test_export_to_torch_script(self):
        """ test export functionality on ResNet18 """

        resnet50 = models.resnet50()
        resnet50.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet50, dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.export('./data/', 'resnet50', dummy_input, onnx_export_args=None)
        with open('./data/resnet50.encodings') as json_file:
            encoding_data = json.load(json_file)

        activation_keys = list(encoding_data["activation_encodings"].keys())
        assert isinstance(encoding_data["activation_encodings"][activation_keys[0]], list)

        param_keys = list(encoding_data["param_encodings"].keys())
        assert param_keys[0] == "conv1.weight"
        assert isinstance(encoding_data["param_encodings"]["conv1.weight"], list)

        with open('./data/resnet50.encodings.yaml') as yaml_file:
            encoding_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        activation_keys = list(encoding_data["activation_encodings"].keys())
        assert activation_keys[0] == "103"
        assert isinstance(encoding_data["activation_encodings"]["103"], list)

        param_keys = list(encoding_data["param_encodings"].keys())
        assert param_keys[0] == "conv1.weight"
        assert isinstance(encoding_data["param_encodings"]["conv1.weight"], list)

    # -------------------------------------------

    def test_export_to_onnx(self):
        """Exporting encodings and model"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()
        sim = QuantizationSimModel(model, dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.model.conv1_a.param_quantizers['weight'].encoding.max = 10
        sim.model.conv1_a.output_quantizers[0].encoding.max = 30

        # save encodings
        sim.export('./data/', 'two_input_model', dummy_input)

        # check the encodings
        with open('./data/two_input_model.encodings', 'r') as fp:
            encodings = json.load(fp)

            activation_encodings = encodings['activation_encodings']
            param_encodings = encodings['param_encodings']
            assert 16 == len(activation_encodings)
            assert 7 == len(param_encodings['conv1_a.weight'][0])
            assert 10 == param_encodings['conv1_a.weight'][0]['max']

        with open('./data/two_input_model.encodings.yaml', 'r') as fp_yaml:
            encodings = yaml.load(fp_yaml, Loader=yaml.FullLoader)

            activation_encodings = encodings['activation_encodings']
            param_encodings = encodings['param_encodings']
            assert 16 == len(activation_encodings)
            assert 7 == len(param_encodings['conv1_a.weight'][0])
            assert 10 == param_encodings['conv1_a.weight'][0]['max']

        # check the exported model
        loaded_model = torch.load('./data/two_input_model.pth')
        loaded_model(torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))

    # -------------------------------------------
    def test_no_fine_tuning_tf_enhanced(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(sim.model, None)

    @pytest.mark.cuda
    def test_multi_gpu_qat(self):
        """"""
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        class OneConvLayerModel(nn.Module):
            def __init__(self):
                super(OneConvLayerModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=2)

            def forward(self, x):
                x = self.conv1(x)
                return x

        model = OneConvLayerModel().to('cuda:0')
        model = model.to('cuda:0')

        dummy_input = torch.ones(2, 1, 3, 3).to('cuda:0')
        dummy_input[1] += 1

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=dummy_input)

        original_weight = sim.model.conv1._module_to_wrap.weight

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1._module_to_wrap.weight.grad

        sim.model = torch.nn.DataParallel(sim.model)

        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))

        weight = sim.model.module.conv1._module_to_wrap.weight

        assert torch.allclose(output_multi_gpu, output_single_gpu)
        assert torch.allclose(original_weight, weight)

        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        assert torch.allclose(sim.model.module.conv1._module_to_wrap.weight.grad, grad_single_gpu)

    # -------------------------------------------
    def test_input_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizers[0].enabled = False
                module.input_quantizers[0].enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert not sim.model.conv1.output_quantizers[0].encoding
        assert sim.model.conv1.input_quantizers[0].encoding

        print(sim.model.conv1.input_quantizers[0])
        print(sim.model.conv1.output_quantizers[0])

    # -------------------------------------------
    def test_input_and_output_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizers[0].enabled = True
                module.input_quantizers[0].enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        assert sim.model.conv1.output_quantizers[0].encoding
        assert sim.model.conv1.input_quantizers[0].encoding

        print(sim.model.conv1.input_quantizers[0])
        print(sim.model.conv1.output_quantizers[0])

    def test_quantizing_models_with_funtional_add_ops(self):
        """
        Testing models with add functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = ya + yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_module_add_ops(self):
        """
        Testing models with add functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)
                self.add1 = elementwise_ops.Add()
                self.add2 = elementwise_ops.Add()

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = self.add1(ya, yb)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = self.add2(ya, yb)
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert not sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.add1.output_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_add_followed_by_split(self):
        """
        Testing models with add functional ops followed by a split
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv4a.input_quantizers[0].enabled
        assert sim.model.conv4b.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_add_followed_by_add(self):
        """
        Testing models with add functional ops followed by a split and then another add.
        This is similar to the resnet architecture where there are no ops on the residual connection
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5, padding=2)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = x
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv4a.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_model_with_input_leading_to_add(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.conv1(x1)
                x = x + x2
                return self.softmax(x)

        model = Net().eval()
        _ = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                 dummy_input=(torch.rand(1, 3, 28, 28), torch.rand(1, 3, 24, 24)))

    def test_quantizing_models_with_mul_ops(self):
        """
        Testing models with elementwise multiply functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya * yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya * yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_div_ops(self):
        """
        Testing models with elementwise division functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya / yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya / yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_quantizing_models_with_concat_ops(self):
        """
        Testing models with concat functional ops
        :return:
        """

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 10, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 10, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 10, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 10, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 28, 28))

        assert sim.model.conv3.input_quantizers[0].enabled
        assert sim.model.conv5.input_quantizers[0].enabled

        print(sim)

    def test_no_finetuning_tf(self):
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 28, 28))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

    def test_per_layer_bitwidths(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 1, 28, 28))
        assert isinstance(sim.model.conv1, QcQuantizeWrapper)
        sim.model.conv1.set_output_bw(16)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(model, None)

        assert 16 == sim.model.conv1.output_quantizers[0].bitwidth
        assert 8 == sim.model.conv2.output_quantizers[0].bitwidth

    # -------------------------------------------
    def test_with_standalone_ops(self):

        model = ModelWithStandaloneOps()
        dummy_input = torch.rand(1, 1, 28, 28)

        sim = QuantizationSimModel(model=model, dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

        # Save encodings
        sim.export("./data/", "encodings_with_standalone_ops", dummy_input)
        with open('./data/encodings_with_standalone_ops.encodings') as json_file:
            encoding_data = json.load(json_file)
        # in onnx definition tensor 16 is output of Reshape, to be ignored
        assert "32" not in encoding_data["activation_encodings"].keys()

    # -------------------------------------------------------------------------------
    def test_layers_to_ignore(self):
        """ Test the  capability to skip quantizing the layers specified by the user"""

        model = SmallMnist()

        dummy_input=torch.rand(1, 1, 28, 28)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        layers_to_ignore = [sim.model.conv1, sim.model.fc2]
        sim.exclude_layers_from_quantization(layers_to_ignore)

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # Check
        assert isinstance(sim.model.conv1, nn.Conv2d)
        assert not isinstance(sim.model.conv2, nn.Conv2d)
        assert isinstance(sim.model.fc2, nn.Linear)

        # export and check encodings file has excluded layers listed as string
        sim.export('./data', 'excluded_layers', dummy_input, propagate_encodings=True)

        with open('./data/excluded_layers.encodings') as f:
            encodings = json.load(f)
            assert 2 == len(encodings['excluded_layers'])

    def check_quant_params(self, model_layer, loaded_model_layer, check_weights):
        output_encoding1 = model_layer.output_quantizers[0].encoding
        output_encoding2 = loaded_model_layer.output_quantizers[0].encoding

        assert model_layer.output_quantizers[0].bitwidth == loaded_model_layer.output_quantizers[0].bitwidth
        assert output_encoding1.max == output_encoding2.max
        assert output_encoding1.min == output_encoding2.min
        assert output_encoding1.delta == output_encoding2.delta
        assert output_encoding1.offset == output_encoding2.offset

        if model_layer.param_quantizers:
            assert next(iter(model_layer.param_quantizers.values())).bitwidth == \
                   next(iter(loaded_model_layer.param_quantizers.values())).bitwidth

        assert model_layer.output_quantizers[0].round_mode == loaded_model_layer.output_quantizers[0].round_mode
        assert model_layer.output_quantizers[0].quant_scheme == loaded_model_layer.output_quantizers[0].quant_scheme

        if check_weights:
            assert np.allclose(model_layer._module_to_wrap.weight.detach().numpy(),
                               loaded_model_layer._module_to_wrap.weight.detach().numpy())

    def test_save_and_load(self):

        model = ModelWithStandaloneOps()

        sim = QuantizationSimModel(model, dummy_input=torch.rand(32, 1, 28, 28))

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)

        # Run some inferences - mimic using a forward pass
        sim.model.eval()
        dummy_input = torch.randn((32, 1, 28, 28))
        output_before_save = sim.model(dummy_input)

        # Save quantized model
        torch.save(sim.model, './data/xx')

        loaded_model = torch.load('./data/xx')
        loaded_model.eval()
        output_after_load = loaded_model(dummy_input)

        self.check_quant_params(sim.model.conv1, loaded_model.conv1, True)
        self.check_quant_params(sim.model.conv2, loaded_model.conv2, True)
        self.check_quant_params(sim.model.conv2_drop, loaded_model.conv2_drop, False)
        self.check_quant_params(sim.model.fc2, loaded_model.fc2, True)

        assert np.allclose(output_before_save.detach().numpy(),
                           output_after_load.detach().numpy())

    def test_ste_gradient_math_tensors(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = [-0.25, -0.25]
        c_enc_max = [1.0, 1.0]
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]])
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_1, grad_out_1)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]])
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]])
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_2, grad_out_2)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]])
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)
        assert np.allclose(expected_grad_3, grad_out_3)

    @pytest.mark.cuda
    def test_ste_gradient_math_tensors_cuda(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = [-0.25, -0.25]
        c_enc_max = [1.0, 1.0]
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]]).cuda()

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]]).cuda()
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]]).cuda()
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_1, grad_out_1)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]]).cuda()
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]]).cuda()
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_2, grad_out_2)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]]).cuda()
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]]).cuda()
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)
        assert torch.allclose(expected_grad_3, grad_out_3)

    def test_ste_gradient_math(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = -0.25
        c_enc_max = 1.0
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]])
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]])
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]])
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]])
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)

        assert np.allclose(expected_grad_1, grad_out_1)
        assert np.allclose(expected_grad_2, grad_out_2)
        assert np.allclose(expected_grad_3, grad_out_3)

    def test_changing_param_quantizer_settings(self):
        """ Test that changing param quantizer settings takes effect after computing encodings is run """
        torch.random.manual_seed(10)
        model = SmallMnist()

        # Skew weights of conv1
        old_weight = model.conv1.weight.detach().clone()
        model.conv1.weight = torch.nn.Parameter(old_weight + .5 * torch.abs(torch.min(old_weight)), requires_grad=False)

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))

        # Check that no encoding is present for param quantizer
        assert not sim.model.conv1.param_quantizers['weight'].encoding

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)
        asym_min = sim.model.conv1.param_quantizers['weight'].encoding.min
        asym_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        assert 8 == sim.model.conv1.param_quantizers['weight'].encoding.bw

        # Check that offset is still symmetric
        assert sim.model.conv1.param_quantizers['weight'].encoding.offset == -128.0

        # Change param quantizer to symmetric and new bitwidth
        sim.model.conv1.param_quantizers['weight'].use_symmetric_encodings = False
        sim.model.conv1.param_quantizers['weight'].bitwidth = 4
        sim.compute_encodings(dummy_forward_pass, None)
        sym_min = sim.model.conv1.param_quantizers['weight'].encoding.min
        sym_max = sim.model.conv1.param_quantizers['weight'].encoding.max
        assert 4 == sim.model.conv1.param_quantizers['weight'].encoding.bw

        # Check that offset is not relatively symmetric
        assert not sim.model.conv1.param_quantizers['weight'].encoding.offset in [-127, -128]

        # Check that mins and maxes have been recomputed
        assert not asym_min == sym_min
        assert not asym_max == sym_max

    def test_compute_encodings_on_subset_of_modules(self):
        """ Test that computing encodings on a subset of modules causes remaining quantized modules to be set to
            passThrough mode. """

        def dummy_forward_pass(model, _):
            conv1_out = model.conv1(torch.randn((1, 1, 28, 28)))
            relu1_out = model.relu1(conv1_out)

        model = SmallMnist()
        model.eval()
        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))
        sim.compute_encodings(dummy_forward_pass, None)
        for name, module in sim.model.named_modules():
            if isinstance(module, StaticGridQuantWrapper):
                assert QcQuantizeOpMode.ACTIVE == module._mode
                if name == 'relu1':
                    assert module.output_quantizers[0].enabled
                elif name in ['conv2', 'conv2_drop', 'relu2', 'relu3', 'dropout', 'fc2', 'log_softmax']:
                    assert not module.output_quantizers[0].enabled

    def test_connected_graph_is_none(self):
        """ Test that an assertion is thrown when connected graph is not able to be built. """

        def raise_trace_error(_self, _model, _inputs):
            raise torch.jit.TracingCheckError(None, None)

        model = SmallMnist()
        model.eval()
        with unittest.mock.patch.object(ConnectedGraph, '__init__', raise_trace_error):
            with unittest.mock.patch.object(ConnectedGraph, '__del__', lambda _self: None):
                with pytest.raises(AssertionError):
                    _ = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28))

    def test_rnn_quantization(self):
        """ Test quantizing a model with rnn layer """
        model = SingleLayerRNNModel()
        dummy_input = torch.randn(10, 1, 3)

        sim = QuantizationSimModel(model, dummy_input)
        assert isinstance(sim.model.rnn, QcQuantizeRecurrent)

    def test_quantizing_qc_quantize_module(self):
        """ Test that qc_quantize_module is identified as not quantizable """
        qc_quantize_module = QcQuantizeRecurrent(torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1), weight_bw=16,
                                                 activation_bw=16, quant_scheme=QuantScheme.post_training_tf,
                                                 round_mode='nearest', data_type=QuantizationDataType.int)
        assert not QuantizationSimModel._is_quantizable_module(qc_quantize_module)

    def test_export_recurrent_model(self):
        """ Test export functionality with recurrent models """
        # models = [TwoLayerBidirectionaRNNModel(), TwoLayerBidirectionalLSTMModel(), TwoLayerBidirectionalGRUModel()]
        models = [TwoLayerBidirectionalLSTMModel()]
        dummy_input = torch.randn(10, 1, 3)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        for model in models:
            sim = QuantizationSimModel(model, dummy_input)

            # Quantize
            sim.compute_encodings(forward_pass, None)

            # Edit part of weights tensor to compare with original model before and after removal of quantize module
            with torch.no_grad():
                sim.model.recurrent.weight_ih_l0[0][0] = 1
            edited_weight = sim.model.recurrent.weight_ih_l0.detach().clone()

            # Check that edited weight is different than original weight in module_to_quantize
            assert not torch.equal(edited_weight, sim.model.recurrent.module_to_quantize.weight_ih_l0)

            sim.export('./data', 'recurrent_save', dummy_input)
            exported_model = torch.load('./data/recurrent_save.pth')

            # Check that weight from quantized module was copied to original module successfully
            assert isinstance(exported_model.recurrent, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
            assert torch.equal(edited_weight, exported_model.recurrent.weight_ih_l0)

            with open('./data/recurrent_save.encodings') as f:
                encodings = json.load(f)
                # verifying the encoding against default eAI HW cfg
                # activation encoding (input only w/o cell state) -- x_l0, h_l0, x_l1 & h_l1
                assert 8 == len(encodings['activation_encodings'])
                # param encoding (weight only w/o bias)  -- W_l0, R_l0, W_l1 & R_l1
                assert 4 == len(encodings['param_encodings'])

            os.remove('./data/recurrent_save.pth')
            os.remove('./data/recurrent_save.onnx')
            os.remove('./data/recurrent_save.encodings')

    def test_export_dict_input_output(self):
        """ test export functionality on dictionary input and output """

        dummy_input = {'a': torch.randn(1, 10, 10, 10),
                       'b': torch.randn(1, 10, 10, 10),
                       'c': torch.randn(1, 10, 10, 10)}

        model = InputOutputDictModel()

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.model.mul1.output_quantizers[0].enabled = True
        sim.model.mul2.output_quantizers[0].enabled = True
        sim.model.mul3.output_quantizers[0].enabled = True

        # Quantize
        sim.compute_encodings(forward_pass, None)

        o_names = ['ab', 'bc', 'ca']
        sim.export('./data/', 'dict_input_output_model', dummy_input,
                   onnx_export_args=OnnxExportApiArgs(input_names=list(dummy_input.keys()),
                                                      output_names=o_names,
                                                      opset_version=12
                                                      ))
        with open('./data/dict_input_output_model.encodings') as json_file:
            encoding_data = json.load(json_file)
            print(encoding_data)

        onnx_model = onnx.load('./data/dict_input_output_model.onnx')
        for inp in onnx_model.graph.input:
            assert inp.name in ['a', 'b', 'c']
        for exp, act in zip(o_names, onnx_model.graph.output):
            assert exp == act.name
        for tensor_name in encoding_data["activation_encodings"].keys():
            assert tensor_name in o_names

    def test_compute_encoding_fp16(self):
        """
        Test encodings generated for fp16
        """
        dummy_input = {'a': torch.randn(1, 10, 10, 10),
                       'b': torch.randn(1, 10, 10, 10),
                       'c': torch.randn(1, 10, 10, 10)}

        model = InputOutputDictModel()

        sim = QuantizationSimModel(model, default_output_bw=16, default_param_bw=16, dummy_input=dummy_input,
                                   default_data_type=QuantizationDataType.float)

        quantizer = sim.model.mul1.input_quantizers[0]
        enc_dict = sim._create_encoding_dict(encoding=None, quantizer=quantizer, propagate_encodings=False)
        assert enc_dict['dtype'] == 'float'
        assert enc_dict['bitwidth'] == 16
        assert 'min' not in enc_dict
        assert 'max' not in enc_dict
        assert 'scale' not in enc_dict
        assert 'offset' not in enc_dict
        assert 'is_symmetric' not in enc_dict

    def test_mapping_encoding_for_torch_module_with_multiple_onnx_ops(self):
        """
         Test the input and output encoding map to input/output at subgraph level when atorch module generates
         multiple onnx ops i.e. a sub-graph
        """

        dummy_input = torch.randn(1, 4, 256, 512)
        model = SoftMaxAvgPoolModel()

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.model.sfmax.output_quantizers[0].enabled = True
        sim.model.sfmax.input_quantizers[0].enabled = True
        sim.model.avgpool.output_quantizers[0].enabled = True
        sim.model.avgpool.input_quantizers[0].enabled = True
        sim.compute_encodings(forward_pass, None)
        sim.export('./data', 'sfmaxavgpool_model', dummy_input)

        with open('./data/sfmaxavgpool_model.encodings') as json_file:
            encoding_data = json.load(json_file)

        assert not set(encoding_data["activation_encodings"].keys()).symmetric_difference(('4', '9', 't.1'))

    def test_transformer_mask_override_tf(self):
        """
        test logic to override mask for a custom block with mask op for tf mode
        :return:
        """
        torch.manual_seed(10)

        class AttnBlock(nn.Module):

            def __init__(self):
                super(AttnBlock, self).__init__()

                self.add = elementwise_ops.Add()
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.add(x1, x2)
                return self.softmax(x)

        class DummyAttnBlockModel(nn.Module):
            def __init__(self):
                super(DummyAttnBlockModel, self).__init__()
                self.block = AttnBlock()

            def forward(self, x1, x2):
                return self.block(x1, x2)

        # update data input to reflect range at add -10000 to ~16.xx
        # this results in max being mapped to zero when econding grid is computed with 8 bit for mask add
        dummy_input = (torch.FloatTensor(32, 1, 100, 100).uniform_(-6000, 15),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-5000, 17))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        # use some dummy custom block type
        model = DummyAttnBlockModel()
        from aimet_common.defs import QuantScheme
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, dummy_input=dummy_input)
        sim.compute_encodings(forward_pass, None)

        old_encoding_min = sim.model.block.add.output_quantizers[0].encoding.min
        old_encoding_max = sim.model.block.add.output_quantizers[0].encoding.max

        print("old encoding min = ", old_encoding_min)
        print("old encoding max = ", old_encoding_max)
        assert int(old_encoding_min) == -11013
        assert int(old_encoding_max) == 0

        # use override registration function
        transformer_utils.register_attention_mask_override('AttnBlock', 'add')
        sim2 = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, dummy_input=dummy_input)

        # compute encodings again to check override takes effect
        sim2.compute_encodings(forward_pass, None)
        new_encoding_min = sim2.model.block.add.output_quantizers[0].encoding.min
        new_encoding_max = sim2.model.block.add.output_quantizers[0].encoding.max
        print("encoding min = ", new_encoding_min)
        print("encoding max = ", new_encoding_max)

        # validate override
        assert int(new_encoding_min) == -6
        assert int(new_encoding_max) == 17
        assert sim2.model.block.add.output_quantizers[0].encoding.bw == 8

    def test_transformer_mask_override_tf_enhanced(self):
        """
        test logic to override mask for a custom block with mask op for tf enhanced mode
        :return:
        """
        torch.manual_seed(10)

        class AttnBlock(nn.Module):

            def __init__(self):
                super(AttnBlock, self).__init__()

                self.add_2 = elementwise_ops.Add()
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.add_2(x1, x2)
                return self.softmax(x)

        class DummyAttnBlockModel(nn.Module):
            def __init__(self):
                super(DummyAttnBlockModel, self).__init__()
                self.block = AttnBlock()

            def forward(self, x1, x2):
                return self.block(x1, x2)

        # update data input to reflect range at add -10000 to ~16.xx
        # this results in max being mapped to zero when econding grid is computed with 8 bit for mask add
        dummy_input = (torch.FloatTensor(32, 1, 100, 100).uniform_(-6000, 15),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-5000, 17))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        # use some dummy custom block type
        model = DummyAttnBlockModel()
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, dummy_input=dummy_input)
        sim.compute_encodings(forward_pass, None)
        old_encoding_min = sim.model.block.add_2.output_quantizers[0].encoding.min
        old_encoding_max = sim.model.block.add_2.output_quantizers[0].encoding.max

        print("old encoding min = ", old_encoding_min)
        print("old encoding max = ", old_encoding_max)
        assert int(old_encoding_min) == -10974
        assert int(old_encoding_max) == 0

        # use override registration function
        transformer_utils.register_attention_mask_override('AttnBlock', 'add_2')
        sim2 = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, dummy_input=dummy_input)

        # compute encodings again to check override takes effect
        sim2.compute_encodings(forward_pass, None)
        new_encoding_min = sim2.model.block.add_2.output_quantizers[0].encoding.min
        new_encoding_max = sim2.model.block.add_2.output_quantizers[0].encoding.max
        print("encoding min = ", new_encoding_min)
        print("encoding max = ", new_encoding_max)
        assert int(new_encoding_min) == -6
        assert int(new_encoding_max) == 17
        assert sim2.model.block.add_2.output_quantizers[0].encoding.bw == 8

    def test_transformer_mask_override_transformers_tf_enhanced(self):
        """
        test logic to override mask for a DistilBERT, RoBERTa, GPT-2 models.
        :return:
        """
        torch.manual_seed(10)

        class MultiHeadSelfAttention(nn.Module):

            def __init__(self):
                super(MultiHeadSelfAttention, self).__init__()

                self.mask_add = elementwise_ops.Add()
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.mask_add(x1, x2)
                return self.softmax(x)

        class RobertaSelfAttention(nn.Module):

            def __init__(self):
                super(RobertaSelfAttention, self).__init__()

                self.mask_add = elementwise_ops.Add()
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.mask_add(x1, x2)
                return self.softmax(x)

        class Attention(nn.Module):

            def __init__(self):
                super(Attention, self).__init__()

                self.mask_add = elementwise_ops.Add()
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.mask_add(x1, x2)
                return self.softmax(x)

        class DummyAttnBlockModel(nn.Module):
            def __init__(self):
                super(DummyAttnBlockModel, self).__init__()
                self.distilbert_block = MultiHeadSelfAttention()
                self.roberta_block = RobertaSelfAttention()
                self.gpt_block = Attention()

            def forward(self, x1, x2, x3, x4, x5, x6):
                a = self.distilbert_block(x1, x2)
                b = self.roberta_block(x3, x4)
                c = self.gpt_block(x5, x6)
                return a + b + c

        # update data input to reflect range at add -10000 to ~16.xx
        # this results in max being mapped to zero when econding grid is computed with 8 bit for mask add
        dummy_input = (torch.FloatTensor(32, 1, 100, 100).uniform_(-6000, 15),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-5000, 17),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-6700, 15),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-5900, 17),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-6100, 15),
                       torch.FloatTensor(32, 1, 100, 100).uniform_(-5700, 17))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        # use some dummy custom block type
        model = DummyAttnBlockModel()
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, dummy_input=dummy_input)
        sim.compute_encodings(forward_pass, None)
        distil_encoding_min = sim.model.distilbert_block.mask_add.output_quantizers[0].encoding.min
        distil_encoding_max = sim.model.distilbert_block.mask_add.output_quantizers[0].encoding.max

        roberta_encoding_min = sim.model.roberta_block.mask_add.output_quantizers[0].encoding.min
        roberta_encoding_max = sim.model.roberta_block.mask_add.output_quantizers[0].encoding.max

        gpt_encoding_min = sim.model.gpt_block.mask_add.output_quantizers[0].encoding.min
        gpt_encoding_max = sim.model.gpt_block.mask_add.output_quantizers[0].encoding.max

        # check min clamped
        assert int(distil_encoding_min) == -6
        assert int(distil_encoding_max) == 17

        assert int(roberta_encoding_min) == -5
        assert int(roberta_encoding_max) == 15

        assert int(gpt_encoding_min) == -6
        assert int(gpt_encoding_max) == 16

        assert sim.model.distilbert_block.mask_add.output_quantizers[0].encoding.bw == 8
        assert sim.model.roberta_block.mask_add.output_quantizers[0].encoding.bw == 8
        assert sim.model.gpt_block.mask_add.output_quantizers[0].encoding.bw == 8

    def test_encodings_propagation_simple_model(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module
        """
        model = PixelShuffleModel()
        dummy_input = torch.randn(1, 4, 8, 8)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        # verifying the encodings propagation works well while using QcQuantizeRecurrent or QcQuantizeWrapper
        sim = QuantizationSimModel(model, dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings
        sim.export('./data', 'encodings_propagation_false', dummy_input)
        with open('./data/encodings_propagation_false.encodings') as f:
            encodings = json.load(f)
        assert len(encodings['activation_encodings']) == 2
        assert 't.1' in encodings['activation_encodings']
        assert '7' in encodings['activation_encodings']

        # Save encodings again - now with propagate encodings flag enabled
        sim.export('./data', 'encodings_propagation_true', dummy_input, propagate_encodings=True)
        with open('./data/encodings_propagation_true.encodings') as f:
            encodings = json.load(f)
        assert len(encodings['activation_encodings']) == 4
        assert '3' in encodings['activation_encodings']
        assert 'min' not in encodings['activation_encodings']['3']
        assert '4' in encodings['activation_encodings']
        assert 'min' not in encodings['activation_encodings']['4']

        pretty_data = json.dumps(encodings, indent=2)
        print(pretty_data)

    def test_encodings_propagation_lstm_model(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module for LSTM layers
        """
        model = TwoLayerBidirectionalLSTMModel()
        dummy_input = torch.randn(10, 1, 3)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        # verifying the encodings propagation works well while using QcQuantizeRecurrent or QcQuantizeWrapper
        sim = QuantizationSimModel(model, dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # Save encodings
        sim.export('./data', 'encodings_propagation_false', dummy_input)
        with open('./data/encodings_propagation_false.encodings') as f:
            encodings = json.load(f)
        assert len(encodings['activation_encodings']) == 8

        # Save encodings again - now with propagate encodings flag enabled
        sim.export('./data', 'encodings_propagation_true', dummy_input, propagate_encodings=True)
        with open('./data/encodings_propagation_true.encodings') as f:
            encodings = json.load(f)
        assert len(encodings['activation_encodings']) == 25
        # assert '3' in encodings['activation_encodings']
        # assert '4' in encodings['activation_encodings']

        pretty_data = json.dumps(encodings, indent=2)
        print(pretty_data)

    def test_change_quant_scheme_tf_enhanced_to_tf(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, 5)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(forward_pass, None)
        assert sim.model.conv2.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf_enhanced

        tensor = torch.rand(1, 10, 24, 24)
        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 1000
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv2.output_quantizers[0].encoding.max < 1.0

        sim.model.conv2.output_quantizers[0].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(1, 10, 24, 24)
        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 1000
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert sim.model.conv2.output_quantizers[0].encoding.max == pytest.approx(1000.0, rel=0.1)

    def test_change_quant_scheme_tf_enhanced_to_tf_per_channel(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, 5)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim1 = QuantizationSimModel(model, dummy_input)
        sim1.model.conv2.enable_per_channel_quantization()
        sim1.compute_encodings(forward_pass, None)
        assert sim1.model.conv2.param_quantizers['weight'].quant_scheme == QuantScheme.post_training_tf_enhanced

        tensor = torch.rand(20, 10, 5, 5)
        sim1.model.conv2.param_quantizers['weight'].reset_encoding_stats()
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 100
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        sim1.model.conv2.param_quantizers['weight'].compute_encoding()
        assert sim1.model.conv2.param_quantizers['weight'].encoding[0].max < 1.0

        sim1.model.conv2.param_quantizers['weight'].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(20, 10, 5, 5)
        sim1.model.conv2.param_quantizers['weight'].reset_encoding_stats()
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        tensor[0, 0, 0, 0] = 100
        sim1.model.conv2.param_quantizers['weight'].update_encoding_stats(tensor)
        sim1.model.conv2.param_quantizers['weight'].compute_encoding()
        assert sim1.model.conv2.param_quantizers['weight'].encoding[0].max == pytest.approx(100.0, rel=0.1)

        # Scenario: we set the scheme before a call to compute_encodings
        sim2 = QuantizationSimModel(model, dummy_input)
        sim2.model.conv2.enable_per_channel_quantization()
        sim2.model.conv2.param_quantizers['weight'].quant_scheme = QuantScheme.post_training_tf
        tensor = torch.rand(20, 10, 5, 5)
        tensor[0, 0, 0, 0] = 100
        sim2.model.conv2._module_to_wrap.weight.data = tensor
        sim2.compute_encodings(forward_pass, None)
        assert sim2.model.conv2.param_quantizers['weight'].encoding[0].max == pytest.approx(100.0, rel=0.1)

    def test_conditional_export(self):
        """ Test exporting a model with conditional paths """
        model = SimpleConditional()
        inp = torch.randn(1, 3)
        true_tensor = torch.tensor([1])
        false_tensor = torch.tensor([0])

        def forward_callback(model, _):
            model(inp, true_tensor)
            model(inp, false_tensor)

        qsim = QuantizationSimModel(model, dummy_input=(inp, true_tensor))
        qsim.compute_encodings(forward_callback, forward_pass_callback_args=None)
        qsim._export_conditional('./data', 'simple_cond', dummy_input=(inp, false_tensor),
                                 forward_pass_callback=forward_callback, forward_pass_callback_args=None)

        with open('./data/simple_cond.encodings') as f:
            encodings = json.load(f)
            # verifying the encoding against default eAI HW cfg
            # activation encodings -- input, linear1 out, prelu1 out, linear2 out, prelu2 out, softmax out
            assert 6 == len(encodings['activation_encodings'])
            # param encoding -- linear 1 & 2 weight & bias, prelu 1 & 2 weight
            assert 4 == len(encodings['param_encodings'])

        if os.path.exists('./data/simple_cond.pth'):
            os.remove('./data/simple_cond.pth')
        if os.path.exists('./data/simple_cond.onnx'):
            os.remove('./data/simple_cond.onnx')
        if os.path.exists('./data/simple_cond.encodings'):
            os.remove('./data/simple_cond.encodings')
        if os.path.exists('./data/simple_cond.encodings.yaml'):
            os.remove('./data/simple_cond.encodings.yaml')

    def test_export_prelu_encoding_and_check_load_encodings(self):
        """ Test that prelu weight is exported correctly """
        model = PreluModel()
        dummy_input = torch.rand(1, 3, 8, 8)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        sim.export('./data', 'prelu_model', dummy_input=dummy_input)
        with open('./data/prelu_model.encodings') as json_file:
            encoding_data = json.load(json_file)
        assert 'prelu.weight' in encoding_data['param_encodings'].keys()

        output = sim.model(copy.deepcopy(dummy_input))
        del sim

        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        encoding_file_path_pytorch = os.path.join('./data', 'prelu_model' + '_torch' + '.encodings')
        load_encodings_to_sim(sim, encoding_file_path_pytorch)

        layer = sim.model.prelu
        if isinstance(layer, QcQuantizeWrapper):
            for input_quantizer in layer.input_quantizers:
                if input_quantizer.enabled:
                    assert input_quantizer.encoding is not None
            for output_quantizer in layer.output_quantizers:
                if output_quantizer.enabled:
                    assert output_quantizer.encoding is not None
            for name in layer.param_quantizers:
                if layer.param_quantizers[name].enabled:
                    assert layer.param_quantizers[name].encoding is not None

        output1 = sim.model(copy.deepcopy(dummy_input))
        assert sum(output1.flatten() - output.flatten()) == 0.0

    def test_fetching_varaible_from_module(self):
        class Model(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

            def forward(self, inputs):
                x = self.conv(inputs)
                return x

        in_channels = out_channels = 10
        model = Model(in_channels, out_channels, 5)
        dummy_input = torch.rand(1, 10, 24, 24)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(forward_pass, None)

        assert hasattr(sim.model.conv, 'in_channels')
        assert getattr(sim.model.conv, 'in_channels') == 10
        assert hasattr(sim.model.conv, 'out_channels')
        assert getattr(sim.model.conv, 'out_channels') == 10

        del sim

    def test_nested_input(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.add = elementwise_ops.Add()

            def forward(self, x, y):
                return self.add(x, y)

        model = Model()

        length = 8
        shape = (256, 256)
        inputs_a = [torch.rand(shape) for _ in range(length)]
        inputs_b = [torch.rand(shape) for _ in range(length)]

        sim = QuantizationSimModel(model, (inputs_a, inputs_b))

        def forward_pass(model, args):
            model.eval()
            model(inputs_a, inputs_b)
        sim.compute_encodings(forward_pass, None)

        assert sim.model.add.input_quantizers[0].encoding is not None
        assert sim.model.add.input_quantizers[1].encoding is not None


class TestQuantizationSimLearnedGrid:

    # -------------------------------------------------------------------------------

    def test_replace_quant_wrapper(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.fc1 = nn.Linear(640, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                return x

        net = Net()
        model = net.to(torch.device('cpu'))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=torch.randn(1, 1, 12, 12))
        assert isinstance(sim.model.conv1, StaticGridQuantWrapper)
        assert isinstance(sim.model.fc1, StaticGridQuantWrapper)
        for _, module in sim.model._modules.items():
            module.input_quantizers[0].enabled = False
            module.output_quantizers[0].enabled = False
            module.param_quantizers['weight'].enabled = False
        sim._replace_quantization_wrapper(sim.model, device='cpu')

        assert isinstance(sim.model.conv1, LearnedGridQuantWrapper)
        assert isinstance(sim.model.fc1, LearnedGridQuantWrapper)

    @pytest.mark.cuda
    def test_multi_gpu_qat(self):
        """"""
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        class OneConvLayerModel(nn.Module):
            def __init__(self):
                super(OneConvLayerModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 5, kernel_size=2)

            def forward(self, x):
                x = self.conv1(x)
                return x

        model = OneConvLayerModel().to('cuda:0')
        model = model.to('cuda:0')

        dummy_input = torch.ones(2, 1, 3, 3).to('cuda:0')
        dummy_input[1] += 1

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        original_weight = sim.model.conv1._module_to_wrap.weight

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                output = model(torch.randn((2, 1, 3, 3)).to('cuda:0'))
            return output

        sim.compute_encodings(forward_pass, None)
        output_single_gpu = sim.model(copy.deepcopy(dummy_input))
        loss = output_single_gpu.flatten().sum()
        loss.backward()
        grad_single_gpu = sim.model.conv1._module_to_wrap.weight.grad

        sim.model = torch.nn.DataParallel(sim.model)

        output_multi_gpu = sim.model(copy.deepcopy(dummy_input))

        weight = sim.model.module.conv1._module_to_wrap.weight

        assert torch.allclose(output_multi_gpu, output_single_gpu)
        assert torch.allclose(original_weight, weight)

        loss = output_multi_gpu.flatten().sum()
        loss.backward()
        assert torch.allclose(sim.model.module.conv1._module_to_wrap.weight.grad, grad_single_gpu)

    def test_copy_properties_between_wrappers_when_inp_output_is_1(self):
        conv1 = nn.Conv2d(3, 10, kernel_size=5)
        post_training_module = StaticGridQuantWrapper(conv1, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=False,
                                                      is_output_quantized=False, activation_bw=8, weight_bw=8)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2

        post_training_module.input_quantizers[0].enabled = True
        post_training_module.input_quantizers[0].encoding = encodings
        post_training_module.param_quantizers['weight'].enabled = False
        post_training_module.param_quantizers['bias'].enabled = False
        dummy_input = torch.randn(1, 3, 12, 12)
        sim = QuantizationSimModel(conv1, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)
        # sim.model.conv1.input_quantizer.enabled = True
        trainable_module = sim._construct_and_initialize_trainable_wrapper(post_training_module, device='cpu')

        assert trainable_module.output_quantizers[0].use_symmetric_encodings == False
        assert trainable_module.output_quantizers[0].enabled == False
        assert trainable_module.input0_encoding_min.item() == -5.0

    def test_copy_properties_for_elementwise_aimet_add_op(self):

        class ElementwiseAdd(nn.Module):
            def __init__(self):
                super(ElementwiseAdd, self).__init__()
                self.add = elementwise_ops.Add()

            def forward(self, *inputs):
                return self.add(inputs[0], inputs[1])

        add = ElementwiseAdd()
        post_training_module = StaticGridQuantWrapper(add, round_mode='nearest',
                                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=False,
                                                      is_output_quantized=False, activation_bw=8, weight_bw=8,
                                                      num_inputs=2, num_outputs=1)
        encodings = libpymo.TfEncoding()
        encodings.bw, encodings.max, encodings.min, encodings.delta, encodings.offset = 8, 5.0, -1.0, 1, 0.2

        for inp_quantizer in post_training_module.input_quantizers:
            inp_quantizer.enabled = True
            inp_quantizer.encoding = encodings

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))
        sim = QuantizationSimModel(add, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        trainable_module = sim._construct_and_initialize_trainable_wrapper(post_training_module, device='cpu')

        assert trainable_module.output_quantizers[0].use_symmetric_encodings == False
        assert trainable_module.output_quantizers[0].enabled == False
        assert trainable_module.input0_encoding_min.item() == -1.0
        assert trainable_module.input1_encoding_max.item() == 5.0

    def test_qc_trainable_wrapper(self):
        torch.manual_seed(0)
        conv1 = nn.Conv2d(1, 32, kernel_size=5)

        trainable_module = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                   is_symmetric=False, is_output_quantized=True, activation_bw=8,
                                                   weight_bw=8, device='cpu', data_type=QuantizationDataType.int)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 3
        encodings.min = -2
        encodings.delta = 1
        encodings.offset = 0.2
        trainable_module.input_quantizers[0].enabled = True
        trainable_module.input_quantizers[0].encoding = encodings

        trainable_module.param_quantizers['weight'].enabled = True
        trainable_module.param_quantizers['weight'].encoding = encodings
        trainable_module.param_quantizers['bias'].enabled = True
        trainable_module.param_quantizers['bias'].encoding = encodings

        trainable_module.output_quantizers[0].enabled = True
        trainable_module.output_quantizers[0].encoding = encodings

        inp = torch.rand((1, 1, 5, 5), requires_grad=True)
        out = trainable_module(inp)
        optimizer = torch.optim.SGD(trainable_module.parameters(), lr=0.05, momentum=0.5)
        loss = out.flatten().sum()
        loss.backward()
        optimizer.step()

        # Checking if encoding min max have changed
        assert not trainable_module.input0_encoding_min.item() == -2.0
        assert not trainable_module.input0_encoding_max.item() == 3.0

        assert not trainable_module.output0_encoding_min.item() == -2.0
        assert not trainable_module.output0_encoding_max.item() == 3.0

        assert not trainable_module.weight_encoding_min.item() == -2.0
        assert not trainable_module.weight_encoding_max.item() == 3.0

        assert not trainable_module.bias_encoding_min.item() == -2.0
        assert not trainable_module.bias_encoding_max.item() == 3.0

    def test_qc_trainable_wrapper_for_model_with_multiple_inputs_with_one_add(self):
        # NOTE: Use asymmetric quantization for parameter, which have gradients both encoding min/max
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        config_file_path = "/tmp/quantsim_config.json"
        with open(config_file_path, "w") as f:
            json.dump(quantsim_config, f)

        dummy_input = (torch.rand(32, 1, 100, 100), torch.rand(32, 10, 22, 22))

        def forward_pass(sim_model, _):
            sim_model.eval()
            with torch.no_grad():
                sim_model(*dummy_input)

        model = ModelWithTwoInputsOneToAdd()

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file=config_file_path)
        # Enable input parameters to add (multiple input parameter exist)
        sim.model.add.input_quantizers[0].enabled = True
        sim.model.add.input_quantizers[1].enabled = True

        sim.compute_encodings(forward_pass, forward_pass_callback_args=None)

        assert len(sim.model.add.input_quantizers) == 2

        out = sim.model(*dummy_input)
        for _, params in sim.model.named_parameters():
            assert params.grad is None

        optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)
        loss = out.flatten().sum()
        loss.backward()
        optimizer.step()
        # All parameters should have a gradient
        for params in sim.model.parameters():
            assert params.grad is not None

        if os.path.exists(config_file_path):
            os.remove(config_file_path)

    def test_set_and_get_encoding_properties(self):
        torch.manual_seed(0)
        conv1 = nn.Conv2d(1, 32, kernel_size=5)

        trainable_module = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                   is_symmetric=False, is_output_quantized=True, activation_bw=8,
                                                   weight_bw=8, device='cpu', data_type=QuantizationDataType.int)

        trainable_module.input_quantizers[0].enabled = True

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2

        # If enabled encoding cannot be None
        with pytest.raises(RuntimeError):
            trainable_module.input_quantizers[0].encoding = None
        trainable_module.input_quantizers[0].encoding = encodings

        # Check if quantizer.encoding is accessible
        print(trainable_module.input_quantizers[0].encoding)
        assert trainable_module.input0_encoding_min.data == encodings.min

    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""

        dummy_input = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   dummy_input=dummy_input)

        # Quantize
        sim.compute_encodings(forward_pass, None)

        assert sim.model.conv1_a.output_quantizers[0].encoding

        # self.assertAlmostEqual(sim.model.conv1_a.output_quantizer.encoding.min,
        #                        sim.model.conv1_a.output0_encoding_min.data)
        # self.assertAlmostEqual(sim.model.conv1_a.output_quantizer.encoding.max,
        #                        sim.model.conv1_a.output0_encoding_max.data)

        forward_pass(sim.model, None)
        print(sim)

    def test_memory_profiler(self):
        """ test using memory profiler """
        # checks PyTorch version before importing torch.profiler (introduced in version 1.8.0), for
        # older versions this test is passthrough.
        if version.parse(torch.__version__) >= version.parse("1.8"):

            from torch.profiler import profile, ProfilerActivity
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(3, 10, 5, 5)
                    self.conv2 = nn.Conv2d(10, 20, 5, 5)
                    self.conv3 = nn.Conv2d(20, 40, 5, 5)

                def forward(self, input):
                    x = self.conv1(input)
                    x = torch.nn.functional.relu(x)
                    x = self.conv2(x)
                    x = torch.nn.functional.relu(x)
                    x = self.conv3(x)
                    x = torch.nn.functional.relu(x)
                    return x

            model = Model()
            from aimet_torch.model_preparer import prepare_model
            model = prepare_model(model)
            in_tensor = torch.rand(32, 3, 224, 224)

            def forward_pass(model, args):
                model(in_tensor)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                       dummy_input=in_tensor)
            sim.compute_encodings(forward_pass, None)

            print("Starting")
            sim.model.train()

            with profile(activities=[ProfilerActivity.CPU],
                         profile_memory=True, record_shapes=True) as prof:

                for _ in range(1):
                    out = sim.model(in_tensor)
                    out = out.sum().backward()

            memory_stats = [event for event in prof.key_averages() if event.key == '[memory]'][0]
            assert abs(memory_stats.cpu_memory_usage) < 1.1 * (100 * (10 ** 6))

            print(memory_stats.cpu_memory_usage)

    def test_accumulator_overflow(self):

        model = models.resnet18(pretrained=True)
        model = model.eval()
        layer, range_used = check_accumulator_overflow(model, 8, 32)

        assert layer == 'layer4.1.conv1'

        # self.assertAlmostEqual(100 * range_used, 0.263623, places=3)

    def test_export_prelu_encoding_and_check_load_encodings(self):
        """ Test that prelu weight is exported correctly """
        model = PreluModel()
        dummy_input = torch.rand(1, 3, 8, 8)
        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        sim.export('./data', 'prelu_model', dummy_input=dummy_input)
        with open('./data/prelu_model.encodings') as json_file:
            encoding_data = json.load(json_file)
        assert 'prelu.weight' in encoding_data['param_encodings'].keys()

        output = sim.model(copy.deepcopy(dummy_input))
        del sim

        sim = QuantizationSimModel(model, dummy_input=dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        encoding_file_path_pytorch = os.path.join('./data', 'prelu_model' + '_torch' + '.encodings')
        load_encodings_to_sim(sim, encoding_file_path_pytorch)

        layer = sim.model.prelu
        if isinstance(layer, QcQuantizeWrapper):
            for input_quantizer in layer.input_quantizers:
                if input_quantizer.enabled:
                    assert input_quantizer.encoding is not None
            for output_quantizer in layer.output_quantizers:
                if output_quantizer.enabled:
                    assert output_quantizer.encoding is not None
            for name in layer.param_quantizers:
                if layer.param_quantizers[name].enabled:
                    assert layer.param_quantizers[name].encoding is not None

        output1 = sim.model(copy.deepcopy(dummy_input))
        assert sum(output1.flatten() - output.flatten()) == 0.0

    def test_load_encodings_multi_input_multi_output_model(self):
        net = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=4, default_output_bw=4)

        sim.model.cust.output_quantizers[0].enabled = False
        sim.compute_encodings(evaluate, dummy_input)

        sim.export('./data/', 'module_with_5_output', dummy_input,
                   onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                   propagate_encodings=False)

        del sim

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=4, default_output_bw=4)
        sim.model.cust.output_quantizers[0].enabled = False
        encoding_file_path_pytorch = os.path.join('./data', 'module_with_5_output' + '_torch' + '.encodings')
        load_encodings_to_sim(sim, encoding_file_path_pytorch)

        layer = sim.model.cust
        if isinstance(layer, QcQuantizeWrapper):
            for input_quantizer in layer.input_quantizers:
                if input_quantizer.enabled:
                    assert input_quantizer.encoding is not None
            for output_quantizer in layer.output_quantizers:
                if output_quantizer.enabled:
                    assert output_quantizer.encoding is not None
            for name in layer.param_quantizers:
                if layer.param_quantizers[name].enabled:
                    assert layer.param_quantizers[name].encoding is not None

    def test_inplace_modification_with_relu(self):
        """
        Test custom function behavior with view+in-place (relu)
         (output of custom function is view when returned as-is)
        """

        class ModelWithInPlaceReLU(torch.nn.Module):
            """ A model with in-place ReLU. Use this model for unit testing purposes.
                Expected inputs: 1 input, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithInPlaceReLU, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.relu(x)
                return x

        # Test with both QAT and QAT-range learning.
        input_shape = (1, 3, 8, 8)
        dummy_input = [torch.randn(*input_shape)]
        model = ModelWithInPlaceReLU().eval()
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]

        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme, default_param_bw=4,
                                             default_output_bw=4)
            quant_sim.compute_encodings(evaluate, dummy_input)
            quant_sim.model(*dummy_input)

    def test_inplace_modification_with_add(self):
        """
        Test custom function behavior with view+in-place (add)
         (output of custom function is view when returned as-is)
        """

        class ModelWithInPlaceAdd(torch.nn.Module):
            """ A model with in-place Add. Use this model for unit testing purposes.
                Expected inputs: 2 inputs, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithInPlaceAdd, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)
                self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                y = self.conv2(inputs[1])
                x += y
                return x

        # Test with both QAT and QAT-range learning.
        input_shape = (1, 3, 8, 8)
        dummy_input = [torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithInPlaceAdd().eval()
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]

        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme, default_param_bw=4,
                                             default_output_bw=4)
            quant_sim.compute_encodings(evaluate, dummy_input)
            quant_sim.model(*dummy_input)

    def test_multi_output_onnx_op(self):
        """
        Test mapping and exporting of output encodings for multiple output onnx op.
        """

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        net = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)

        sim = QuantizationSimModel(net, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   default_param_bw=4, default_output_bw=4)

        sim.model.cust.output_quantizers[0].enabled = False
        sim.compute_encodings(evaluate, dummy_input)

        sim.export('./data/', 'module_with_5_output', dummy_input,
                   onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                   propagate_encodings=False)
        with open('./data/module_with_5_output.encodings') as json_file:
            activation_encodings = json.load(json_file)['activation_encodings']
            assert '7' not in activation_encodings
            assert set(['8', '9', '10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple(self):
        """

        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        cust_model = CustModelV1Simple()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                         default_param_bw=8, default_output_bw=8)

        quant_sim.compute_encodings(evaluate, dummy_input)

        print(quant_sim)
        print(quant_sim.model)

        quant_sim.export('./data/', 'cust_v1_simple', dummy_input,
                         onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                         propagate_encodings=True)
        with open('./data/cust_v1_simple.encodings') as json_file:
            activation_encodings = json.load(json_file)['activation_encodings']
            assert set(['10', '11', 't.1']).issubset(activation_encodings.keys())

    def test_custom_op_simple_v2(self):
        """

        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        cust_model = CustomOpV2()

        input_shape = (1, 10, 24, 24)
        dummy_input = torch.randn(*input_shape)

        output = cust_model(dummy_input)

        quant_sim = QuantizationSimModel(cust_model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                         default_param_bw=4, default_output_bw=4)

        quant_sim.compute_encodings(evaluate, dummy_input)

        print(quant_sim)
        print(quant_sim.model)

        a, b, c, d, e = quant_sim.model(dummy_input)

        quant_sim.export('./data/', 'cust_v2_simple', dummy_input,
                         onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                         propagate_encodings=False)

        with open('./data/cust_v2_simple.encodings') as json_file:
            activation_encodings = json.load(json_file)['activation_encodings']
            assert set(['13', '17', '21', '4', '6', 't.1']).issubset(activation_encodings.keys())

        module_names = { module_name for module_name, _ in cust_model.named_modules()}
        onnx_model = onnx.load('./data/cust_v2_simple.onnx')

        for node in onnx_model.graph.node:
            if not node.name.startswith('.'):
                name = node.name.split('#')[0]
                assert '.'.join(name.split('.')[:-1]) in module_names
        onnx.checker.check_model(onnx_model)

    def test_quant_roi_model(self):
        roi_model = RoiModel(height=7, width=7, scale=0.25)
        x = torch.rand(1, 1, 6, 6)
        rois = torch.tensor([ [0, -2.0, -2.0, 22.0, 22.0], ])
        dummy_input = (x, rois)
        torch.onnx.export(roi_model, dummy_input, './roi.onnx', enable_onnx_checker=False, opset_version=11)
        sim = QuantizationSimModel(roi_model, dummy_input=dummy_input)
        for q in sim.model.roi.input_quantizers:
            q.enabled = False

        def forward_pass(model, _):
            model.eval()
            with torch.no_grad():
                model(*dummy_input)

        sim.compute_encodings(forward_pass, None)
        sim.export("./data/", "roi_model", dummy_input,
                   onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),  propagate_encodings=True)

        with open('./data/roi_model.encodings') as json_file:
            encodings = json.load(json_file)['activation_encodings']
            assert set(['5', '6', '7', '9', '11',]).issubset(encodings.keys())
            assert 'scale' in encodings['11'][0]

class CustModelV1Simple(torch.nn.Module):
    def __init__(self):
        super(CustModelV1Simple, self).__init__()
        self.cust = CustomOp()

    def forward(self, x):
        k1, k2 = self.cust(x)
        return k1, k2


class CustomOp(torch.nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.size = 8
        self.mul1 = elementwise_ops.Multiply()
        self.mul2 = elementwise_ops.Multiply()

    def forward(self, x):
        y = self.mul1(x, self.size)
        z = self.mul2(x, 5)
        return y, z


class CustModelV2Simple(torch.nn.Module):
    def __init__(self):
        super(CustModelV2Simple, self).__init__()
        self.cust = CustomOpV2()

    def forward(self, x):
        k1, k2, k3, k4, k5 = self.cust(x)
        return k1, k2, k3, k4, k5


class CustomOpV2(torch.nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.size = 8
        self.mul = elementwise_ops.Multiply()
        self.add = elementwise_ops.Add()
        self.sub = elementwise_ops.Subtract()
        self.clamp = Clamp()

    def forward(self, x):
        y = self.mul(x, self.size)
        z = x * 5 # functional op
        a = self.clamp(z)
        b = self.add(y, z)
        c = self.sub(y, z)
        return y, z, a, b, c


class Clamp(torch.nn.Module):
    """ Custom module for a functional clamp"""

    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for add op
        """
        return x.clamp(0)
