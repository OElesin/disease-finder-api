import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
import torchvision
import pickle


input_shape = (3, 224, 224)
model_onnx_path = "torch_model.onnx"

model = pickle.load(open('../api/model_dir/plant-disease-model-cpu.pt', 'rb'))

dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(
    model, dummy_input,
    model_onnx_path, verbose=False
)
