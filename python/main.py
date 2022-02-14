# This is a sample Python script.
import torch
import torchvision
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.to(device)
classifier = model.classifier

# add softmax output
classifier.add_module("output", nn.Softmax(dim=1))

print(model)

model.eval()

example = torch.rand(1, 3, 224, 224).to(device)
traced_script_module = torch.jit.trace(model, example)

from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_traced_model = optimize_for_mobile(traced_script_module)

optimized_traced_model._save_for_lite_interpreter("test.ptl")


