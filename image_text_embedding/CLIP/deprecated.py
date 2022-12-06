import towhee
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import onnxruntime
import sys

sys.path.append('../../')
import utils


op = towhee.ops.image_embedding.timm(model_name='resnet50').get_op()
model = op.model

dummy_input = torch.randn(1,3,224,224)
torch_out = model(dummy_input)


torch.onnx.export(model,
                  dummy_input,
                  'out.onnx',
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=12)

tfms = transforms.Compose([
           transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(
              (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
           ])

img = Image.open('example.webp')
img_tensor = tfms(img)

ort_session = onnxruntime.InferenceSession("out.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

