import sys
import towhee

#from utils import FormatTest, to_numpy 
def init():
    sys.path.append('../../')
    from autils import FormatTest, to_numpy 
    sys.path.pop()
    return FormatTest, to_numpy 
FormatTest, to_numpy = init() 

from torchvision import transforms
import torch
import ipdb
from torch import nn

formalized_test= FormatTest('image_captioning', 'camel')
formalized_test.start_eval()

# our clip implementation use the jit in default which could cause the failure for onnx.
ipdb.set_trace()

op = towhee.ops.image_captioning.camel(model_name='camel_mesh').get_op()

#sanity check begin
print(op.model.__dict__)
model = op.image_model
img = torch.randn(1,3,384,384) 

class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, img):
        out = self.model(img)
        return out

img_model = CaptionModel()
emb = img_model(img)

formalized_test.set_model(img_model)
formalized_test.set_input_shape([1,3,384,384])

tfms = transforms.Compose([
           transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.CenterCrop(384),
           transforms.ToTensor(),
           transforms.Normalize(
              (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
           ])

formalized_test.set_tfms(tfms)
formalized_test.to_onnx()
formalized_test.inference_torch()

def default_inference_onnx(session, dummy_input):
   ort_inputs = {session.get_inputs()[0].name: to_numpy(dummy_input)}
   ort_outs = session.run(None, ort_inputs)
   return ort_outs[0]
 
formalized_test.inference_onnx(default_inference_onnx)

formalized_test.check_output()
formalized_test.dump()
