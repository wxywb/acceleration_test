import sys
import towhee
import ipdb

sys.path.append('../../')
import autils
from autils import FormatTest, to_numpy 
from torchvision import transforms
import torch
from torch import nn

formalized_test= FormatTest('image_text_embedding', 'blip')
formalized_test.start_eval()

# our clip implementation use the jit in default which could cause the failure for onnx.
op = towhee.ops.image_text_embedding.blip(model_name='blip_base',modality='image').get_op()

#sanity check begin
model = op.model
img = torch.randn(1,3,224,224) 

class BLIPImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.device = op.device

    def forward(self, img):
        caption = ''
        image_feature = self.model(img, caption, mode='image', device=self.device)[0,0]
        return image_feature
#sanity check end
img_model = BLIPImageModel()
emb = img_model(img)

formalized_test.set_model(img_model)
formalized_test.set_input_shape([1,3,224,224])

tfms = transforms.Compose([
           transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(
              (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
           ])

formalized_test.set_tfms(tfms)
formalized_test.to_onnx()

def default_inference_torch(model, inp):
    out = model(inp)
    return out

formalized_test.inference_torch()

def default_inference_onnx(session, dummy_input):
   ort_inputs = {session.get_inputs()[0].name: to_numpy(dummy_input)}
   ort_outs = session.run(None, ort_inputs)
   return ort_outs[0]
 
formalized_test.inference_onnx(default_inference_onnx)

formalized_test.check_output()
formalized_test.dump()
