import sys
import towhee
from PIL import Image

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

formalized_test= FormatTest('image_captioning', 'clipcap')
formalized_test.start_eval()

# our clip implementation use the jit in default which could cause the failure for onnx.

op = towhee.ops.image_captioning.clipcap(model_name='clipcap_coco').get_op()

img = torch.randn(1,3,224,224) 

class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = op.clip_model
        self.model = op.model
        self.clip_model.eval()
        self.model.eval()

    def forward(self, img):
        clip_feat = self.clip_model.encode_image(img)
        prefix_length = 10
        prefix_embed = self.model.clip_project(clip_feat).reshape(1, prefix_length, -1)
        return prefix_embed

#sanity check begin
ipdb.set_trace()
img_model = CaptionModel()
emb = img_model(img)

formalized_test.set_model(img_model)
formalized_test.set_input_shape([1,3,224,224])
im1 = Image.open('dog.png')

tfms = transforms.Compose([
           transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(
              (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
           ])

#emb = img_model(tfms(im1).unsqueeze(0))
emb = img_model(img)

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
