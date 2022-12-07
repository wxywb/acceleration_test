import sys
import towhee

sys.path.append('../../')
import utils
from utils import FormatTest, to_numpy 
from torchvision import transforms

formalized_test= FormatTest('image_embedding', 'timm')
formalized_test.start_eval()

op = towhee.ops.image_embedding.timm(model_name='resnet50').get_op()
formalized_test.set_model(op.model)
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

formalized_test.inference_torch()

def default_inference_onnx(session, dummy_input):
   ort_inputs = {session.get_inputs()[0].name: to_numpy(dummy_input)}
   ort_outs = session.run(None, ort_inputs)
   return ort_outs[0]
 
formalized_test.inference_onnx(default_inference_onnx)

formalized_test.check_output()
formalized_test.dump()


