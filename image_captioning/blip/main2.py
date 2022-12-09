import torch
import ipdb

ipdb.set_trace()
D_in = 5
H = 10
D_out = 3
features = torch.randn(1,5)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),  # problem with dropout layer
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),  # problem with dropout layer
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid()
)

model.eval()
input_names = ['input_1', 'input_2']
output_names = ['output_1']
#for key, module in model._modules.items():
#    input_names.append("l_{}_".format(key) + module._get_name())
print(input_names)
torch_out = torch.onnx.export(model,
                         features,
                         "onnx_model.onnx",
                         export_params = True,
                         verbose = True,
                         input_names = input_names,
                         output_names = output_names,
                        )
