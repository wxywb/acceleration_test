import torch
import time
import onnxruntime
import platform
import snakemd
import os
import json
import numpy as np
import subprocess
from datetime import date
from snakemd import Document, Table

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
 
        return json.JSONEncoder.default(self, obj)

def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        return subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        return subprocess.check_output(command, shell=True).strip()
    return ""

def default_inference_torch(model, inp):
    out = model(inp)
    return out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def default_inference_onnx(session, dummy_input):
    ort_inputs = {session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = session.run(None, ort_inputs)
    return ort_outs

class EnvironmentInspector():
    def __init__(self):
        pass

    def get_env_info_cpu(self):
        ver = str(torch.__version__)
        os_ver = '{}_{}'.format(platform.system(), platform.release())
        cpu_type = get_processor_info().decode("utf-8") 
        info = {'type':'cpu', 'os':os_ver, 'device':cpu_type, 'torch_version':ver}
        return info

    def get_env_info_gpu(self):
        devices_torch = torch.cuda.get_device_name()
        ver = str(torch.__version__)
        os_ver = '{}_{}'.format(platform.system(), platform.release)
        info = {'type':'gpu', 'os':os_ver, 'device':devices_torch, 'torch_version':ver}

    def get_env_info(self):
        if torch.cuda.is_available():
            return self.get_env_info_gpu()
        else: 
            return self.get_env_info_cpu()

class FormatTest():
    def __init__(self, taskname, operator_name, inf_iter=5):
        self.taskname = taskname
        self.operator_name = operator_name
        self.input_shape = None
        self.inf_iter = inf_iter

        self.torch_out = []
        self.session_out = []
        self.envir_inspector = EnvironmentInspector()

        self._meta = None
        if os.path.exists('./meta.json') is True:
            fw = open('./meta.json')
            self._meta = json.loads(fw.read())
        else:
            self._meta = self.init_meta() 

    def init_meta(self):
        meta_info = {}
        meta_info['taskname'] = self.taskname
        meta_info['operator_name'] = self.operator_name
        meta_info['records'] = []
        return meta_info

    def start_eval(self):        
        today = date.today()
        d4 = today.strftime("%Y-%b-%d")
        self.latest_record = {}
        self.latest_record['time'] = d4
        env_info = self.envir_inspector.get_env_info()
        for k in env_info:
            self.latest_record[k] = env_info[k]
        
    def set_input_shape(self, shape): 
        self.input_shape = shape
        self.dummy_input = torch.randn(*shape)
        self.latest_record['inf_shape'] = str(shape)

    def set_tfms(self, tfms):
        self.tfms = tfms

    def set_model(self, model):
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.latest_record['params'] = params
        

    def inference_torch(self, inference_hook = default_inference_torch):
        self.model.eval()
        elapsed_times = []
        with torch.no_grad():
            for i in range(self.inf_iter):
                start_time = time.time()
                out = inference_hook(self.model, self.dummy_input)
                self.torch_out.extend(out)
                end_time = time.time()
                elapsed_times.append(end_time - start_time)
        self.torch_out.extend(out)
        tt = 0
        for et in elapsed_times:
            tt += et
        avg_time = tt / len(elapsed_times)
        self.latest_record['torch_inf_time'] = avg_time
        return avg_time

    def inference_onnx(self, inference_hook = default_inference_onnx):
        elapsed_times = []

        for i in range(self.inf_iter):
            start_time = time.time()
            ort_outs = inference_hook(self.session, self.dummy_input)
            end_time = time.time()
            elapsed_times.append(end_time - start_time)
        self.session_out.extend(ort_outs)
        tt = 0
        for et in elapsed_times:
            tt += et
        avg_time = tt / len(elapsed_times)
        self.latest_record['onnx_inf_time'] = avg_time
        return avg_time

    def to_onnx(self , input_names=["input"], output_names=["output"],opset_version=12):
        torch.onnx.export(self.model,
                  self.dummy_input,
                  'out.onnx',
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=opset_version)
        file_stats = os.stat('out.onnx')
        file_size = file_stats.st_size / (1000 * 1000)
        self.latest_record['onnx_size'] = str(file_stats.st_size / (1000 * 1000)) + 'MB'
        self.latest_record['opset'] = opset_version

        self.session = onnxruntime.InferenceSession("out.onnx")

    def check_output(self, rtol = 1e-03, atol=1e-05):
        self._rtol = rtol
        self._atol = atol
        check_pass = True
        for tout, oout in zip(self.torch_out, self.session_out):
            try: 
                p = True
                np.testing.assert_allclose(to_numpy(tout), oout, rtol=rtol, atol=atol)
            except Exception as e:
                p = False
            check_pass = check_pass and p

        if check_pass is True:
            self.latest_record['numerical_test'] = 'PASS'
        else :
            self.latest_record['numerical_test'] = 'FAILED'
        return check_pass

    def dump(self):
        meta = self._meta
        doc = snakemd.new_doc("tmp_README")
        doc.add_header("{} / {}".format(self.taskname, self.operator_name))
        p = doc.add_paragraph(
          """
          evaluation records.
          """
        )

        p.insert_link("{}/{}".format(self.taskname, self.operator_name), "https://towhee.io/{}/{}".format(self.taskname.replace('-','_'),self.operator_name.replace('-','_')))

        column_names = ['time', 'type', 'os', 'device', 'inf_shape', 'params', 'opset', 'onnx_size', 'numerical_test', 'torch_inf_time', 'onnx_inf_time'] 

        rows = []
        for row in meta['records']:
            row_data = [row[col_item] for col_item in column_names]
            rows.append(row_data)

        latest_row =  [self.latest_record[col_item] for col_item in column_names]

        rows.append(latest_row)
        doc.add_table(
            column_names,
            rows,
            [Table.Align.LEFT for i in range(len(column_names))],
            0
        )
        meta['records'] = rows
        doc.output_page('./')
        os.system('mv tmp_README.md README.md')
        with open('meta.json', 'w') as fw:
            fw.write(json.dumps(meta, indent=4, cls=NpEncoder))


        



