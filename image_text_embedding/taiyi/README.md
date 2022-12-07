# image_text_embedding / taiyi

evaluation records.

| time        | type | os            | device                                     | inf_shape        | params      | opset | onnx_size    | numerical_test | torch_inf_time      | onnx_inf_time      |
| :---------- | :--- | :------------ | :----------------------------------------- | :--------------- | :---------- | :---- | :----------- | :------------- | :------------------ | :----------------- |
| 2022-Dec-07 | cpu  | Darwin_20.5.0 | Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz | [1, 3, 224, 224] | 151277313.0 | 12    | 351.583742MB | PASS           | 0.15347843170166015 | 0.0471135139465332 |