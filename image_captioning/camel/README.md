# image_captioning / camel

evaluation records.

| time        | type | os            | device                                     | inf_shape        | params    | opset | onnx_size    | numerical_test | torch_inf_time    | onnx_inf_time      |
| :---------- | :--- | :------------ | :----------------------------------------- | :--------------- | :-------- | :---- | :----------- | :------------- | :---------------- | :----------------- |
| 2022-Dec-09 | cpu  | Darwin_20.5.0 | Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz | [1, 3, 384, 384] | 167328912 | 12    | 544.486299MB | FAILED         | 1.694301176071167 | 0.5769555568695068 |