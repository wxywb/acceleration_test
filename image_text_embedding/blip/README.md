# image_text_embedding / blip

evaluation records.

| time        | type | os            | device                                     | inf_shape        | params    | opset | onnx_size    | numerical_test | torch_inf_time      | onnx_inf_time      |
| :---------- | :--- | :------------ | :----------------------------------------- | :--------------- | :-------- | :---- | :----------- | :------------- | :------------------ | :----------------- |
| 2022-Dec-07 | cpu  | Darwin_20.5.0 | Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz | [1, 3, 224, 224] | 223057152 | 12    | 343.334914MB | PASS           | 0.48071789741516113 | 0.2009571075439453 |