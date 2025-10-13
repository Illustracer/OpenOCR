import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
import numpy as np

from onnxruntime.quantization import (
    quantize_dynamic,
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

from onnxruntime.transformers.float16 import convert_float_to_float16


def check_model_ops(model_path):
    model = onnx.load(model_path)
    op_types = set()

    for node in model.graph.node:
        op_types.add(node.op_type)

    print("Model contains these operation types:")
    for op_type in sorted(op_types):
        print(f"  - {op_type}")

    return op_types


class MyDataReader(CalibrationDataReader):
    def __init__(self, img_list, cfg, batch_size=1):
        """
        Args:
            img_list: 图像列表, 可以是numpy数组列表或图像路径列表
            batch_size: 批处理大小
        """
        from openrec.preprocess import create_operators, transform
        from tools.infer_rec import build_rec_process

        self.cfg = cfg
        self.img_list = img_list
        self.batch_size = batch_size
        self.transform_func = transform
        transforms, _ = build_rec_process(self.cfg)
        self.ops = create_operators(transforms, self.cfg["Global"])
        self.current_idx = 0

    def get_next(self):
        if self.current_idx >= len(self.img_list):
            return None

        try:
            batch_data = []
            max_width, max_height = 0, 0
            # 计算当前批次的结束索引
            end_idx = min(self.current_idx + self.batch_size, len(self.img_list))
            # 处理当前批次的图像
            for img_idx in range(self.current_idx, end_idx):
                img = self.img_list[img_idx]

                # 如果是路径，读取图像
                if isinstance(img, str):
                    with open(img, "rb") as f:
                        img_data = f.read()
                        data = {"image": img_data}
                    # 应用变换（如果提供了变换函数）
                    if self.transform_func and self.ops:
                        data = self.transform_func(data, self.ops[:1])
                        batch = self.transform_func(data, self.ops[1:])
                        processed_img = batch[0]
                    else:
                        # 如果没有变换函数，假设img_data已经是处理好的numpy数组
                        processed_img = img_data
                else:
                    # 如果已经是numpy数组
                    if self.transform_func and self.ops:
                        data = {"image": img}
                        batch = self.transform_func(data, self.ops[1:])
                        processed_img = batch[0]
                    else:
                        processed_img = img

                # 确保是numpy数组
                if not isinstance(processed_img, np.ndarray):
                    processed_img = processed_img.numpy()

                # 获取图像尺寸
                h, w = processed_img.shape[-2:]
                max_width = max(max_width, w)
                max_height = max(max_height, h)
                batch_data.append(processed_img)

            # 创建填充后的批次数据
            padded_batch = np.zeros(
                (len(batch_data), 3, max_height, max_width), dtype=np.float32
            )
            for i, img in enumerate(batch_data):
                h, w = img.shape[-2:]
                padded_batch[i, :, :h, :w] = img

            # 更新索引
            self.current_idx = end_idx
            # 返回字典格式，键名需要与ONNX模型的输入名称匹配
            return {"input": padded_batch}

        except Exception as e:
            print(f"Error processing batch starting at index {self.current_idx}: {e}")
            return None

    def rewind(self):
        """重置数据读取器到开始位置"""
        self.current_idx = 0


def dynamic_quant(model_path, output_path):
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=[
            "MatMul",  # 全连接层
            "LayerNormalization",  # LayerNorm 层
            "Add",  # 加法操作
            "Mul",  # 乘法操作
            "Gather",  # 索引操作
        ],
    )


def static_quant(model_path, output_path, calibration_data_reader):
    model = onnx.load(model_path)
    nodes_to_quantize = [
        node.name 
        for node in model.graph.node 
        if node.name.startswith('/decoder/') and node.op_type == 'MatMul'
    ]

    nodes_to_quantize.extend(
        [
            "/decoder/svtr_encoder/conv1/conv/Conv",
            "/decoder/svtr_encoder/conv2/conv/Conv",
            "/decoder/svtr_encoder/conv3/conv/Conv",
            "/decoder/svtr_encoder/conv4/conv/Conv",
            "/decoder/svtr_encoder/conv1x1/conv/Conv",
            # "/encoder/blocks5/blocks5.0/dw_conv/reparam_conv/Conv",
            # "/encoder/blocks5/blocks5.0/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks5/blocks5.1/dw_conv/reparam_conv/Conv",
            "/encoder/blocks5/blocks5.1/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks5/blocks5.2/dw_conv/reparam_conv/Conv",
            "/encoder/blocks5/blocks5.2/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks5/blocks5.3/dw_conv/reparam_conv/Conv",
            "/encoder/blocks5/blocks5.3/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks5/blocks5.4/dw_conv/reparam_conv/Conv",
            "/encoder/blocks5/blocks5.4/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks6/blocks6.0/dw_conv/reparam_conv/Conv",
            "/encoder/blocks6/blocks6.0/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks6/blocks6.1/dw_conv/reparam_conv/Conv",
            "/encoder/blocks6/blocks6.1/pw_conv/reparam_conv/Conv",
            # "/encoder/blocks6/blocks6.2/dw_conv/reparam_conv/Conv",
            "/encoder/blocks6/blocks6.2/pw_conv/reparam_conv/Conv",
            "/encoder/blocks6/blocks6.3/dw_conv/reparam_conv/Conv",
            "/encoder/blocks6/blocks6.3/pw_conv/reparam_conv/Conv",
        ]
    )

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,    # 激活 int8
        weight_type=QuantType.QInt8,        # 权重 int8，可自动 per-channel
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,  # MinMax(只支持)
        nodes_to_quantize = nodes_to_quantize,
        # op_types_to_quantize=[
        #     "MatMul",  # 全连接层
        #     "Conv",
        # ],
    )

    # 步骤2: 转 FP16，但保护量化节点
    exit()
    model = onnx.load(output_path)
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        # 关键：阻止这些算子转换为 FP16
        op_block_list=[
            'QuantizeLinear',      # 量化节点
            'DequantizeLinear',    # 反量化节点
        ],
    )
    onnx.save(model_fp16, './output/add_dict/model_mixed_final.onnx')


def convert_to_fp16(model_path, output_path):
    """转换模型为FP16"""
    model = onnx.load(model_path)

    # 转换为FP16
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,  # 保持输入输出为FP32
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, output_path)
    print(f"模型已转换为FP16: {output_path}")


def quant_debug(float_model_path, qdq_model_path):
    from onnxruntime.quantization.qdq_loss_debug import (
        compute_weight_error,
        create_weight_matching,
    )

    print("------------------------------------------------\n")
    print("Comparing weights of float model vs qdq model.....")

    matched_weights = create_weight_matching(float_model_path, qdq_model_path)
    weights_error = compute_weight_error(matched_weights)
    for weight_name, err in weights_error.items():
        if err > 30. :
            print(f"Cross model error of '{weight_name}': {err}")


def check_onnx_model(model_path):
    model = onnx.load(model_path)
    # 打印所有节点名称和类型
    for node in model.graph.node:
        if node.op_type in ["Conv"]:
            print(f"Node name: {node.name}, Op type: {node.op_type}")
            # print(f"  Inputs: {node.input}")
            # print(f"  Outputs: {node.output}")


if __name__ == "__main__":
    # 检查模型
    model_path = "./output/add_dict/rec_model_rep.onnx"
    # model_path = "./output/rec_model_rep.onnx"
    ops = check_model_ops(model_path)
    check_onnx_model(model_path)
    # exit()

    # Debug
    # quant_debug(model_path, "./output/rec_model_rep_int8_static.onnx")
    # exit()

    # 直接转换为 FP16
    # output_path = "./output/add_dict/rec_model_rep_fp16.onnx"
    # convert_to_fp16(model_path, output_path)
    # exit()

    # 动态量化
    # output_path = "./output/add_dict/rec_model_rep_int8_dynamic.onnx"
    # dynamic_quant(model_path, output_path)
    # exit()

    # 静态量化
    import yaml
    from tools.utils.utility import get_image_file_list

    yaml_path = "./configs/rec/paddle/v4_rec_add_dict.yml"
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # img_path = "/Users/ght/Projects/Dustbin/QQHodor/resources/wx1507/images"
    img_path = "/Users/ght/Projects/Dustbin/PaddleOCR2Pytorch/doc/imgs_words/ch"
    image_list = get_image_file_list(img_path)
    reader = MyDataReader(image_list, config)

    output_path = "./output/add_dict/rec_model_rep_int8_static.onnx"
    static_quant(model_path, output_path, reader)
