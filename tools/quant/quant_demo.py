import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

import torch

from torch import nn
from openrec.modeling.encoders.rec_lcnetv3 import PPLCNetV3
from openrec.modeling.encoders.rec_lcnetv3_quant import QuantizablePPLCNetV3
from openrec.modeling.decoders.ctc_decoder_quant import QuantizedCTCDecoder, CTCDecoder


def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def quantize_model(model: nn.Module, backend: str, dummy_input) -> None:
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == "fbgemm":
        default_qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        default_qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_weight_observer,
        )

    # 先设置全局配置
    model.qconfig = default_qconfig
    model.fuse_model()  # type: ignore[operator]
    torch.ao.quantization.prepare(model, inplace=True)

    # 如果模型有禁用量化的方法，调用它
    if hasattr(model, "disable_svtr_quantization"):
        model.disable_svtr_quantization()
    if hasattr(model, "disable_lab_quantization"):
        model.disable_lab_quantization()
    if hasattr(model, "disable_quant_layers"):
        model.disable_quant_layers()

    model(dummy_input)
    torch.ao.quantization.convert(model, inplace=True)


def prepare_qat_model(model: nn.Module, backend: str) -> None:
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.train()

    # 设置全局配置
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model.fuse_model(is_qat=True)  # type: ignore[operator]
    torch.ao.quantization.prepare_qat(model, inplace=False)

    # 如果模型有禁用量化的方法，调用它
    if hasattr(model, "disable_svtr_quantization"):
        model.disable_svtr_quantization()
    if hasattr(model, "disable_lab_quantization"):
        model.disable_lab_quantization()
    if hasattr(model, "disable_quant_layers"):
        model.disable_quant_layers()
    print("=== QAT模型准备完成 ===")


def finalize_qat_model(qat_model):
    """
    完成QAT训练后, 转换为最终的量化模型
    """
    print("=== 完成QAT量化 ===")
    qat_model = qat_model.to("cpu")
    qat_model.eval()
    quantized_model = torch.quantization.convert(qat_model, inplace=False)
    print("=== QAT量化完成 ===")
    return quantized_model


def export_onnx_model(quantized_model, dummy_input, dynamic_axes, onnx_path):
    """直接导出量化模型 - 可能不完全支持"""
    input_axis_name = ["batch_size", "channel", "in_width", "int_height"]
    output_axis_name = ["batch_size", "channel", "out_width", "out_height"]
    torch.onnx.export(
        quantized_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,  # 或更高版本
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {axis: input_axis_name[axis] for axis in dynamic_axes},
            "output": {axis: output_axis_name[axis] for axis in dynamic_axes},
        },
    )
    print(f"模型已导出到: {onnx_path}")
    return True


def find_layers_by_name(model, layer_name):
    """根据类名字符串查找层"""
    found_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == layer_name:
            found_layers.append((name, module))
    return found_layers


def debug_quantized_model(model, dummy_input):
    """调试量化模型"""
    with torch.no_grad():
        _ = model(dummy_input)
    print("量化模型前向传播成功")
    return True


def debug_quantization_status(model, prefix=""):
    """递归检查所有模块的量化状态"""
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name

        # 检查 qconfig
        if hasattr(module, "qconfig"):
            print(f"{full_name}: qconfig = {module.qconfig}")

        # 检查自定义量化标志
        if hasattr(module, "quantization_enabled"):
            print(f"{full_name}: quantization_enabled = {module.quantization_enabled}")

        # 检查是否是量化相关的模块
        if any(x in str(type(module)) for x in ["Quant", "DeQuant", "FloatFunctional"]):
            print(
                f"{full_name}: {type(module)} - qconfig = {getattr(module, 'qconfig', 'None')}"
            )


def quant_lcnet():
    backend = "qnnpack"

    # ======== 模型导出定义 =========
    dynamic_axes = [0, 3]
    example_inputs = torch.rand(1, 3, 48, 320).to("cpu")

    # ======== 原始模型定义 =========
    origin_model = PPLCNetV3(scale=0.95)
    rep_layers = find_layers_by_name(origin_model, "LearnableRepLayer")
    for name, layer in rep_layers:
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()
    export_onnx_model(
        origin_model, example_inputs, dynamic_axes, "./output/PPLCNet-v3.onnx"
    )

    # ======== 步骤1: 量化模型定义 =========
    quant_model = QuantizablePPLCNetV3(scale=0.95)
    rep_layers = find_layers_by_name(quant_model, "QuantizableLearnableRepLayer")
    for name, layer in rep_layers:
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()
    _replace_relu(quant_model)

    # ======== 步骤2: 模型量化(静态) =========
    quantize_model(quant_model, backend, example_inputs)
    debug_quantized_model(quant_model, example_inputs)
    export_onnx_model(
        quant_model, example_inputs, dynamic_axes, "./output/PPLCNet-v3-quant.onnx"
    )


def quant_ctc():
    backend = "qnnpack"

    # ======== 模型导出定义 =========
    dynamic_axes = [0, 3]
    example_inputs = torch.rand(1, 120, 8, 32).to("cpu")

    svtr_config = {
        "dims": 120,
        "depth": 2,
        "kernel_size": [1, 3],
        "hidden_dims": 120,
        "use_guide": True,
        "use_pool": False,
        "support_ppocr_v4": True,
    }

    # ======== 原始模型定义 =========
    origin_model = CTCDecoder(
        in_channels=120,
        out_channels=6625,
        mid_channels=None,
        return_feats=False,
        svtr_encoder=svtr_config,
    )
    export_onnx_model(
        origin_model, example_inputs, dynamic_axes, "./output/CTCDecoder.onnx"
    )

    # ======== 步骤1: 量化模型定义 =========
    quant_model = QuantizedCTCDecoder(
        in_channels=120,
        out_channels=6625,
        mid_channels=None,
        return_feats=False,
        svtr_encoder=svtr_config,
    )
    _replace_relu(quant_model)

    # ======== 步骤2: 模型量化 =========
    quantize_model(quant_model, backend, example_inputs)
    debug_quantized_model(quant_model, example_inputs)
    export_onnx_model(
        quant_model, example_inputs, dynamic_axes, "./output/CTCDecoder-quant.onnx"
    )


def quant_rec_model(quant_method="static"):
    import yaml

    from openrec.modeling import build_model as build_rec_model
    from openrec.postprocess import build_post_process as build_rec_post_process

    backend = "qnnpack"
    yaml_path = "./configs/rec/paddle/v4_rec_quant.yml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # build post process
    post_process_class = build_rec_post_process(cfg["PostProcess"], cfg["Global"])

    # ======== 模型导出定义 =========
    dynamic_axes = [0, 3]
    example_inputs = torch.rand(1, 3, 48, 320).to("cpu")

    # ======== 步骤1: 量化模型定义 =========
    char_num = post_process_class.get_character_num()
    cfg["Architecture"]["Decoder"]["out_channels"] = char_num
    quant_model = build_rec_model(cfg["Architecture"])

    # ======== 加载模型参数 ========
    checkpoint = torch.load(
        "./output/ppocr_v4_rec.pth", map_location="cpu", weights_only=True
    )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    quant_model.load_state_dict(state_dict)
    print("=== 模型参数加载完成 ===")

    rep_layers = find_layers_by_name(quant_model, "QuantizableLearnableRepLayer")
    for name, layer in rep_layers:
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()
    _replace_relu(quant_model)

    # ======== 步骤2: 模型量化(静态) =========
    if quant_method == "static":
        quantize_model(quant_model, backend, example_inputs)
        debug_quantized_model(quant_model, example_inputs)
        export_onnx_model(
            quant_model, example_inputs, dynamic_axes, "./output/PPOCR-v4-quant.onnx"
        )

    # ======== 步骤2: 模型量化(QAT) =========
    if quant_method == "qat":
        prepare_qat_model(quant_model, backend)
        # TODO: 增加 QAT 训练
        finalize_qat_model(quant_model)
        debug_quantized_model(quant_model, example_inputs)
        export_onnx_model(
            quant_model, example_inputs, dynamic_axes, "./output/PPLCNet-v3-quant.onnx"
        )


if __name__ == "__main__":
    # quant_lcnet()
    # quant_ctc()
    quant_rec_model(quant_method="static")
