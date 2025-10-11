import os
import sys
import torch
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process as build_rec_post_process


class PPOCRv4RecConverter:
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PPOCRv4RecConverter, self).__init__()
        self.cfg = config

        # build post process
        self.post_process_class = build_rec_post_process(
            self.cfg["PostProcess"], self.cfg["Global"]
        )

        # build model
        # for rec algorithm
        char_num = self.post_process_class.get_character_num()
        self.cfg["Architecture"]["Decoder"]["out_channels"] = char_num
        self.net = build_rec_model(self.cfg["Architecture"])

        # load paddle model
        self.load_paddle_model(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_model(self, weights_path):
        para_state_dict, opti_state_dict = self.read_paddle_weights(weights_path)
        para_state_dict = self.del_invalid_state_dict(para_state_dict)
        self.load_paddle_weights([para_state_dict, opti_state_dict])

    def save_pytorch_weights(self, weights_path):
        checkpoint = {
            "state_dict": self.net.state_dict(),
        }
        try:
            torch.save(
                checkpoint,
                weights_path,
                _use_new_zipfile_serialization=False,
            )
        except:  # pylint: disable=bare-except
            torch.save(
                checkpoint, weights_path
            )  # _use_new_zipfile_serialization=False for torch>=1.6.0
        print("model is saved: {}".format(weights_path))

    def convert_param_name(self, paddle_name):
        """参数名称转换"""
        conversions = [
            ("._mean", ".running_mean"),
            ("._variance", ".running_var"),
            ("backbone.", "encoder."),
            ("head.ctc_head.", "decoder."),
            ("head.ctc_encoder.encoder.", "decoder.svtr_encoder."),
        ]

        converted = paddle_name
        for old, new in conversions:
            converted = converted.replace(old, new)
        return converted

    def read_paddle_weights(self, weights_path):
        import paddle

        para_state_dict = paddle.load(weights_path)
        opti_state_dict = None
        return para_state_dict, opti_state_dict

    def del_invalid_state_dict(self, para_state_dict):
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        # NOTE: 这里本质是在删除 NRTRHead
        for i, (k, v) in enumerate(para_state_dict.items()):
            if k.startswith("head.gtc_head."):
                continue
            elif k.startswith("head.before_gtc"):
                continue
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights

        for k, v in para_state_dict.items():
            ptname = k
            ptname = self.convert_param_name(ptname)

            try:
                if (
                    k.endswith("fc1.weight")
                    or k.endswith("fc2.weight")
                    or k.endswith("fc.weight")
                    or k.endswith("qkv.weight")
                    or k.endswith("proj.weight")
                ):
                    self.net.state_dict()[ptname].copy_(torch.Tensor(v.T.cpu().numpy()))
                else:
                    self.net.state_dict()[ptname].copy_(torch.Tensor(v.cpu().numpy()))

            except Exception as e:
                print("exception:")
                print("pytorch: {}, {}".format(ptname, self.net.state_dict()[v].size()))
                print("paddle: {}, {}".format(k, v.shape))
                raise e

        print("model is loaded.")


def find_layers_by_name(model, layer_name):
    """根据类名字符串查找层"""
    found_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == layer_name:
            found_layers.append((name, module))
    return found_layers


def to_onnx(model, dummy_input, dynamic_axes, sava_path="model.onnx"):
    input_axis_name = ["batch_size", "channel", "in_width", "int_height"]
    output_axis_name = ["batch_size", "channel", "out_width", "out_height"]
    torch.onnx.export(
        model.to("cpu"),
        dummy_input,
        sava_path,
        input_names=["input"],
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {axis: input_axis_name[axis] for axis in dynamic_axes},
            "output": {axis: output_axis_name[axis] for axis in dynamic_axes},
        },
    )


if __name__ == "__main__":
    from torchsummary import summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument("--export_dir", type=str)
    parser.add_argument("--export_onnx_model", type=bool, default=False)
    
    args = parser.parse_args()

    with open(args.yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    converter = PPOCRv4RecConverter(config, args.src_model_path)
    summary(converter.net, (3, 48, 320))

    os.makedirs(args.export_dir, exist_ok=True)
    converter.save_pytorch_weights(weights_path=os.path.join(args.export_dir, "ppocr_v4_rec_add_dict.pth"))

    # 导出 onnx
    if args.export_onnx_model:
        # 查找 LearnableRepLayer
        # NOTE: 导出 ONNX 模型需要执行该操作
        rep_layers = find_layers_by_name(converter.net, "LearnableRepLayer")
        for name, layer in rep_layers:
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()
        # summary(converter.net, (3, 48, 320))

        dynamic_axes = [0, 3]
        dummy_input = torch.randn([1, 3, 48, 320], device="cpu")
        save_path = os.path.join(args.export_dir, "rec_model_rep.onnx")
        os.makedirs(args.export_dir, exist_ok=True)
        to_onnx(converter.net, dummy_input, dynamic_axes, save_path)
        print(f"成功导出 ONNX 模型: {save_path}")