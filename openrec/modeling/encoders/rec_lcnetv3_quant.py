import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Union
from openrec.modeling.encoders.rec_lcnetv3 import (
    Act,
    SELayer,
    ConvBNLayer,
    LearnableAffineBlock,
    LearnableRepLayer,
    LCNetV3Block,
    PPLCNetV3,
    make_divisible,
)


def _fuse_modules(
    model: nn.Module, modules_to_fuse: Union[list[str], list[list[str]]], is_qat: Optional[bool], **kwargs: Any
):
    if is_qat is None:
        is_qat = model.training
    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    return method(model, modules_to_fuse, **kwargs)


class PACT(nn.Module):
    """
    PACT: Parameterized Clipping Activation for Quantization
    Paper: https://arxiv.org/abs/1805.06085
    
    将激活值裁剪到 [-alpha, alpha] 范围内，alpha 是可学习参数
    """
    def __init__(self, alpha_init=20.0, learn_alpha=True):
        super(PACT, self).__init__()
        
        # 创建可学习的 alpha 参数
        if learn_alpha:
            self.alpha = nn.Parameter(
                torch.tensor([alpha_init], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'alpha', 
                torch.tensor([alpha_init], dtype=torch.float32)
            )
        self.learn_alpha = learn_alpha

    def forward(self, x):
        """
        将 x 裁剪到 [-alpha, alpha] 范围
        等价于: x = clip(x, -alpha, alpha)
        """
        x = torch.clamp(x, min=-self.alpha, max=self.alpha)
        return x
    
    def extra_repr(self):
        return f'alpha={self.alpha.item():.4f}, learn_alpha={self.learn_alpha}'


class QuantizableAct(Act):
    def __init__(self, act="hard_swish", lr_mult=1.0, lab_lr=0.1):
        super().__init__(act, lr_mult, lab_lr)

        # 替换为量化版本
        self.lab = QuantizableLearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        x = self.act(x)
        x = self.lab(x)
        return x

    def fuse_model(self, is_qat=None):
        """融合激活函数相关操作"""
        if hasattr(self, "lab"):
            self.lab.fuse_model(is_qat)

    def disable_lab_quantization(self):
        """禁用 LAB 相关模块的量化"""
        if hasattr(self, "lab"):
            self.lab.disable_lab_quantization()


class QuantizableSELayer(SELayer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__(channel, reduction, lr_mult)

        # 添加量化友好的乘法操作
        self.mul = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        # 使用量化友好的乘法
        x = self.mul.mul(x, identity)
        return x

    def fuse_model(self, is_qat=None):
        """融合Conv+ReLU操作"""
        _fuse_modules(
            self, [["conv1", "relu"]], inplace=True, is_qat=is_qat
        )


class QuantizableConvBNLayer(ConvBNLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, lr_mult=1.0
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, groups, lr_mult
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def fuse_model(self, is_qat=None):
        """融合Conv+BN操作"""
        _fuse_modules(
            self, [["conv", "bn"]], inplace=True, is_qat=is_qat
        )


class QuantizableLearnableAffineBlock(LearnableAffineBlock):
    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.1):
        super().__init__(scale_value, bias_value, lr_mult, lab_lr)

        # 添加量化友好的运算操作: 不支持导出 ONNX
        # self.mul_add = torch.nn.quantized.FloatFunctional()

        # 添加量化控制
        # self.input_dequant = torch.quantization.DeQuantStub()
        # self.output_quant = torch.quantization.QuantStub()

    def forward(self, x):
        # x = self.input_dequant(x)
        x = self.scale * x + self.bias
        # x = self.output_quant(x)
        return x

    def fuse_model(self, is_qat=None):
        """LearnableAffineBlock通常不需要融合操作"""
        pass

    def disable_lab_quantization(self):
        """禁用 LAB 量化"""
        self.qconfig = None


class QuantizableLearnableRepLayer(LearnableRepLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        num_conv_branches=1,
        lr_mult=1.0,
        lab_lr=0.1,
        use_pact=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            num_conv_branches,
            lr_mult,
            lab_lr,
        )

        # 替换为量化版本的子模块
        # self.lab = QuantizableLearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        # self.act = QuantizableAct(lr_mult=lr_mult, lab_lr=lab_lr)

        # 添加量化友好的加法操作（用于非rep状态）
        self.add = torch.nn.quantized.FloatFunctional()

        # 新增：为 reparam_conv 添加 quant/dequant stub
        self.conv_quant = torch.quantization.QuantStub()
        self.conv_dequant = torch.quantization.DeQuantStub()

        # PACT 预处理层（在量化之前）
        self.use_pact = use_pact
        if use_pact:
            pact_alpha_init = 20.0
            self.pact = PACT(alpha_init=pact_alpha_init)

    def rep(self):
        """调用基类的rep方法"""
        super().rep()

    def forward(self, x):
        # rep后的模型（只考虑这种情况）
        if self.is_repped:
            # 修改：在 reparam_conv 前后添加 quant/dequant
            if self.use_pact:
                x = self.pact(x)
            out = self.conv_quant(x)
            out = self.reparam_conv(out)
            out = self.conv_dequant(out)
            out = self.lab(out)
            if self.stride != 2:
                out = self.act(out)
            return out

        # 非rep状态的量化友好实现
        out = 0
        if self.identity is not None:
            out = self.add.add(out, self.identity(x))
        if self.conv_1x1 is not None:
            out = self.add.add(out, self.conv_1x1(x))
        for conv in self.conv_kxk:
            out = self.add.add(out, conv(x))

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

    def fuse_model(self, is_qat=None):
        """融合rep后的conv操作"""
        if self.is_repped:
            # rep后只有一个conv，无需融合
            pass

    def disable_lab_quantization(self):
        """禁用 LAB 相关模块的量化"""
        if hasattr(self, "act") and hasattr(self.act, "disable_lab_quantization"):
            self.act.disable_lab_quantization()
        if hasattr(self, "lab") and hasattr(self.lab, "disable_lab_quantization"):
            self.lab.disable_lab_quantization()

    # 新增：禁用 reparam_conv 的量化
    def disable_conv_quantization(self):
        """禁用 reparam_conv 的量化"""
        self.conv_quant.qconfig = None
        self.conv_dequant.qconfig = None
        if self.use_pact:
            self.pact.qconfig = None


class QuantizableLCNetV3Block(LCNetV3Block):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        conv_kxk_num=4,
        lr_mult=1.0,
        lab_lr=0.1,
        use_pact=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            stride,
            dw_size,
            use_se,
            conv_kxk_num,
            lr_mult,
            lab_lr,
        )

        # 替换为量化版本的子模块
        self.dw_conv = QuantizableLearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
            use_pact=use_pact,
        )

        # SE 模块不能量化
        # if use_se:
            # self.se = QuantizableSELayer(in_channels, lr_mult=lr_mult)

        self.pw_conv = QuantizableLearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
            use_pact=use_pact,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

    def fuse_model(self, is_qat=None):
        """融合子模块"""
        if hasattr(self, "dw_conv"):
            self.dw_conv.fuse_model(is_qat)
        if hasattr(self, "se") and self.use_se and hasattr(self.se, "fuse_model"):
            self.se.fuse_model(is_qat)
        if hasattr(self, "pw_conv"):
            self.pw_conv.fuse_model(is_qat)

    def disable_lab_quantization(self):
        """禁用 LAB 相关模块的量化"""
        self.dw_conv.disable_lab_quantization()
        self.pw_conv.disable_lab_quantization()

    # 新增：禁用 conv 的量化
    def disable_conv_quantization(self):
        """禁用 dw_conv 和 pw_conv 的量化"""
        if hasattr(self.dw_conv, 'disable_conv_quantization'):
            self.dw_conv.disable_conv_quantization()
        if hasattr(self.pw_conv, 'disable_conv_quantization'):
            self.pw_conv.disable_conv_quantization()


class QuantizablePPLCNetV3(PPLCNetV3):
    def __init__(
        self,
        scale=1.0,
        conv_kxk_num=4,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        lab_lr=0.1,
        det=False,
        **kwargs,
    ):
        super().__init__(scale, conv_kxk_num, lr_mult_list, lab_lr, det, **kwargs)

        # NOTE: 第一层不量化
        # self.conv1 = QuantizableConvBNLayer(
        #     in_channels=3,
        #     out_channels=make_divisible(16 * scale),
        #     kernel_size=3,
        #     stride=2,
        #     lr_mult=self.lr_mult_list[0],
        # )

        # 替换所有的 LCNetV3Block 为量化版本
        self.blocks2 = nn.Sequential(
            *[
                QuantizableLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[1],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                QuantizableLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[2],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                QuantizableLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[3],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                QuantizableLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[4],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                QuantizableLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[5],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )

        # 检测模式下的量化友好层
        if self.det:
            # 将普通Conv2d替换为量化友好的版本
            self.layer_list = nn.ModuleList(
                [
                    QuantizableConvBNLayer(
                        in_channels=self.out_channels[0],
                        out_channels=int(mv_c[0] * scale),
                        kernel_size=1,
                        stride=1,
                        lr_mult=1.0,
                    )
                    for i, mv_c in enumerate([16, 24, 56, 480])
                ]
            )

        # 新增：只在检测模式下添加输出的 dequant
        if self.det:
            self.output_dequants = nn.ModuleList([
                torch.quantization.DeQuantStub() for _ in range(4)
            ])
        else:
            self.output_dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        out_list = []
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.det:
            # 检测模式：对每个输出应用对应的层
            for i in range(len(out_list)):
                out_list[i] = self.layer_list[i](out_list[i])

            # 修改：只在输出时反量化
            out_list = [self.output_dequants[i](out) for i, out in enumerate(out_list)]
            return out_list

        # 识别模式
        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x

    def fuse_model(self, is_qat=None):
        """融合整个模型的所有层"""
        # 融合第一层
        if hasattr(self, "conv1") and hasattr(self.conv1, "fuse_model"):
            self.conv1.fuse_model(is_qat)

        # 融合所有block组
        for blocks in [
            self.blocks2,
            self.blocks3,
            self.blocks4,
            self.blocks5,
            self.blocks6,
        ]:
            for block in blocks:
                if hasattr(block, "fuse_model"):
                    block.fuse_model(is_qat)

        # 融合检测模式的额外层
        if self.det and hasattr(self, "layer_list"):
            for layer in self.layer_list:
                if hasattr(layer, "fuse_model"):
                    layer.fuse_model(is_qat)

    def disable_lab_quantization(self):
        """
        禁用所有 QuantizableLCNetV3Block 中的 LAB 量化
        """
        # 获取所有包含 QuantizableLCNetV3Block 的 Sequential 模块
        block_sequences = [
            self.blocks2, self.blocks3, self.blocks4, 
            self.blocks5, self.blocks6
        ]

        # 遍历所有 block sequences
        for blocks in block_sequences:
            for block in blocks:
                if hasattr(block, 'disable_lab_quantization'):
                    block.disable_lab_quantization()

    # 选择性禁用 conv 量化
    def set_quantization_config(self, quant_blocks):
        """
        设置哪些 block 需要量化

        Args:
            quant_blocks: 需要量化的 block 列表，例如:
                ['blocks6.0', 'blocks6.2', 'blocks6.3']
        """
        # 默认禁用所有 block 的 conv 量化
        all_blocks = {
            'blocks2': self.blocks2,
            'blocks3': self.blocks3,
            'blocks4': self.blocks4,
            'blocks5': self.blocks5,
            'blocks6': self.blocks6,
        }

        for block_name, blocks in all_blocks.items():
            for i, block in enumerate(blocks):
                full_name = f"{block_name}.{i}"

                if full_name in quant_blocks:
                    # 需要量化：不做任何操作（保持默认配置）
                    print(f"启用量化: {full_name}")
                else:
                    # 不需要量化：禁用 conv 量化
                    if hasattr(block, 'disable_conv_quantization'):
                        block.disable_conv_quantization()
                        print(f"禁用量化: {full_name}")
