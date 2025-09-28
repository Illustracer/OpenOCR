import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Union
from openrec.modeling.common import Mlp
from openrec.modeling.encoders.svtrnet import Attention, ConvBNLayer
from openrec.modeling.decoders.ctc_decoder import EncoderWithSVTR, CTCDecoder


def _fuse_modules(
    model: nn.Module,
    modules_to_fuse: Union[list[str], list[list[str]]],
    is_qat: Optional[bool],
    **kwargs: Any,
):
    if is_qat is None:
        is_qat = model.training
    method = (
        torch.ao.quantization.fuse_modules_qat
        if is_qat
        else torch.ao.quantization.fuse_modules
    )
    return method(model, modules_to_fuse, **kwargs)


class QuantizedMlp(Mlp):
    # MLP 模型无需编写量化代码
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__(in_features, hidden_features, out_features, act_layer, drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def fuse_model(self, is_qat=None):
        """无需融合"""
        pass


class QuantizedConvBNLayer(ConvBNLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act="swish",  # 固定为swish
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias, groups, act
        )
        # NOTE: 这里必须这样写, 否则有问题
        self.act = nn.Hardswish(inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

    def fuse_model(self, is_qat=None):
        """融合Conv+BN操作"""
        _fuse_modules(self, [["conv", "norm"]], inplace=True, is_qat=is_qat)


class QuantizedEncoderWithSVTR(EncoderWithSVTR):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        kernel_size=[3, 3],
        qk_scale=None,
        use_pool=True,
        support_ppocr_v4=False,
    ):
        # 先初始化父类以获得所有基本属性
        super().__init__(
            in_channels=in_channels,
            dims=dims,
            depth=depth,
            hidden_dims=hidden_dims,
            use_guide=use_guide,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path=drop_path,
            kernel_size=kernel_size,
            qk_scale=qk_scale,
            use_pool=use_pool,
            support_ppocr_v4=support_ppocr_v4,
        )

        # 替换ConvBNLayer为量化版本
        self.conv1 = QuantizedConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act="swish",
            bias=False,
        )

        self.conv2 = QuantizedConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act="swish", bias=False
        )

        self.conv3 = QuantizedConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act="swish", bias=False
        )

        # conv4根据support_ppocr_v4参数设置
        if support_ppocr_v4:
            self.conv4 = QuantizedConvBNLayer(
                2 * in_channels, in_channels // 8, padding=1, act="swish", bias=False
            )
        else:
            self.conv4 = QuantizedConvBNLayer(
                2 * in_channels,
                in_channels // 8,
                kernel_size=kernel_size,
                padding=[kernel_size[0] // 2, kernel_size[1] // 2],
                act="swish",
                bias=False,
            )

        self.conv1x1 = QuantizedConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act="swish", bias=False
        )

        # 添加量化相关的FloatFunctional操作
        self.cat_func = torch.nn.quantized.FloatFunctional()  # 用于torch.concat

        # svtr_block和norm保持原样，不进行量化
        # 为Block模块添加量化/反量化保护
        self.block_dequant = torch.quantization.DeQuantStub()
        self.block_quant = torch.quantization.QuantStub()

    def forward(self, x):
        if self.use_pool:
            x = self.pool_h_2(x)

        # for use guide
        if self.use_guide:
            z = x.detach()
        else:
            z = x

        # for short cut
        h = z

        # reduce dim - 使用量化的ConvBN层
        z = self.conv1(z)
        z = self.conv2(z)

        z = self.block_dequant(z)
        # SVTR global block - 保持原有精度
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2).contiguous()

        # 需要考虑 svtr_block 如何进行量化
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)

        # last stage - 使用量化的ConvBN层
        z = z.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        z = self.block_quant(z)

        z = self.conv3(z)

        # 使用FloatFunctional进行concat操作
        z = self.cat_func.cat([h, z], dim=1)
        z = self.conv1x1(self.conv4(z))
        return z

    def fuse_model(self, is_qat=None):
        """融合"""
        self.conv1.fuse_model(is_qat)
        self.conv2.fuse_model(is_qat)
        self.conv3.fuse_model(is_qat)
        self.conv4.fuse_model(is_qat)
        self.conv1x1.fuse_model(is_qat)

    def disable_svtr_quantization(self):
        """禁用SVTR相关模块的量化"""
        # 禁用LayerNorm
        if hasattr(self, "norm"):
            self.norm.qconfig = None

        # 禁用SVTR blocks
        for blk in self.svtr_block:
            blk.qconfig = None
            # 递归禁用block内部的所有子模块
            for module in blk.modules():
                module.qconfig = None


class QuantizedCTCDecoder(CTCDecoder):
    """简化的CTC解码器 - 继承原始CTCDecoder"""
    # example_inputs = torch.rand(1, 120, 8, 32).to("cpu")
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        mid_channels=None,
        return_feats=False,
        svtr_encoder=None,
        **kwargs,
    ):
        # 临时保存svtr_encoder配置
        if svtr_encoder is not None:
            # 暂时移除svtr_encoder配置，让父类不创建编码器
            svtr_encoder_ = None
            super().__init__(120, out_channels, mid_channels, return_feats, svtr_encoder_, **kwargs)

            # 手动创建量化版本的SVTR编码器
            svtr_encoder["in_channels"] = in_channels  # in_channels
            self.svtr_encoder = QuantizedEncoderWithSVTR(**svtr_encoder)
        else:
            super().__init__(in_channels, out_channels, mid_channels, return_feats, svtr_encoder, **kwargs)

        # 添加量化控制
        self.input_quant = torch.quantization.QuantStub()
        self.output_dequant = torch.quantization.DeQuantStub()

        # 为softmax添加量化控制
        self.softmax_dequant = torch.quantization.DeQuantStub()
        self.softmax_quant = torch.quantization.QuantStub()

    def forward(self, x):
        # 输入量化
        x = self.input_quant(x)

        # SVTR编码器处理（如果存在）
        if self.svtr_encoder is not None:
            x = self.svtr_encoder(x)
            x = x.flatten(2).transpose(1, 2)

        # 全连接层处理
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        # 返回结果处理
        if self.return_feats:
            # 反量化用于返回
            x_float = self.output_dequant(x.clone())
            predicts_float = self.output_dequant(predicts.clone())
            result = (x_float, predicts_float)
        else:
            result = predicts

        # 推理时的softmax
        if not self.training:
            predicts_float = self.softmax_dequant(predicts)
            predicts_float = F.softmax(predicts_float, dim=2)
            result = predicts_float  # softmax结果通常需要浮点精度

        return result

    def fuse_model(self, is_qat=None):
        """融合模型"""
        if self.svtr_encoder is not None:
            self.svtr_encoder.fuse_model(is_qat)

        # Linear层通常不需要特殊的融合操作
        # 但如果需要，可以在这里添加

    def disable_svtr_quantization(self):
        """禁用SVTR相关模块的量化"""
        self.svtr_encoder.disable_svtr_quantization()


class QuantizedEncoderModel(nn.Module):
    # example_inputs = torch.rand(1, 512, 48, 320).to("cpu")
    def __init__(self, in_channels=512, dims=64):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.encoder = QuantizedEncoderWithSVTR(
            in_channels=in_channels,
            dims=dims,
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.0,
            kernel_size=[3, 3],
            qk_scale=None,
            use_pool=True,
            support_ppocr_v4=False,
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.encoder(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat=None):
        """融合"""
        self.encoder.fuse_model(is_qat)

    def disable_svtr_quantization(self):
        """禁用SVTR相关模块的量化"""
        self.encoder.disable_svtr_quantization()


class QuantizedConvBNModel(nn.Module):
    # example_inputs = torch.rand(1, 3, 48, 320).to("cpu")
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv_bn = QuantizedConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
            act="swish",
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv_bn(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat=None):
        """融合"""
        self.conv_bn.fuse_model(is_qat)


class QuantizedModel(nn.Module):
    # 测试量化 MLP 模块
    # NOTE: 这个可以直接量化
    # example_inputs = torch.rand(1, 128, 768).to("cpu")
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.mlp_block = QuantizedMlp(768, 3072, 768, nn.GELU, 0.1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.mlp_block(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat=None):
        """无需融合"""
        pass


class QuantizedAttentionModel(nn.Module):
    # 测试量化 Attention 模块
    # example_inputs = torch.rand(1, 128, 768).to("cpu")
    # NOTE: 不支持 q = q * self.scale
    # 需要写 self.mul_scale = torch.nn.quantized.FloatFunctional()
    # 但是不支持导出 ONNX
    def __init__(self, dim=768, num_heads=12, HW=None, mixer="Global"):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            mixer=mixer,
            HW=HW,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.attention(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat=None):
        """无需融合"""
        pass
