import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.signal import windows

import seisbench.util as sbu
from .base import ActivationLSTMCell, CustomLSTM, WaveformModel


# For implementation, potentially follow: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
class CrossTwoStreamEQT(WaveformModel):
    """
    The EQTransformer from Mousavi et al. (2020)

    Implementation adapted from the Github repository https://github.com/smousavi05/EQTransformer
    Assumes padding="same" and activation="relu" as in the pretrained EQTransformer models

    By instantiating the model with `from_pretrained("original")` a binary compatible version of the original
    EQTransformer with the original weights from Mousavi et al. (2020) can be loaded.

    .. document_args:: seisbench.models EQTransformer

    :param in_channels: Number of input channels, by default 3.
    :param in_samples: Number of input samples per channel, by default 6000.
                       The model expects input shape (in_channels, in_samples)
    :param classes: Number of output classes, by default 2. The detection channel is not counted.
    :param phases: Phase hints for the classes, by default "PS". Can be None.
    :param res_cnn_blocks: Number of residual convolutional blocks
    :param lstm_blocks: Number of LSTM blocks
    :param drop_rate: Dropout rate
    :param original_compatible: If True, uses a few custom layers for binary compatibility with original model
                                from Mousavi et al. (2020).
                                This option defaults to False.
                                It is usually recommended to stick to the default value, as the custom layers show
                                slightly worse performance than the PyTorch builtins.
                                The exception is when loading the original weights using :py:func:`from_pretrained`.
    :param norm: Data normalization strategy, either "peak" or "std".
    :param kwargs: Keyword arguments passed to the constructor of :py:class:`WaveformModel`.
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.1)
    _annotate_args["detection_threshold"] = ("Detection threshold", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (500, 500),
    )
    # Overwrite default stacking method
    _annotate_args["stacking"] = (
        "Stacking method for overlapping windows (only for window prediction models). "
        "Options are 'max' and 'avg'. ",
        "max",
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 3000)

    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
    ]

    def __init__(
            self,
            in_channels=3,
            in_samples=6000,
            classes=2,
            phases="PS",
            cnn_blocks=7,
            res_cnn_blocks=7,
            lstm_blocks=3,
            drop_rate=0.3,
            original_compatible=False,
            sampling_rate=100,
            norm="std",
            eqt_mode="twostream",
            logits=False,
            TF=False,
            **kwargs,
    ):

        citation = (
            "Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L, Y., and Beroza, G, C. "
            "Earthquake transformer—an attentive deep-learning model for simultaneous earthquake "
            "detection and phase picking. Nat Commun 11, 3952 (2020). "
            "https://doi.org/10.1038/s41467-020-17591-w"
        )

        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        # Blinding defines how many samples at beginning and end of the prediction should be ignored
        # This is usually required to mitigate prediction problems from training properties, e.g.,
        # if all picks in the training fall between seconds 5 and 55.
        super().__init__(
            citation=citation,
            output_type="array",
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=["Detection"] + list(phases),
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.cnn_blocks = cnn_blocks
        self.res_cnn_blocks = res_cnn_blocks
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.norm = norm
        self.eqt_mode = eqt_mode
        self.logits = logits
        self.TF = TF

        print(eqt_mode)
        print(cnn_blocks)
        print(res_cnn_blocks)

        # Add options for conservative and the true original - see https://github.com/seisbench/seisbench/issues/96#issuecomment-1155158224
        if original_compatible == True:
            warnings.warn(
                "Using the non-conservative 'original' model, set `original_compatible='conservative' to use the more conservative model"
            )
            original_compatible = "non-conservative"

        if original_compatible:
            eps = 1e-7  # See Issue #96 - original models use tensorflow default epsilon of 1e-7
        else:
            eps = 1e-5
        self.original_compatible = original_compatible

        if original_compatible and in_samples != 6000:
            raise ValueError("original_compatible=True requires in_samples=6000.")

        self._phases = phases
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )

        # Parameters from EQTransformer repository
        self.all_filters = [8, 16, 16, 32, 32, 64, 64]
        self.all_kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.all_res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        def trim_list(lst, length):
            return lst[:length - 1] + [lst[-1]] if length < len(lst) else lst

        # Determine effective blocks
        effective_cnn_blocks = min(self.cnn_blocks, len(self.all_filters))
        effective_res_cnn_blocks = min(self.res_cnn_blocks, len(self.all_res_cnn_kernels))

        # Select filters and kernels
        self.filters = trim_list(self.all_filters, effective_cnn_blocks)
        self.kernel_sizes = trim_list(self.all_kernel_sizes, effective_cnn_blocks)
        self.res_cnn_kernels = trim_list(self.all_res_cnn_kernels, effective_res_cnn_blocks)

        # print(self.filters, self.kernel_sizes, self.res_cnn_kernels)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize with a value of 0.5


        # TODO: Add regularizers when training model
        # kernel_regularizer=keras.regularizers.l2(1e-6),
        # bias_regularizer=keras.regularizers.l1(1e-4),

        # Fencoder stack
        self.fencoder = FrequenceDim(
            in_channels=self.in_channels,
            in_samples=self.in_samples,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            res_cnn_kernels = self.res_cnn_kernels,
            lstm_blocks = self.lstm_blocks,
            drop_rate=self.drop_rate,
            original_compatible=self.original_compatible
        )

        self.tencoder = TimeDim(
            in_channels=self.in_channels,
            in_samples=self.in_samples,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            res_cnn_kernels=self.res_cnn_kernels,
            lstm_blocks=self.lstm_blocks,
            drop_rate=self.drop_rate,
            original_compatible=self.original_compatible
        )

        self.transformer_d = Transformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )

        self.corss_transformer =CrossAttentionTransformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )

        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            original_compatible=original_compatible,
        )
        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
        )

        # Picking branches
        self.pick_lstms = []
        self.pick_attentions = []
        self.pick_decoders = []
        self.pick_convs = []
        self.dropout = nn.Dropout(drop_rate)

        for _ in range(self.classes):
            if original_compatible == "conservative":
                # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
                lstm = CustomLSTM(ActivationLSTMCell, 16, 16, bidirectional=False)
            else:
                lstm = nn.LSTM(16, 16, bidirectional=False)
            self.pick_lstms.append(lstm)

            attention = SeqSelfAttention(input_size=16, attention_width=3, eps=eps)
            self.pick_attentions.append(attention)

            decoder = Decoder(
                input_channels=16,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=in_samples,
                original_compatible=original_compatible,
            )
            self.pick_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
            )
            self.pick_convs.append(conv)

        self.pick_lstms = nn.ModuleList(self.pick_lstms)
        self.pick_attentions = nn.ModuleList(self.pick_attentions)
        self.pick_decoders = nn.ModuleList(self.pick_decoders)
        self.pick_convs = nn.ModuleList(self.pick_convs)

    def forward(self, x):
        global xf
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        try:
            flag=self.logits
        except AttributeError:
            flag=False
        # print("eqt_mode:", self.eqt_mode)
        if self.eqt_mode=="twostream" or self.eqt_mode=="frequence":
            xf = self.fencoder(x)
            if self.eqt_mode=="frequence":
                xf,_=self.transformer_d(xf)
                x = xf
        if self.eqt_mode=="twostream" or self.eqt_mode=="time":
            x = self.tencoder(x)
            if self.eqt_mode=="time":
                x,_ = self.transformer_d(x)
        if self.eqt_mode=="twostream" :
            # x = self.alpha * x + (1 - self.alpha) * xf  # add
            x = self.corss_transformer(x,xf)

        # Detection part
        detection = self.decoder_d(x)
        if flag:
            detection = self.conv_d(detection)
        else:
            detection = torch.sigmoid(self.conv_d(detection))
        detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

        outputs = [detection]

        # Pick parts
        for lstm, attention, decoder, conv in zip(
            self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(
                2, 0, 1
            )  # From batch, channels, sequence to sequence, batch, channels
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(
                1, 2, 0
            )  # From sequence, batch, channels to batch, channels, sequence
            px, _ = attention(px)
            px = decoder(px)
            if flag:
                pred = conv(px)
            else:
                pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)  # Remove channel dimension

            outputs.append(pred)

        return tuple(outputs)

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to correct shape
        batch = torch.stack(batch, dim=-1)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm_detrend:
            batch = sbu.torch_detrend(batch)
        if self.norm_amp_per_comp:
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        else:
            if self.norm == "std":
                std = batch.std(axis=(-1, -2), keepdims=True)
                batch = batch / (std + 1e-10)
            elif self.norm == "peak":
                peak = batch.abs().max(axis=-1, keepdims=True)[0]
                batch = batch / (peak + 1e-10)

        # Cosine taper (very short, i.e., only six samples on each side)
        tap = 0.5 * (
            1 + torch.cos(torch.linspace(np.pi, 2 * np.pi, 6, device=batch.device))
        )
        batch[:, :, :6] *= tap
        batch[:, :, -6:] *= tap.flip(dims=(0,))

        return batch

    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return list(range(self.classes))

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete picks using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`
        and to discrete detections using :py:func:`~seisbench.models.base.WaveformModel.detections_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".
        Trigger onset thresholds for detections are derived from the argdict at key "detection_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks, list of detections
        """
        picks = sbu.PickList()
        for phase in self.phases:
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )
        picks = sbu.PickList(sorted(picks))

        detections = self.detections_from_annotations(
            annotations.select(channel=f"{self.__class__.__name__}_Detection"),
            argdict.get(
                "detection_threshold", self._annotate_args.get("detection_threshold")[1]
            ),
        )

        return sbu.ClassifyOutput(self.name, picks=picks, detections=detections)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "sampling_rate",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["in_samples"] = self.in_samples
        model_args["classes"] = self.classes
        model_args["phases"] = self.phases
        model_args["cnn_blocks"] = self.cnn_blocks
        model_args["res_cnn_blocks"] = self.res_cnn_blocks
        model_args["lstm_blocks"] = self.lstm_blocks
        model_args["drop_rate"] = self.drop_rate
        model_args["original_compatible"] = self.original_compatible
        model_args["sampling_rate"] = self.sampling_rate
        model_args["eqt_mode"] = self.eqt_mode
        model_args["logits"] = self.logits
        model_args["TF"] = self.TF
        model_args["norm"] = self.norm
        model_args["norm_amp_per_comp"] = self.norm_amp_per_comp
        model_args["norm_detrend"] = self.norm_detrend

        return model_args

class FrequenceDim(nn.Module):
    def __init__(self, in_channels, in_samples, filters, kernel_sizes, res_cnn_kernels, lstm_blocks,
                 drop_rate, original_compatible, fs=100, nperseg=80, sigma=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.res_cnn_kernels = res_cnn_kernels
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.original_compatible = original_compatible
        self.fs = fs
        self.nperseg = nperseg
        self.sigma = sigma

        if original_compatible:
            eps = 1e-7
        else:
            eps = 1e-5

        # Define frequency range
        self.f_min = 1
        self.f_max = self.fs / 2
        self.f = torch.linspace(self.f_min, self.f_max, self.nperseg, device='cuda')

        # Define time range (adjust this if necessary)
        self.t = None

        # Encoder stack
        self.encoder = Encoder(
            input_channels=(self.nperseg//2+1) * 3,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=in_samples,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=drop_rate,
        )

        # BiLSTM stack
        self.bi_lstm_stack = BiLSTMStack(
            blocks=self.lstm_blocks,
            input_size=self.filters[-1],
            drop_rate=self.drop_rate,
            original_compatible=original_compatible,
        )

        # Global attention - two transformers
        self.transformer_d0 = Transformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.in_channels, -1)

        xf = torch.stft(
            x,
            n_fft=self.nperseg,
            hop_length=self.nperseg // 2,
            return_complex=True)
        xf = xf.view(batch_size, self.in_channels, xf.shape[1], xf.shape[2])
        xf = torch.abs(xf)
        xf = F.interpolate(xf, size=(xf.shape[2], self.in_samples), mode='bilinear')
        xf = xf.view(batch_size, xf.shape[1] * xf.shape[2], xf.shape[-1])

        # Pass through the rest of the model
        xf = self.encoder(xf)
        xf = self.res_cnn_stack(xf)
        xf = self.bi_lstm_stack(xf)
        xf, _ = self.transformer_d0(xf)

        return xf

class TimeDim(nn.Module):
    def __init__(self, in_channels, in_samples , filters, kernel_sizes, res_cnn_kernels, lstm_blocks,
                 drop_rate, original_compatible):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.res_cnn_kernels = res_cnn_kernels
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.original_compatible = original_compatible

        # TODO: Add regularizers when training model
        # kernel_regularizer=keras.regularizers.l2(1e-6),
        # bias_regularizer=keras.regularizers.l1(1e-4),

        if original_compatible:
            eps = 1e-7  # See Issue #96 - original models use tensorflow default epsilon of 1e-7
        else:
            eps = 1e-5

        # Encoder stack
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        # BiLSTM stack
        self.bi_lstm_stack = BiLSTMStack(
            blocks=self.lstm_blocks,
            input_size=self.filters[-1],
            # input_size=self.fusion_inputsize[self.eqt_mode],
            drop_rate=self.drop_rate,
            original_compatible=original_compatible,
        )

        # Global attention - two transformers
        self.transformer_d0 = Transformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)

        return x

class Encoder(nn.Module):
    """
    Encoder stack
    """

    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()

        convs = []
        pools = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        for conv, pool, padding in zip(self.convs, self.pools, self.paddings):
            x = torch.relu(conv(x))
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]

            x = F.relu(conv(x))

        return x


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))

        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)

        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class BiLSTMStack(nn.Module):
    def __init__(
        self, blocks, input_size, drop_rate, hidden_size=16, original_compatible=False
    ):
        super().__init__()

        # First LSTM has a different input size as the subsequent ones
        self.members = nn.ModuleList(
            [
                BiLSTMBlock(
                    input_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
            ]
            + [
                BiLSTMBlock(
                    hidden_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate, original_compatible=False):
        super().__init__()

        if original_compatible == "conservative":
            # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
            self.lstm = CustomLSTM(ActivationLSTMCell, input_size, hidden_size)
        elif original_compatible == "non-conservative":
            self.lstm = CustomLSTM(
                ActivationLSTMCell,
                input_size,
                hidden_size,
                gate_activation=torch.sigmoid,
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv1d(2 * hidden_size, hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size, eps=1e-3)

    def forward(self, x):
        x = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = x.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        x = self.conv(x)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_size, drop_rate, attention_width=None, eps=1e-5):
        super().__init__()

        self.attention = SeqSelfAttention(
            input_size, attention_width=attention_width, eps=eps
        )
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x):
        y, weight = self.attention(x)
        y = x + y
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y2)

        return y2, weight


class SeqSelfAttention(nn.Module):
    """
    Additive self attention
    """

    def __init__(self, input_size, units=32, attention_width=None, eps=1e-5):
        super().__init__()
        self.attention_width = attention_width

        self.Wx = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.Wt = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.bh = nn.Parameter(torch.zeros(units))

        self.Wa = nn.Parameter(uniform(-0.02, 0.02, units, 1))
        self.ba = nn.Parameter(torch.zeros(1))

        self.eps = eps

    def forward(self, x):
        # x.shape == (batch, channels, time)

        x = x.permute(0, 2, 1)  # to (batch, time, channels)

        q = torch.unsqueeze(
            torch.matmul(x, self.Wt), 2
        )  # Shape (batch, time, 1, channels)
        k = torch.unsqueeze(
            torch.matmul(x, self.Wx), 1
        )  # Shape (batch, 1, time, channels)

        h = torch.tanh(q + k + self.bh)

        # Emissions
        e = torch.squeeze(
            torch.matmul(h, self.Wa) + self.ba, -1
        )  # Shape (batch, time, time)

        # This is essentially softmax with an additional attention component.
        e = (
            e - torch.max(e, dim=-1, keepdim=True).values
        )  # In versions <= 0.2.1 e was incorrectly normalized by max(x)
        e = torch.exp(e)
        if self.attention_width is not None:
            lower = (
                torch.arange(0, e.shape[1], device=e.device) - self.attention_width // 2
            )
            upper = lower + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, e.shape[1], device=e.device), 1)
            mask = torch.logical_and(lower <= indices, indices < upper)
            e = torch.where(mask, e, torch.zeros_like(e))

        a = e / (torch.sum(e, dim=-1, keepdim=True) + self.eps)

        v = torch.matmul(a, x)

        v = v.permute(0, 2, 1)  # to (batch, channels, time)

        return v, a


def uniform(a, b, *args):
    return a + (b - a) * torch.rand(*args)


class LayerNormalization(nn.Module):
    def __init__(self, filters, eps=1e-14):
        super().__init__()

        gamma = torch.ones(filters, 1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.zeros(filters, 1)
        self.beta = nn.Parameter(beta)
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        var = torch.mean((x - mean) ** 2, 1, keepdim=True) + self.eps
        std = torch.sqrt(var)
        outputs = (x - mean) / std

        outputs = outputs * self.gamma
        outputs = outputs + self.beta

        return outputs


class FeedForward(nn.Module):
    def __init__(self, io_size, drop_rate, hidden_size=128):
        super().__init__()

        self.lin1 = nn.Linear(io_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, io_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # To (batch, time, channel)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = x.permute(0, 2, 1)  # To (batch, channel, time)

        return x


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x

class CrossAttention(nn.Module):
    def __init__(self, input_size, units=32, attention_width=None, eps=1e-5):
        super().__init__()
        self.attention_width = attention_width

        # Query, Key, Value权重矩阵
        self.Wq = nn.Parameter(torch.Tensor(input_size, units))
        self.Wk = nn.Parameter(torch.Tensor(input_size, units))
        self.Wv = nn.Parameter(torch.Tensor(input_size, units))
        self.bh = nn.Parameter(torch.zeros(units))

        # 注意力权重矩阵
        self.Wa = nn.Parameter(torch.Tensor(units, 1))
        self.ba = nn.Parameter(torch.zeros(1))

        self.eps = eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.Wq, -0.02, 0.02)
        nn.init.uniform_(self.Wk, -0.02, 0.02)
        nn.init.uniform_(self.Wv, -0.02, 0.02)
        nn.init.uniform_(self.Wa, -0.02, 0.02)

    def forward(self, y1, y2):
        # y1, y2的形状: (batch_size, channels, samples)
        y1 = y1.permute(0, 2, 1)  # (batch_size, samples, channels)
        y2 = y2.permute(0, 2, 1)  # (batch_size, samples, channels)

        # 计算Query, Key, Value
        q1 = torch.matmul(y1, self.Wq)  # (batch_size, samples, units)
        k2 = torch.matmul(y2, self.Wk)  # (batch_size, samples, units)
        v2 = torch.matmul(y2, self.Wv)  # (batch_size, samples, units)

        # 计算注意力得分
        d_k = q1.size(-1)  # 获取单位维度
        e = torch.matmul(q1, k2.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # (batch_size, samples, samples)

        # (batch_size, samples, samples)

        if self.attention_width is not None:
            lower = torch.arange(0, e.shape[-1], device=e.device) - self.attention_width // 2
            upper = lower + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, e.shape[-1], device=e.device), 1)
            mask = torch.logical_and(lower <= indices, indices < upper)
            e = torch.where(mask, e, torch.full_like(e, float('-inf')))

        a = F.softmax(e, dim=-1)  # 归一化注意力权重

        # 计算加权值
        out = torch.matmul(a, v2)  # (batch_size, samples, units)

        out = out.permute(0, 2, 1)  # 转换回 (batch_size, units, samples)

        return out,a


class CrossAttentionTransformer(nn.Module):
    def __init__(self, input_size, drop_rate, attention_width=None, eps=1e-5):
        super().__init__()

        self.transformer1 = Transformer(input_size, drop_rate, attention_width, eps)
        self.transformer2 = Transformer(input_size, drop_rate, attention_width, eps)
        self.cross_attention = CrossAttention(input_size, units=16,attention_width=attention_width, eps=eps)
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x1, x2):

        y1, _ = self.transformer1(x1)
        y2, _ = self.transformer2(x2)

        y,_= self.cross_attention(y1, y2)
        y = y1 + y  # 残差连接
        y = self.norm1(y)
        yy = self.ff(y)
        yy = y + yy
        yy = self.norm2(yy)

        return yy