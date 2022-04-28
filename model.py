import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl


# define the LightningModule
class GRUCNN(pl.LightningModule):
    def __init__(
            self,
            input_length: int,
            encoding: nn.Embedding,
            l1_kernel_length: int = 8,
            num_l1_kernels: int = 32,
            l2_kernel_length: int = 8,
            num_l2_kernels: int = 16,
            l2_stride: int = 4,
            l3_kernel_length: int = 8,
            num_l3_kernels: int = 16,
            l3_stride: int = 4,
            l4_kernel_length: int = 8,
            num_l4_kernels: int = 16,
            l4_stride: int = 4,
            l5_kernel_length: int = 8,
            num_l5_kernels: int = 16,
            l5_stride: int = 4,
            dense_size: int = 512,
            out_classes: int = 2
    ):
        super().__init__()

        if input_length % (l2_stride*l3_stride*l4_stride*l5_stride) != 0:
            raise ValueError(
                "Input length must be evenly divided by strides of "
                "intermediate layers."
            )

        kernels_l1 = [nn.GRU(
            input_size=l1_kernel_length,
            hidden_size=1,
            bidirectional=True
        )] * num_l1_kernels
        kernels_l2 = [nn.GRU(
            input_size=l2_kernel_length,
            hidden_size=1,
            bidirectional=True
        )] * num_l2_kernels
        kernels_l3 = [nn.GRU(
            input_size=l3_kernel_length,
            hidden_size=1,
            bidirectional=True
        )] * num_l3_kernels
        kernels_l4 = [nn.GRU(
            input_size=l4_kernel_length,
            hidden_size=1,
            bidirectional=True
        )] * num_l4_kernels
        kernels_l5 = [nn.GRU(
            input_size=l5_kernel_length,
            hidden_size=1,
            bidirectional=True
        )] * num_l5_kernels


        conv2d_l2 = [nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(l2_stride, num_l1_kernels),
            stride=(l3_stride, 0),
            padding=0
        )] * num_l2_kernels
        conv2d_l3 = [nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(l3_stride, num_l2_kernels),
            stride=(l3_stride, 0),
            padding=0
        )] * num_l3_kernels
        conv2d_l4 = [nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(l4_stride, num_l3_kernels),
            stride=(l4_stride, 0),
            padding=0
        )] * num_l4_kernels
        conv2d_l5 = [nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(l5_stride, num_l4_kernels),
            stride=(l5_stride, 0),
            padding=0
        )] * num_l5_kernels

        input_reduction = (l2_stride*l3_stride*l4_stride*l5_stride)
        dense = nn.Linear(
            in_features=num_l5_kernels * input_length // input_reduction,
            out_features=dense_size
        )
        output = nn.Softmax(out_classes)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
