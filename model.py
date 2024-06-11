from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(177, 96, 15, padding=7, padding_mode='replicate'),
                                     nn.MaxPool1d(2),
                                     nn.ReLU(),

                                     nn.Conv1d(96, 128, 15, padding=7, padding_mode='replicate'),
                                     nn.MaxPool1d(2),
                                     nn.ReLU(),

                                     nn.Conv1d(128, 256, 15, padding=7, padding_mode='replicate'),
                                     nn.MaxPool1d(2),
                                     nn.Tanh())

        self.decoder = nn.Sequential(nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='linear'),
                                     nn.ConvTranspose1d(256, 128, 15, padding=7),

                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='linear'),
                                     nn.ConvTranspose1d(128, 96, 15, padding=7),

                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode='linear'),
                                     nn.ConvTranspose1d(96, 177, 15, padding=7))

        # self.encoder = nn.Sequential(nn.Conv1d(93, 96, 15, padding=7, padding_mode='replicate'),
        #                              nn.MaxPool1d(2),
        #                              nn.Tanh(),
        #
        #                              nn.Conv1d(96, 128, 15, padding=7, padding_mode='replicate'),
        #                              nn.MaxPool1d(2),
        #                              nn.Tanh(),
        #
        #                              nn.Conv1d(128, 256, 15, padding=7, padding_mode='replicate'),
        #                              nn.MaxPool1d(2),
        #                              nn.Tanh())
        #
        # self.decoder = nn.Sequential(nn.Tanhshrink(),
        #                              nn.Upsample(scale_factor=2, mode='linear'),
        #                              nn.ConvTranspose1d(256, 128, 15, padding=7),
        #
        #                              nn.Tanhshrink(),
        #                              nn.Upsample(scale_factor=2, mode='linear'),
        #                              nn.ConvTranspose1d(128, 96, 15, padding=7),
        #
        #                              nn.Tanhshrink(),
        #                              nn.Upsample(scale_factor=2, mode='linear'),
        #                              nn.ConvTranspose1d(96, 93, 15, padding=7))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x