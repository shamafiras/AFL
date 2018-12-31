from torch import nn

class TransBlockDual(nn.Module):
    def __init__(self, size):
        super(TransBlockDual, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2*size, size, 3, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.Conv2d(size, size, 3, padding=1),
            nn.BatchNorm2d(size)
        )

    def forward(self, input):
        return self.main(input)

class GeneratorAFL(nn.Module):
    def __init__(self):
        super(GeneratorAFL, self).__init__()
        dim = 128
        self.trans_block0 = TransBlockDual(8 * dim)
        self.trans_block1 = TransBlockDual(4 * dim)
        self.trans_block2 = TransBlockDual(2 * dim)
        self.trans_block3 = TransBlockDual(1 * dim)

    def set_input_disc(self, layers_input):
        self.feedback0 = layers_input[3]
        self.feedback1 = layers_input[2]
        self.feedback2 = layers_input[1]
        self.feedback3 = layers_input[0]


