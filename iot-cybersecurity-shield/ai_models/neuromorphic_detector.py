"""
Brain-inspired Spiking Neural Network for low-power detection
"""
import snntorch as snn

class NeuroIDS(snn.Module):
    def __init__(self):
        super().__init__()
        self.lif1 = snn.Leaky(beta=0.8)
        self.lif2 = snn.Leaky(beta=0.7)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        spk1, mem1 = self.lif1(x, mem1)
        spk2, _ = self.lif2(spk1)
        return spk2
