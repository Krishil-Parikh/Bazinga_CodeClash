"""
Simulates 100+ IoT devices with constrained resources using NS-3
"""
import ns.core
from ns.network import NodeContainer, InternetStackHelper

class IoTSimulation:
    def __init__(self, num_devices=100):
        self.nodes = NodeContainer()
        self.nodes.Create(num_devices)
        
    def configure_energy(self):
        """Apply energy constraints to IoT devices"""
        energy_source = ns.energy.BasicEnergySourceHelper()
        energy_source.Set("BasicEnergySourceInitialEnergyJ", ns.core.DoubleValue(500))
        energy_source.Install(self.nodes)
    
    def run(self, duration=3600):
        """Run network simulation"""
        ns.core.Simulator.Stop(ns.core.Seconds(duration))
        ns.core.Simulator.Run()