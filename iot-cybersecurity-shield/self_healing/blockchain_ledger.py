"""
Blockchain-based device authentication using Hyperledger
"""
from web3 import Web3

class IoTBlockchain:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
        self.contract = self.w3.eth.contract(
            abi='...',
            address='0x...'
        )
    
    def validate_device(self, device_id):
        return self.contract.functions.checkAuthorization(device_id).call()
    
    def blacklist_attacker(self, ip):
        self.contract.functions.addToBlacklist(ip).transact()