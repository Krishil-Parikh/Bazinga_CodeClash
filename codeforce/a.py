import os
import json

def create_directory_structure():
    # Define the directory structure
    structure = {
        "core": {
            "simulation": {
                "ns3_iot_simulator.py": "",
                "attack_emulator": {
                    "quantum_sidechannel.py": "",
                    "ai_driven_apt_attack.py": ""
                }
            }
        },
        "edge_processing": {
            "tiny_mmdetection": {
                "onnx_model_quantizer.py": ""
            },
            "federated_trainer.py": "",
            "homomorphic_encryption.py": ""
        },
        "ai_models": {
            "spacetime_gnn.py": "",
            "neuromorphic_detector.py": "",
            "shap_lime_integrator.py": ""
        },
        "self_healing": {
            "quantum_key_rotator.py": "",
            "blockchain_ledger.py": ""
        },
        "zero_trust": {
            "behavioral_biometrics.py": "",
            "honeypot_generator.py": ""
        },
        "configs": {
            "pqc_params.yaml": "# Post-Quantum Cryptography Parameters\n",
            "iot_device_profiles.json": "{\n  \"devices\": []\n}"
        },
        "tests": {
            "adversarial_testbench.py": "",
            "resilience_benchmark.py": ""
        }
    }

    # Create directories and files based on the structure
    def create_structure(structure, current_path=""):
        for name, content in structure.items():
            path = os.path.join(current_path, name)
            
            if isinstance(content, dict):  # It's a directory
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
                create_structure(content, path)
            else:  # It's a file
                # Create the file with some placeholder content
                with open(path, "w") as f:
                    f.write(content if content else f"# {name}\n# Add your code here\n")
                print(f"Created file: {path}")

    # Start creating from the current directory
    create_structure(structure)
    print("Directory structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()