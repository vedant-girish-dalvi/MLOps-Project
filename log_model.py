import wandb
import json

run = wandb.init(project="semantic-segmentation", name="log_model_artifact")

model_path = "./artifacts/deeplabv3plus_mitb5.onnx"

config = {
    "input_size": [640, 640],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "class_map": {
        "Crack": 0, "ACrack": 1, "Wetspot": 2, "Efflorescence": 3,"Rust": 4, "Rockpocket": 5, "Hollowareas": 6,
        "Cavity": 7, "Spalling": 8, "Graffiti": 9,"Weathering": 10, "Restformwork": 11, "ExposedRebars": 12,
        "Bearing": 13,"EJoint": 14,"Drainage": 15,"PEquipment": 16,"JTape": 17,"WConccor": 18}, 
}
with open("preprocess_config.json", "w") as f:
    json.dump(config, f)

model_artifact = wandb.Artifact(
    name="trained-model",
    type="model",
    description="Fine-tuned concrete damage segmentation model",
    metadata={"framework": "pytorch", "validation_mIOU": 0.38}
)

model_artifact.add_file(model_path)
model_artifact.add_file("preprocess_config.json")
wandb.log_artifact(model_artifact)

wandb.finish()