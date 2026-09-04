import json
import wandb
import onnxruntime as ort


def load_model():
    api = wandb.Api()
    artifact = api.artifact(
        "vedant_girish-dalvi-hochschule-m-nchen/semantic-segmentation/trained-model:latest"
    )
    artifact_dir = artifact.download()

    with open(f"{artifact_dir}/preprocess_config.json") as f:
        config = json.load(f)

    session = ort.InferenceSession(
        f"{artifact_dir}/deeplabv3plus_mitb5.onnx",
        providers=["CPUExecutionProvider"],
    )

    return session, config
