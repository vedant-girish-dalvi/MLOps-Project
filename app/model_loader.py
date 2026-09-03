import json
import wandb
import onnxruntime as ort
from dotenv import load_dotenv
load_dotenv()


def load_model():
    run = wandb.init(
        project="semantic-segmentation",
        name="log_model_artifact")
    artifact = run.use_artifact("trained-model:latest")
    artifact_dir = artifact.download()

    with open(f"{artifact_dir}/preprocess_config.json") as f:
        config = json.load(f)

    session = ort.InferenceSession(
        f"{artifact_dir}/deeplabv3plus_mitb5.onnx",
        providers=["CPUExecutionProvider"],
    )

    wandb.finish()
    return session, config
