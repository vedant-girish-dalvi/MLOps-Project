# Concrete Damage Detection — MLOps Pipeline

A production-style inference service for concrete structural damage segmentation, built around a fine-tuned DeepLabV3+ (mit_b5 encoder) model covering 19 damage classes (cracks, spalling, corrosion, efflorescence, exposed rebar, and more). This project focuses on the *deployment and operations* side of ML: model versioning, containerized serving, CI/CD, and monitoring — not model training.

## What it does

Given an image of a concrete surface, the service detects all present damage types and returns a color-coded, labeled segmentation overlay via a REST API — each damage class rendered in a distinct color with its label drawn on the detected region.

## Architecture

- **Model**: DeepLabV3+ (mit_b5), 19 damage classes, exported to ONNX for lightweight, framework-agnostic inference
- **Model registry**: [Weights & Biases Artifacts](https://wandb.ai/) — model weights and preprocessing config (input size, normalization, class map) are versioned together and pulled at service startup, not baked into the image
- **Serving**: FastAPI + ONNX Runtime
- **Containerization**: Docker
- **CI/CD**: GitHub Actions — lint, build, smoke-test on every PR; build & push to GHCR on merge to `main`
- **Monitoring**: request latency and prediction statistics logged for drift tracking *(in progress)*

```
[Client] --image--> [FastAPI /predict] --> [ONNX Runtime] --> [labeled multi-class damage overlay PNG]
                            |
                     [W&B Artifact]
                    (model + config,
                   pulled at startup)
```

## Running locally

```bash
pip install -r requirements.txt
export WANDB_API_KEY=your_key
uvicorn app.main:app --reload
```

## Running with Docker

```bash
docker build -t damage-detector .
docker run -p 8000:8000 -e WANDB_API_KEY=your_key damage-detector
```

## Usage

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict --output result.png
```

## Detected damage classes

Crack, ACrack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars, Bearing, EJoint, Drainage, PEquipment, JTape, WConccor

## Project status

- [x] Model versioning via W&B Artifacts
- [x] FastAPI inference service (ONNX Runtime)
- [x] Dockerized
- [x] CI/CD via GitHub Actions
- [ ] Cloud deployment
- [ ] Monitoring & drift detection

## Notes

This project deliberately excludes heavier inference strategies (test-time augmentation, sliding-window tiling) used in the original research pipeline, in favor of a simpler single-pass approach suited to a real-time API.
