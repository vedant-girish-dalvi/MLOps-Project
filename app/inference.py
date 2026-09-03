import cv2
import numpy as np

DEFAULT_THRESHOLD = 0.5


def read_image_rgb_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_colormap(num_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def overlay_all_damages(class_masks: dict, image: np.ndarray, colormap: np.ndarray,
                        class_map: dict, alpha=0.5) -> np.ndarray:
    """
    class_masks: {class_idx: binary_mask (H, W)}
    colormap: array of shape (num_classes, 3), RGB colors
    class_map: {class_name: class_idx}
    """
    idx_to_name = {v: k for k, v in class_map.items()}
    overlay = image.copy()

    for class_idx, mask in class_masks.items():
        if mask.sum() == 0:
            continue

        color = tuple(int(c) for c in colormap[class_idx])
        contours, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(overlay, [cnt], -1, color, -1)

    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # draw labels at the centroid of each detected class region
    for class_idx, mask in class_masks.items():
        if mask.sum() == 0:
            continue

        color = tuple(int(c) for c in colormap[class_idx])
        ys, xs = np.where(mask > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = idx_to_name.get(class_idx, str(class_idx))

        cv2.putText(blended, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(blended, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, thickness=1, lineType=cv2.LINE_AA)

    return blended


def predict_all_damages(image: np.ndarray, session, config,
                        threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Single-pass ONNX Runtime inference across all damage classes.
    image: RGB numpy array, any size
    session: onnxruntime.InferenceSession
    config: dict from preprocess_config.json (expects 'input_size' and 'class_map')
    """
    H, W = image.shape[:2]
    input_size = tuple(config["input_size"])
    class_map = config["class_map"]  # {class_name: class_idx}
    num_classes = len(class_map)

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: tensor})[
        0]  # (1, num_classes, h, w)

    probs = sigmoid(out[0])  # (num_classes, h, w)

    class_masks = {}
    for class_idx in range(num_classes):
        prob_full = cv2.resize(
            probs[class_idx], (W, H), interpolation=cv2.INTER_LINEAR)
        mask = (prob_full >= threshold).astype(np.uint8)
        if mask.sum() > 0:
            class_masks[class_idx] = mask

    colormap = generate_colormap(num_classes)
    overlay = overlay_all_damages(class_masks, image, colormap, class_map)

    detected_labels = [k for k, v in class_map.items() if v in class_masks]

    return {
        "class_masks": class_masks,
        "overlay": overlay,
        "detected_labels": detected_labels,
    }
