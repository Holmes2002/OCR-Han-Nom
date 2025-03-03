import cv2
import numpy as np
import onnxruntime as ort


class ONNXObjectDetection:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def load_onnx(self):
        """
        Load the ONNX model.
        """
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

    def preprocess(self, img):
        """
        Preprocess the image for the ONNX model.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        ori_shape = img.shape[:2]

        # Resize while keeping aspect ratio
        target_size = (1333, 800)
        scale_factor = min(target_size[0] / ori_shape[1], target_size[1] / ori_shape[0])
        new_size = (int(ori_shape[1] * scale_factor), int(ori_shape[0] * scale_factor))
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        # Pad to make dimensions divisible by 32
        pad_h = (32 - img_resized.shape[0] % 32) % 32
        pad_w = (32 - img_resized.shape[1] % 32) % 32
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        # Normalize
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_normalized = (img_padded - mean) / std

        # Transpose to (C, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Metadata
        meta = {
            "valid_ratio": (new_size[1] / img_padded.shape[1], new_size[0] / img_padded.shape[0]),
            "ori_shape": ori_shape,
            "pad_shape": img_padded.shape[:2],
            "img_shape": new_size,
            "scale_factor": scale_factor,
        }

        return img_batch.astype(np.float32), meta

    def infer_onnx(self, input_tensor):
        """
        Run inference using the ONNX model.
        """
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        return output

    def postprocess(self, rects, metadata):
        """
        Convert bounding boxes to the original image scale and return them.
        """
        boxes = []
        scores = []
        for box in rects[0]:
            x1, y1, x2, y2, score = box
            if score > 0.5:  # Filter based on confidence
                pad_h = (32 - metadata["pad_shape"][0] % 32) % 32
                pad_w = (32 - metadata["pad_shape"][1] % 32) % 32

                # Convert coordinates to the original scale
                x1 = (x1 - pad_w // 2) / metadata["scale_factor"]
                y1 = (y1 - pad_h // 2) / metadata["scale_factor"]
                x2 = (x2 - pad_w // 2) / metadata["scale_factor"]
                y2 = (y2 - pad_h // 2) / metadata["scale_factor"]

                # Clip coordinates to the original image size
                x1 = max(0, min(int(x1), metadata["ori_shape"][1] - 1))
                y1 = max(0, min(int(y1), metadata["ori_shape"][0] - 1))
                x2 = max(0, min(int(x2), metadata["ori_shape"][1] - 1))
                y2 = max(0, min(int(y2), metadata["ori_shape"][0] - 1))

                boxes.append((x1, y1, x2, y2))
                scores.append(score)
        return boxes, scores

    def pipeline(self, image):
        """
        Run the entire pipeline: load, preprocess, infer, and postprocess.
        Returns bounding boxes and scores.
        """
        # Preprocess the image
        self.load_onnx()
        input_tensor, metadata = self.preprocess(image)

        # Perform inference
        rects, _, _ = self.infer_onnx(input_tensor)

        # Postprocess results
        boxes, scores = self.postprocess(rects, metadata)

        return boxes, scores


# Example usage
if __name__ == "__main__":
    # Paths to the image and ONNX model
    image_path = "photo_2024-10-08_15-56-22 (2).jpg"
    model_path = "/home1/data/congvu/HanNom/Chinese_Accident_Docs/ort/end2end.onnx"

    # Initialize the class
    detector = ONNXObjectDetection(model_path)
    detector.load_onnx()

    # Run the pipeline
    results = detector.pipeline(image_path)

    # Print results
    print("Bounding boxes with scores:", results)

    # Draw results on the image
    image = cv2.imread(image_path)
    for box in results:
        x1, y1, x2, y2, score = box
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add confidence score
        cv2.putText(
            image, f"{score:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Save the result image
    cv2.imwrite("result_pipeline.jpg", image)
