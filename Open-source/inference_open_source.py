import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional, Tuple, Union
import requests
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

def send_image_to_endpoint(file_path, url="http://10.9.3.239:4020/Chinese_detection"):
    """
    Sends an image to a specified FastAPI endpoint, processes the response, and returns the annotated image.

    Parameters:
    - file_path (str): The path to the image file to be sent.
    - url (str): The URL of the FastAPI endpoint. Defaults to "http://10.9.3.239:4020/Chinese_detection".

    Returns:
    - np.ndarray: The annotated image with rectangles drawn around detected boxes.
    - dict: The JSON response from the server.
    """
    try:
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        boxes = []
        if image is None:
            raise ValueError("Failed to read the image file.")
        
        # Open the image file in binary mode
        with open(file_path, "rb") as image_file:
            # Prepare the files dictionary for the request
            files = {"file": image_file}
            
            # Send a POST request with the file
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Return the parsed JSON response
            data = response.json()
            
            # Extract the split lines from the 'text'
            text_lines = data['text'].split('\n')

            # Create a dictionary of box information for quick lookup
            box_texts_dict = {entry[-1]: entry for entry in data['info']}

            # Order the boxes based on the split lines
            ordered_boxes = [box_texts_dict[line] for line in text_lines if line in box_texts_dict]

            # Draw rectangles on the image based on the ordered boxes
            for idx, box in enumerate(ordered_boxes):
                x1, y1, x2, y2, _ = box
                boxes.append([x1, y1, x2, y2])
                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(image, str(idx), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return boxes
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except KeyError as e:
        print(f"Unexpected response structure: {e}")
    except ValueError as e:
        print(f"Image reading error: {e}")



class DocumentBuilder():
    """Implements a document builder

    Args:
    ----
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
    """

    def __init__(
        self,
        resolve_lines: bool = True,
        resolve_blocks: bool = True,
        paragraph_break: float = 0.035,
        export_as_straight_boxes: bool = False,
    ) -> None:
        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.export_as_straight_boxes = export_as_straight_boxes

    @staticmethod
    # def _sort_boxes(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort(), boxes
    def _sort_boxes(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort(), boxes
    def sort_boxes_right_to_left(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Sort by xmax in descending order
        sorted_indices = np.argsort(-boxes[:, 2])
        return sorted_indices, boxes[sorted_indices]
    def sort_boxes_top_to_bottom(boxes: np.ndarray, y_max:float) -> Tuple[np.ndarray, np.ndarray]:
        # Sort by ymin in ascending order
        # Filter boxes where ymin < y_max
        filtered_boxes = boxes[boxes[:, 1] < y_max]
        
        # Check if any boxes meet the condition
        if filtered_boxes.size == 0:
            return []  # Return an empty list if no boxes meet the condition
        
        # Sort the filtered boxes by ymin in ascending order
        sorted_indices = np.argsort(filtered_boxes[:, 1])
        
        # Return the first value of sorted_indices
        return sorted_indices[0]
    def overlap_check(self, box1, box2):
        # Calculate the width of each box
        width1 = box1[3] - box1[1]
        width2 = box2[3] - box2[1]

        # Calculate the overlap in the x-axis
        overlap_xmin = max(box1[1], box2[1])
        overlap_xmax = min(box1[3], box2[3])
        overlap_width = overlap_xmax - overlap_xmin

        # Check if the overlap is at least 0.5 of either box's width
        if overlap_width >= 0.5 * width1 or overlap_width >= 0.5 * width2:
            return True
        return False
    def _resolve_sub_lines(self, boxes: np.ndarray, word_idcs: List[int]) -> List[List[int]]:
        """Split a line in sub_lines

        Args:
        ----
            boxes: bounding boxes of shape (N, 4)
            word_idcs: list of indexes for the words of the line

        Returns:
        -------
            A list of (sub-)lines computed from the original line (words)
        """
        lines = []
        # Sort words horizontally
        word_idcs = [word_idcs[idx] for idx in boxes[word_idcs, 0].argsort().tolist()]

        # Eventually split line horizontally
        if len(word_idcs) < 2:
            lines.append(word_idcs)
        else:
            sub_line = [word_idcs[0]]
            for i in word_idcs[1:]:
                horiz_break = True

                prev_box = boxes[sub_line[-1]]
                # Compute distance between boxes
                dist = boxes[i, 0] - prev_box[2]
                # If distance between boxes is lower than paragraph break, same sub-line
                if dist < self.paragraph_break:
                    horiz_break = False

                if horiz_break:
                    lines.append(sub_line)
                    sub_line = []

                sub_line.append(i)
            lines.append(sub_line)

        return lines


    def _resolve_lines(self, boxes: np.ndarray) -> List[List[int]]:
        """Order boxes to group them in lines

        Args:
        ----
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2) in case of rotated bbox

        Returns:
        -------
            nested list of box indices
        """
        # Sort boxes, and straighten the boxes if they are rotated
        idxs, boxes = self._sort_boxes(boxes)

        # Compute median for boxes heights
        lines = []
        words = [idxs[0]]  # Assign the top-left word to the first line
        # Define a mean y-center for the line
        for idx in (idxs[1:]):
            vert_break = True
            bool_overlap = self.overlap_check( boxes[words[-1] ], boxes[idx] )
            if  bool_overlap:
                vert_break = False

            if vert_break:
                # Compute sub-lines (horizontal split)
                lines.append(words)
                words = []
                y_center_sum = 0

            words.append(idx)

        # Use the remaining words to form the last(s) line(s)
        if len(words) > 0:
            # Compute sub-lines (horizontal split)
            lines.append(words)
        return lines


class ONNXObjectDetection:
    def __init__(self, model_path = '/home1/vudinh/NomNaOCR/weights/detection/open-source/cascadercnn.onnx'):
        self.model_path = model_path
        self.session = None
        self.sort_method = DocumentBuilder()
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

    def postprocess(self, rects, metadata, thr_sl = 0.5):
        """
        Convert bounding boxes to the original image scale and return them.
        """
        boxes = []
        scores = []
        for box in rects[0]:
            x1, y1, x2, y2, score = box
            if score > thr_sl:  # Filter based on confidence
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
        boxes, scores = self.postprocess(rects, metadata, thr_sl = 0.01)

        return boxes, scores
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        xmin, ymin, xmax ,ymax = pts
        height, width, _ = image.shape
        decode_box = [ymin/height, xmin/width, ymax/height, xmax/width]
        return image[ymin:ymax, xmin:xmax,:], decode_box

    def inference_API(self, image):
        boxes, scores = self.pipeline(image)
        draw_image = image.copy()
        decoded_boxes = []
        patch_images = []
        for box, score in zip(boxes, scores):
            crop_image, decode_box = self.four_point_transform(image.copy(), box)
            patch_images.append(crop_image)
            decoded_boxes.append(np.array(decode_box))
        decoded_boxes = np.array(decoded_boxes)
        index_order_boxes = self.sort_method._resolve_lines(decoded_boxes )   
        flatten_index = []
        index_order_boxes.reverse()
        for index_list in index_order_boxes:
                flatten_index += index_list
        for idx, idx_box in enumerate(flatten_index):
                    box = boxes[idx_box]
                    xmin, ymin, xmax, ymax = [ int(i) for i in box]
                    cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)  # Draw rectangle in red
                    cv2.putText(draw_image, str(idx), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return draw_image, patch_images, flatten_index, boxes
    def inference_API(self, image):
        cv2.imwrite('tmp.jpg', image)
        boxes = send_image_to_endpoint('tmp.jpg', url = 'http://10.9.3.191:10002/api/v1/ocr')
        draw_image = image.copy()
        decoded_boxes = []
        patch_images = []
        for box in (boxes):
            crop_image, decode_box = self.four_point_transform(image.copy(), box)
            patch_images.append(crop_image)
            decoded_boxes.append(np.array(decode_box))
        decoded_boxes = np.array(decoded_boxes)
        index_order_boxes = self.sort_method._resolve_lines(decoded_boxes )   
        flatten_index = []
        index_order_boxes.reverse()
        for index_list in index_order_boxes:
                flatten_index += index_list
        for idx, idx_box in enumerate(flatten_index):
                    box = boxes[idx_box]
                    xmin, ymin, xmax, ymax = [ int(i) for i in box]
                    cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)  # Draw rectangle in red
                    cv2.putText(draw_image, str(idx), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return draw_image, patch_images, flatten_index, boxes

def inference_Open_source(image):
    model_onnx = ONNXObjectDetection()
    draw_image, patch_images, flatten_index, boxes = model_onnx.inference_API(image)
    return draw_image, patch_images, flatten_index, boxes
# Example usage
if __name__ == "__main__":
    # Paths to the image and ONNX model
    image_path = "/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Pages/Handwritten_dataset/Tung/BNTwEHieafhul1897.1.5.jpg"
    model_path = "cascadercnn.onnx"

    # Initialize the class
    detector = ONNXObjectDetection(model_path)

    # Run the pipeline
    image = cv2.imread(image_path)
    boxes, scores = detector.pipeline(image)

    # Print results

    # Draw results on the image
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add confidence score
        cv2.putText(
            image, f"{score:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Save the result image
    cv2.imwrite("result_pipeline.jpg", image)
