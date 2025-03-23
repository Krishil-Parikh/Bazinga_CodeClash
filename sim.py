import cv2
import numpy as np
import time
import onnxruntime as ort
from traditional_enhancement import ImprovedEnhancement

class RealTimeDetector:
    def __init__(self, 
                 model_path="/Users/krishilparikh/Desktop/CodeClash/bdd100k_augmented/yolov8m_fast_train/weights/best.onnx",
                 img_size=416,
                 conf_threshold=0.01,  # Reduced confidence threshold for debugging
                 iou_threshold=0.45,
                 enable_enhancement=False):  # Disabled enhancement for cleaner debugging
        """
        Initialize real-time object detector with traditional enhancement
        
        Args:
            model_path: Path to ONNX model
            img_size: Input size for model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            enable_enhancement: Enable image enhancement
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_enhancement = enable_enhancement
        
        # Initialize enhancement pipeline
        if self.enable_enhancement:
            self.enhancer = ImprovedEnhancement(
                use_clahe=True,
                use_bilateral=True,
                use_gamma=True,
                use_unsharp=True,
                use_dehazing=True,
                use_contrast_stretch=True,
                use_adaptive_gamma=True,
                clahe_clip=3.0,
                gamma=1.8,
                unsharp_strength=0.5,
                dehazing_strength=0.6,
                night_mode=True
            )
        
        # Print model path for debugging
        print(f"Loading model from: {model_path}")
        
        # Initialize ONNX Runtime
        self._setup_onnx(model_path)
        
        # Get class names (adjust based on your training data)
        self.class_names = ['person', 'car', 'bus', 'truck']
        
        # Initialize performance tracking
        self.frame_times = []
        self.total_frames = 0
    
    def _setup_onnx(self, model_path):
        """Set up ONNX Runtime session"""
        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self._is_cuda_available() else ['CPUExecutionProvider']
            print(f"Using providers: {providers}")
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Print model info for debugging
            print(f"Model input name: {self.input_name}")
            print(f"Model input shape: {self.input_shape}")
            print(f"Model output names: {self.output_names}")
            
            # Warmup
            self._warmup()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _is_cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            return cuda_available
        except:
            print("Torch not available, defaulting to CPU")
            return False
    
    def _warmup(self):
        """Warm up the model"""
        print("Warming up model...")
        dummy_input = np.random.rand(1, 3, self.img_size, self.img_size).astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: dummy_input})
        print(f"Warmup output shapes: {[out.shape for out in outputs]}")
    
    def preprocess(self, frame):
        """Preprocess frame for model input"""
        # Apply enhancement if enabled
        if self.enable_enhancement:
            frame = self.enhancer.process(frame)
        
        # Get original dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Calculate scale and new dimensions
        scale = min(self.img_size / orig_h, self.img_size / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox (center image on black background)
        letterbox = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        
        # Calculate offsets for centering
        offset_h, offset_w = (self.img_size - new_h) // 2, (self.img_size - new_w) // 2
        
        # Place resized image on letterbox
        letterbox[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized
        
        # Convert BGR to RGB, transpose to CHW format, and normalize
        img = letterbox[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img) / 255.0
        img = img.astype(np.float32)
        
        return img, (orig_h, orig_w, scale, (offset_h, offset_w))
    
    def detect(self, frame):
        """Detect objects in frame"""
        # Start timer
        start_time = time.time()
        
        # Preprocess frame
        img, meta = self.preprocess(frame)
        
        # Debug info
        print(f"Preprocessed input shape: {img.shape}")
        print(f"Input range: min={img.min():.4f}, max={img.max():.4f}")
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img[None, ...]})
        
        # Debug info
        print(f"Model output shapes: {[out.shape for out in outputs]}")
        if len(outputs) > 0 and outputs[0].size > 0:
            print(f"Output[0] sample: {outputs[0][0, 0, :5] if len(outputs[0].shape) > 2 else outputs[0][0, :5]}")
            print(f"Max confidence value: {np.max(outputs[0]):.6f}")
        
        # Postprocess results
        detections = self._postprocess(outputs, meta)
        print(f"Detections found: {len(detections)}")
        
        # Calculate FPS
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        self.total_frames += 1
        
        # Only keep the last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return detections, fps
    
    def _postprocess(self, outputs, meta):
        """Process ONNX model outputs with proper box handling for YOLOv8"""
        orig_h, orig_w, scale, (offset_h, offset_w) = meta
        
        # Get the predictions
        predictions = outputs[0]
        
        # Handle YOLOv8 output format
        if len(predictions.shape) == 3:
            predictions = predictions.squeeze(0)
        
        # Extract boxes and scores
        boxes = predictions[:, :4].copy()
        scores = predictions[:, 4:]
        
        # Get max scores and corresponding class indices
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence threshold
        mask = max_scores > self.conf_threshold
        if not np.any(mask):
            return []
        
        # Apply mask to get confident detections
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]
        
        # Convert from center format to corner format (x_center, y_center, width, height -> x1, y1, x2, y2)
        converted_boxes = []
        for box in filtered_boxes:
            center_x, center_y, width, height = box
            
            # Convert to corner format (already in pixel coordinates)
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            converted_boxes.append([x1, y1, x2, y2])
        
        boxes_to_use = np.array(converted_boxes)
        
        # Important: Validate boxes before rescaling - ensure positive width and height
        valid_mask = (boxes_to_use[:, 2] > boxes_to_use[:, 0]) & (boxes_to_use[:, 3] > boxes_to_use[:, 1])
        boxes_to_use = boxes_to_use[valid_mask]
        filtered_scores = filtered_scores[valid_mask]
        filtered_class_ids = filtered_class_ids[valid_mask]
        
        if len(boxes_to_use) == 0:
            return []
        
        # Rescale boxes to original image dimensions
        boxes_to_use = self._rescale_boxes(boxes_to_use, (offset_h, offset_w), scale, (orig_h, orig_w))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_to_use.tolist(), filtered_scores.tolist(), self.conf_threshold, self.iou_threshold
        )
        
        # Format results
        detections = []
        
        # Handle OpenCV version differences in NMS output format
        if isinstance(indices, np.ndarray) and len(indices.shape) == 2:
            indices = indices.flatten()
        
        for i in indices:
            # For OpenCV < 4.5.4
            if isinstance(i, list) or (isinstance(i, np.ndarray) and i.size > 1):
                i = i[0]
                
            box = boxes_to_use[i].round().astype(np.int32)
            cls_id = int(filtered_class_ids[i])
            score = float(filtered_scores[i])
            
            # Safety check for class index
            label_idx = min(cls_id, len(self.class_names) - 1)
            label = f"{self.class_names[label_idx]} {score:.2f}"
            
            # Final validation check - ensure the box is within image bounds
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(orig_w, box[2])
            box[3] = min(orig_h, box[3])
            
            # Only add if box has positive width and height
            if box[2] > box[0] and box[3] > box[1]:
                detections.append({
                    'box': box.tolist(),
                    'class_id': cls_id,
                    'score': score,
                    'label': label
                })
        
        return detections
    
    def _rescale_boxes(self, boxes, offsets, scale, img_shape):
        """Properly rescale boxes to original image dimensions"""
        offset_h, offset_w = offsets
        orig_h, orig_w = img_shape
        
        # Make a copy to avoid modifying the original array
        boxes = boxes.copy()
        
        # Adjust for letterboxing offset
        boxes[:, [0, 2]] -= offset_w
        boxes[:, [1, 3]] -= offset_h
        
        # Rescale
        boxes /= scale
        
        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        return boxes
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            box = det['box']
            label = det['label']
            
            # Draw box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label background
            text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame, 
                (box[0], box[1] - text_h - 4), 
                (box[0] + text_w, box[1]), 
                (0, 255, 0), 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (box[0], box[1] - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2  # Reduced thickness for better readability
            )
        
        return frame
    
    def process_video(self, source=0, output=None, show=True):
        """Process video source"""
        # Open video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties: {width}x{height} at {fps} FPS")
        
        # Create video writer if needed
        writer = None
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        max_frames = 30  # Only process this many frames for debugging
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"\n--- Processing frame {frame_count} ---")
            
            # Run detection
            detections, fps_rate = self.detect(frame)
            
            # Draw detections
            result = self.draw_detections(frame.copy(), detections)
            
            # Add FPS counter
            cv2.putText(
                result, 
                f"FPS: {fps_rate:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            # Write frame
            if writer:
                writer.write(result)
            
            # Show frame
            if show:
                cv2.imshow("Real-time Object Detection", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame for analysis
                    cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
                    cv2.imwrite(f"debug_result_{frame_count}.jpg", result)
                    print(f"Saved debug frames for frame {frame_count}")
            
            frame_count += 1
            
            # Slow down processing for debugging
            time.sleep(0.5)
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    
    def process_image(self, image_path, output=None, show=True):
        """Process a single image"""
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        print(f"Image shape: {frame.shape}")
        
        # Run detection
        detections, fps = self.detect(frame)
        
        # Draw detections
        result = self.draw_detections(frame.copy(), detections)
        
        # Add FPS info
        cv2.putText(
            result, 
            f"Processing time: {1000/fps:.1f}ms", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # Save result
        if output:
            cv2.imwrite(output, result)
            print(f"Saved result to {output}")
        
        # Show result
        if show:
            # Create comparison
            height = max(frame.shape[0], result.shape[0])
            width = frame.shape[1] + result.shape[1]
            comparison = np.zeros((height, width, 3), dtype=np.uint8)
            comparison[:frame.shape[0], :frame.shape[1]] = frame
            comparison[:result.shape[0], frame.shape[1]:] = result
            
            # Add labels
            cv2.putText(
                comparison, 
                "Original", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            cv2.putText(
                comparison, 
                "Processed + Detection", 
                (frame.shape[1] + 10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            cv2.imshow("Object Detection", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result

# Example usage
if __name__ == "__main__":
    # Create detector
    detector = RealTimeDetector(
        model_path="bdd100k_augmented/yolov8m_fast_train/weights/best.onnx",
        conf_threshold=0.01,  # Very low threshold for debugging
        enable_enhancement=False  # Disable enhancement for cleaner debugging
    )
    
    # Process a single image first (easier to debug)
    print("\n--- Processing test image ---")
    detector.process_image(
        image_path="foggy_image.jpg",  # Use an image with clear objects
        output="debug_detection.jpg",
        show=True
    )
    
    # Then try video if image works
    # print("\n--- Processing video ---")
    # detector.process_video(
    #     source="854204-hd_1920_1080_30fps.mp4",
    #     output="debug_detection.mp4",
    #     show=True
    # )