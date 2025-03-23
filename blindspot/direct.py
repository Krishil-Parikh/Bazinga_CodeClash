import cv2
import time
from ultralytics import YOLO

class SimpleYOLOv8Detector:
    def __init__(self, model_path="yolov8m.pt", conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLOv8 detector using the official Ultralytics implementation
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load the model directly using the Ultralytics YOLO class
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize performance tracking
        self.frame_times = []
        self.total_frames = 0
    
    def detect(self, frame):
        """Detect objects in frame"""
        # Start timer
        start_time = time.time()
        
        # Run inference - Ultralytics handles preprocessing internally
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                cls_name = result.names[cls_id]
                
                # Create detection object
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': cls_id,
                    'score': conf,
                    'label': f"{cls_name} {conf:.2f}"
                })
        
        # Calculate FPS
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        self.total_frames += 1
        
        # Only keep the last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return detections, fps
    
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
                2
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
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                    cv2.imshow("YOLOv8 Detection", result)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
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
            cv2.imshow("YOLOv8 Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result

# Example usage
if __name__ == "__main__":
    # Create detector - use the default yolov8m.pt model or specify your own path
    detector = SimpleYOLOv8Detector(
        model_path="yolov8m.pt",  # Will download if not present
        conf_threshold=0.25
    )
    
    # Process a single image
    detector.process_image(
        image_path="foggy_image.jpg",
        output="detection_result.jpg",
        show=True
    )
    
    # Or process video
    # detector.process_video(
    #     source="foggy_video.mp4",  # or 0 for webcam
    #     output="detection_result.mp4",
    #     show=True
    # )
