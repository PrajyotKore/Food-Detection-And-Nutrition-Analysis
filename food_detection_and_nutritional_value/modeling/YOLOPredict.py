import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import base64
from pathlib import Path
import numpy as np


class YOLOInference:
    def __init__(self, model_path, conf_threshold=0):
        """
        Initialize YOLO inference class
        
        Args:
            model_path (str): Path to YOLO model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def process_image(self, image_path):
        """
        Process a single image and return detections
        """
        # Check if the input is a PIL Image, base64 string, or file path
        if isinstance(image_path, Image.Image):  # If it's a PIL image
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        elif isinstance(image_path, str):  # If it's a file path or base64 string
            try:
                # Try reading as a file path
                image = cv2.imread(image_path)
            except:
                # If it fails, treat it as base64
                binary_data = base64.b64decode(image_path)
                np_array = np.frombuffer(binary_data, dtype=np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported image input type.")

        # Check if the image was successfully read
        if image is None:
            raise ValueError(f"Could not process image input: {image_path}")
            
        # Run inference
        results = self.model(image)[0]
        
        # Process results
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.conf_threshold:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'score': float(score),
                    'class': results.names[int(class_id)]
                })
        

        image = self.annotate(image, detections)
            
        return image, detections
        
    def process_directory(self, input_dir, output_dir=None):
        """
        Process all images in a directory
        
        Args:
            input_dir (str): Input directory containing images
            output_dir (str, optional): Output directory for processed images
        
        Returns:
            dict: Dictionary with results for each image
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
        results = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        cnt = 0
        for img_path in input_dir.glob('*'):
            if cnt >200:
                break
            if img_path.suffix.lower() in image_extensions:
                try:
                    # Process image
                    processed_img, detections = self.process_image(str(img_path))
                    print(detections)
                    # Save results
                    results[img_path.name] ={
                        'detection': detections,
                        'image': processed_img
                        } 
                     
                    
                    # Save processed image if output directory is specified
                    # if output_dir:
                    #     output_path = output_dir / img_path.name
                    #     cv2.imwrite(str(output_path), processed_img)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            cnt +=1
                    
        return results
    
    def annotate(self, image, detections):
        """
        Annotate the image with bounding boxes and labels from detections.
        
        Args:
            image (numpy.ndarray): The input image to annotate.
            detections (list): A list of detections, where each detection is a dictionary
                            with 'bbox', 'class', and 'score'.
        
        Returns:
            numpy.ndarray: The annotated image.
        """
        if not detections:  # If no detections, return the original image
            return image

        default_color = (144, 238, 144)   # Light green
        overlay = image.copy()
        final_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            score = det['score']

            # Get color for class or use default
            color = default_color

            # Convert bbox coordinates to integers
            x1, y1, x2, y2 = map(int, bbox)

            # Draw semi-transparent background
            alpha = 0.3
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, final_image, 1 - alpha, 0, final_image)

            # Draw gradient border
            border_thickness = 2
            for i in range(border_thickness):
                alpha = (border_thickness - i) / border_thickness
                current_color = tuple(int(c * alpha) for c in color)
                cv2.rectangle(final_image, 
                            (x1 - i, y1 - i), 
                            (x2 + i, y2 + i), 
                            current_color, 
                            1)

            # Create beautiful label background
            label = f"{class_name}: {score:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness)

            # Draw label background with rounded corners
            padding = 5
            label_bg = np.full((text_height + 2 * padding, text_width + 2 * padding, 3),
                            color, dtype=np.uint8)
            label_bg = cv2.addWeighted(label_bg, 0.7, 
                                    np.full_like(label_bg, 255), 0.3, 0)

            # Position label
            label_x = max(x1, 10)
            label_y = max(y1 - 10, text_height + 2 * padding)

            # Blend label background into image
            bg_y1 = label_y - (text_height + 2 * padding)
            bg_y2 = label_y
            bg_x1 = label_x
            bg_x2 = label_x + text_width + 2 * padding

            # Ensure coordinates are within image bounds
            bg_y1 = max(0, bg_y1)
            bg_x1 = max(0, bg_x1)
            bg_y2 = min(image.shape[0], bg_y2)
            bg_x2 = min(image.shape[1], bg_x2)

            # Add label background
            final_image[bg_y1:bg_y2, bg_x1:bg_x2] = cv2.addWeighted(
                final_image[bg_y1:bg_y2, bg_x1:bg_x2], 0.5,
                label_bg[:(bg_y2-bg_y1), :(bg_x2-bg_x1)], 0.5, 0)

            # Add text
            cv2.putText(final_image, label,
                        (label_x + padding, label_y - padding),
                        font, font_scale, (0, 0, 0), font_thickness + 1)
            cv2.putText(final_image, label,
                        (label_x + padding, label_y - padding),
                        font, font_scale, (255, 255, 255), font_thickness)

        return final_image

        
    def process_camera(self, camera_index=0, max_frames=400):
        """
        Process frames from a live camera feed.
        
        Args:
            camera_index (int): Index of the camera (default is 0 for the primary camera).
            max_frames (int): Maximum number of frames to process (default is 200).
            
        Returns:
            dict: Dictionary with results for each frame.
        """
        print()
        # Initialize the camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        results = {}
        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from the camera.")
                break
            
            # if frame_count%1==0:
                # try:
            # Process the frame
            processed_frame, detections = self.process_image(frame)
            print(f"Frame {frame_count}: {detections}")
        
            # Save results
            results[frame_count] = {
                'detection': detections,
                'image': processed_frame,
            }

            # Display the processed frame (opti onal)
            cv2.imshow("Processed Frame", processed_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

                # except Exception as e:
                #     print(f"Error processing frame {frame_count}: {str(e)}")
                #     continue

            frame_count += 1

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        return results


            
        
            
