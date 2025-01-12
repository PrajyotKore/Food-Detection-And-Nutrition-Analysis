import io
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
from food_detection_and_nutritional_value.modeling import LLAMA, YOLOPredict


def main(image):
    # Example usage
    model_path = "models/last (1) (1).pt"
    input_directory = "data/raw/archive (1)"
    output_directory = "data/processed"

    # Initialize detector
    detector = YOLOPredict.YOLOInference(model_path)
    llama = LLAMA.llamaOutput()
    
    # Process image
    result = detector.process_image(image)
    
    detections = result[1]
    response = {
        'detected_items': [],
        'nutrition_info': '',
        'annotated_image': ''
    }
    
    processed_img = result[0]  # This should be a numpy array from YOLO
    
    if detections:
        # Process detections
        for det in detections:
            print(f"s- {det['class']}: {det['score']:.2f}")
            print(f"Getting Nutritional information for {det['class']}")
            nutrition = llama.nutrition_from_yolo_pred_class(det['class'])
            formatted_table = llama.parse_nutrition_data(nutrition)
            print(formatted_table)
            response['detected_items'].append(det['class'])
            response['nutrition_info'] = formatted_table
        
        # Convert numpy array to image and then to base64
        try:
            # # Convert numpy array to PIL Image
            # if isinstance(processed_img, np.ndarray):
            #     img_pil = Image.fromarray(processed_img)
            # else:
            #     img_pil = processed_img
            
            # # Create a byte stream
            # img_byte_arr = io.BytesIO()
            
            # # Save the image as JPEG to the byte stream
            # img_pil.save(img_byte_arr, format='JPEG', quality=95)
            
            # # Get the byte array and encode it to base64
            # img_byte_arr = img_byte_arr.getvalue()
            # base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # response['annotated_image'] = base64_str
             # Convert numpy array to PIL Image
            if isinstance(processed_img, np.ndarray):
                img_pil = Image.fromarray(processed_img)
            else:
                img_pil = processed_img
            # Resize the image to 128x128
            resized_img = img_pil.resize((480, 480), Image.Resampling.LANCZOS)

            # Save the resized image locally (optional)
            resized_img.save("resized_annotated.jpg", format="JPEG", quality=95)

            # Create a byte stream
            img_byte_arr = io.BytesIO()

            # Save the resized image as JPEG to the byte stream
            resized_img.save(img_byte_arr, format='JPEG', quality=95)

            # Get the byte array and encode it to Base64
            img_byte_arr = img_byte_arr.getvalue()
            base64_str = base64.b64encode(img_byte_arr).decode('utf-8')

            response['annotated_image'] = base64_str
            
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            response['annotated_image'] = ''
    
    if not response['detected_items']:
        response['nutrition_info'] = 'No foods being detected!! :('
    
    return response
            


    # # Print results
    # for image_name, detections in results.items():
    #     print(f"\nDetections for {image_name}:")
    #     # print(detections)
    #     if detections:
    #         df[image_name] = {}
    #         processed_img =  detections['image']
    #         for det in detections:
    #             nutrition= llama.nutrition_from_yolo_pred_class(det['class'])
    #             # print(nutrition)
    #             # print(f"- {det['class']}: {det['score']:.2f}")
    #             formatted_table = llama.parse_nutrition_data(nutrition)
    #             print(formatted_table)
    #             df[image_name][det['class']] = formatted_table
    #             llama.put_nutrition_and_save(processed_img, formatted_table,output_directory, image_name, det['class'])
    #         cv2.imshow("Processed Frame", processed_img)
    #         cv2.destroyAllWindows()
        
    # pd.DataFrame(df).T.to_csv('validation_output.csv')
