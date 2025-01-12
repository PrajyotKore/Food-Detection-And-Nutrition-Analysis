
import cv2
import pandas as pd
from food_detection_and_nutritional_value.modeling import LLAMA, YOLOPredict


if __name__ == "__main__":
    # Example usag
    model_path = "models/last (1) (1).pt"
    input_directory = "data/raw/archive (1)"
    output_directory = "data/processed"

    # Initialize detector
    detector = YOLOPredict.YOLOInference(model_path)
    llama = LLAMA.llamaOutput()
    # Process all images
    results = detector.process_directory(input_directory, output_directory)
    # results =  detector.process_camera()


    df = {}

    # Print results
    for image_name, detections in results.items():
        print(f"\nDetections for {image_name}:")
        # print(detections)
        if detections['detection']:
            df[image_name] = {}
            processed_img =  detections['image']
            for det in detections['detection']:
                nutrition= llama.nutrition_from_yolo_pred_class(det['class'])
                # print(nutrition)
                # print(f"- {det['class']}: {det['score']:.2f}")
                formatted_table = llama.parse_nutrition_data(nutrition)
                print(formatted_table)
                df[image_name][det['class']] = formatted_table
                llama.put_nutrition_and_save(processed_img, formatted_table,output_directory, image_name, det['class'])
            cv2.imshow("Processed Frame", processed_img)
            cv2.destroyAllWindows()
        
    # pd.DataFrame(df).T.to_csv('validation_output.csv')
