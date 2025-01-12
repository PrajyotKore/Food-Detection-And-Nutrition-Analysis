import re
import cv2
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class llamaOutput:
    def __init__(self ):
        
        self.base_model = 'llama-3.2-transformers-1b-instruct-v1'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,

            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",  # Use Accelerate to handle devices
            trust_remote_code=True,
        )

        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            # model = 'llama-3.2-transformers-1b-instruct-v1',
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )


    def nutrition_from_yolo_pred_class(self,yolo_clas):
        messages = [{"role": "user", "content": f"What is nutritional value of {yolo_clas} per 100gm?, also, keep it short and concise"}]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
            outputs = self.pipe(prompt, max_new_tokens=400, do_sample=True)
        except:

            outputs = self.pipe(f"What is nutritional value of {yolo_clas} per 100gm?, also, keep it short and concise", max_new_tokens=400, do_sample=True)

        # print(outputs[0]["generated_text"])
        return outputs[0]["generated_text"]


    def parse_nutrition_data(self, text):
        """
        Parse nutrition data from the given text format and create a formatted table.
        
        Args:
            text (str): Input text containing nutrition information
        
        Returns:
            str: Formatted table with nutrition information
        """
        # Remove the header tokens and split by sections
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # Initialize lists to store data
        nutrition_data = []
        current_section = ""
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if it's a section header
            if line.endswith(':') or line.startswith('**') and line.endswith('**'):
                current_section = line.strip('*: ')
                continue
                
            # Extract nutrient and value using regex
            match = re.search(r'\*\*(.+?):\*\* (.+)', line)
            if match:
                nutrient, value = match.groups()
                # Clean up the value
                value = value.strip()
                # Add section information if available
                if current_section and current_section not in ["Serving size", "Nutritional Value"]:
                    nutrient = f"{current_section} - {nutrient}"
                nutrition_data.append([nutrient, value])
            
            # Handle bullet points without bold markers
            match = re.search(r'\* (.+?):', line)
            if match:
                description = match.group(1)
                # Get the rest of the text after the colon
                value = line.split(':', 1)[1].strip()
                nutrition_data.append([description, value])

        # Create table using tabulate
        headers = ["Nutrient/Component", "Value"]
        table = tabulate(nutrition_data, headers=headers, tablefmt="grid")
        return table


    def put_nutrition_and_save(self, processed_img, nutrition, output_directory, image_name,class_name ):
        """
        Overlay nutrition information on an image as a smaller, semi-transparent table and save it.

        Args:
            processed_img (numpy.ndarray): The image to annotate.
            nutrition (str): Nutrition information as a multi-line string.
            output_directory (str): Path to save the processed image.
            image_name (str): Name of the image file.
            font: OpenCV font type.
            font_scale (float): Font scale for text.
            font_thickness (int): Font thickness for text.
            font_color (tuple): Color of the text in (B, G, R) format.
            bg_color (tuple): Background color for the nutrition box in (B, G, R) format.
            transparency (float): Transparency level of the background (0.0 to 1.0).
        """
        font=cv2.FONT_HERSHEY_SIMPLEX; font_scale=0.40 ;font_thickness=1
        font_color=(255, 255, 255); bg_color=(0, 0, 0); transparency=0.5
        # Split nutrition data into lines
        nutrition = f'Nutritional Value for {class_name}:' + nutrition
        lines = nutrition.split("\n")
        
        # Position for the table (top-left corner)
        x_offset, y_offset = 10, 10
        line_height = 10  # Smaller line height for reduced size
        
        # Calculate dimensions of the background rectangle
        max_line_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines])
        table_height = line_height * len(lines)
        
        # Create a transparent overlay
        overlay = processed_img.copy()
        
        # Draw the background rectangle on the overlay
        cv2.rectangle(
            overlay,
            (x_offset, y_offset),
            (x_offset + max_line_width + 10, y_offset + table_height + 10),
            bg_color,
            -1,
        )
        
        # Blend the overlay with the original image for transparency
        cv2.addWeighted(overlay, transparency, processed_img, 1 - transparency, 0, processed_img)
        
        # Overlay each line of text on the image
        for i, line in enumerate(lines):
            y_line = y_offset + (i + 1) * line_height
            cv2.putText(processed_img, line, (x_offset + 5, y_line), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Save the processed image
        output_path = Path(output_directory) / image_name
        Path(output_directory).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        cv2.imwrite(str(output_path), processed_img)
