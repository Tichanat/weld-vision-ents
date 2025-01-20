import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
from PIL import Image
import json
import sys
import logging
import argparse 
import numpy as np
import configparser
import warnings
import os 
import cv2


def setup_logging(verbose):
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

def load_config(config_file_path):
    """Reads the configuration from the specified file."""
    config = configparser.ConfigParser()
    config.read(config_file_path)
    
    # Access parameters
    model_path = config["model"]["model_path"]
    grey_confidence_threshold= float(config["model"]["grey_confidence_threshold"])
    white_confidence_threshold= float(config["model"]["white_confidence_threshold"])
    result_folder = config["result"]["result_folder"]
    
    return model_path, grey_confidence_threshold, white_confidence_threshold,result_folder

def load_model(model_path):
    """Load the trained Faster R-CNN model."""
    logging.info("Loading the Faster R-CNN model...")
    model = torch.jit.load(model_path)
    logging.info("Model loaded successfully.")
    return model

def preprocess_image(image_path):
    """Load and preprocess the input image."""
    logging.info(f"Loading image from: {image_path}")
    # Convert the image to a tensor  
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(img)
    logging.info("Image preprocessed successfully.")
    return image_tensor

# post processing 
def class_rule_based_on_pixel_value(label, score, box, image,grey_confidence_threshold, white_confidence_threshold):
    """
    Adjust the predicted class based on the average pixel value within the bounding box.
    
    :param label: The original predicted label.
    :param score: The original prediction score.
    :param box: The bounding box [x_min, y_min, x_max, y_max].
    :param image: The image tensor.
    :return: Adjusted label and score.
    """
    
    if image.shape[0] == 3:  # Check if image has 3 channels (RGB)
    # Convert to grayscale manually using the standard weights
        grayscale_image = (0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :])
    else:
        # If the image is already grayscale, skip the conversion
        grayscale_image = image[0, :, :]
    
    if len(box) != 4:
        box = box[0] 
    x_min, y_min, x_max, y_max = map(int, box)  
    cropped_region = grayscale_image[ y_min:y_max, x_min:x_max]
    average_pixel_value = cropped_region.mean().item()
    
    # define pixel value to determine black or grey in config file
    #---------------------------------------------------------------------------

    gray_cond_val = grey_confidence_threshold
    white_cond_val = white_confidence_threshold
    #---------------------------------------------------------------------------

    # class and index 

    # {'cluster_porosity': 1, 'lack_of_fusion': 2, 'porosity_slag': 3, 'slag_line': 4, 'tungsten_inclusion': 5}
    if average_pixel_value <= gray_cond_val: 
        new_label = 4 # change to class Slag line
        return new_label, score
    elif gray_cond_val < average_pixel_value <= white_cond_val:
        new_label = 2 # change to class lack of fusion
        return new_label, score
    elif average_pixel_value > white_cond_val: # tungsten most white
        new_label = 5  # change to class Tungsten 
        return new_label, score
    # If no rule matched, return the original label and score
    return label, score

def apply_rules_to_predictions_for_predict(pred_labels, pred_scores, pred_boxes, images,grey_confidence_threshold, white_confidence_threshold):
    adjusted_labels = []
    adjusted_scores = []
    # class_to_idx {'cluster_porosity': 1, 'lack_of_fusion': 2, 'porosity_slag': 3, 'slag_line': 4, 'tungsten_inclusion': 5}
    class_label = [2,4,5]
    image = images[0] 
    for label, score, box in zip(pred_labels, pred_scores, pred_boxes):
        if len(label) > 1:
            # Iterate over each element in the array
            for single_label, single_score, single_box in zip(label, score, box):
                # print(single_label, single_score, single_box)
                if single_label in class_label : 
                    new_label, new_score = class_rule_based_on_pixel_value(single_label, single_score, single_box, image[0],grey_confidence_threshold, white_confidence_threshold)
                    adjusted_labels.append(new_label)
                    adjusted_scores.append(new_score)
                else: # not in class label  
                    adjusted_labels.append(single_label)
                    adjusted_scores.append(single_score)
        else:
            # single predict 
            if label in class_label: 
                new_label, new_score = class_rule_based_on_pixel_value(label, score, box, image[0])
                adjusted_labels.append(new_label)
                adjusted_scores.append(new_score)
            else:
                # No rule, keep the original prediction
                adjusted_labels.append(label)
                adjusted_scores.append(score)
            adjusted_labels = adjusted_labels[0]
            adjusted_scores = adjusted_scores[0]
    return adjusted_labels, adjusted_scores

def make_json_serializable(data):
    serializable_data = []
    for item in data:
        serializable_data.append({
            'box': item['box'].tolist(),  # Convert ndarray to list
            'label': int(item['label']),  # Convert np.int64 to int
            'score': float(item['score']),  # Convert np.float32 to float
            'class': item['class'] # class no change
        })
    return serializable_data

def predict(model, image_tensor, grey_confidence_threshold, white_confidence_threshold,threshold=0.5):
    """Make predictions on the image tensor."""
    logging.info("Generating predictions...")
    with torch.no_grad():
        model.eval()
        predictions = model([image_tensor])

    # print(predictions[1][0])
    predictions = predictions[1][0] # Single batch output
    # print(predictions)
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    adjusted_labels, adjusted_scores = apply_rules_to_predictions_for_predict(
                [labels], [scores], [boxes], [[image_tensor.cpu()]] ,grey_confidence_threshold, white_confidence_threshold
            )

    raw_results = []
    class_to_idx =  {'cluster_porosity': 1, 'lack_of_fusion': 2, 'porosity_slag': 3, 'slag_line': 4, 'tungsten_inclusion': 5}
    for box, label, score in zip(boxes, adjusted_labels, adjusted_scores):
        if score >= threshold:
            logging.debug(f"Prediction - Box: {box}, Label: {label}, Score: {score}")
            raw_results.append({
                'box': box,  # [xmin, ymin, xmax, ymax]
                'label': label,  # Class label
                'score': score , # Confidence score
                'class': next((k for k, v in class_to_idx.items() if v == label), None)
                # map class to class_name
            })
    json_serializable_result = make_json_serializable(raw_results)
    logging.info(f"Generated {len(raw_results)} predictions above the threshold {threshold}.")
    return raw_results,json_serializable_result

def visualize_predictions_img(image_path, prediction, class_to_idx, result_folder, threshold=0.5):
    # Create a reverse dictionary to map indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = [i['box'] for i in prediction]
    scores = [i['score'] for i in prediction]

    for box, score in zip(boxes, scores):
        if score > threshold:
            # Determine the color of the bounding box based on confidence levels
            if score < 0.1:
                color = (255, 0, 0)  # Red
            elif 0.1 <= score < 0.3:
                color = (255, 165, 0)  # Orange
            elif 0.3 <= score < 0.5:
                color = (255, 255, 0)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            # Draw the bounding box without any text
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    filename = f"{os.path.basename(image_path).split('.')[0]}_bbox.jpg"
    save_path = os.path.join(result_folder, filename)

    # Convert the image back to BGR before saving with OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    return filename

def main(image_path, threshold=0.5, verbose=False):
    setup_logging(verbose)
    logging.info("loading configuration")
    warnings.filterwarnings("ignore", message="RCNN always returns a (Losses, Detections) tuple in scripting")

    config_file_path = "config.ini"   
    class_to_idx={'cluster_porosity': 1, 'lack_of_fusion': 2, 'porosity_slag': 3, 'slag_line': 4, 'tungsten_inclusion': 5}
    # Load the config
    model_path, grey_confidence_threshold, white_confidence_threshold,result_folder_ = load_config(config_file_path)
    logging.info(f"Read configuration from :{config_file_path}")
    logging.info(f"CONFIG :: model path :: {model_path}")
    logging.info(f"CONFIG :: grey_confidence_threshold :: {grey_confidence_threshold}")
    logging.info(f"CONFIG :: white_confidence_threshold :: {white_confidence_threshold}")
    logging.info(f"CONFIG :: result_folder_path :: {result_folder_}")

    logging.info("Starting the prediction process...")
    # Load model
    model = load_model(model_path)

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Get predictions
    raw_results, predictions = predict(model, image_tensor, grey_confidence_threshold, white_confidence_threshold,threshold)

    # print(predictions)
    # Save result in json file in result folder read from config.ini
    result_folder =result_folder_
    os.makedirs(result_folder,exist_ok=True)
    output_json_filename = os.path.join(result_folder,'r_'+os.path.basename(image_path).split('.')[0]+'.json')
    with open(output_json_filename, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)

    result_img_path = visualize_predictions_img(image_path,raw_results,class_to_idx,result_folder,threshold )
    logging.info("Prediction process completed successfully.")
    logging.info(f"Output JSON generated at path :: {output_json_filename}")
    logging.info(f"Result Image with bbox generated at path :: {result_folder}\{result_img_path}")
    return predictions
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on an image using a trained Faster R-CNN model.")

    parser.add_argument("image_path", type=str, help="Path to the input image.") 
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging (DEBUG level).")

    args = parser.parse_args()

    try:
        main(args.image_path, threshold=args.threshold, verbose=args.verbose,)
    except Exception as e:
        logging.error(f"An error occurred during prediction: {str(e)}")
        sys.exit(1)
