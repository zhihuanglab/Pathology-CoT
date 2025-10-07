import json
import os
import glob
from tqdm import tqdm
import argparse
import http.client
from google import genai
from google.genai import types

import numpy as np

# --- Dependency Check ---
try:
    import cv2
except ImportError:
    print("Error: The 'opencv-python' library is required for image analysis.")
    print("Please install it by running: pip install opencv-python")
    exit()

# --- API Configuration ---
# Using Google GenAI for the analysis
GENAI_API_KEY = "" 
# GENAI_MODEL = "models/gemini-2.5-pro-preview-03-25"
GENAI_MODEL = ""

# Using a separate API for JSON cleaning as a fallback
CLEANING_API_HOST = "api.shubiaobiao.com"
CLEANING_API_PATH = "/v1/chat/completions"
CLEANING_API_KEY = "sk-Liazegu1UIRGErTuc0kkshTzbdqTetjYDpyShOGuFbOE9chu"
CLEANING_API_MODEL = "gpt-4o-mini"
CLEANING_MAX_TOKENS = 5000

# --- Helper Functions ---

def is_image_background(image_path, threshold=230):
    """
    Checks if an image is mostly background by analyzing its average color.
    Returns True if the image is likely background, False otherwise.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # If the image cannot be read, return False
        if img is None:
            print(f"  - WARNING: Unable to read image {image_path}.")
            return False
        
        # Calculate the average color (RGB channels)
        avg_color = np.mean(img, axis=(0, 1))
        
        # Calculate the overall average intensity across the RGB channels
        overall_avg = np.mean(avg_color)

        # If the overall average intensity is above the threshold, it's considered background
        if overall_avg > threshold:
            print(f"  - INFO: {os.path.basename(image_path)} is background (avg: {overall_avg:.2f}) and will be excluded.")
            return True
        return False
    except Exception as e:
        print(f"  - ERROR: Could not analyze image {image_path}. Error: {e}")
        return False

def clean_json_string(input_string):
    """
    Cleans a string containing JSON mixed with other content to return only valid JSON.
    Uses a secondary API as a fallback if the primary parsing fails.
    """
    prompt = f"""You are a JSON cleaning assistant. Your task is to extract only the valid JSON from the input text.
    Remove any surrounding text, comments, or other content that is not part of the JSON.
    Return ONLY the valid JSON string, nothing else. Do not add any explanations or additional text.
    
    Input text:
    {input_string}
    """
    
    payload = json.dumps({
        "model": CLEANING_API_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CLEANING_MAX_TOKENS
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {CLEANING_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection(CLEANING_API_HOST)
        conn.request("POST", CLEANING_API_PATH, payload, headers)
        res = conn.getresponse()
        response_body = res.read().decode("utf-8")
        
        if 200 <= res.status < 300:
            response_data = json.loads(response_body)
            cleaned_json = response_data['choices'][0]['message']['content']
            json.loads(cleaned_json)
            return cleaned_json
        else:
            raise Exception(f"Cleaning API Error {res.status}: {response_body}")
            
    except Exception as e:
        raise Exception(f"Error cleaning JSON: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def parse_gemini_result(s):
    """
    Parses the JSON output from the Gemini API, cleaning it if necessary.
    """
    s = s.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"Initial JSON parsing failed. Attempting to clean...")
        try:
            cleaned_s = clean_json_string(s)
            return json.loads(cleaned_s)
        except Exception as e:
            print(f"Fatal: Could not parse JSON even after cleaning: {e}")
            return {"error": "Failed to parse VLM output", "raw_output": s}


def generate_analysis_for_box(thumb_w_box_path, box_path):
    """
    Calls the Google GenAI API to get analysis for a single thumbnail/box pair.
    """
    print(f"  - Analyzing box: {os.path.basename(box_path)}")
    client = genai.Client(api_key=GENAI_API_KEY)
    
    try:
        # Upload files to the Gemini API
        thumb_file = client.files.upload(file=thumb_w_box_path)
        box_file = client.files.upload(file=box_path)

        prompt = """We record a pathologist's behavior when using the image viewer (the green box indicates where the pathologist zoomed into), the second image is the zoomed-in view.

This is a Whole Slide Image of a Colorectal Cancer case, consisting of lymph nodes and primary tumor sections.
- For lymph nodes, the task is to find positive nodes.
- For the primary tumor, the task is to find potential metastasis and determine the T stage.

Please provide your analysis in JSON format:
{
    "what_you_see_from_this_thumbnail_your_first_impression": "...",
    "if_you_are_a_pathologist_why_you_zoom_into_this_area": "...",
    "in_this_zoom-in_area_what_you_see_what_is_your_impression": "..."
}

Instructions for the JSON content:
- "what_you_see_from_this_thumbnail_your_first_impression": Describe the number of lymph nodes and overall tissue structure.
- "if_you_are_a_pathologist_why_you_zoom_into_this_area": Explain the reasoning for examining this specific area (e.g., "this is the Nth lymph node from the top").
"""

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=thumb_file.uri, mime_type=thumb_file.mime_type),
                    types.Part.from_uri(file_uri=box_file.uri, mime_type=box_file.mime_type),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        # Streaming call to the API
        response = client.models.generate_content(
            model=GENAI_MODEL,
            contents=contents
        )
        return response.text

    except Exception as e:
        print(f"  - ERROR: An exception occurred during API call for {os.path.basename(box_path)}: {e}")
        return None
    finally:
        # Clean up uploaded files
        if 'thumb_file' in locals():
            client.files.delete(name=thumb_file.name)
        if 'box_file' in locals():
            client.files.delete(name=box_file.name)


def find_image_path(base_path):
    """Checks for an image at a given path, preferring .png, then .jpg, then .jpeg."""
    for ext in ['.png', '.jpg', '.jpeg']:
        path = f"{base_path}{ext}"
        if os.path.exists(path):
            return path
    return None

def process_case_folder(case_folder_path, debug=False):
    """
    Main function to process a single case folder. It generates VLM analysis
    for each box and then structures it into a final conversation.json file.
    """
    case_id = os.path.basename(os.path.normpath(case_folder_path))
    print(f"Processing case: {case_id}")

    # --- Stage 1: Generate Analysis for all boxes ---
    single_case_o3_data = {}
    box_files = []
    for ext in ['png', 'jpg', 'jpeg']:
        box_files.extend(sorted(glob.glob(os.path.join(case_folder_path, f"box_*.{ext}"))))

    if not box_files:
        print(f"Warning: No 'box_*.png', 'box_*.jpg', or 'box_*.jpeg' files found in {case_folder_path}. Cannot generate conversation.")
        return

    if debug:
        print("--- DEBUG MODE ENABLED: Processing only the first 2 boxes. ---")
        box_files = box_files[:2]

    for box_path in tqdm(box_files, desc="  Analyzing boxes"):
        box_i = os.path.basename(box_path).split('.')[0].split('_')[-1]
        thumb_base_path = os.path.join(case_folder_path, f"thumbnail_with_box_{box_i}")
        thumb_path = find_image_path(thumb_base_path)

        if thumb_path:
            analysis_text = generate_analysis_for_box(thumb_path, box_path)
            if analysis_text:
                single_case_o3_data[box_i] = {
                    "box": box_path,
                    "thumb": thumb_path,
                    "analysis": parse_gemini_result(analysis_text),
                }
        else:
            print(f"Warning: Thumbnail for box {box_i} (png, jpg, or jpeg) not found. Skipping.")

    if not single_case_o3_data:
        print("Error: Failed to generate analysis for any boxes. Aborting.")
        return

    # --- Stage 2: Process Analysis into a Conversation ---
    print("  - Structuring data into conversation format...")
    
    parent_dir = os.path.dirname(os.path.normpath(case_folder_path))
    boxes_json_path = os.path.join(parent_dir, "boxes.json")
    try:
        with open(boxes_json_path, 'r') as f:
            all_boxes_data = json.load(f)
        case_boxes = all_boxes_data.get(case_id, [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse '{boxes_json_path}': {e}. Coordinates will be missing.")
        case_boxes = []

    main_thumbnail_path = find_image_path(os.path.join(case_folder_path, "thumbnail"))
    if not main_thumbnail_path:
        print("Error: Main thumbnail image (png, jpg, or jpeg) not found. Cannot create conversation.")
        return
        
    first_analysis = single_case_o3_data[list(sorted(single_case_o3_data.keys(), key=int))[0]]["analysis"]
    overview = first_analysis.get("what_you_see_from_this_thumbnail_your_first_impression", "No overview available.")

    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": "This is a HE WSI of a CRC case, consisting of lymph nodes or primary tumor section. What is your initial impression of this image?"},
            {"type": "image", "image": main_thumbnail_path}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": overview}]}
    ]

    zoom_reasons = []
    zoom_coords = []
    for box_num in sorted(single_case_o3_data.keys(), key=int):
        analysis = single_case_o3_data[box_num].get("analysis", {})
        reason = analysis.get("if_you_are_a_pathologist_why_you_zoom_into_this_area", f"Reason for box {box_num} not available.")
        zoom_reasons.append(reason)
        if int(box_num) - 1 < len(case_boxes):
            zoom_coords.append(case_boxes[int(box_num) - 1])
        else:
            zoom_coords.append("[coordinates not found]")

    conversation.append({"role": "user", "content": [{"type": "text", "text": "Are there any places you want to look into further? Give me reasons why you want to look into each region."}]})
    conversation.append({"role": "assistant", "content": [{"type": "text", "text": " ".join([f"{reason} I want to zoom into {coord}." for reason, coord in zip(zoom_reasons, zoom_coords)])}]})

    for i, box_num in enumerate(sorted(single_case_o3_data.keys(), key=int)):
        analysis = single_case_o3_data[box_num].get("analysis", {})
        impression = analysis.get("in_this_zoom-in_area_what_you_see_what_is_your_impression", "No impression available.")
        
        # Find the corresponding cyto_box image and check if it's background
        cyto_box_base_path = os.path.join(case_folder_path, f"cyto_box_{box_num}")
        cyto_box_path = find_image_path(cyto_box_base_path)
        
        system_content = []
        # Include cyto_box only if it exists and is not mostly background
        if cyto_box_path and not is_image_background(cyto_box_path):
            system_content = [
                {"type": "text", "text": f"This is ROI {i+1} and its 40x cyto pattern. What do you observe when you zoom in?"},
                {"type": "image", "image": single_case_o3_data[box_num]["box"]},
                {"type": "image", "image": cyto_box_path}
            ]
        else:
            # Fallback if no cyto_box is found or it is background
            system_content = [
                {"type": "text", "text": f"This is ROI {i+1}. What do you observe when you zoom in?"},
                {"type": "image", "image": single_case_o3_data[box_num]["box"]}
            ]

        conversation.extend([
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": [{"type": "text", "text": impression}]}
        ])

    # --- Stage 3: Save the Final Conversation ---
    output_path = os.path.join(case_folder_path, "conversation.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(conversation, f, indent=4)
        print(f"  - Successfully saved conversation to: {output_path}")
    except IOError as e:
        print(f"  - Error: Could not write final JSON to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a structured conversation.json for a given case folder by analyzing its images with a VLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "case_folder_path",
        help="The absolute path to the case folder containing thumbnail.png, box_*.png, and thumbnail_with_box_*.png files."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to process only the first 2 boxes for faster testing."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.case_folder_path):
        print(f"Error: The provided path '{args.case_folder_path}' is not a valid directory.")
    else:
        process_case_folder(args.case_folder_path, args.debug)
    
    print("\nProcessing complete.")
