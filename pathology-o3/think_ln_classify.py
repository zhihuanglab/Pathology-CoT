#!/usr/bin/env python3
"""
Think-style Lymph Node Classification with Pre-extracted ROI Images
"""

import json
import os
import base64
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openai
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
from io import BytesIO

# --- Configuration ---
def get_api_server(model_name: str) -> str:
    """Determine API server based on model name."""
    model_name = model_name.lower()
    if model_name.startswith('gpt'):
        return 'oai'
    elif model_name.startswith('grok'):
        return 'xai'
    elif model_name.startswith('gemini'):
        return 'gemini'
    elif model_name == 'qwen/qwen2.5-vl-72b-instruct:free':
        return 'openrouter'
    else:
        return 'openrouter'  # default for other models

def initialize_client(api_server: str):
    """Initialize the OpenAI client based on the API server."""
    if api_server == 'oai':
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('OPENAI_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.openai.com/v1"
        )
    elif api_server == 'sbb':
        OPENAI_API_KEY = os.environ.get('SBB_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('SBB_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.shubiaobiao.com/v1"
        )
    elif api_server == 'xai':
        OPENAI_API_KEY = os.environ.get('XAI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('XAI_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    elif api_server == 'gemini':
        OPENAI_API_KEY = os.environ.get('GEMINI_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('GEMINI_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif api_server == 'openrouter':
        OPENAI_API_KEY = os.environ.get('OPENROUTER_API_KEY')
        if not OPENAI_API_KEY:
            raise ValueError('OPENROUTER_API_KEY environment variable not set')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError('Invalid API server configuration')

# Prompts
DIAGNOSTIC_CRITERIA = '''
Diagnostic Criteria for Lymph Node Metastasis:

POSITIVE indicators:
- Clear clusters of atypical epithelial cells with enlarged, irregular nuclei
- Obvious architectural disruption by tumor cells
- Glandular structures typical of adenocarcinoma
- Sheets or nests of malignant cells replacing lymphoid tissue

NEGATIVE indicators:
- Predominantly normal lymphoid tissue with small, uniform lymphocytes
- Preserved lymph node architecture with intact sinuses
- No clear atypical cell clusters
- Processing artifacts without viable tumor cells

TUMOR DEPOSIT (not positive lymph node):
- Entire node replaced by necrotic/dead tumor tissue
- No remaining viable lymph node architecture
'''

THUMB_PROMPT = '''
This is an H&E WSI thumbnail of a CRC case. The task is to find all positive lymph nodes.
An expert pathologist has identified the regions of interest marked in the image for closer examination.
What is your initial impression of the overall image?

''' + DIAGNOSTIC_CRITERIA + '''

Format your response as: <impression>your overall impression of the slide</impression>'''

ZOOM_IN_PROMPT = '''What is your impression on this region? (This is from a colorectal cancer lymph node section)  
Is lymph node positive for metastatic carcinoma?

''' + DIAGNOSTIC_CRITERIA + '''

Please consider that space between cells is not always cytoplasm, but could be an artifact from processing. 
Lymph node that is dead or completely occupied by tumor / or dead tumor cell, this should be called tumor deposit not a positive lymph node.'''

SUMMARY_PROMPT = """
Please provide a comprehensive final pathological impression and diagnosis based on all the above analyses. Consider:
1. The overall tissue architecture and morphology
2. Findings from each specific region
3. Any patterns or correlations between regions
4. Your final diagnostic impression, please do not consider suspicious region as positive

Format your response EXACTLY as follows:
<final_impression>Your comprehensive final diagnostic impression</final_impression>
<recommendations>Any additional recommendations</recommendations>
<diagnostic_info>
PT_or_LN: "PT" if this is a primary tumor section, "LN" if this is a lymph node section
t_stage: [1-4] if primary tumor, 0 if lymph node
lymph_node_positive: Is lymph node positive for metastatic carcinoma? true/false
positive_regions: if lymph_node_positive is true, give a [1,2,3] like list. if false just say []
suspicious_regions: []
</diagnostic_info>"""

def image_to_base64(image_path: str):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()
    img_str = base64.b64encode(img_data).decode()
    return f"data:image/jpeg;base64,{img_str}"

def load_case_data(case_dir: str):
    """Load thumbnail and ROI images for a case."""
    thumbnail_path = os.path.join(case_dir, "thumbnail.jpg")
    if not os.path.exists(thumbnail_path):
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")
    
    # Find all ROI images
    roi_images = []
    roi_idx = 1
    while True:
        roi_path = os.path.join(case_dir, f"roi_{roi_idx}.jpg")
        if os.path.exists(roi_path):
            roi_images.append(roi_path)
            roi_idx += 1
        else:
            break
    
    if not roi_images:
        raise FileNotFoundError(f"No ROI images found in {case_dir}")
    
    return thumbnail_path, roi_images

def save_image_with_boxes(thumbnail_path: str, num_rois: int, output_path: str):
    """Create thumbnail with numbered boxes indicating ROI locations."""
    # Since we don't have exact bbox coordinates, we'll create a simple numbered overlay
    thumbnail = Image.open(thumbnail_path)
    img_with_boxes = thumbnail.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        font = ImageFont.truetype("Arial", 36)
    except:
        font = ImageFont.load_default()
    
    # Add text indicating number of ROIs analyzed
    text = f"ROIs analyzed: {num_rois}"
    draw.text((10, 10), text, fill="red", font=font)
    
    img_with_boxes.save(output_path, "JPEG", quality=95)
    return output_path

def analyze_roi_regions(roi_images: list, output_dir: str, initial_messages: list = None, model_name: str = "gpt-4o", 
                       image_history_mode: str = "current_only", client: openai.OpenAI = None):
    """
    Analyze each ROI region.
    
    image_history_mode options:
    - "current_only": Only send current ROI image (default)
    - "full_history": Send all previous images in conversation history
    """
    responses = []
    messages_history = initial_messages.copy() if initial_messages else []
    
    for i, roi_path in enumerate(roi_images):
        # Copy ROI image to output directory
        roi_filename = os.path.join(output_dir, f"roi_{i+1}.jpg")
        roi_image = Image.open(roi_path)
        roi_image.save(roi_filename, "JPEG", quality=95)
        
        roi_base64 = image_to_base64(roi_path)
        
        prompt = f"""
        This is the {i+1}th region of interest. {ZOOM_IN_PROMPT}
        """
        
        user_message = {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": roi_base64}},
                {"type": "text", "text": prompt},
            ]
        }
        
        if image_history_mode == "full_history":
            # Include full conversation history
            current_messages = messages_history + [user_message]
        else:
            # Only current ROI image and prompt
            current_messages = [user_message]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=current_messages
        )
        
        region_response = response.choices[0].message.content
        responses.append({
            'roi_index': i + 1,
            'response': region_response
        })
        
        # Add to history for next round if using full history mode
        if image_history_mode == "full_history":
            messages_history.append(user_message)
            messages_history.append({"role": "assistant", "content": region_response})
        
        print(f"ROI {i+1}: {region_response}\n")
    
    return responses

def get_final_summary(initial_response: str, region_responses: list, model_name: str = "gpt-4o", client: openai.OpenAI = None):
    """Generate final diagnostic summary."""
    summary_text = f"""Based on the following pathological analysis conversation:

INITIAL OVERALL IMPRESSION:
{initial_response}

DETAILED REGIONAL ANALYSES:
"""
    for region_data in region_responses:
        roi_idx = region_data['roi_index']
        response = region_data['response']
        summary_text += f"\nROI {roi_idx}:\n{response}\n"
    
    summary_text += SUMMARY_PROMPT
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": summary_text}
        ]
    )
    return response.choices[0].message.content

def create_pdf_report(chat_history: list, output_dir: str, output_filename: str = "pathology_report.pdf"):
    """Create PDF report from chat history."""
    output_path = os.path.join(output_dir, output_filename)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='UserStyle', parent=styles['Normal'], textColor=colors.darkred, spaceAfter=12))
    styles.add(ParagraphStyle(name='AssistantStyle', parent=styles['Normal'], textColor=colors.black, spaceAfter=12))
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
    
    story = [Paragraph("Pathology Analysis Report", title_style), Spacer(1, 20)]
    
    for message in chat_history:
        role = message['role']
        content = message['content']
        if isinstance(content, str):
            style = styles['UserStyle'] if role == 'user' else styles['AssistantStyle']
            story.append(Paragraph(content.replace("\n", "<br/>"), style))
        elif isinstance(content, list):
            for item in content:
                if item['type'] == 'text':
                    style = styles['UserStyle'] if role == 'user' else styles['AssistantStyle']
                    story.append(Paragraph(item['text'].replace("\n", "<br/>"), style))
                elif item['type'] == 'image':
                    img_path = item['image']
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        max_width, max_height = 6 * inch, 8 * inch
                        scale_factor = min(max_width / img_width, max_height / img_height)
                        new_width, new_height = img_width * scale_factor, img_height * scale_factor
                        img_rl = RLImage(img_path, width=new_width, height=new_height)
                        story.append(img_rl)
                        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"PDF report saved as {output_path}")

def analyze_case(case_dir: str, output_folder: str, model_name: str = "gpt-4o", 
                image_history_mode: str = "current_only", client: openai.OpenAI = None) -> dict:
    """Analyze a single case with pre-extracted ROI images."""
    
    # 1. Setup output directory
    case_id = os.path.basename(case_dir)
    output_dir = os.path.join(output_folder, f"{case_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load case data
    try:
        thumbnail_path, roi_images = load_case_data(case_dir)
    except FileNotFoundError as e:
        print(f"Error loading case data: {e}")
        return None
    
    print(f"Analyzing case: {case_id}")
    print(f"Found {len(roi_images)} ROI images")
    print(f"Image history mode: {image_history_mode}")
    
    # 3. Copy thumbnail and create version with ROI indicators
    thumbnail_output = os.path.join(output_dir, "thumbnail.jpg")
    thumbnail_image = Image.open(thumbnail_path)
    thumbnail_image.save(thumbnail_output, "JPEG", quality=95)
    
    thumbnail_with_indicators = os.path.join(output_dir, "thumbnail_with_indicators.jpg")
    save_image_with_boxes(thumbnail_path, len(roi_images), thumbnail_with_indicators)
    
    # 4. Initial analysis of thumbnail
    print("Sending initial thumbnail to VLM...")
    thumbnail_base64 = image_to_base64(thumbnail_with_indicators)
    
    initial_user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": THUMB_PROMPT},
            {"type": "image_url", "image_url": {"url": thumbnail_base64}}
        ]
    }
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[initial_user_message]
    )
    initial_response = response.choices[0].message.content
    print(f"Initial VLM Impression: {initial_response}\n")
    
    # 5. Analyze each ROI
    print("Analyzing each ROI region...")
    initial_messages = [initial_user_message, {"role": "assistant", "content": initial_response}]
    region_responses = analyze_roi_regions(roi_images, output_dir, initial_messages, model_name, image_history_mode, client)
    
    # 6. Get final summary
    print("Getting final summary...")
    final_summary = get_final_summary(initial_response, region_responses, model_name, client)
    print(f"Final Summary: {final_summary}\n")
    
    # 7. Structure chat history for output
    clean_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": THUMB_PROMPT}, 
            {"type": "image", "image": thumbnail_with_indicators}
        ]},
        {"role": "assistant", "content": initial_response}
    ]
    
    for i, region_data in enumerate(region_responses):
        roi_path = os.path.join(output_dir, f"roi_{i+1}.jpg")
        clean_messages.append({
            "role": "user", 
            "content": [{"type": "image", "image": roi_path}]
        })
        clean_messages.append({
            "role": "assistant", 
            "content": region_data['response']
        })
    
    clean_messages.append({"role": "assistant", "content": final_summary})
    
    # 8. Extract diagnostic info
    import re
    diagnostic_info_match = re.search(r'<diagnostic_info>(.*?)</diagnostic_info>', final_summary, re.DOTALL)
    if diagnostic_info_match:
        diagnostic_info = diagnostic_info_match.group(1)
        pt_or_ln = re.search(r'PT_or_LN:\s*"([^"]+)"', diagnostic_info).group(1) if re.search(r'PT_or_LN:\s*"([^"]+)"', diagnostic_info) else "Unknown"
        t_stage = int(re.search(r't_stage:\s*(\d+)', diagnostic_info).group(1)) if re.search(r't_stage:\s*(\d+)', diagnostic_info) else 0
        lymph_node_positive = re.search(r'lymph_node_positive:\s*(true|false)', diagnostic_info).group(1).lower() == "true" if re.search(r'lymph_node_positive:\s*(true|false)', diagnostic_info) else False
    else:
        pt_or_ln, t_stage, lymph_node_positive = "Unknown", 0, False
    
    # 9. Create final output structure
    output = {
        "case_id": case_id,
        "chat_history": clean_messages,
        "roi_count": len(roi_images),
        "PT_or_LN": pt_or_ln,
        "t_stage": t_stage,
        "lymph_node_positive": lymph_node_positive,
        "model_name": model_name,
        "image_history_mode": image_history_mode,
        "timestamp": datetime.now().isoformat()
    }
    
    # 10. Save JSON and PDF report
    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    create_pdf_report(clean_messages, output_dir)
    print(f"All outputs saved to directory: {output_dir}")
    
    return output

def load_annotation_data(annotation_file: str = "annotation_summary_merged_remove_unknown_exclude.csv"):
    """Load the annotation data to get ground truth labels."""
    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file {annotation_file} not found")
        return {}
    
    df = pd.read_csv(annotation_file)
    labels = {}
    
    for _, row in df.iterrows():
        slide_name = row['slide_name']
        case_id = os.path.splitext(slide_name)[0]
        n_positive_ln = row['n_positive_LN']
        ground_truth = 1 if n_positive_ln > 0 else 0
        labels[case_id] = ground_truth
    
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Think-style analysis of lymph node cases with pre-extracted ROI images.')
    parser.add_argument('-d', '--data-dir', default="LNCO2", help='Data directory containing case folders (default: LNCO2)')
    parser.add_argument('-o', '--output', required=True, help='Output folder path')
    parser.add_argument('-m', '--model', default="gpt-4o", help='Model name to use (default: gpt-4o)')
    parser.add_argument('-n', '--num-cases', type=int, default=5, help='Number of cases to test (default: 5)')
    parser.add_argument('--annotation-file', default="annotation_summary_merged_remove_unknown_exclude.csv", 
                       help='Path to annotation CSV file')
    parser.add_argument('--image-history', choices=['current_only', 'full_history'], default='current_only',
                       help='Image history mode: current_only (default) or full_history')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize API client
    api_server = get_api_server(args.model)
    client = initialize_client(api_server)

    # Load annotation data for ground truth
    ground_truth_labels = load_annotation_data(args.annotation_file)

    # Find available cases
    case_dirs = []
    if os.path.exists(args.data_dir):
        for item in os.listdir(args.data_dir):
            case_path = os.path.join(args.data_dir, item)
            if os.path.isdir(case_path):
                # Check if it has both thumbnail and at least one ROI
                thumbnail_path = os.path.join(case_path, "thumbnail.jpg")
                roi_1_path = os.path.join(case_path, "roi_1.jpg")
                if os.path.exists(thumbnail_path) and os.path.exists(roi_1_path):
                    case_dirs.append(case_path)

    if not case_dirs:
        print(f"No valid cases found in {args.data_dir}")
        exit(1)

    print(f"Found {len(case_dirs)} valid cases")
    
    # Limit to requested number of cases
    case_dirs = case_dirs[:args.num_cases]
    
    results = []
    for i, case_dir in enumerate(case_dirs):
        print(f"\n--- Processing case {i+1}/{len(case_dirs)}: {os.path.basename(case_dir)} ---")
        
        try:
            result = analyze_case(case_dir, args.output, args.model, args.image_history, client)
            if result:
                # Add ground truth if available
                case_id = result['case_id']
                if case_id in ground_truth_labels:
                    result['ground_truth'] = ground_truth_labels[case_id]
                    result['predicted'] = 1 if result['lymph_node_positive'] else 0
                    result['correct'] = result['ground_truth'] == result['predicted']
                
                results.append(result)
        except Exception as e:
            print(f"Error processing case {case_dir}: {e}")
            continue
    
    # Save summary results
    summary_path = os.path.join(args.output, f"think_analysis_summary_{args.model}_{args.image_history}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate performance metrics if ground truth available
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    total_with_gt = sum(1 for r in results if 'ground_truth' in r)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total cases processed: {len(results)}")
    print(f"Image history mode: {args.image_history}")
    if total_with_gt > 0:
        accuracy = correct_predictions / total_with_gt
        print(f"Cases with ground truth: {total_with_gt}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
    
    print(f"Summary saved to: {summary_path}")