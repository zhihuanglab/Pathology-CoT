#!/usr/bin/env python3
import argparse
import json
import os
import re
from glob import glob


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_text_content(msg):
    if not isinstance(msg, dict):
        return None
    parts = msg.get('content', [])
    for p in parts:
        if isinstance(p, dict) and p.get('type') == 'text':
            return p.get('text', '').strip()
    return None


def extract_impression_and_reasons(messages):
    # Impression: first assistant text before first system
    # Reasons: last assistant text before first system
    first_system_idx = None
    for i, m in enumerate(messages):
        if m.get('role') == 'system':
            first_system_idx = i
            break
    if first_system_idx is None:
        first_system_idx = len(messages)

    impression = None
    reasons = None
    for i in range(first_system_idx):
        m = messages[i]
        if m.get('role') == 'assistant':
            t = find_text_content(m)
            if t:
                if impression is None:
                    impression = t
                reasons = t
    return impression or '', reasons or ''


def split_reasons_by_marker(reasons_text: str) -> list:
    """Split the long 'why zoom' rationale into per-ROI segments.
    We split on the explicit marker 'I want to zoom into [coordinates not found].'
    The marker is not included in the returned segments.
    """
    if not reasons_text:
        return []
    parts = re.split(r"\s*I want to zoom into \[coordinates not found\]\.[\s\n]*", reasons_text.strip())
    # Remove empty tails
    return [p.strip() for p in parts if p and p.strip()]


roi_re = re.compile(r"ROI\s*(\d+)")


def extract_roi_items(messages):
    # Pair each system ROI block with its following assistant text
    roi_to_text = {}
    for i, m in enumerate(messages):
        if m.get('role') == 'system':
            t = find_text_content(m) or ''
            m_roi = roi_re.search(t)
            if not m_roi:
                continue
            idx = int(m_roi.group(1))
            # find next assistant text
            for j in range(i + 1, len(messages)):
                if messages[j].get('role') == 'assistant':
                    roi_to_text[idx] = find_text_content(messages[j]) or ''
                    break
    return roi_to_text


def build_items_for_case(source_root, case_dir, impression, reasons_by_roi_list, roi_texts, use_absolute=True, data_prefix='data'):
    items = []
    # expected files: thumbnail_with_box_{i}.jpeg, box_{i}.jpeg, cyto_box_{i}.jpeg
    # count until missing
    # Determine caseId relative to source_root
    case_rel = os.path.relpath(case_dir, start=source_root)

    def path_for(filename):
        p = os.path.join(case_dir, filename)
        if use_absolute:
            return p
        return os.path.join(data_prefix, case_rel, filename)

    # Try discover max i by existing box_i
    indices = []
    for path in glob(os.path.join(case_dir, 'box_*.jpeg')):
        m = re.search(r'box_(\d+)\.jpeg$', path)
        if m:
            indices.append(int(m.group(1)))
    for path in glob(os.path.join(case_dir, 'box_*.jpg')):
        m = re.search(r'box_(\d+)\.jpg$', path)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        return items
    for ord_idx, i in enumerate(sorted(set(indices)), start=1):
        box_name = f'box_{i}.jpeg'
        cyto_name = f'cyto_box_{i}.jpeg'
        thumb_name = f'thumbnail_with_box_{i}.jpeg'
        box_path = os.path.join(case_dir, box_name)
        cyto_path = os.path.join(case_dir, cyto_name)
        if not os.path.exists(box_path) or not os.path.exists(cyto_path):
            continue
        if not os.path.exists(os.path.join(case_dir, thumb_name)):
            # fallback to shared thumbnail with boxes or plain thumbnail
            if os.path.exists(os.path.join(case_dir, 'thumbnail_with_boxes.jpeg')):
                thumb_name = 'thumbnail_with_boxes.jpeg'
            else:
                thumb_name = 'thumbnail.jpeg'
        # Choose whyDraft by order if available
        why_draft = ''
        if reasons_by_roi_list:
            # Map by ordinal order across discovered ROI indices
            if ord_idx - 1 < len(reasons_by_roi_list):
                why_draft = reasons_by_roi_list[ord_idx - 1]

        item = {
            'caseId': case_rel,
            'roiIndex': i,
            'thumb': path_for(thumb_name),
            'box': path_for(box_name),
            'cyto': path_for(cyto_name),
            'thumbDraft': impression,
            'whyDraft': why_draft,
            'boxDraft': roi_texts.get(i, ''),
            'cytoDraft': roi_texts.get(i, ''),
        }
        items.append(item)
    return items


def main():
    ap = argparse.ArgumentParser(description='Generate manifest.json for HIL app from conversation.json and images')
    ap.add_argument('--source-root', help='Root folder containing case folders (e.g., .../all_jpg)')
    ap.add_argument('--output', required=True, help='Output manifest.json path')
    ap.add_argument('--relative', action='store_true', help='Emit relative paths under data/<case> instead of absolute paths')
    ap.add_argument('--cases', nargs='*', help='Explicit list of case directories (each contains conversation.json and images)')
    args = ap.parse_args()

    if not args.source_root and not args.cases:
      ap.error('Provide --source-root or --cases list')

    source_root = os.path.abspath(args.source_root) if args.source_root else None
    out_path = os.path.abspath(args.output)
    use_absolute = not args.relative

    if args.cases:
        case_dirs = [os.path.abspath(p) for p in args.cases]
        conv_paths = [os.path.join(p, 'conversation.json') for p in case_dirs]
    else:
        conv_paths = sorted(glob(os.path.join(source_root, '**', 'conversation.json'), recursive=True))
    manifest = []
    for conv in conv_paths:
        case_dir = os.path.dirname(conv)
        try:
            conv_data = read_json(conv)
        except Exception as e:
            print(f"Skip {conv}: {e}")
            continue
        impression, reasons_long = extract_impression_and_reasons(conv_data)
        reasons_by_roi_list = split_reasons_by_marker(reasons_long)
        roi_texts = extract_roi_items(conv_data)
        items = build_items_for_case(source_root or os.path.dirname(case_dir), case_dir, impression, reasons_by_roi_list, roi_texts, use_absolute=use_absolute)
        # Skip cases with no ROIs
        if items:
            manifest.extend(items)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(manifest)} items to {out_path}")


if __name__ == '__main__':
    main()


