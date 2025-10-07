import os
import json
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BehaviorRecord:
    """Class to store WSI viewing behavior data"""
    wsi_path: str
    image_width: int
    image_height: int
    duration: datetime
    start_time: datetime
    end_time: datetime
    zoom_in_count: int
    zoom_out_count: int
    zoom_chain: List[str]
    zoom_times: List[int]  # microseconds between operations
    point_chain: List[Tuple[int, int]]
    zoom_levels: List[float]
    bounding_boxes: List[Tuple[int, int, int, int]]

    @property
    def zoom_levels_count(self) -> int:
        return len(self.zoom_levels)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for JSON serialization"""
        return {
            'wsi_path': self.wsi_path,
            'image_size': {
                'width': self.image_width,
                'height': self.image_height
            },
            'duration_seconds': self.duration.total_seconds(),
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'zoom_in_count': self.zoom_in_count,
            'zoom_out_count': self.zoom_out_count,
            'zoom_chain': self.zoom_chain,
            'zoom_times_microseconds': self.zoom_times,
            'point_chain': self.point_chain,
            'zoom_levels': self.zoom_levels,
            'zoom_levels_count': self.zoom_levels_count,
            'bounding_boxes': self.bounding_boxes
        }

class NucleiioLog:
    def __init__(self,image_path:str,log:BehaviorRecord):
        self.image_path = image_path
        self.log = log

    def log(self, message):
        print(message)

def parse_wsi_logs(log_file_path: str) -> List[BehaviorRecord]:
    """Parse WSI log file and return a list of BehaviorRecord objects"""
    
    # First pass: Parse raw events (This part is correct)
    wsi_events = {}
    current_wsi = None

    with open(log_file_path, 'r') as file:
        for line in file:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', line)
            if not timestamp_match:
                continue

            timestamp = timestamp_match.group(1)

            if '*** Open slide ***' in line:
                filename_match = re.search(r'filename: ([^,]+)', line)
                resolution_match = re.search(r'(\d+)x(\d+) \[', line)

                if filename_match:
                    current_wsi = filename_match.group(1)
                    if current_wsi not in wsi_events:
                        wsi_events[current_wsi] = {
                            'resolution': {
                                'width': int(resolution_match.group(1)) if resolution_match else None,
                                'height': int(resolution_match.group(2)) if resolution_match else None
                            },
                            'roi_changes': []
                        }

            elif current_wsi:
                if '*** showImage ***' in line:
                    roi_match = re.search(
                        r'\[x1:(-?\d+), y1:(-?\d+), x2:(-?\d+), y2:(-?\d+)\].*currentZoom: ([\d.]+)',
                        line)
                    if roi_match:
                        roi_data = {
                            'timestamp': timestamp,
                            'x1': int(roi_match.group(1)),
                            'y1': int(roi_match.group(2)),
                            'x2': int(roi_match.group(3)),
                            'y2': int(roi_match.group(4)),
                            'zoom': float(roi_match.group(5)) if roi_match.group(5) else None
                        }
                        wsi_events[current_wsi]['roi_changes'].append(roi_data)

    # Second pass: Convert to BehaviorRecords
    behavior_records = []
    
    for wsi_path, data in wsi_events.items():
        if not data['roi_changes']:
            continue

        roi_changes = sorted(data['roi_changes'], 
                           key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S.%f'))

        start_time = datetime.strptime(roi_changes[0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(roi_changes[-1]['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        duration = end_time - start_time
        
        # --- BUG FIX 1: Process ALL events, not just compressed zoom changes ---
        # The 'compressed_events' logic has been removed. We now process every navigation event.
        
        all_events = [(
            roi['zoom'],
            datetime.strptime(roi['timestamp'], '%Y-%m-%d %H:%M:%S.%f'),
            ((roi['x1'] + roi['x2']) // 2, (roi['y1'] + roi['y2']) // 2)
        ) for roi in roi_changes]

        bounding_boxes = [((roi['x1'], roi['y1']), (roi['x2'], roi['y2'])) for roi in roi_changes]
        
        zoom_in_count = 0
        zoom_out_count = 0
        zoom_chain = []
        zoom_times = []
        
        # All zoom levels and points are now taken directly from the full event list
        zoom_levels = [evt[0] for evt in all_events]
        point_chain = [evt[2] for evt in all_events]

        # --- BUG FIX 2: Calculate time difference between EVERY consecutive event ---
        if len(all_events) > 1:
            for i in range(1, len(all_events)):
                # Calculate time difference from the previous event to the current one
                time_diff = all_events[i][1] - all_events[i-1][1]
                microseconds = int(time_diff.total_seconds() * 1_000_000)
                zoom_times.append(microseconds)

                if all_events[i][0] > all_events[i-1][0]:
                    zoom_in_count += 1
                    zoom_chain.append('in')
                elif all_events[i][0] < all_events[i-1][0]:
                    zoom_out_count += 1
                    zoom_chain.append('out')
                else:
                    # This is a panning event
                    zoom_chain.append('pan')
        
        # NOTE: zoom_times will have one less element than the other lists,
        # because it represents the time *between* events.
        # This is expected and can be handled during visualization.

        record = BehaviorRecord(
            wsi_path=wsi_path,
            image_width=data['resolution']['width'],
            image_height=data['resolution']['height'],
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            zoom_in_count=zoom_in_count,
            zoom_out_count=zoom_out_count,
            zoom_chain=zoom_chain,
            zoom_times=zoom_times,
            point_chain=point_chain,
            zoom_levels=zoom_levels,
            bounding_boxes=bounding_boxes
        )
        
        behavior_records.append(record)

    return behavior_records

if __name__ == '__main__':
    plasma_logs = json.load(open('plasma_logs.json', 'r'))

    for log_file in plasma_logs:
        behavior_records = parse_wsi_logs(log_file)
        print(behavior_records)