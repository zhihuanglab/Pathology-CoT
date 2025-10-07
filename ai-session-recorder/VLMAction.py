import numpy as np
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Literal

# Assuming the BehaviorRecord class and the log parser from your previous code are available
from NucleiioLog import BehaviorRecord, parse_wsi_logs

@dataclass
class VLMAction:
    """Represents a single, VLM-friendly action derived from user behavior."""
    action_type: Literal['StayInspect', 'PanInspect']
    magnification_bin: Literal['5x', '10x']
    start_time: float  # As timestamp
    end_time: float    # As timestamp
    bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    @property
    def vlm_command(self) -> str:
        """Returns the action formatted as a VLM command string."""
        return f"<{self.magnification_bin}-{self.action_type.lower()}>"

    def __repr__(self):
        return (f"VLMAction(command='{self.vlm_command}', "
                f"duration={self.end_time - self.start_time:.2f}s, "
                f"bbox={self.bounding_box})")

class BehaviorProcessor:
    """
    Processes a raw BehaviorRecord to extract a sequence of VLM-friendly actions.
    """
    def __init__(self, static_inspect_threshold_s=1.0, pan_inspect_threshold_s=2.0):
        self.STATIC_INSPECT_SECONDS = static_inspect_threshold_s
        self.PAN_INSPECT_SECONDS = pan_inspect_threshold_s

    def _bin_magnification(self, zoom_level: float) -> str:
        """Bins a zoom level to 5x or 10x only."""
        if zoom_level < 7.5:  # Threshold between 5x and 10x
            return '5x'
        else:
            return '10x'

    def process_record(self, record: BehaviorRecord) -> List[VLMAction]:
        """
        Converts a BehaviorRecord into a list of VLMAction objects.
        
        Note: This implementation uses heuristics based on the provided text.
        It prioritizes clarity and can be tuned for more complex scenarios.
        """
        actions = []
        if not record.zoom_times:
            return actions

        # Convert times to seconds for easier comparison
        dwell_times_s = [t / 1_000_000 for t in record.zoom_times]

        # --- Identify "StayInspect" and "PanInspect" actions ---
        i = 0
        while i < len(dwell_times_s):
            start_idx = i
            
            # Check for static inspect: dwell time > threshold
            if dwell_times_s[i] > self.STATIC_INSPECT_SECONDS:
                current_action_end_idx = i
                action_type = "StayInspect"
            # Check for panning inspect: sequence of 'pan' movements > threshold
            elif record.zoom_chain[i:i+1] == ['pan']: # Check if it's the start of a pan sequence
                pan_duration = 0
                pan_end_idx = i
                while pan_end_idx < len(record.zoom_chain) and record.zoom_chain[pan_end_idx] == 'pan':
                    pan_duration += dwell_times_s[pan_end_idx]
                    pan_end_idx += 1
                
                if pan_duration > self.PAN_INSPECT_SECONDS:
                    current_action_end_idx = pan_end_idx -1
                    action_type = "PanInspect"
                else:
                    i += 1
                    continue # Not a long enough pan, move to next event
            else:
                i += 1
                continue # Not an inspect event, move on

            # --- Consolidate the identified action ---
            # Get all points and bboxes within this action's time window
            points_in_action = record.point_chain[start_idx : current_action_end_idx + 1]
            
            if action_type == "StayInspect":
                # For StayInspect, use the original bounding box from the record
                # Since it's a single static event, we use the bounding box from the start_idx
                if hasattr(record, 'bounding_boxes') and record.bounding_boxes:
                    # merged_bbox = record.bounding_boxes[start_idx]
                    # merged_bbox = [merged_bbox[0][0],merged_bbox[0][1],merged_bbox[0][0]+merged_bbox[1][0],merged_bbox[0][1]+merged_bbox[1][1]]
                    # print(merged_bbox)
                    center_x, center_y = points_in_action[0]
                    zoom_level = record.zoom_levels[start_idx]
                    _l = zoom_level * 500
                    # Use a smaller viewport size for stay inspect (e.g., 512x512)
                    merged_bbox = (center_x - _l, center_y - _l, center_x + _l, center_y + _l)
                else:
                    # Fallback: use the single point with a reasonable viewport size
                    center_x, center_y = points_in_action[0]
                    # Use a smaller viewport size for stay inspect (e.g., 512x512)
                    merged_bbox = (center_x - 256, center_y - 256, center_x + 256, center_y + 256)
            else:  # PanInspect
                # For PanInspect, calculate the minimal bounding box that covers all viewports in the action
                all_x = [p[0] for p in points_in_action]
                all_y = [p[1] for p in points_in_action]
                
                # We assume a standard viewport size, e.g., 1024x1024 for calculation
                # A more robust solution might use the actual bounding_boxes from the record
                x_min = min(all_x) - 512
                y_min = min(all_y) - 512
                x_max = max(all_x) + 512
                y_max = max(all_y) + 512
                merged_bbox = (x_min, y_min, x_max, y_max)
            
            avg_zoom = np.mean(record.zoom_levels[start_idx : current_action_end_idx + 1])
            
            action_start_time = record.start_time.timestamp() + sum(dwell_times_s[:start_idx])
            action_end_time = record.start_time.timestamp() + sum(dwell_times_s[:current_action_end_idx+1])

            actions.append(VLMAction(
                action_type=action_type,
                magnification_bin=self._bin_magnification(avg_zoom),
                start_time=action_start_time,
                end_time=action_end_time,
                bounding_box=merged_bbox
            ))
            
            # Move index past the action we just processed
            i = current_action_end_idx + 1
            
        # Sort final actions by start time
        actions.sort(key=lambda x: x.start_time)
        
        return actions

if __name__ == '__main__':
    # --- Example Usage ---
    # 1. First, you need to parse a log file to get a BehaviorRecord
    # log_file = "path/to/your/log_file.log"
    # records = parse_wsi_logs(log_file)
    # example_record = records[0]

    # 2. Create a processor and process the record
    # processor = BehaviorProcessor()
    # vlm_actions = processor.process_record(example_record)

    # 3. Print the results
    # print(f"Processed {len(vlm_actions)} VLM-friendly actions:")
    # for action in vlm_actions:
    #     print(action)
    
    # This part is commented out because we need an actual 'BehaviorRecord' object to run it.
    # You can uncomment and adapt this once you have your records loaded.
    pass