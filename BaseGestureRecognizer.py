import csv
import logging
import os

from collections import defaultdict
from typing import List, Dict
from Misc import Point, assert_return_type_point
from pythontuio import Cursor as TuioCursor


class BaseGestureRecognizer:
    unique_cursor_ids = set()

    def __init__(self, is_data_collection_mode: bool, gesture_type: str,
                 verbose: bool = False, path: str = r'../data/dollar_gesture_capture.csv'):
        self.path_data_collection = path
        self.is_data_collection_mode = is_data_collection_mode

        self.templates = []

        self.verbose = verbose
        self.is_recognizer_ready = False
        self.gesture_type_data_collection = gesture_type
        self.can_start_prediction = False

        if not self.is_data_collection_mode:
            self.is_recognizer_ready = self.load_templates()

    def add_template(self, name: str, points: List[Point], data_type: str):
        #each child will implement their own template addition
        pass

    @assert_return_type_point
    def convert_history_to_points(self, history: List[dict]) -> List[Point]:
        points: List[Point] = []
        for entry in history:
            x = entry["position_x"]
            y = entry["position_y"]
            session_id = entry["session_id"]

            point = Point(x=x, y=y, session_id=session_id)
            points.append(point)

        return points

    def load_templates(self) -> bool:
        try:
            with open(self.path_data_collection, 'r', newline='') as f:
                csv_reader = csv.reader(f)
                gesture_data = defaultdict(list)

                for row in csv_reader:
                    if len(row) >= 4:
                        try:
                            gesture_name = row[0]
                            x = float(row[1])
                            y = float(row[2])
                            session_id = int(row[3])

                            point = Point(x, y, session_id)
                            # since we are using a default dict, it will just create an empty list for the key that we
                            # used, if the key isn't present in the dictionary already. This way we can build upon the
                            # same gesture_name iteratively.
                            gesture_data[gesture_name].append(point)
                        except (IndexError, ValueError) as e:
                            raise RuntimeError(f"Error processing row: {e}")
                    else:
                        raise RuntimeError(f"Invalid row format: {row}")

                # for key, value in gesture_data.items():
                #     print(f'{key}: has {len(value)} point instances in it')

                for name, points_list in gesture_data.items():
                    self.add_template(name=name, points=points_list, data_type='Template Data')

                unique_gestures = self.get_unique_gestures(gesture_data)
                self.output_unique_gestures(unique_gestures)

            return True
        except FileNotFoundError:
            print(f'No file with path {self.path_data_collection} exists! Please check the path and try again.')
        except AssertionError as e:
            logging.error(e)

        return False

    @staticmethod
    def get_unique_gestures(gesture_data: Dict[str, List[Point]]) -> List[str]:
        unique_gestures = list(gesture_data.keys())
        return unique_gestures

    @staticmethod
    def output_unique_gestures(unique_gestures: List[str]) -> None:
        logging.info(f'\n--------------------------------------------------')
        logging.info(f"Unique Gestures ({len(unique_gestures)}) available are:")
        for gesture in unique_gestures:
            logging.info(f"- {gesture}")
        logging.info(f'--------------------------------------------------\n')

    def collect_data(self, cursors: List[TuioCursor], gesture_type: str) -> None:
        try:
            with open(self.path_data_collection, "a", newline='') as capture_file:
                csv_writer = csv.writer(capture_file)
                for cursor in cursors:
                    csv_writer.writerow([
                        gesture_type,
                        cursor.position[0],
                        cursor.position[1],
                        cursor.session_id
                    ])
        except FileNotFoundError as fe:
            print(fe)

    @staticmethod
    def remove_gesture(gesture_name: str, path_to_file: str = r'../data/dollar_gesture_capture.csv') -> None:
        try:
            temp_file = path_to_file + '.temp'
            with open(path_to_file, 'r', newline='') as infile, \
                    open(temp_file, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                row_len = 0
                for row in reader:
                    if row and row[0].lower() == gesture_name.lower():
                        row_len += 1
                        continue
                    writer.writerow(row)

            os.replace(temp_file, path_to_file)
            logging.info(f"\nRemoved all rows ({row_len}) associated with gesture '{gesture_name}'.")
            exit(1)
        except FileNotFoundError:
            logging.error(f'\nNo file with path {path_to_file} exists! Please check the path and try again.')
        except Exception as e:
            logging.error(f"\nError removing gesture '{gesture_name}': {e}")
