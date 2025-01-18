from __future__ import annotations
import logging
import math
import uuid
from functools import wraps
from typing import Callable, List, TypeVar, Tuple

T = TypeVar('T')


def assert_return_type_point(func: Callable[..., List[Point]]) -> Callable[..., List[Point]]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> List[Point]:
        result = func(*args, **kwargs)

        if not all(isinstance(point, Point) for point in result):
            for idx, point in enumerate(result):
                if not isinstance(point, Point):
                    error_message = (
                        f"Expected type Point for function '{func.__name__}', "
                        f"but the element at index {idx} is of type {type(point)}.\n"
                    )
                    if hasattr(point, '__dict__'):
                        error_message += "Dumping attributes for faulty object:\n"
                        for attr, value in point.__dict__.items():
                            error_message += f"{attr}: {value}\n"

                    raise TypeError(error_message)

        return result

    return wrapper


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0, session_id: int = -1):
        self.x = x
        self.y = y
        self.session_id = session_id

    def __str__(self):
        return f"Point(x={self.x}, y={self.y}, session_id={self.session_id})"


class RecognitionResult:
    def __init__(self, name: str = "Unknown", score: float = 0):
        self.name = name
        self.score = score
        self.unique_id = str(uuid.uuid4())

    def __str__(self):
        return f"RecognitionResult(name={self.name}, score={self.score}, unique_id: {self.unique_id})"


class Object:
    def __init__(self, name: str, position: Tuple[float, float], scale_factor: float,
                 obj_id: int = -1, color: Tuple[int, int, int] = (0, 0, 0), rotation_angle: float = 0):
        self.name = name
        self.center = position
        self.color = color
        self.scale_factor = scale_factor
        self.id = obj_id
        self.rotation_angle = rotation_angle

    def __str__(self):
        return (f'Object(name:{self.name}), position:{self.center}, color:{self.color}, '
                f'size:{self.scale_factor}, id:{self.id}')

    def calculate_rectangle_bounding_box(self, screen_width: int, screen_height: int) -> Tuple[
        Tuple[int, int], Tuple[int, int]]:
        #calc coordinates of the top-left and bottom-right corners of the rectangle
        center_x = int(self.center[0] * screen_width)
        center_y = int(self.center[1] * screen_height)
        half_width = int(self.scale_factor / 2 * screen_width)  #half of the rectangle's width
        half_height = int(self.scale_factor / 2 * screen_height)  #half of the rectangle's height

        top_left = (center_x - half_width, center_y - half_height)
        bottom_right = (center_x + half_width, center_y + half_height)

        return top_left, bottom_right


def verbose_output(obj: object, string: str) -> None:
    if hasattr(obj, 'verbose') and obj.verbose and isinstance(obj.verbose, bool):
        logging.debug(f'\033[97m{string}\033[0m')
