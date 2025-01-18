import math
import numpy as np

from typing import List
from Misc import assert_return_type_point, verbose_output, Point, RecognitionResult

from BaseGestureRecognizer import BaseGestureRecognizer


class Rectangle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class GestureTemplate:
    def __init__(self, name: str, points: List[Point]):
        self.name = name
        self.points = points


class OneDollarGestureRecognizer(BaseGestureRecognizer):
    def __init__(self, is_data_collection_mode: bool, gesture_type: str, verbose: bool = False,
                 path: str = r'../data/dollar_gesture_capture.csv'):
        self.ignore_rotation = False
        self.half_diagonal = 0.5 * math.sqrt((250.0 ** 2) + (250.0 ** 2))
        self.angle_range = 30.0
        self.angle_precision = 2.0
        self.golden_ratio = 0.5 * (-1.0 + math.sqrt(5.0))
        self.square_size = 250
        super().__init__(path=path, is_data_collection_mode=is_data_collection_mode, gesture_type=gesture_type,
                         verbose=verbose)

    def add_template(self, name: str, points: List[Point], data_type: str) -> None:
        try:
            verbose_output(self, f'\nAdding template with name:{name} for {data_type}')
            verbose_output(self, f'Sending {data_type} for normalization')

            points = self.normalize_path(points, data_type=data_type)

            self.templates.append(GestureTemplate(name, points))
        except AssertionError as e:
            print(e)

    def bound_box(self, points: List[Point], data_type: str) -> Rectangle:
        verbose_output(self, f'Bounding box for {data_type}')

        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for point in points:
            if isinstance(point, Point):
                min_x = min(min_x, point.x)
                max_x = max(max_x, point.x)
                min_y = min(min_y, point.y)
                max_y = max(max_y, point.y)

        width = max_x - min_x
        height = max_y - min_y
        return Rectangle(min_x, min_y, width, height)

    @assert_return_type_point
    def calculate_centroid(self, points: List[Point], data_type: str) -> List[Point]:
        verbose_output(self, f'Calculating centroid for {data_type} with len {len(points)} points')
        count = 0
        total_x = 0.0
        total_y = 0.0
        for point in points:
            if isinstance(point, Point):
                total_x += point.x
                total_y += point.y
                count = count + 1
        if len(points) > 0:
            centroid_x = total_x / len(points)
            centroid_y = total_y / len(points)
            return [Point(centroid_x, centroid_y)]
        else:
            return [Point(0.0, 0.0)]

    @assert_return_type_point
    def normalize_path(self, input_points: List[Point], data_type: str) -> List[Point]:
        verbose_output(self, f'\nNormalizing points for {data_type} ....')
        verbose_output(self, f'Length of input points passed for {data_type} to normalize_path is:{len(input_points)}')

        resampled_points = self.resample(input_points, data_type)

        if self.ignore_rotation:
            resampled_points = self.rotate_to_zero(resampled_points, data_type)

        resampled_points = self.scale_to_square(resampled_points, data_type)
        resampled_points = self.translate_to_origin(resampled_points, data_type)

        verbose_output(self, f'Result from normalize_path for {data_type} is of length:{len(resampled_points)}')

        return resampled_points

    @assert_return_type_point
    def resample(self, input_points: List[Point], data_type: str, desired_length: int = 60) -> List[Point]:
        verbose_output(self, f'\nResampling {data_type} points ....')
        verbose_output(self, f'Length of {data_type} passed to resample is:{len(input_points)}')

        interval = self.calculate_path_length(input_points) / (desired_length - 1)
        D = 0.0
        newPoints = []
        new_length = desired_length

        for i in range(1, len(input_points)):
            if isinstance(input_points[i], Point):
                currentPoint = input_points[i]
                previousPoint = input_points[i - 1]
                d = self.calculate_euclidean_distance(previousPoint, currentPoint)
                if D + d >= interval:
                    if d == 0:
                        continue
                    qx = previousPoint.x + ((interval - D) / d) * (currentPoint.x - previousPoint.x)
                    qy = previousPoint.y + ((interval - D) / d) * (currentPoint.y - previousPoint.y)
                    newPoints.append(Point(qx, qy, 0))
                    input_points.insert(i, Point(qx, qy, 0))
                    D = 0.0
                    new_length -= 1
                    if new_length <= 1:
                        break
                else:
                    D += d

        while len(newPoints) < desired_length:
            newPoints.append(input_points[-1])

        verbose_output(self, f'Return of resampled points from resample for {data_type} is of length:{len(newPoints)}')
        return newPoints[:desired_length]

    @assert_return_type_point
    def scale_to_square(self, points: List[Point], data_type: str) -> List[Point]:
        verbose_output(self, f'\nScaling points for {data_type} with length:{len(points)}')
        box = self.bound_box(points, data_type)
        count = 0
        if box.width == 0 or box.height == 0:
            return points
        newPoints = []
        for point in points:
            scaledX = point.x * (self.square_size / box.width)
            scaledY = point.y * (self.square_size / box.height)
            newPoints.append(Point(scaledX, scaledY))
            count = count + 1
        verbose_output(self, f'Return of scaled points for {data_type} is of length:{len(newPoints)}')
        return newPoints

    @assert_return_type_point
    def translate_to_origin(self, points: List[Point], data_type: str) -> List[Point]:
        verbose_output(self, f'\nTranslation for {data_type} with length:{len(points)}')
        c = self.calculate_centroid(points, data_type)[0]
        newPoints = []
        for point in points:
            qx = point.x - c.x
            qy = point.y - c.y
            newPoints.append(Point(qx, qy))
        verbose_output(self, f'Return of translated points for {data_type} is of length:{len(newPoints)}')
        return newPoints

    @staticmethod
    def calculate_euclidean_distance(p1: Point, p2: Point) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt((dx ** 2) + (dy ** 2))

    @assert_return_type_point
    def rotate_to_zero(self, points: List[Point], data_type: str) -> List[Point]:
        verbose_output(self, f'Rotating all points to zero for {data_type} with length:{len(points)}')
        c = self.calculate_centroid(points, data_type)[0]
        rotation = math.atan2(c.y - points[0].y, c.x - points[0].x)
        return self.rotate_all_points_by(points, -rotation, data_type)

    @assert_return_type_point
    def rotate_all_points_by(self, points: List[Point], rotation: float, data_type: str) -> List[Point]:
        verbose_output(self, f'Rotating all points for {data_type} by {rotation} with length:{len(points)}')
        c = self.calculate_centroid(points, data_type)[0]
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        newPoints = []
        for point in points:
            qx = (point.x - c.x) * cosine - (point.y - c.y) * sine + c.x
            qy = (point.x - c.x) * sine + (point.y - c.y) * cosine + c.y
            newPoints.append(Point(qx, qy))

        verbose_output(self, f'Result of rotating all points for {data_type} is of length:{len(newPoints)}')
        return newPoints

    def calculate_path_length(self, points: List[Point]) -> float:
        length = np.sum([self.calculate_euclidean_distance(points[i - 1], points[i]) for i in range(1, len(points))])
        return length

    def calculate_path_distance(self, input_points: List[Point], points_from_template: List[Point]) -> float:
        verbose_output(self, f'\nLength of template points passed to path_distance is:{len(points_from_template)}')
        verbose_output(self, f'Length of input points passed to path_distance is:{len(input_points)}')

        if len(input_points) != len(points_from_template):
            raise ValueError("Paths must have the same length")

        distance = 0.0
        for i in range(len(input_points)):
            distance += self.calculate_euclidean_distance(input_points[i], points_from_template[i])

        return distance / len(input_points)

    def calculate_distance_at_angle(self, input_points: List[Point], loaded_template: GestureTemplate,
                                    rotation: float, data_type: str) -> float:
        verbose_output(self, f'\nLength of input points passed to distanceAtAngle is:{len(input_points)}')
        verbose_output(self,
                       f'Length of loaded template points passed to distanceAtAngle is:{len(loaded_template.points)}')

        newPoints = self.rotate_all_points_by(input_points, rotation, data_type)
        return self.calculate_path_distance(newPoints, loaded_template.points)

    def calculate_distance_at_best_angle(self, input_points: List[Point], loaded_template: GestureTemplate,
                                         data_type: str) -> float:
        verbose_output(self, f'\nLength of input points passed to distanceAtBestAngle is:{len(input_points)}')
        verbose_output(self,
                       f'Length of loaded template points passed to distanceAtBestAngle is:{len(loaded_template.points)}')

        startRange = -self.angle_range
        endRange = self.angle_range
        x1 = self.golden_ratio * startRange + (1.0 - self.golden_ratio) * endRange
        f1 = self.calculate_distance_at_angle(input_points, loaded_template, x1, data_type)
        x2 = (1.0 - self.golden_ratio) * startRange + self.golden_ratio * endRange
        f2 = self.calculate_distance_at_angle(input_points, loaded_template, x2, data_type)
        while abs(endRange - startRange) > self.angle_precision:
            if f1 < f2:
                endRange = x2
                x2 = x1
                f2 = f1
                x1 = self.golden_ratio * startRange + (1.0 - self.golden_ratio) * endRange
                f1 = self.calculate_distance_at_angle(input_points, loaded_template, x1, data_type)
            else:
                startRange = x1
                x1 = x2
                f1 = f2
                x2 = (1.0 - self.golden_ratio) * startRange + self.golden_ratio * endRange
                f2 = self.calculate_distance_at_angle(input_points, loaded_template, x2, data_type)
        return min(f1, f2)

    def recognize(self, history: List[dict], data_type: str) -> RecognitionResult:
        try:
            input_points = self.convert_history_to_points(history=history)

            if not self.templates:
                verbose_output(self, "No templates loaded so no symbols to match.")
                return RecognitionResult("Unknown", 0.0)

            verbose_output(self, f'\nLength of the points passed to recognize is: {len(input_points)}')
            verbose_output(self, f'Length of the Templates in recognize is: {len(self.templates)}')
            verbose_output(self,
                           f'The template names in recognize are: {[template.name for template in self.templates]}')

            points = self.normalize_path(input_points, data_type='Real World Data')

            bestDistance = float('inf')
            indexOfBestMatch = -1

            for i, template in enumerate(self.templates):
                distance = self.calculate_distance_at_best_angle(points, template, data_type)
                if distance < bestDistance:
                    bestDistance = distance
                    indexOfBestMatch = i

            if indexOfBestMatch == -1:
                verbose_output(self, "Couldn't find a good match.")
                return RecognitionResult("Unknown", 1)

            score = 1.0 - (bestDistance / self.half_diagonal)
            return RecognitionResult(self.templates[indexOfBestMatch].name, score)
        except AssertionError as e:
            print(e)

    def set_rotation_invariance(self, ignore_rotation: bool) -> None:
        self.ignore_rotation = ignore_rotation
        if self.ignore_rotation:
            self.angle_range = 45.0
        else:
            self.angle_range = 15.0
