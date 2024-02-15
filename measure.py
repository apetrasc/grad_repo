import numpy as np

class Measurement:
    @staticmethod
    def calculate_centerline(left_y, right_y):
        return (left_y + right_y) / 2

    @staticmethod
    def calculate_distance_from_center(car_y, centerline_y):
        return abs(car_y - centerline_y)

    @staticmethod
    def evaluate_trajectory(car_trajectory, left_trajectory, right_trajectory):
        centerline_trajectory = Measurement.calculate_centerline(left_trajectory, right_trajectory)
        total_distance = 0
        for car_y, center_y in zip(car_trajectory, centerline_trajectory):
            total_distance += Measurement.calculate_distance_from_center(car_y, center_y)
        return total_distance / len(car_trajectory)

    @staticmethod
    def cal_dev(v, dt, omega):
        return abs(v * dt * 0.001 * np.tan(omega * dt * 0.001))
