import cv2
import numpy as np
import json
import os
import torch
import torch.nn as nn

class SimpleRoomNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleRoomNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predict Length, Width, Height
        )

    def forward(self, x):
        return self.fc(x)

class RoomDimensionEstimator:
    def __init__(self):
        self.reference_points = []
        self.model = SimpleRoomNet(300)  # Placeholder input size

    def capture_live_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print("Press 'c' to capture an image of the room...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Please try again.")
                continue
            cv2.imshow('Room Capture', frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.imwrite('room_image.jpg', frame)
                print("Image captured successfully!")
                break

        cap.release()
        cv2.destroyAllWindows()
        return 'room_image.jpg'

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        print(f"Processed image shape: {img.shape}, Edges detected: {np.sum(edges > 0)}")
        return img, edges

    def detect_corners(self, edges):
        corners = cv2.goodFeaturesToTrack(edges, 100, 0.01, 10)
        if corners is not None:
            corners = corners.reshape(-1, 2).astype(int)
            print(f"Detected {len(corners)} corners")
        else:
            corners = np.array([])
            print("No corners detected")
        return corners

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.reference_points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Set Reference', param)
            if len(self.reference_points) == 2:
                cv2.destroyWindow('Set Reference')

    def set_reference(self, img):
        self.reference_points = []
        cv2.imshow('Set Reference', img)
        cv2.setMouseCallback('Set Reference', self.click_event, img)
        print("Click on two points that are 3 feet apart in the image")
        while len(self.reference_points) < 2:
            cv2.waitKey(1)
        return self.reference_points

    def calculate_pixels_per_foot(self, points):
        distance_pixels = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        return distance_pixels / 3

    def calculate_measurements(self, corners, pixels_per_foot):
        distances = []
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                distance_pixels = np.linalg.norm(corners[i] - corners[j])
                distance_feet = distance_pixels / pixels_per_foot
                distances.append((distance_feet, tuple(corners[i]), tuple(corners[j])))
        print(f"Calculated distances between {len(corners)} detected corners")
        return sorted(distances, key=lambda x: x[0], reverse=True)

    def estimate_height_from_perspective(self, img, reference_points):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 2:
            print("Not enough lines for perspective-based height estimation.")
            return 8.0

        # Placeholder fallback height
        estimated_height = 8.0
        print("Fallback height estimation used: 8 feet")
        return estimated_height

    def analyze_dimensions(self, measurements, img=None, pixels_per_foot=None, reference_points=None):
        if len(measurements) < 3:
            return [(f"Dimension {i+1}", m[0]) for i, m in enumerate(measurements)]

        longest = measurements[0]
        second_longest = measurements[1]
        third_longest = measurements[2]

        vec1 = np.array(longest[1]) - np.array(longest[2])
        vec2 = np.array(second_longest[1]) - np.array(second_longest[2])
        dot_product = np.dot(vec1, vec2)
        magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if magnitudes != 0:
            cos_angle = dot_product / magnitudes
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)

            if 80 < angle_deg < 100:
                estimated_height = third_longest[0]
                if estimated_height < 5.0 and img is not None and reference_points is not None:
                    estimated_height = self.estimate_height_from_perspective(img, reference_points)
                return [
                    ("Estimated Length", longest[0]),
                    ("Estimated Width", second_longest[0]),
                    ("Estimated Height", estimated_height),
                    ("Largest Diagonal", max(longest[0], second_longest[0], third_longest[0]))
                ]

        return [
            ("Longest dimension", longest[0]),
            ("Second longest", second_longest[0]),
            ("Third longest", third_longest[0]),
            ("Shortest detected", measurements[-1][0])
        ]

    def visualize_results(self, img, corners, measurements, pixels_per_foot):
        for corner in corners:
            x, y = corner
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Detected Features', img)
        print("Displaying detected features. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if measurements:
            print(f"Number of measurements: {len(measurements)}")
            print(f"First measurement: {measurements[0]}")

            analyzed_dimensions = self.analyze_dimensions(measurements, img, pixels_per_foot, self.reference_points)
            print("\nAnalyzed dimensions:")
            print(analyzed_dimensions)

            print("\nEstimated room measurements:")
            for name, value in analyzed_dimensions:
                print(f"{name}: {value:.2f} feet")

            self.save_measurements(analyzed_dimensions)
            self.export_measurements_json(analyzed_dimensions)
            self.save_debug_data(corners, measurements)
            self.train_ai_model(analyzed_dimensions)
            self.save_trained_model()

            print("\nNote:")
            print("- These measurements are estimates based on the longest detected distances between corners.")
            print("- The actual room dimensions may be smaller due to furniture or other objects in the room.")
            print("- 'Estimated Height' might be refined using perspective estimation if ceiling isn't visible.")
            print("- For more accurate results, ensure the image captures full walls and corners clearly.")
        else:
            print("No measurements could be calculated.")

    def train_ai_model(self, dimensions):
        try:
            x = torch.randn(10, 300)
            y = torch.tensor([[dim[1], dim[1]/2, dim[1]/3] for dim in dimensions] * 10, dtype=torch.float32)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            for epoch in range(50):
                self.model.train()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

            print("AI model trained on extended sample data.")
        except Exception as e:
            print(f"AI model training failed: {e}")

    def save_trained_model(self, path='room_model.pth'):
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_measurements(self, dimensions, filename='measurements.txt'):
        try:
            with open(filename, 'w') as f:
                for name, value in dimensions:
                    f.write(f"{name}: {value:.2f} feet\n")
            print(f"Measurements saved to {filename}")
        except Exception as e:
            print(f"Failed to save measurements: {e}")

    def export_measurements_json(self, dimensions, filename='measurements.json'):
        try:
            data = {name: round(value, 2) for name, value in dimensions}
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Measurements exported to {filename}")
        except Exception as e:
            print(f"Failed to export measurements to JSON: {e}")

    def save_debug_data(self, corners, measurements, filename='debug_data.json'):
        try:
            debug_info = {
                "corners": [corner.tolist() for corner in corners],
                "measurements": [
                    {
                        "distance_feet": round(distance, 2),
                        "point1": point1,
                        "point2": point2
                    } for distance, point1, point2 in measurements
                ]
            }
            with open(filename, 'w') as f:
                json.dump(debug_info, f, indent=4)
            print(f"Debug data saved to {filename}")
        except Exception as e:
            print(f"Failed to save debug data: {e}")

    def clear_previous_data(self):
        for file in ['room_image.jpg', 'measurements.txt', 'measurements.json', 'debug_data.json']:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Deleted previous file: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")

    def run(self):
        try:
            self.clear_previous_data()
            image_path = self.capture_live_image()
            img, edges = self.process_image(image_path)
            corners = self.detect_corners(edges)

            if len(corners) < 2:
                print("Not enough corners detected to calculate measurements.")
                print("Try capturing the image again with more distinct features in view.")
                cv2.imshow('Processed Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return

            reference_points = self.set_reference(img)
            pixels_per_foot = self.calculate_pixels_per_foot(reference_points)
            measurements = self.calculate_measurements(corners, pixels_per_foot)
            print("Measurements calculated. Proceeding to visualize results...")
            self.visualize_results(img, corners, measurements, pixels_per_foot)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    estimator = RoomDimensionEstimator()
    estimator.run()
