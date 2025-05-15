# Room Dimension Estimator

Room Dimension Estimator is a Python-based tool that leverages computer vision and artificial intelligence to estimate room dimensions from a single webcam image. It detects corners, calculates distances using a known reference scale, and applies a neural network model to provide refined dimension estimates.

## Features

* Capture room image via webcam
* Detect edges and corners using OpenCV
* Set reference scale interactively
* Estimate Length, Width, and Height
* Train a simple neural network for AI-based predictions
* Export measurements in both TXT and JSON formats
* Save debug data for analysis
* Estimate height using image perspective when necessary

## Requirements

* Python 3.7+
* OpenCV
* NumPy
* PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python room_dimension_estimator.py
```

Follow on-screen instructions to capture an image and set a reference scale. Results are displayed and saved locally.

## Output

* `measurements.txt`: Human-readable dimensions
* `measurements.json`: Dimensions in JSON format
* `debug_data.json`: Corner and measurement data for debugging
* `room_image.jpg`: Captured room image

## Contributing

Contributions are welcome! If you'd like to improve the model, enhance image processing, or add new features:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a new Pull Request

Please include tests and documentation with your contributions when possible.

## License

This project is licensed under the MIT License.
