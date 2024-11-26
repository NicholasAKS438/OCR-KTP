import cv2
import numpy as np

def compute_focus_measure(image: np.ndarray) -> float:
    """
    Compute the focus measure of an image using the Laplacian variance method.
    A higher variance indicates a sharper (less blurry) image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def detect_blur_with_grid(image: np.ndarray, grid_size: int = 5) -> float:
    """
    Detect blur in an image by subdividing it into a grid, computing focus measures
    for each grid cell, and aggregating the results.
    
    Args:
    - image (np.ndarray): The input image.
    - grid_size (int): The number of subdivisions along each axis (e.g., 5x5 grid).

    Returns:
    - float: The aggregated focus measure for the full image.
    """
    h, w, _ = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    focus_values = []

    # Subdivide the image into grid cells and compute focus measure for each cell
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = image[y1:y2, x1:x2]
            focus_values.append(compute_focus_measure(cell))

    # Aggregate focus measures (e.g., using max, mean, or median)
    aggregated_focus = np.mean(focus_values)
    return aggregated_focus

def is_blurry(image: np.ndarray, threshold: float, grid_size: int = 5) -> bool:
    """
    Determine if an image is blurry based on a threshold for the focus measure.

    Args:
    - image (np.ndarray): The input image.
    - threshold (float): The threshold below which the image is considered blurry.
    - grid_size (int): The number of subdivisions along each axis.

    Returns:
    - bool: True if the image is blurry, False otherwise.
    """
    focus_measure = detect_blur_with_grid(image, grid_size=grid_size)
    print(f"Focus Measure: {focus_measure}")
    return focus_measure < threshold

# Example usage
if __name__ == "__main__":
    # Load the image
    image_path = "C:\\OCR-KTP\\OCRR\\OCR-KTP\\ktp\\clear\\ktpIMG-20221222-WA0003_jpg.rf.40b635b596d23e517421a74e6a4ddd4d.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    # Set the threshold for blurriness and check
    blur_threshold = 100.0  # Adjust based on experiments
    grid_size = 5  # Subdivide the image into a 5x5 grid
    result = is_blurry(image, blur_threshold, grid_size)
    
    if result:
        print("The image is blurry.")
    else:
        print("The image is sharp.")
