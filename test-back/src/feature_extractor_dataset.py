import os
import numpy as np
import cv2
import json
import pickle
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
from scipy.stats import moment
import joblib

class FeatureExtractor:
    def __init__(self, rsscn7_path='RSSCN7', cache_path='feature_cache'):
        """
        Initialize feature extractor with paths to dataset and cache
        
        :param rsscn7_path: Path to the RSSCN7 dataset
        :param cache_path: Path to store precomputed features
        """
        self.rsscn7_path = rsscn7_path
        self.cache_path = cache_path
        
        # Ensure cache directory exists
        os.makedirs(cache_path, exist_ok=True)

    def _load_image(self, image_path):
        """
        Load and preprocess image
        
        :param image_path: Path to the image file
        :return: Preprocessed image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def extract_color_histogram(self, image):
        """
        Extract color histogram features
        
        :param image: RGB image
        :return: Dictionary of color histograms for each channel
        """
        hist = {
            'blue': cv2.calcHist([image], [0], None, [256], [0, 256]).flatten(),
            'green': cv2.calcHist([image], [1], None, [256], [0, 256]).flatten(),
            'red': cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        }
        
        # Normalize histograms
        for channel in hist:
            hist[channel] = hist[channel] / np.sum(hist[channel])
        
        return hist

    def extract_dominant_colors(self, image, num_colors=5):
        """
        Extract dominant color features
        
        :param image: RGB image
        :param num_colors: Number of dominant colors to extract
        :return: Dictionary with color percentages
        """
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate percentages
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        return {
            'colors': centers.astype(int).tolist(),
            'percentages': percentages
        }

    def extract_gabor_features(self, image, num_orientations=8, num_scales=5):
        """
        Extract Gabor texture features
        
        :param image: RGB image
        :param num_orientations: Number of orientation filters
        :param num_scales: Number of scale filters
        :return: Dictionary of Gabor features
        """
        gray = rgb2gray(image)
        features = []
        
        for scale in range(num_scales):
            for orientation in range(num_orientations):
                kernel = cv2.getGaborKernel(
                    (21, 21), 
                    3.0, 
                    np.pi * orientation / num_orientations, 
                    10.0 * scale + 1, 
                    0.5, 
                    0
                )
                filtered_image = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(np.mean(filtered_image))
        
        # Normalize features
        features = np.array(features)
        features = (features - features.mean()) / features.std()
        
        return {'features': features}

    def extract_hu_moments(self, image):
        """
        Extract Hu moments invariant features
        
        :param image: RGB image
        :return: Dictionary of Hu moments
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform to handle small values
        for i in range(len(hu_moments)):
            hu_moments[i] = -np.sign(hu_moments[i]) * np.log(abs(hu_moments[i])) if hu_moments[i] != 0 else 0
        
        return {'moments': hu_moments}

    def extract_lbp_features(self, image, radius=1, n_points=8):
        """
        Extract Local Binary Pattern (LBP) features
        
        :param image: RGB image
        :param radius: Radius of circle
        :param n_points: Number of points to sample around the center
        :return: Dictionary of LBP histogram
        """
        gray = rgb2gray(image)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return {'histogram': hist}

    def extract_hog_features(self, image, orientations=9, pixels_per_cell=(16, 16)):
        """
        Extract Histogram of Oriented Gradients (HOG) features
        
        :param image: RGB image
        :param orientations: Number of orientation bins
        :param pixels_per_cell: Size of cell
        :return: Dictionary of HOG features
        """
        gray = rgb2gray(image)
        features = hog(
            gray, 
            orientations=orientations, 
            pixels_per_cell=pixels_per_cell, 
            cells_per_block=(1, 1), 
            visualize=False
        )
        
        return {'features': features}

    def extract_glcm_features(self, image):
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) features
        
        :param image: RGB image
        :return: Dictionary of GLCM statistical features
        """
        gray = rgb2gray(image) * 255
        gray = gray.astype(np.uint8)
        
        glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
        glcm = glcm / np.sum(glcm)
        
        return {
            'mean': np.mean(glcm),
            'std': np.std(glcm),
            'energy': np.sum(glcm**2),
            'entropy': -np.sum(glcm * np.log2(glcm + 1e-10))
        }

    def extract_all_features(self, image_path):
        """
        Extract all features for a given image
        
        :param image_path: Path to the image file
        :return: Dictionary of all extracted features
        """
        image = self._load_image(image_path)
        
        return {
            'color_histogram': self.extract_color_histogram(image),
            'dominant_colors': self.extract_dominant_colors(image),
            'gabor_features': self.extract_gabor_features(image),
            'hu_moments': self.extract_hu_moments(image),
            'lbp_features': self.extract_lbp_features(image),
            'hog_features': self.extract_hog_features(image),
            'glcm_features': self.extract_glcm_features(image)
        }

    def precompute_features(self, categories=None):
        """
        Precompute and cache features for all images in the dataset
        
        :param categories: List of categories to process. If None, process all.
        """
        if categories is None:
            categories = [d for d in os.listdir(self.rsscn7_path) 
                          if os.path.isdir(os.path.join(self.rsscn7_path, d))]
        
        for category in categories:
            category_path = os.path.join(self.rsscn7_path, category)
            cache_category_path = os.path.join(self.cache_path, category)
            
            # Create category cache directory
            os.makedirs(cache_category_path, exist_ok=True)
            
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(category_path, image_file)
                    cache_path = os.path.join(cache_category_path, f"{os.path.splitext(image_file)[0]}_features.pkl")
                    
                    try:
                        # Skip if features already computed
                        if os.path.exists(cache_path):
                            continue
                        
                        # Extract and cache features
                        features = self.extract_all_features(image_path)
                        
                        # Save features using pickle
                        with open(cache_path, 'wb') as f:
                            pickle.dump(features, f)
                        
                        print(f"Processed features for {image_file}")
                    
                    except Exception as e:
                        print(f"Error processing {image_file}: {e}")
    
    def load_cached_features(self, image_path):
        """
        Load precomputed features for a given image
        
        :param image_path: Path to the image file
        :return: Cached features or None if not found
        """
        # Derive cache path based on original image path
        relative_path = os.path.relpath(image_path, self.rsscn7_path)
        category = os.path.dirname(relative_path)
        image_name = os.path.splitext(os.path.basename(relative_path))[0]
        
        cache_path = os.path.join(self.cache_path, category, f"{image_name}")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        return None

# Example usage
if __name__ == "__main__":
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Precompute features for all categories
    extractor.precompute_features()
    
    print("Feature extraction complete. Features cached in feature_cache directory.")