import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from typing import Dict, List, Tuple
import mahotas as mt

class ImageFeatureExtractor:
    """Enhanced feature extractor with multiple descriptors."""
    
    def __init__(self):
        self.gabor_filters = self._create_gabor_filters()
    
    def _create_gabor_filters(self) -> List[Tuple]:
        """Create Gabor filters with different orientations and frequencies."""
        orientations = 8
        frequencies = [0.1, 0.3, 0.5, 0.7]
        return [(frequency, theta) for theta in np.arange(0, np.pi, np.pi/orientations) 
                for frequency in frequencies]

    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> Dict[str, np.ndarray]:
        """Extract color histogram features for each channel."""
        histograms = {}
        for i, channel in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms[channel] = hist
        return histograms

    def extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> Dict[str, np.ndarray]:
        """Extract dominant colors using K-means clustering."""
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                      cv2.KMEANS_RANDOM_CENTERS)
        # Calculate percentage of each dominant color
        percentages = np.bincount(labels.flatten()) / len(labels)
        return {
            'colors': centers,
            'percentages': percentages
        }

    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Gabor texture features with enhanced parameters."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        feature_names = []
        
        for frequency, theta in self.gabor_filters:
            filt_real, filt_imag = gabor(gray, frequency=frequency, theta=theta)
            features.extend([
                np.mean(filt_real), np.std(filt_real),
                np.mean(filt_imag), np.std(filt_imag)
            ])
            feature_names.extend([
                f'gabor_real_mean_f{frequency}_t{theta}',
                f'gabor_real_std_f{frequency}_t{theta}',
                f'gabor_imag_mean_f{frequency}_t{theta}',
                f'gabor_imag_std_f{frequency}_t{theta}'
            ])
        
        return {
            'features': np.array(features),
            'feature_names': feature_names
        }

    def extract_hu_moments(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Hu moments for shape description."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
        log_moments = -np.sign(moments) * np.log10(np.abs(moments))
        return {
            'moments': log_moments,
            'names': [f'hu_moment_{i+1}' for i in range(len(log_moments))]
        }

    def extract_lbp_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Local Binary Pattern features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                             range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return {
            'histogram': hist,
            'parameters': {
                'radius': radius,
                'n_points': n_points
            }
        }

    def extract_glcm_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Gray Level Co-occurrence Matrix (GLCM) features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        features = {
            'contrast': graycoprops(glcm, 'contrast')[0],
            'dissimilarity': graycoprops(glcm, 'dissimilarity')[0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0],
            'energy': graycoprops(glcm, 'energy')[0],
            'correlation': graycoprops(glcm, 'correlation')[0]
        }
        
        return features

    def extract_hog_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Histogram of Oriented Gradients (HOG) features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win_size = (64, 64)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        # Resize image to match window size
        gray = cv2.resize(gray, win_size)
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               cell_size, num_bins)
        features = hog.compute(gray)
        
        return {
            'features': features.flatten(),
            'parameters': {
                'window_size': win_size,
                'cell_size': cell_size,
                'block_size': block_size,
                'num_bins': num_bins
            }
        }

    def extract_all_features(self, image_path: str) -> Dict[str, Dict]:
        """Extract all features from an image and return structured results."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.resize(image, (224, 224))  # Standardize size
        
        return {
            'color_histogram': self.extract_color_histogram(image),
            'dominant_colors': self.extract_dominant_colors(image),
            'gabor_features': self.extract_gabor_features(image),
            'hu_moments': self.extract_hu_moments(image),
            'lbp_features': self.extract_lbp_features(image),
            'hog_features': self.extract_hog_features(image),
            'glcm_features': self.extract_glcm_features(image)
        }