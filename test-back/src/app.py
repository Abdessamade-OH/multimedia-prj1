from flask import Flask, request, jsonify, send_file, url_for
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
import os
import json
import cv2
import numpy as np
import pickle
from flask import url_for
from flask_swagger_ui import get_swaggerui_blueprint
from feature_extractor import ImageFeatureExtractor

app = Flask(__name__)
api = Api(app)

# Configure folders
UPLOAD_FOLDER = 'upload_folder'
TRANSFORMED_FOLDER = 'transformed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSFORMED_FOLDER'] = TRANSFORMED_FOLDER

# Swagger configuration
SWAGGER_URL = '/docs'  # URL for accessing Swagger UI
API_URL = '/static/swagger.json'  # Path to your Swagger JSON file

# Create the blueprint for Swagger UI
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI endpoint
    API_URL,  # Swagger specification URL
    config={'app_name': "My Flask API"}  # Optional: Customize Swagger UI
)

# Initialize CBIR system
feature_extractor = ImageFeatureExtractor()

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
class ExtractFeaturesResource(Resource):
    def post(self, image_name):
        """Extract features from a specific image and find similar images using cached features"""
        try:
            # Get category from form data
            category = request.form.get('category')
            # Get number of similar images to return (default: 10)
            k = int(request.form.get('k', 10))
            
            # Validate category is present
            if not category:
                return {
                    'status': 'error',
                    'message': 'Category is required'
                }, 400

            # Validate category enum values
            valid_categories = [
                'aGrass', 'bField', 'cIndustry', 
                'dRiverLake', 'eForest', 'fResident', 'gParking'
            ]
            if category not in valid_categories:
                return {
                    'status': 'error',
                    'message': f'Invalid category. Must be one of: {", ".join(valid_categories)}'
                }, 400

            # Look for query image in transformed folder
            query_image_path = os.path.join(app.config['TRANSFORMED_FOLDER'], category, image_name)
            
            if not os.path.exists(query_image_path):
                return {
                    'status': 'error',
                    'message': f'Image {image_name} not found in category {category}'
                }, 404

            # Extract features for query image from transformed folder
            query_features = feature_extractor.extract_all_features(query_image_path)

            # Generate URL for the query image (from transformed folder)
            image_url = url_for(
                'static',
                filename=os.path.join('transformed', category, image_name),
                _external=True
            )

            # Initialize list to store similarity results
            similarity_results = []

            # Define base path for feature cache
            feature_cache_base = os.path.join(os.getcwd(), 'feature_cache')
            
            print(f"Searching for features in: {feature_cache_base}")  # Debug print

            # Load and compare with cached features from RSSCN7 dataset
            for dataset_category in valid_categories:
                cache_category_path = os.path.join(feature_cache_base, dataset_category)
                
                print(f"Checking category path: {cache_category_path}")  # Debug print
                
                if not os.path.exists(cache_category_path):
                    print(f"Category path does not exist: {cache_category_path}")  # Debug print
                    continue

                # Process each cached feature file in the category
                for feature_file in os.listdir(cache_category_path):
                    if not feature_file.endswith('.pkl'):
                        continue

                    try:
                        # Load cached features
                        feature_path = os.path.join(cache_category_path, feature_file)
                        print(f"Loading features from: {feature_path}")  # Debug print
                        
                        with open(feature_path, 'rb') as f:
                            dataset_features = pickle.load(f)
                        
                        # Get original image filename (remove .pkl extension)
                        img_file = feature_file[:-4]
                        if not any(img_file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                            img_file += '.jpg'  # Add default extension if missing
                        
                        # Calculate similarity score
                        similarity_score = self._calculate_similarity(query_features, dataset_features)
                        
                        print(f"Similarity score for {img_file}: {similarity_score}")  # Debug print

                        # Add to results with reference to RSSCN7 dataset image
                        similarity_results.append({
                            'image_path': os.path.join('RSSCN7', dataset_category, img_file),
                            'category': dataset_category,
                            'similarity_score': similarity_score,
                            'image_url': url_for('static', 
                                               filename=os.path.join('RSSCN7', dataset_category, img_file),
                                               _external=True)
                        })
                    except Exception as e:
                        print(f"Error processing cached features {feature_path}: {str(e)}")  # Debug print
                        continue

            # Sort results by similarity score (descending)
            similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)

            print(f"Total results found: {len(similarity_results)}")  # Debug print

            # Ensure features are JSON serializable
            serializable_features = {}
            for feature_type, feature_data in query_features.items():
                serializable_features[feature_type] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in feature_data.items()
                }

            return {
                'status': 'success',
                'image_name': image_name,
                'category': category,
                'features': serializable_features,
                'query_image_url': image_url,
                'similar_images': similarity_results[:k],
                'total_images_processed': len(similarity_results)
            }, 200

        except Exception as e:
            print(f"Main error: {str(e)}")  # Debug print
            return {
                'status': 'error',
                'message': str(e)
            }, 500

    def _calculate_similarity(self, query_features, dataset_features):
        """
        Calculate weighted similarity score between query image and dataset image features
        using improved comparison metrics and normalization
        """
        try:
            total_similarity = 0.0
            total_weight = 0.0  # Track total weight to handle missing features
            
            # Adjusted weights based on feature importance
            feature_weights = {
                'color_histogram': 0.30,  # Increased weight for color features
                'dominant_colors': 0.20,
                'gabor_features': 0.15,
                'hu_moments': 0.10,
                'lbp_features': 0.15,
                'hog_features': 0.05,
                'glcm_features': 0.05
            }

            # Validate that both feature sets have the same structure
            common_features = set(query_features.keys()) & set(dataset_features.keys())
            
            for feature_type in common_features:
                weight = feature_weights.get(feature_type, 0.0)
                if weight == 0.0:
                    continue
                    
                total_weight += weight
                feature_similarity = 0.0

                try:
                    if feature_type == 'color_histogram':
                        hist_similarity = 0
                        channels = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        
                        for channel in channels:
                            hist1 = np.array(query_features[feature_type][channel], dtype=np.float32)
                            hist2 = np.array(dataset_features[feature_type][channel], dtype=np.float32)
                            
                            # Ensure non-zero sums
                            sum1 = np.sum(hist1)
                            sum2 = np.sum(hist2)
                            if sum1 > 0 and sum2 > 0:
                                hist1 /= sum1
                                hist2 /= sum2
                                
                                # Use Bhattacharyya distance
                                hist_similarity += np.sum(np.sqrt(hist1 * hist2))
                        
                        feature_similarity = hist_similarity / len(channels) if channels else 0

                    elif feature_type == 'dominant_colors':
                        if 'percentages' in query_features[feature_type] and 'percentages' in dataset_features[feature_type]:
                            percentages1 = np.array(query_features[feature_type]['percentages'])
                            percentages2 = np.array(dataset_features[feature_type]['percentages'])
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(percentages1)
                            norm2 = np.linalg.norm(percentages2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(percentages1, percentages2) / (norm1 * norm2)

                    elif feature_type in ['gabor_features', 'hog_features']:
                        if 'features' in query_features[feature_type] and 'features' in dataset_features[feature_type]:
                            feat1 = np.array(query_features[feature_type]['features'])
                            feat2 = np.array(dataset_features[feature_type]['features'])
                            
                            # Ensure same length
                            min_len = min(len(feat1), len(feat2))
                            feat1 = feat1[:min_len]
                            feat2 = feat2[:min_len]
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(feat1)
                            norm2 = np.linalg.norm(feat2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(feat1, feat2) / (norm1 * norm2)

                    elif feature_type == 'hu_moments':
                        if 'moments' in query_features[feature_type] and 'moments' in dataset_features[feature_type]:
                            moments1 = np.array(query_features[feature_type]['moments'])
                            moments2 = np.array(dataset_features[feature_type]['moments'])
                            
                            # Calculate normalized similarity
                            diff = np.abs(moments1 - moments2)
                            feature_similarity = np.mean(1 / (1 + diff))

                    elif feature_type == 'lbp_features':
                        if 'histogram' in query_features[feature_type] and 'histogram' in dataset_features[feature_type]:
                            hist1 = np.array(query_features[feature_type]['histogram'])
                            hist2 = np.array(dataset_features[feature_type]['histogram'])
                            
                            # Calculate intersection similarity
                            feature_similarity = np.sum(np.minimum(hist1, hist2))

                    elif feature_type == 'glcm_features':
                        common_keys = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        if common_keys:
                            similarities = []
                            for key in common_keys:
                                val1 = query_features[feature_type][key]
                                val2 = dataset_features[feature_type][key]
                                max_val = max(abs(val1), abs(val2))
                                if max_val > 0:
                                    similarities.append(1 - abs(val1 - val2) / max_val)
                            
                            feature_similarity = np.mean(similarities) if similarities else 0

                    # Ensure feature similarity is in [0, 1] range
                    feature_similarity = max(0.0, min(1.0, feature_similarity))
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Normalize by actual total weight used
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                return max(0.0, min(1.0, final_similarity))
            
            return 0.0

        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

class ImageTransformResource(Resource):
    def post(self, image_name):
        """Transform an image with specified operations"""
        # Validate input parameters
        if not request.is_json:
            return {
                'status': 'error',
                'message': 'Request must include JSON data'
            }, 400
            
        # Get category from query parameters
        category = request.args.get('category', 'uncategorized')
        
        # Construct the full image path using the category
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], category, image_name)
        
        # Validate that the image exists in the upload folder
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': f'Image {image_name} not found in category {category}'
            }, 404
            
        try:
            # Parse transformations array
            transformations = request.json.get('transformations', [])
            if not transformations:
                return {
                    'status': 'error',
                    'message': 'No transformations specified'
                }, 400
                
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'status': 'error',
                    'message': f'Failed to load image from {image_path}'
                }, 500
                
            transformation_details = {}
            
            # Apply each transformation in sequence
            for transform in transformations:
                if transform == 'crop':
                    # Get crop coordinates
                    crop_coords = request.json.get('crop_coordinates', {})
                    x = crop_coords.get('x', 0)
                    y = crop_coords.get('y', 0)
                    
                    # Use image dimensions if width/height are 0 or not specified
                    width = crop_coords.get('width', image.shape[1]) or image.shape[1]
                    height = crop_coords.get('height', image.shape[0]) or image.shape[0]
                    
                    # Ensure crop coordinates are within image bounds
                    x = max(0, min(x, image.shape[1] - 1))
                    y = max(0, min(y, image.shape[0] - 1))
                    width = min(width, image.shape[1] - x)
                    height = min(height, image.shape[0] - y)
                    
                    # Perform crop
                    image = image[y:y+height, x:x+width]
                    
                    transformation_details['crop'] = {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    }
                
                elif transform == 'resize':
                    # Get resize dimensions
                    resize_dims = request.json.get('resize_dimensions', {})
                    width = resize_dims.get('width', image.shape[1]) or image.shape[1]
                    height = resize_dims.get('height', image.shape[0]) or image.shape[0]
                    
                    # Resize image
                    image = cv2.resize(
                        image, 
                        (width, height), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    transformation_details['resize'] = {
                        'width': width,
                        'height': height
                    }
                
                elif transform == 'rotate':
                    # Get rotation angle
                    angle = request.json.get('rotation_angle', 0)
                    
                    # Perform rotation
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    image = cv2.warpAffine(
                        image, 
                        rotation_matrix, 
                        (image.shape[1], image.shape[0])
                    )
                    
                    transformation_details['rotate'] = {
                        'angle': angle
                    }
            
            # Verify the image is not empty after transformations
            if image is None or image.size == 0:
                return {
                    'status': 'error',
                    'message': 'Transformation resulted in an empty image'
                }, 500
            
            # Create output path maintaining category structure
            output_filename = f"{image_name}"
            output_category_path = os.path.join(app.config['TRANSFORMED_FOLDER'], category)
            output_path = os.path.join(output_category_path, output_filename)
            
            # Ensure the output directory exists
            os.makedirs(output_category_path, exist_ok=True)
            
            # Save the transformed image
            success = cv2.imwrite(output_path, image)
            if not success:
                return {
                    'status': 'error',
                    'message': 'Failed to save transformed image'
                }, 500
            
            # Generate URL for the transformed image
            transformed_image_url = url_for(
                'static',
                filename=os.path.join('transformed', category, output_filename),
                _external=True
            )
            
            return {
                'status': 'success',
                'transformed_image_url': transformed_image_url,
                'transformation_details': transformation_details
            }, 200
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500
        
class RelevanceFeedbackSearchResource(Resource):
    def post(self):
        """
        Process a single image using Query Point Movement (QPM) 
        relevance feedback method with JSON request parsing
        """
        try:
            # Validate request data
            if not request.is_json:
                return {
                    'status': 'error',
                    'message': 'Request must include JSON data'
                }, 400

            # Validate input data exists
            image_data = request.json
            if not image_data:
                return {
                    'status': 'error', 
                    'message': 'No image data provided'
                }, 400

            # Get parameters with default values
            k = int(request.args.get('k', image_data.get('k', 10)))
            alpha = float(image_data.get('alpha', 1.0))
            beta = float(image_data.get('beta', 0.65))
            gamma = float(image_data.get('gamma', 0.35))

            # Valid categories
            valid_categories = [
                'aGrass', 'bField', 'cIndustry', 
                'dRiverLake', 'eForest', 'fResident', 'gParking'
            ]

            # Validate required fields
            if not all(key in image_data for key in ['name', 'category']):
                return {
                    'status': 'error',
                    'message': 'Image must have name and category specified'
                }, 400

            image_name = image_data['name']
            category = image_data['category']
            
            # Validate and process relevant and non-relevant image paths
            def _validate_image_paths(paths, base_path=''):
                validated_paths = []
                for path in paths:
                    full_path = os.path.join(base_path, path)
                    if os.path.exists(full_path):
                        validated_paths.append(full_path)
                    else:
                        print(f"Warning: Image path not found - {full_path}")
                return validated_paths

            relevant_images = _validate_image_paths(
                image_data.get('relevant_images', [])
            )
            non_relevant_images = _validate_image_paths(
                image_data.get('non_relevant_images', [])
            )

            # Validate category
            if category not in valid_categories:
                return {
                    'status': 'error',
                    'message': f'Invalid category. Must be one of: {", ".join(valid_categories)}'
                }, 400

            # Look for image in transformed folder
            image_path = os.path.join(app.config['TRANSFORMED_FOLDER'], category, image_name)
            
            if not os.path.exists(image_path):
                return {
                    'status': 'error',
                    'message': f'Image {image_name} not found in category {category}'
                }, 404

            # Extract features for query image
            query_features = feature_extractor.extract_all_features(image_path)

            # Apply Query Point Movement if feedback is provided
            updated_features = self._apply_qpm(
                query_features,
                relevant_images,
                non_relevant_images,
                alpha, beta, gamma
            )

            # Generate URL for the query image
            image_url = url_for(
                'static',
                filename=os.path.join('transformed', category, image_name),
                _external=True
            )

            # Initialize similarity results
            similarity_results = []

            # Define base path for feature cache
            feature_cache_base = os.path.join('feature_cache')
            
            # Process cached features from each category
            for dataset_category in valid_categories:
                cache_category_path = os.path.join(feature_cache_base, dataset_category)
                
                if not os.path.exists(cache_category_path):
                    continue

                # Process each cached feature file
                for feature_file in os.listdir(cache_category_path):
                    if not feature_file.endswith('.pkl'):
                        continue

                    try:
                        # Load cached features
                        feature_path = os.path.join(cache_category_path, feature_file)
                        with open(feature_path, 'rb') as f:
                            dataset_features = pickle.load(f)

                        # Get original image filename
                        img_file = feature_file[:-4]
                        if not any(img_file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                            img_file += '.jpg'

                        # Calculate similarity score
                        similarity_score = self._calculate_similarity(updated_features, dataset_features)

                        similarity_results.append({
                            'image_path': os.path.join('RSSCN7', dataset_category, img_file),
                            'category': dataset_category,
                            'similarity_score': similarity_score,
                            'image_url': url_for('static',
                                               filename=os.path.join('RSSCN7', dataset_category, img_file),
                                               _external=True)
                        })
                    except Exception as e:
                        print(f"Error processing cached features {feature_path}: {str(e)}")
                        continue

            # Sort results by similarity score
            similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Make features JSON serializable
            serializable_features = {}
            for feature_type, feature_data in updated_features.items():
                if isinstance(feature_data, dict):
                    serializable_features[feature_type] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in feature_data.items()
                    }
                else:
                    serializable_features[feature_type] = feature_data

            return {
                'status': 'success',
                'image_name': image_name,
                'category': category,
                'query_image_url': image_url,
                'features': serializable_features,
                'similar_images': similarity_results[:k],
                'total_images_processed': len(similarity_results),
                'feedback_applied': bool(relevant_images or non_relevant_images)
            }, 200

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                'status': 'error',
                'message': f"An unexpected error occurred: {str(e)}"
            }, 500

    def _apply_qpm(self, original_features, relevant_images, non_relevant_images, alpha, beta, gamma):
        """
        Apply Query Point Movement method to update feature vector based on relevance feedback
        """
        # If no feedback, return original features
        if not relevant_images and not non_relevant_images:
            return original_features

        updated_features = {}
        
        # Process each feature type
        for feature_type in original_features.keys():
            if feature_type == 'color_histogram':
                updated_features[feature_type] = self._apply_qpm_to_histogram(
                    original_features[feature_type],
                    relevant_images,
                    non_relevant_images,
                    alpha, beta, gamma
                )
            elif feature_type in ['gabor_features', 'hog_features']:
                updated_features[feature_type] = self._apply_qpm_to_vector(
                    feature_type,
                    original_features[feature_type],
                    relevant_images,
                    non_relevant_images,
                    alpha, beta, gamma
                )
            else:
                # For other features, keep original values
                updated_features[feature_type] = original_features[feature_type]

        return updated_features

    def _apply_qpm_to_histogram(self, original_hist, relevant_images, non_relevant_images, alpha, beta, gamma):
        """Apply QPM to histogram features"""
        updated_hist = {'blue': None, 'green': None, 'red': None}
        
        for channel in ['blue', 'green', 'red']:
            # Convert to numpy array for calculations
            q = np.array(original_hist[channel])
            
            # Calculate centroids for relevant and non-relevant images
            r_centroid = np.zeros_like(q)
            nr_centroid = np.zeros_like(q)
            
            if relevant_images:
                for img_path in relevant_images:
                    features = feature_extractor.extract_all_features(img_path)
                    r_centroid += np.array(features['color_histogram'][channel])
                r_centroid /= len(relevant_images)
            
            if non_relevant_images:
                for img_path in non_relevant_images:
                    features = feature_extractor.extract_all_features(img_path)
                    nr_centroid += np.array(features['color_histogram'][channel])
                nr_centroid /= len(non_relevant_images)
            
            # Apply Rocchio formula
            updated_hist[channel] = (
                alpha * q +
                (beta * r_centroid if relevant_images else 0) -
                (gamma * nr_centroid if non_relevant_images else 0)
            )
            
            # Normalize histogram
            updated_hist[channel] = np.clip(updated_hist[channel], 0, 1)
            updated_hist[channel] /= updated_hist[channel].sum()
            
        return updated_hist

    def _apply_qpm_to_vector(self, feature_type, original_features, relevant_images, non_relevant_images, alpha, beta, gamma):
        """Apply QPM to vector features (Gabor, HOG)"""
        # Get original feature vector
        q = np.array(original_features['features'])
        
        # Calculate centroids for relevant and non-relevant images
        r_centroid = np.zeros_like(q)
        nr_centroid = np.zeros_like(q)
        
        if relevant_images:
            for img_path in relevant_images:
                features = feature_extractor.extract_all_features(img_path)
                r_centroid += np.array(features[feature_type]['features'])
            r_centroid /= len(relevant_images)
        
        if non_relevant_images:
            for img_path in non_relevant_images:
                features = feature_extractor.extract_all_features(img_path)
                nr_centroid += np.array(features[feature_type]['features'])
            nr_centroid /= len(non_relevant_images)
        
        # Apply Rocchio formula
        updated_vector = (
            alpha * q +
            (beta * r_centroid if relevant_images else 0) -
            (gamma * nr_centroid if non_relevant_images else 0)
        )
        
        # Normalize vector
        updated_vector = updated_vector / np.linalg.norm(updated_vector)
        
        return {'features': updated_vector}

    def _calculate_similarity(self, query_features, dataset_features):
        """Calculate similarity score between query image and dataset image"""
        try:
            total_similarity = 0.0
            total_weight = 0.0
            
            # Adjusted weights based on feature importance
            feature_weights = {
                'color_histogram': 0.30,  # Increased weight for color features
                'dominant_colors': 0.20,
                'gabor_features': 0.15,
                'hu_moments': 0.10,
                'lbp_features': 0.15,
                'hog_features': 0.05,
                'glcm_features': 0.05
            }

            # Validate that both feature sets have the same structure
            common_features = set(query_features.keys()) & set(dataset_features.keys())
            
            for feature_type in common_features:
                weight = feature_weights.get(feature_type, 0.0)
                if weight == 0.0:
                    continue
                    
                total_weight += weight
                feature_similarity = 0.0

                try:
                    if feature_type == 'color_histogram':
                        hist_similarity = 0
                        channels = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        
                        for channel in channels:
                            hist1 = np.array(query_features[feature_type][channel], dtype=np.float32)
                            hist2 = np.array(dataset_features[feature_type][channel], dtype=np.float32)
                            
                            # Ensure non-zero sums
                            sum1 = np.sum(hist1)
                            sum2 = np.sum(hist2)
                            if sum1 > 0 and sum2 > 0:
                                hist1 /= sum1
                                hist2 /= sum2
                                
                                # Use Bhattacharyya distance
                                hist_similarity += np.sum(np.sqrt(hist1 * hist2))
                        
                        feature_similarity = hist_similarity / len(channels) if channels else 0

                    elif feature_type == 'dominant_colors':
                        if 'percentages' in query_features[feature_type] and 'percentages' in dataset_features[feature_type]:
                            percentages1 = np.array(query_features[feature_type]['percentages'])
                            percentages2 = np.array(dataset_features[feature_type]['percentages'])
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(percentages1)
                            norm2 = np.linalg.norm(percentages2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(percentages1, percentages2) / (norm1 * norm2)

                    elif feature_type in ['gabor_features', 'hog_features']:
                        if 'features' in query_features[feature_type] and 'features' in dataset_features[feature_type]:
                            feat1 = np.array(query_features[feature_type]['features'])
                            feat2 = np.array(dataset_features[feature_type]['features'])
                            
                            # Ensure same length
                            min_len = min(len(feat1), len(feat2))
                            feat1 = feat1[:min_len]
                            feat2 = feat2[:min_len]
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(feat1)
                            norm2 = np.linalg.norm(feat2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(feat1, feat2) / (norm1 * norm2)

                    elif feature_type == 'hu_moments':
                        if 'moments' in query_features[feature_type] and 'moments' in dataset_features[feature_type]:
                            moments1 = np.array(query_features[feature_type]['moments'])
                            moments2 = np.array(dataset_features[feature_type]['moments'])
                            
                            # Calculate normalized similarity
                            diff = np.abs(moments1 - moments2)
                            feature_similarity = np.mean(1 / (1 + diff))

                    elif feature_type == 'lbp_features':
                        if 'histogram' in query_features[feature_type] and 'histogram' in dataset_features[feature_type]:
                            hist1 = np.array(query_features[feature_type]['histogram'])
                            hist2 = np.array(dataset_features[feature_type]['histogram'])
                            
                            # Calculate intersection similarity
                            feature_similarity = np.sum(np.minimum(hist1, hist2))

                    elif feature_type == 'glcm_features':
                        common_keys = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        if common_keys:
                            similarities = []
                            for key in common_keys:
                                val1 = query_features[feature_type][key]
                                val2 = dataset_features[feature_type][key]
                                max_val = max(abs(val1), abs(val2))
                                if max_val > 0:
                                    similarities.append(1 - abs(val1 - val2) / max_val)
                            
                            feature_similarity = np.mean(similarities) if similarities else 0

                    # Ensure feature similarity is in [0, 1] range
                    feature_similarity = max(0.0, min(1.0, feature_similarity))
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Normalize by actual total weight used
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                return max(0.0, min(1.0, final_similarity))
            
            return 0.0

        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

class BatchFeatureExtractionResource(Resource):
    def post(self):
        """Extract features from multiple images in a batch using cached features for comparison"""
        try:
            # Validate request data
            if not request.is_json:
                return {
                    'status': 'error',
                    'message': 'Request must include JSON data'
                }, 400

            # Handle both list and dictionary input formats
            if isinstance(request.json, list):
                batch_data = request.json
            else:
                batch_data = request.json.get('images', [])
            
            if not batch_data:
                return {
                    'status': 'error',
                    'message': 'No images provided in the batch'
                }, 400

            # Get number of similar images to return (default: 10)
            k = int(request.args.get('k', 10))

            # Valid categories for validation
            valid_categories = [
                'aGrass', 'bField', 'cIndustry', 
                'dRiverLake', 'eForest', 'fResident', 'gParking'
            ]

            # Define base path for feature cache
            feature_cache_base = os.path.join(os.getcwd(), 'feature_cache')

            # Process each image in the batch
            batch_results = []
            for image_data in batch_data:
                # Validate required fields
                if 'name' not in image_data or 'category' not in image_data:
                    batch_results.append({
                        'status': 'error',
                        'message': 'Each image must have name and category specified',
                        'image_data': image_data
                    })
                    continue

                image_name = image_data['name']
                category = image_data['category']

                # Validate category
                if category not in valid_categories:
                    batch_results.append({
                        'status': 'error',
                        'message': f'Invalid category. Must be one of: {", ".join(valid_categories)}',
                        'image_data': image_data
                    })
                    continue

                # Look for image in transformed folder
                image_path = os.path.join(app.config['TRANSFORMED_FOLDER'], category, image_name)
                
                if not os.path.exists(image_path):
                    batch_results.append({
                        'status': 'error',
                        'message': f'Image {image_name} not found in category {category}',
                        'image_data': image_data
                    })
                    continue

                try:
                    # Extract features for query image
                    query_features = feature_extractor.extract_all_features(image_path)

                    # Generate URL for the query image
                    image_url = url_for(
                        'static',
                        filename=os.path.join('transformed', category, image_name),
                        _external=True
                    )

                    # Initialize list to store similarity results
                    similarity_results = []

                    # Compare with cached features from each category
                    for dataset_category in valid_categories:
                        cache_category_path = os.path.join(feature_cache_base, dataset_category)
                        
                        if not os.path.exists(cache_category_path):
                            continue

                        # Process each cached feature file
                        for feature_file in os.listdir(cache_category_path):
                            if not feature_file.endswith('.pkl'):
                                continue

                            try:
                                # Load cached features
                                feature_path = os.path.join(cache_category_path, feature_file)
                                with open(feature_path, 'rb') as f:
                                    dataset_features = pickle.load(f)

                                # Get original image filename
                                img_file = feature_file[:-4]
                                if not any(img_file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                                    img_file += '.jpg'

                                # Calculate similarity score
                                similarity_score = self._calculate_similarity(query_features, dataset_features)

                                similarity_results.append({
                                    'image_path': os.path.join('RSSCN7', dataset_category, img_file),
                                    'category': dataset_category,
                                    'similarity_score': similarity_score,
                                    'image_url': url_for('static',
                                                       filename=os.path.join('RSSCN7', dataset_category, img_file),
                                                       _external=True)
                                })
                            except Exception as e:
                                print(f"Error processing cached features {feature_path}: {str(e)}")
                                continue

                    # Sort results by similarity score
                    similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)

                    # Make features JSON serializable
                    serializable_features = {}
                    for feature_type, feature_data in query_features.items():
                        serializable_features[feature_type] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in feature_data.items()
                        }

                    # Add successful result
                    batch_results.append({
                        'status': 'success',
                        'image_name': image_name,
                        'category': category,
                        'features': serializable_features,
                        'query_image_url': image_url,
                        'similar_images': similarity_results[:k],
                        'total_images_processed': len(similarity_results)
                    })

                except Exception as e:
                    batch_results.append({
                        'status': 'error',
                        'message': str(e),
                        'image_data': image_data
                    })

            return {
                'status': 'success',
                'batch_results': batch_results,
                'total_processed': len(batch_results)
            }, 200

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500

    def _calculate_similarity(self, query_features, dataset_features):
        """
        Calculate weighted similarity score between query image and dataset image features
        using improved comparison metrics and normalization
        """
        try:
            total_similarity = 0.0
            total_weight = 0.0

            # Adjusted weights based on feature importance
            feature_weights = {
                'color_histogram': 0.30,
                'dominant_colors': 0.20,
                'gabor_features': 0.15,
                'hu_moments': 0.10,
                'lbp_features': 0.15,
                'hog_features': 0.05,
                'glcm_features': 0.05
            }

            # Validate that both feature sets have the same structure
            common_features = set(query_features.keys()) & set(dataset_features.keys())
            
            for feature_type in common_features:
                weight = feature_weights.get(feature_type, 0.0)
                if weight == 0.0:
                    continue
                    
                total_weight += weight
                feature_similarity = 0.0

                try:
                    if feature_type == 'color_histogram':
                        hist_similarity = 0
                        channels = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        
                        for channel in channels:
                            hist1 = np.array(query_features[feature_type][channel], dtype=np.float32)
                            hist2 = np.array(dataset_features[feature_type][channel], dtype=np.float32)
                            
                            # Ensure non-zero sums
                            sum1 = np.sum(hist1)
                            sum2 = np.sum(hist2)
                            if sum1 > 0 and sum2 > 0:
                                hist1 /= sum1
                                hist2 /= sum2
                                
                                # Use Bhattacharyya distance
                                hist_similarity += np.sum(np.sqrt(hist1 * hist2))
                        
                        feature_similarity = hist_similarity / len(channels) if channels else 0

                    elif feature_type == 'dominant_colors':
                        if 'percentages' in query_features[feature_type] and 'percentages' in dataset_features[feature_type]:
                            percentages1 = np.array(query_features[feature_type]['percentages'])
                            percentages2 = np.array(dataset_features[feature_type]['percentages'])
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(percentages1)
                            norm2 = np.linalg.norm(percentages2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(percentages1, percentages2) / (norm1 * norm2)

                    elif feature_type in ['gabor_features', 'hog_features']:
                        if 'features' in query_features[feature_type] and 'features' in dataset_features[feature_type]:
                            feat1 = np.array(query_features[feature_type]['features'])
                            feat2 = np.array(dataset_features[feature_type]['features'])
                            
                            # Ensure same length
                            min_len = min(len(feat1), len(feat2))
                            feat1 = feat1[:min_len]
                            feat2 = feat2[:min_len]
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(feat1)
                            norm2 = np.linalg.norm(feat2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(feat1, feat2) / (norm1 * norm2)

                    elif feature_type == 'hu_moments':
                        if 'moments' in query_features[feature_type] and 'moments' in dataset_features[feature_type]:
                            moments1 = np.array(query_features[feature_type]['moments'])
                            moments2 = np.array(dataset_features[feature_type]['moments'])
                            
                            # Calculate normalized similarity
                            diff = np.abs(moments1 - moments2)
                            feature_similarity = np.mean(1 / (1 + diff))

                    elif feature_type == 'lbp_features':
                        if 'histogram' in query_features[feature_type] and 'histogram' in dataset_features[feature_type]:
                            hist1 = np.array(query_features[feature_type]['histogram'])
                            hist2 = np.array(dataset_features[feature_type]['histogram'])
                            
                            # Calculate intersection similarity
                            feature_similarity = np.sum(np.minimum(hist1, hist2))

                    elif feature_type == 'glcm_features':
                        common_keys = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        if common_keys:
                            similarities = []
                            for key in common_keys:
                                val1 = query_features[feature_type][key]
                                val2 = dataset_features[feature_type][key]
                                max_val = max(abs(val1), abs(val2))
                                if max_val > 0:
                                    similarities.append(1 - abs(val1 - val2) / max_val)
                            
                            feature_similarity = np.mean(similarities) if similarities else 0

                    # Ensure feature similarity is in [0, 1] range
                    feature_similarity = max(0.0, min(1.0, feature_similarity))
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Normalize by actual total weight used
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                return max(0.0, min(1.0, final_similarity))
            
            return 0.0

        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

class BatchRelevanceFeedbackSearchResource(Resource):
    def post(self):
        """
        Process multiple images in a batch using Query Point Movement (QPM) 
        relevance feedback method with precomputed features
        """
        try:
            # Validate request data
            if not request.is_json:
                return {
                    'status': 'error',
                    'message': 'Request must include JSON data'
                }, 400

            # Handle both list and dictionary input formats
            if isinstance(request.json, list):
                batch_data = request.json
            else:
                batch_data = request.json.get('images', [])
            
            if not batch_data:
                return {
                    'status': 'error',
                    'message': 'No images provided in the batch'
                }, 400

            # Get number of similar images to return (default: 10)
            k = int(request.args.get('k', 10))
            
            # QPM parameters
            alpha = float(request.args.get('alpha', 1.0))  # Weight for original query
            beta = float(request.args.get('beta', 0.65))   # Weight for relevant images
            gamma = float(request.args.get('gamma', 0.35))  # Weight for non-relevant images

            # Valid categories for validation
            valid_categories = [
                'aGrass', 'bField', 'cIndustry', 
                'dRiverLake', 'eForest', 'fResident', 'gParking'
            ]

            # Define base path for feature cache
            feature_cache_base = os.path.join(os.getcwd(), 'feature_cache')

            # Process each image in the batch
            batch_results = []
            for image_data in batch_data:
                # Validate required fields
                if not all(key in image_data for key in ['name', 'category']):
                    batch_results.append({
                        'status': 'error',
                        'message': 'Each image must have name and category specified',
                        'image_data': image_data
                    })
                    continue

                image_name = image_data['name']
                category = image_data['category']
                relevant_images = image_data.get('relevant_images', [])
                non_relevant_images = image_data.get('non_relevant_images', [])

                # Validate category
                if category not in valid_categories:
                    batch_results.append({
                        'status': 'error',
                        'message': f'Invalid category. Must be one of: {", ".join(valid_categories)}',
                        'image_data': image_data
                    })
                    continue

                # Look for image in transformed folder
                image_path = os.path.join(app.config['TRANSFORMED_FOLDER'], category, image_name)
                
                if not os.path.exists(image_path):
                    batch_results.append({
                        'status': 'error',
                        'message': f'Image {image_name} not found in category {category}',
                        'image_data': image_data
                    })
                    continue

                try:
                    # Extract features for query image
                    original_features = feature_extractor.extract_all_features(image_path)

                    # Apply Query Point Movement if feedback is provided
                    updated_features = self._apply_qpm(
                        original_features,
                        relevant_images,
                        non_relevant_images,
                        alpha, beta, gamma
                    )

                    # Generate URL for the query image
                    image_url = url_for(
                        'static',
                        filename=os.path.join('transformed', category, image_name),
                        _external=True
                    )

                    # Search for similar images with updated features
                    similarity_results = []
                    
                    # Process cached features from each category
                    for dataset_category in valid_categories:
                        cache_category_path = os.path.join(feature_cache_base, dataset_category)
                        
                        if not os.path.exists(cache_category_path):
                            continue

                        # Process each cached feature file
                        for feature_file in os.listdir(cache_category_path):
                            if not feature_file.endswith('.pkl'):
                                continue

                            try:
                                # Load cached features
                                feature_path = os.path.join(cache_category_path, feature_file)
                                with open(feature_path, 'rb') as f:
                                    dataset_features = pickle.load(f)

                                # Get original image filename
                                img_file = feature_file[:-4]
                                if not any(img_file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                                    img_file += '.jpg'

                                # Calculate similarity score
                                similarity_score = self._calculate_similarity(updated_features, dataset_features)

                                similarity_results.append({
                                    'image_path': os.path.join('RSSCN7', dataset_category, img_file),
                                    'category': dataset_category,
                                    'similarity_score': similarity_score,
                                    'image_url': url_for('static',
                                                       filename=os.path.join('RSSCN7', dataset_category, img_file),
                                                       _external=True)
                                })
                            except Exception as e:
                                print(f"Error processing cached features {feature_path}: {str(e)}")
                                continue

                    # Sort results by similarity score
                    similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)

                    # Make features JSON serializable
                    serializable_features = {}
                    for feature_type, feature_data in updated_features.items():
                        if isinstance(feature_data, dict):
                            serializable_features[feature_type] = {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in feature_data.items()
                            }
                        else:
                            serializable_features[feature_type] = feature_data

                    # Add successful result
                    batch_results.append({
                        'status': 'success',
                        'image_name': image_name,
                        'category': category,
                        'query_image_url': image_url,
                        'features': serializable_features,
                        'similar_images': similarity_results[:k],
                        'total_images_processed': len(similarity_results),
                        'feedback_applied': bool(relevant_images or non_relevant_images)
                    })

                except Exception as e:
                    batch_results.append({
                        'status': 'error',
                        'message': str(e),
                        'image_data': image_data
                    })

            return {
                'status': 'success',
                'batch_results': batch_results,
                'total_processed': len(batch_results)
            }, 200

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500

    def _apply_qpm(self, original_features, relevant_images, non_relevant_images, alpha, beta, gamma):
        """
        Apply Query Point Movement method to update feature vector based on relevance feedback
        """
        # If no feedback, return original features
        if not relevant_images and not non_relevant_images:
            return original_features

        updated_features = {}
        
        # Process each feature type
        for feature_type in original_features.keys():
            if feature_type == 'color_histogram':
                updated_features[feature_type] = self._apply_qpm_to_histogram(
                    original_features[feature_type],
                    relevant_images,
                    non_relevant_images,
                    alpha, beta, gamma
                )
            elif feature_type in ['gabor_features', 'hog_features']:
                updated_features[feature_type] = self._apply_qpm_to_vector(
                    feature_type,
                    original_features[feature_type],
                    relevant_images,
                    non_relevant_images,
                    alpha, beta, gamma
                )
            else:
                # For other features, keep original values
                updated_features[feature_type] = original_features[feature_type]

        return updated_features

    def _apply_qpm_to_histogram(self, original_hist, relevant_images, non_relevant_images, alpha, beta, gamma):
        """Apply QPM to histogram features"""
        updated_hist = {'blue': None, 'green': None, 'red': None}
        
        for channel in ['blue', 'green', 'red']:
            # Convert to numpy array for calculations
            q = np.array(original_hist[channel])
            
            # Calculate centroids for relevant and non-relevant images
            r_centroid = np.zeros_like(q)
            nr_centroid = np.zeros_like(q)
            
            if relevant_images:
                for img_path in relevant_images:
                    features = feature_extractor.extract_all_features(img_path)
                    r_centroid += np.array(features['color_histogram'][channel])
                r_centroid /= len(relevant_images)
            
            if non_relevant_images:
                for img_path in non_relevant_images:
                    features = feature_extractor.extract_all_features(img_path)
                    nr_centroid += np.array(features['color_histogram'][channel])
                nr_centroid /= len(non_relevant_images)
            
            # Apply Rocchio formula
            updated_hist[channel] = (
                alpha * q +
                (beta * r_centroid if relevant_images else 0) -
                (gamma * nr_centroid if non_relevant_images else 0)
            )
            
            # Normalize histogram
            updated_hist[channel] = np.clip(updated_hist[channel], 0, 1)
            updated_hist[channel] /= updated_hist[channel].sum()
            
        return updated_hist

    def _apply_qpm_to_vector(self, feature_type, original_features, relevant_images, non_relevant_images, alpha, beta, gamma):
        """Apply QPM to vector features (Gabor, HOG)"""
        # Get original feature vector
        q = np.array(original_features['features'])
        
        # Calculate centroids for relevant and non-relevant images
        r_centroid = np.zeros_like(q)
        nr_centroid = np.zeros_like(q)
        
        if relevant_images:
            for img_path in relevant_images:
                features = feature_extractor.extract_all_features(img_path)
                r_centroid += np.array(features[feature_type]['features'])
            r_centroid /= len(relevant_images)
        
        if non_relevant_images:
            for img_path in non_relevant_images:
                features = feature_extractor.extract_all_features(img_path)
                nr_centroid += np.array(features[feature_type]['features'])
            nr_centroid /= len(non_relevant_images)
        
        # Apply Rocchio formula
        updated_vector = (
            alpha * q +
            (beta * r_centroid if relevant_images else 0) -
            (gamma * nr_centroid if non_relevant_images else 0)
        )
        
        # Normalize vector
        updated_vector = updated_vector / np.linalg.norm(updated_vector)
        
        return {'features': updated_vector}

    def _calculate_similarity(self, query_features, dataset_features):
        """
        Calculate weighted similarity score between query image and dataset image features
        using improved comparison metrics and normalization
        """
        try:
            total_similarity = 0.0
            total_weight = 0.0

            # Adjusted weights based on feature importance
            feature_weights = {
                'color_histogram': 0.30,
                'dominant_colors': 0.20,
                'gabor_features': 0.15,
                'hu_moments': 0.10,
                'lbp_features': 0.15,
                'hog_features': 0.05,
                'glcm_features': 0.05
            }

            # Validate that both feature sets have the same structure
            common_features = set(query_features.keys()) & set(dataset_features.keys())
            
            for feature_type in common_features:
                weight = feature_weights.get(feature_type, 0.0)
                if weight == 0.0:
                    continue
                    
                total_weight += weight
                feature_similarity = 0.0

                try:
                    if feature_type == 'color_histogram':
                        hist_similarity = 0
                        channels = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        
                        for channel in channels:
                            hist1 = np.array(query_features[feature_type][channel], dtype=np.float32)
                            hist2 = np.array(dataset_features[feature_type][channel], dtype=np.float32)
                            
                            # Ensure non-zero sums
                            sum1 = np.sum(hist1)
                            sum2 = np.sum(hist2)
                            if sum1 > 0 and sum2 > 0:
                                hist1 /= sum1
                                hist2 /= sum2
                                
                                # Use Bhattacharyya distance
                                hist_similarity += np.sum(np.sqrt(hist1 * hist2))
                        
                        feature_similarity = hist_similarity / len(channels) if channels else 0

                    elif feature_type == 'dominant_colors':
                        if 'percentages' in query_features[feature_type] and 'percentages' in dataset_features[feature_type]:
                            percentages1 = np.array(query_features[feature_type]['percentages'])
                            percentages2 = np.array(dataset_features[feature_type]['percentages'])
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(percentages1)
                            norm2 = np.linalg.norm(percentages2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(percentages1, percentages2) / (norm1 * norm2)

                    elif feature_type in ['gabor_features', 'hog_features']:
                        if 'features' in query_features[feature_type] and 'features' in dataset_features[feature_type]:
                            feat1 = np.array(query_features[feature_type]['features'])
                            feat2 = np.array(dataset_features[feature_type]['features'])
                            
                            # Ensure same length
                            min_len = min(len(feat1), len(feat2))
                            feat1 = feat1[:min_len]
                            feat2 = feat2[:min_len]
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(feat1)
                            norm2 = np.linalg.norm(feat2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(feat1, feat2) / (norm1 * norm2)

                    elif feature_type == 'hu_moments':
                        if 'moments' in query_features[feature_type] and 'moments' in dataset_features[feature_type]:
                            moments1 = np.array(query_features[feature_type]['moments'])
                            moments2 = np.array(dataset_features[feature_type]['moments'])
                            
                            # Calculate normalized similarity
                            diff = np.abs(moments1 - moments2)
                            feature_similarity = np.mean(1 / (1 + diff))

                    elif feature_type == 'lbp_features':
                        if 'histogram' in query_features[feature_type] and 'histogram' in dataset_features[feature_type]:
                            hist1 = np.array(query_features[feature_type]['histogram'])
                            hist2 = np.array(dataset_features[feature_type]['histogram'])
                            
                            # Calculate intersection similarity
                            feature_similarity = np.sum(np.minimum(hist1, hist2))

                    elif feature_type == 'glcm_features':
                        common_keys = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        if common_keys:
                            similarities = []
                            for key in common_keys:
                                val1 = query_features[feature_type][key]
                                val2 = dataset_features[feature_type][key]
                                max_val = max(abs(val1), abs(val2))
                                if max_val > 0:
                                    similarities.append(1 - abs(val1 - val2) / max_val)
                            
                            feature_similarity = np.mean(similarities) if similarities else 0

                    # Ensure feature similarity is in [0, 1] range
                    feature_similarity = max(0.0, min(1.0, feature_similarity))
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Normalize by actual total weight used
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                return max(0.0, min(1.0, final_similarity))
            
            return 0.0

        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

# Register API routes
api.add_resource(ExtractFeaturesResource, '/extract_features/<string:image_name>')
api.add_resource(ImageTransformResource, '/transform/<string:image_name>')
api.add_resource(RelevanceFeedbackSearchResource, '/relevance-feedback-search')
api.add_resource(BatchFeatureExtractionResource, '/extract_features/batch')
api.add_resource(BatchRelevanceFeedbackSearchResource, '/relevance-feedback-search/batch')

# Register the Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TRANSFORMED_FOLDER, exist_ok=True)

    app.run(debug=True)