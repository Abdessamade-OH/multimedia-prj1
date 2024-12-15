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
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], category, image_name)
                
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
                        filename=os.path.join('upload_folder', category, image_name),
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
        Calculate weighted similarity score between query image and dataset features
        with improved metrics, category-aware weighting, and dimension handling
        """
        try:
            total_similarity = 0.0
            total_weight = 0.0

            # Adjusted weights to emphasize more discriminative features
            feature_weights = {
                'color_histogram': 0.25,
                'dominant_colors': 0.20,
                'gabor_features': 0.25,
                'hu_moments': 0.10,
                'lbp_features': 0.15,
                'hog_features': 0.03,
                'glcm_features': 0.02
            }

            # Validate feature sets
            common_features = set(query_features.keys()) & set(dataset_features.keys())
            
            for feature_type in common_features:
                weight = feature_weights.get(feature_type, 0.0)
                if weight == 0.0:
                    continue
                    
                total_weight += weight
                feature_similarity = 0.0

                try:
                    if feature_type == 'color_histogram':
                        hist1 = np.array(query_features[feature_type].get('histogram', []))
                        hist2 = np.array(dataset_features[feature_type].get('histogram', []))
                        
                        if hist1.size > 0 and hist2.size > 0:
                            # Ensure same dimensions through interpolation
                            target_size = max(hist1.size, hist2.size)
                            if hist1.size != target_size:
                                hist1 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, hist1.size), hist1)
                            if hist2.size != target_size:
                                hist2 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, hist2.size), hist2)
                            
                            # Normalize histograms
                            hist1 = hist1 / (np.sum(hist1) + 1e-10)
                            hist2 = hist2 / (np.sum(hist2) + 1e-10)
                            
                            # Combine multiple similarity metrics
                            intersection = np.sum(np.minimum(hist1, hist2))
                            correlation = np.corrcoef(hist1, hist2)[0, 1]
                            chi_square = np.sum(np.where(hist1 + hist2 != 0, 
                                                    (hist1 - hist2) ** 2 / (hist1 + hist2), 0))
                            
                            feature_similarity = (0.4 * intersection + 
                                            0.4 * (1 / (1 + chi_square)) +
                                            0.2 * max(0, correlation))

                    elif feature_type == 'dominant_colors':
                        colors1 = np.array(query_features[feature_type].get('colors', []))
                        colors2 = np.array(dataset_features[feature_type].get('colors', []))
                        
                        if colors1.size > 0 and colors2.size > 0:
                            # Ensure both color arrays are 2D
                            if colors1.ndim == 1:
                                colors1 = colors1.reshape(-1, 3)
                            if colors2.ndim == 1:
                                colors2 = colors2.reshape(-1, 3)
                                
                            # Calculate color distance matrix
                            distances = np.zeros((len(colors1), len(colors2)))
                            for i, c1 in enumerate(colors1):
                                for j, c2 in enumerate(colors2):
                                    distances[i, j] = np.sqrt(np.sum((c1 - c2) ** 2))
                            
                            # Find minimum distances for each color
                            min_distances = np.minimum(np.min(distances, axis=0).mean(),
                                                    np.min(distances, axis=1).mean())
                            
                            feature_similarity = 1 / (1 + min_distances)

                    elif feature_type == 'gabor_features':
                        feat1 = np.array(query_features[feature_type].get('features', []))
                        feat2 = np.array(dataset_features[feature_type].get('features', []))
                        
                        if feat1.size > 0 and feat2.size > 0:
                            # Interpolate features to match dimensions
                            target_size = max(feat1.size, feat2.size)
                            if feat1.size != target_size:
                                feat1 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, feat1.size), feat1)
                            if feat2.size != target_size:
                                feat2 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, feat2.size), feat2)
                            
                            # Normalize features
                            feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
                            feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
                            
                            # Calculate cosine similarity
                            cosine_sim = np.dot(feat1_norm, feat2_norm)
                            
                            # Calculate L2 distance similarity
                            l2_sim = 1 / (1 + np.sqrt(np.sum((feat1_norm - feat2_norm) ** 2)))
                            
                            feature_similarity = 0.6 * cosine_sim + 0.4 * l2_sim

                    elif feature_type == 'lbp_features':
                        hist1 = np.array(query_features[feature_type].get('histogram', []))
                        hist2 = np.array(dataset_features[feature_type].get('histogram', []))
                        
                        if hist1.size > 0 and hist2.size > 0:
                            # Match histogram sizes through interpolation
                            target_size = max(hist1.size, hist2.size)
                            if hist1.size != target_size:
                                hist1 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, hist1.size), hist1)
                            if hist2.size != target_size:
                                hist2 = np.interp(np.linspace(0, 1, target_size), 
                                                np.linspace(0, 1, hist2.size), hist2)
                            
                            # Normalize
                            hist1 = hist1 / (np.sum(hist1) + 1e-10)
                            hist2 = hist2 / (np.sum(hist2) + 1e-10)
                            
                            # Combine multiple similarity metrics
                            intersection = np.sum(np.minimum(hist1, hist2))
                            chi_square = np.sum(np.where(hist1 + hist2 != 0,
                                                    (hist1 - hist2) ** 2 / (hist1 + hist2),
                                                    0))
                            
                            feature_similarity = 0.6 * intersection + 0.4 * (1 / (1 + chi_square))

                    # Ensure feature similarity is in [0, 1] range and handle NaN
                    feature_similarity = 0.0 if np.isnan(feature_similarity) else feature_similarity
                    feature_similarity = max(0.0, min(1.0, feature_similarity))
                    
                    # Apply non-linear transformation to emphasize stronger matches
                    feature_similarity = np.power(feature_similarity, 0.5)
                    
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Calculate final similarity score
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                
                # Apply stronger contrast enhancement
                final_similarity = np.power(final_similarity, 0.5)
                
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
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], category, image_name)
                
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
                        filename=os.path.join('upload_folder', category, image_name),
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
        with improved error handling and shape matching
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
                            
                            # Resize histograms to match shapes if necessary
                            if hist1.shape != hist2.shape:
                                target_size = min(len(hist1), len(hist2))
                                hist1 = cv2.resize(hist1, (target_size, 1)).flatten()
                                hist2 = cv2.resize(hist2, (target_size, 1)).flatten()
                            
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
                            percentages1 = np.array(query_features[feature_type]['percentages'], dtype=np.float32)
                            percentages2 = np.array(dataset_features[feature_type]['percentages'], dtype=np.float32)
                            
                            # Match lengths if necessary
                            min_len = min(len(percentages1), len(percentages2))
                            percentages1 = percentages1[:min_len]
                            percentages2 = percentages2[:min_len]
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(percentages1)
                            norm2 = np.linalg.norm(percentages2)
                            if norm1 > 0 and norm2 > 0:
                                feature_similarity = np.dot(percentages1, percentages2) / (norm1 * norm2)

                    elif feature_type in ['gabor_features', 'hog_features']:
                        if 'features' in query_features[feature_type] and 'features' in dataset_features[feature_type]:
                            feat1 = np.array(query_features[feature_type]['features'], dtype=np.float32)
                            feat2 = np.array(dataset_features[feature_type]['features'], dtype=np.float32)
                            
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
                            moments1 = np.array(query_features[feature_type]['moments'], dtype=np.float32)
                            moments2 = np.array(dataset_features[feature_type]['moments'], dtype=np.float32)
                            
                            # Match lengths if necessary
                            min_len = min(len(moments1), len(moments2))
                            moments1 = moments1[:min_len]
                            moments2 = moments2[:min_len]
                            
                            # Calculate normalized similarity
                            diff = np.abs(moments1 - moments2)
                            feature_similarity = float(np.mean(1 / (1 + diff)))

                    elif feature_type == 'lbp_features':
                        if 'histogram' in query_features[feature_type] and 'histogram' in dataset_features[feature_type]:
                            hist1 = np.array(query_features[feature_type]['histogram'], dtype=np.float32)
                            hist2 = np.array(dataset_features[feature_type]['histogram'], dtype=np.float32)
                            
                            # Resize histograms to match shapes if necessary
                            if hist1.shape != hist2.shape:
                                target_size = min(len(hist1), len(hist2))
                                hist1 = cv2.resize(hist1, (target_size, 1)).flatten()
                                hist2 = cv2.resize(hist2, (target_size, 1)).flatten()
                            
                            # Normalize histograms
                            sum1 = np.sum(hist1)
                            sum2 = np.sum(hist2)
                            if sum1 > 0 and sum2 > 0:
                                hist1 /= sum1
                                hist2 /= sum2
                                
                                # Calculate intersection similarity
                                feature_similarity = float(np.sum(np.minimum(hist1, hist2)))

                    elif feature_type == 'glcm_features':
                        common_keys = set(query_features[feature_type].keys()) & set(dataset_features[feature_type].keys())
                        if common_keys:
                            similarities = []
                            for key in common_keys:
                                val1 = query_features[feature_type][key]
                                val2 = dataset_features[feature_type][key]
                                
                                # Convert to numpy arrays and ensure float32 type
                                val1 = np.array(val1, dtype=np.float32).flatten()
                                val2 = np.array(val2, dtype=np.float32).flatten()
                                
                                # Ensure same length
                                min_len = min(len(val1), len(val2))
                                val1 = val1[:min_len]
                                val2 = val2[:min_len]
                                
                                # Calculate element-wise similarities
                                max_vals = np.maximum(np.abs(val1), np.abs(val2))
                                # Avoid division by zero
                                mask = max_vals > 0
                                if np.any(mask):
                                    element_similarities = 1 - np.abs(val1[mask] - val2[mask]) / max_vals[mask]
                                    similarities.append(float(np.mean(element_similarities)))
                                else:
                                    similarities.append(1.0)  # If both values are zero, consider them similar
                            
                            feature_similarity = np.mean(similarities) if similarities else 0.0

                    # Ensure feature similarity is in [0, 1] range
                    feature_similarity = float(max(0.0, min(1.0, feature_similarity)))
                    total_similarity += feature_similarity * weight

                except Exception as e:
                    print(f"Error calculating similarity for {feature_type}: {str(e)}")
                    total_weight -= weight
                    continue

            # Normalize by actual total weight used
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
                return float(max(0.0, min(1.0, final_similarity)))
            
            return 0.0

        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0