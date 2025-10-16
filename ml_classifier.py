import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

class MLClassifier:
    def __init__(self, n_clusters=2, random_state=42):
        """
        Initialize ML classifier with K-Means clustering
        
        Args:
            n_clusters (int): Number of clusters for K-Means (default: 2 for Satellite/Meteor)
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize models
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
        # Feature columns for ML processing (removed plane_score)
        self.feature_columns = [
            'avg_speed', 'speed_consistency', 'duration', 'linearity',
            'direction_changes', 'size_consistency', 'avg_acceleration',
            'blinking_score',  # Brightness variation indicator
            'satellite_score', 'meteor_score'
        ]
    
    def classify_objects(self, features_df):
        """
        Classify detected objects using unsupervised learning
        
        Args:
            features_df (pd.DataFrame): DataFrame with extracted features
            
        Returns:
            pd.DataFrame: DataFrame with classifications and confidence scores
        """
        if features_df.empty:
            return pd.DataFrame()
        
        # Step 0: Check for star groups first (highest priority)
        # Stars are identified by group movement analysis
        if 'is_star_group' in features_df.columns:
            star_mask = features_df['is_star_group'] == 1
            if star_mask.any():
                # Classify stars separately and return
                results_df = features_df.copy()
                results_df['classification'] = ''
                results_df['confidence'] = 0.0
                results_df['cluster'] = 0
                
                # Classify stars
                results_df.loc[star_mask, 'classification'] = 'Star'
                results_df.loc[star_mask, 'confidence'] = 0.9
                
                # Classify non-stars using normal pipeline
                non_star_df = features_df[~star_mask].copy()
                if not non_star_df.empty and len(non_star_df) >= 2:
                    non_star_results = self._classify_non_stars(non_star_df)
                    for col in ['classification', 'confidence', 'cluster']:
                        if col in non_star_results.columns:
                            results_df.loc[~star_mask, col] = non_star_results[col].values
                elif not non_star_df.empty:
                    # Single non-star object, use rule-based
                    single_result = self._classify_single_object(non_star_df)
                    for col in ['classification', 'confidence', 'cluster']:
                        if col in single_result.columns:
                            results_df.loc[~star_mask, col] = single_result[col].values
                
                return results_df
        
        # No stars detected, use normal classification pipeline
        return self._classify_non_stars(features_df)
    
    def _classify_non_stars(self, features_df):
        """Classify non-star objects using the normal ML pipeline"""
        if features_df.empty:
            return pd.DataFrame()
        
        # Prepare features for ML
        X = self._prepare_features(features_df)
        
        if X.shape[0] < 2:
            # Not enough data for clustering
            return self._classify_single_object(features_df)
        
        # Step 1: K-Means Clustering for primary classification
        # Adjust number of clusters based on available samples
        n_samples = X.shape[0]
        n_clusters = min(self.n_clusters, n_samples)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Only use KMeans if we have enough samples
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
        else:
            # Fall back to rule-based for single sample
            return self._classify_single_object(features_df)
        
        # Create results dataframe
        results_df = features_df.copy()
        results_df['cluster'] = cluster_labels
        
        # Step 2: Interpret clusters and assign classifications
        results_df = self._interpret_clusters(results_df)
        
        # Step 3: Calculate final confidence scores
        results_df = self._calculate_confidence_scores(results_df)
        
        return results_df
    
    def _prepare_features(self, features_df):
        """Prepare feature matrix for ML algorithms"""
        # Select relevant features and handle missing values
        X = features_df[self.feature_columns].copy()
        X = X.fillna(0)  # Fill NaN values with 0
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.values
    
    def _classify_single_object(self, features_df):
        """Handle classification when only one object is detected"""
        results_df = features_df.copy()
        
        # Use rule-based classification for single objects
        for idx, row in results_df.iterrows():
            classification, confidence = self._rule_based_classification(row)
            results_df.loc[idx, 'classification'] = classification
            results_df.loc[idx, 'confidence'] = confidence
            results_df.loc[idx, 'cluster'] = 0
        
        return results_df
    
    def _interpret_clusters(self, results_df):
        """Interpret K-Means clusters and assign object classifications"""
        cluster_interpretations = {}
        
        # Get actual number of clusters from the data
        n_actual_clusters = results_df['cluster'].nunique()
        
        for cluster_id in range(n_actual_clusters):
            cluster_data = results_df[results_df['cluster'] == cluster_id]
            
            if cluster_data.empty:
                cluster_interpretations[cluster_id] = ('Junk', 0.5)
                continue
            
            # Analyze cluster characteristics
            avg_speed = cluster_data['avg_speed'].mean()
            avg_consistency = cluster_data['speed_consistency'].mean()
            avg_linearity = cluster_data['linearity'].mean()
            avg_duration = cluster_data['duration'].mean()
            avg_satellite_score = cluster_data['satellite_score'].mean()
            avg_meteor_score = cluster_data['meteor_score'].mean()
            
            # Determine most likely classification for this cluster (Satellite, Meteor, or Junk)
            scores = {
                'Satellite': avg_satellite_score,
                'Meteor': avg_meteor_score,
                'Junk': 0.1  # Base score for junk category
            }
            
            # Apply minimal heuristics - mostly trust the base ML scores
            # Only boost when there's VERY strong evidence
            if avg_speed > 30 and avg_linearity > 0.85 and avg_duration < 1.5:
                # Extremely fast, linear, brief -> Definitely Meteor
                scores['Meteor'] *= 1.8
            elif avg_consistency < 0.3 or avg_linearity < 0.3:
                # Very poor quality trajectory -> Likely Junk
                scores['Junk'] *= 1.5
            
            # Select best classification
            best_class = max(scores, key=scores.get)
            confidence = min(scores[best_class], 0.95)  # Cap confidence at 95%
            
            cluster_interpretations[cluster_id] = (best_class, confidence)
        
        # Assign classifications based on cluster interpretations
        results_df['classification'] = ''
        results_df['confidence'] = 0.0
        
        for idx, row in results_df.iterrows():
            cluster_id = row['cluster']
            classification, confidence = cluster_interpretations[cluster_id]
            
            # CRITICAL OVERRIDE: Only force EXTREMELY slow objects to Junk (prevents false positives)
            # Very permissive threshold to allow distant slow satellites
            avg_speed = row.get('avg_speed', 0)
            if avg_speed < 0.15:  # Lowered from 0.3 to allow slower satellites
                classification = 'Junk'
                confidence = 0.6  # Low confidence for filtered objects
            
            results_df.loc[idx, 'classification'] = classification
            results_df.loc[idx, 'confidence'] = confidence
        
        return results_df
    
    def _rule_based_classification(self, row):
        """Rule-based classification for edge cases - trust the feature scores"""
        # CRITICAL OVERRIDE: Only force EXTREMELY slow objects to Junk (prevents false positives)
        # Very permissive threshold to allow distant slow satellites
        avg_speed = row.get('avg_speed', 0)
        if avg_speed < 0.15:  # Lowered from 0.3 to allow slower satellites
            return 'Junk', 0.6  # Low confidence for filtered objects
        
        # Use the pre-computed scores from feature_extractor
        satellite_score = row.get('satellite_score', 0)
        meteor_score = row.get('meteor_score', 0)
        
        # Determine best classification based on scores (Satellite, Meteor, or Junk)
        scores = {
            'Satellite': satellite_score,
            'Meteor': meteor_score,
            'Junk': 0.1
        }
        
        # Apply minimal adjustments for extreme cases
        if avg_speed > 30 and row.get('linearity', 0) > 0.85 and row.get('duration', 0) < 1.5:
            scores['Meteor'] *= 1.8
        elif row.get('speed_consistency', 0) < 0.3 or row.get('linearity', 0) < 0.3:
            scores['Junk'] *= 1.5
        
        # Select best classification
        best_class = max(scores, key=scores.get)
        confidence = min(scores[best_class], 0.85)  # Cap at 85% for single objects
        
        return best_class, confidence
    
    def _calculate_confidence_scores(self, results_df):
        """Calculate and refine confidence scores based on multiple factors"""
        for idx, row in results_df.iterrows():
            base_confidence = row.get('confidence', 0.5)
            
            # Adjust confidence based on detection count (with safe fallback)
            detection_count = row.get('detection_count', 1)
            if detection_count < 3:
                base_confidence *= 0.8  # Lower confidence for few detections
            elif detection_count > 10:
                base_confidence = min(base_confidence * 1.1, 0.95)  # Higher confidence for many detections
            
            # Adjust confidence based on feature quality
            linearity = row.get('linearity', 0)
            speed_consistency = row.get('speed_consistency', 0)
            
            if linearity > 0.9 and speed_consistency > 0.8:
                base_confidence = min(base_confidence * 1.2, 0.95)  # High quality trajectory
            elif linearity < 0.3 or speed_consistency < 0.3:
                base_confidence *= 0.7  # Poor quality trajectory
            
            results_df.loc[idx, 'confidence'] = round(base_confidence, 3)
        
        return results_df
