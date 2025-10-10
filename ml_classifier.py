import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

class MLClassifier:
    def __init__(self, n_clusters=4, contamination=0.1, random_state=42):
        """
        Initialize ML classifier with K-Means clustering and Isolation Forest
        
        Args:
            n_clusters (int): Number of clusters for K-Means (default: 4)
            contamination (float): Expected proportion of anomalies (default: 0.1)
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize models
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # Feature columns for ML processing
        self.feature_columns = [
            'avg_speed', 'speed_consistency', 'duration', 'linearity',
            'direction_changes', 'size_consistency', 'avg_acceleration',
            'satellite_score', 'meteor_score', 'plane_score'
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
        
        # Step 2: Isolation Forest for anomaly detection
        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        anomaly_outlier_scores = self.isolation_forest.score_samples(X_scaled)
        
        # Create results dataframe
        results_df = features_df.copy()
        results_df['cluster'] = cluster_labels
        results_df['is_anomaly'] = anomaly_scores == -1
        results_df['anomaly_score'] = -anomaly_outlier_scores  # Convert to positive scores
        
        # Step 3: Interpret clusters and assign classifications
        results_df = self._interpret_clusters(results_df)
        
        # Step 4: Override classifications for detected anomalies
        results_df.loc[results_df['is_anomaly'], 'classification'] = 'ANOMALY_UAP'
        results_df.loc[results_df['is_anomaly'], 'confidence'] = 0.9
        
        # Step 5: Calculate final confidence scores
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
            results_df.loc[idx, 'is_anomaly'] = classification == 'ANOMALY_UAP'
            results_df.loc[idx, 'anomaly_score'] = 0.5
        
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
            avg_plane_score = cluster_data['plane_score'].mean()
            
            # Determine most likely classification for this cluster
            scores = {
                'Satellite': avg_satellite_score,
                'Meteor': avg_meteor_score,
                'Plane': avg_plane_score,
                'Junk': 0.1  # Base score for junk category
            }
            
            # Additional heuristics
            if avg_speed > 20 and avg_linearity > 0.8 and avg_duration < 2:
                scores['Meteor'] *= 2
            elif avg_speed < 10 and avg_consistency > 0.7 and avg_linearity > 0.7:
                scores['Satellite'] *= 2
            elif avg_duration > 5 and avg_consistency > 0.6:
                scores['Plane'] *= 2
            else:
                scores['Junk'] *= 2
            
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
            results_df.loc[idx, 'classification'] = classification
            results_df.loc[idx, 'confidence'] = confidence
        
        return results_df
    
    def _rule_based_classification(self, row):
        """Rule-based classification for edge cases"""
        # High-speed, linear, short duration -> Meteor
        if row['avg_speed'] > 25 and row['linearity'] > 0.8 and row['duration'] < 3:
            return 'Meteor', 0.8
        
        # Low speed, consistent, linear -> Satellite
        elif row['avg_speed'] < 15 and row['speed_consistency'] > 0.7 and row['linearity'] > 0.6:
            return 'Satellite', 0.7
        
        # Moderate speed, consistent, longer duration -> Plane
        elif 10 < row['avg_speed'] < 25 and row['speed_consistency'] > 0.6 and row['duration'] > 3:
            return 'Plane', 0.7
        
        # High anomaly indicators -> Potential UAP
        elif row['anomaly_indicators'] > 2:
            return 'ANOMALY_UAP', 0.6
        
        # Default to Junk
        else:
            return 'Junk', 0.5
    
    def _calculate_confidence_scores(self, results_df):
        """Calculate and refine confidence scores based on multiple factors"""
        for idx, row in results_df.iterrows():
            base_confidence = row['confidence']
            
            # Adjust confidence based on detection count
            if row['detection_count'] < 3:
                base_confidence *= 0.8  # Lower confidence for few detections
            elif row['detection_count'] > 10:
                base_confidence = min(base_confidence * 1.1, 0.95)  # Higher confidence for many detections
            
            # Adjust confidence based on feature quality
            if row['linearity'] > 0.9 and row['speed_consistency'] > 0.8:
                base_confidence = min(base_confidence * 1.2, 0.95)  # High quality trajectory
            elif row['linearity'] < 0.3 or row['speed_consistency'] < 0.3:
                base_confidence *= 0.7  # Poor quality trajectory
            
            # Special handling for anomalies
            if row['classification'] == 'ANOMALY_UAP':
                # Higher anomaly score = higher confidence in anomaly detection
                anomaly_confidence = min(0.6 + (row['anomaly_score'] * 0.3), 0.95)
                base_confidence = max(base_confidence, anomaly_confidence)
            
            results_df.loc[idx, 'confidence'] = round(base_confidence, 3)
        
        return results_df
