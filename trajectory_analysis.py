import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def predict_trajectory(detections_df, object_id, method='linear'):
    """
    Predict object trajectory and compare with actual path
    
    Args:
        detections_df: DataFrame with detection data
        object_id: Object ID to analyze
        method: 'linear' or 'polynomial' prediction
        
    Returns:
        dict with prediction results and metrics
    """
    obj_data = detections_df[detections_df['object_id'] == object_id].copy()
    
    if len(obj_data) < 3:
        return None
    
    obj_data = obj_data.sort_values('frame_number')
    
    frames = obj_data['frame_number'].values.reshape(-1, 1)
    x_positions = obj_data['center_x'].values
    y_positions = obj_data['center_y'].values
    
    if method == 'polynomial':
        poly = PolynomialFeatures(degree=2)
        frames_poly = poly.fit_transform(frames)
        
        model_x = LinearRegression()
        model_y = LinearRegression()
        model_x.fit(frames_poly, x_positions)
        model_y.fit(frames_poly, y_positions)
        
        x_pred = model_x.predict(frames_poly)
        y_pred = model_y.predict(frames_poly)
    else:
        model_x = LinearRegression()
        model_y = LinearRegression()
        model_x.fit(frames, x_positions)
        model_y.fit(frames, y_positions)
        
        x_pred = model_x.predict(frames)
        y_pred = model_y.predict(frames)
    
    errors = np.sqrt((x_positions - x_pred)**2 + (y_positions - y_pred)**2)
    
    rmse_x = np.sqrt(mean_squared_error(x_positions, x_pred))
    rmse_y = np.sqrt(mean_squared_error(y_positions, y_pred))
    rmse_total = np.sqrt(rmse_x**2 + rmse_y**2)
    
    r2_x = r2_score(x_positions, x_pred)
    r2_y = r2_score(y_positions, y_pred)
    
    return {
        'object_id': object_id,
        'frames': frames.flatten(),
        'actual_x': x_positions,
        'actual_y': y_positions,
        'predicted_x': x_pred,
        'predicted_y': y_pred,
        'errors': errors,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_total': rmse_total,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'method': method
    }

def analyze_all_trajectories(detections_df, method='linear'):
    """
    Analyze trajectories for all objects
    
    Args:
        detections_df: DataFrame with all detection data
        method: Prediction method ('linear' or 'polynomial')
        
    Returns:
        list of prediction results
    """
    results = []
    unique_objects = detections_df['object_id'].unique()
    
    for obj_id in unique_objects:
        prediction = predict_trajectory(detections_df, obj_id, method)
        if prediction:
            results.append(prediction)
    
    return results

def create_trajectory_comparison_plot(prediction_result, classification=None):
    """
    Create interactive plot comparing predicted vs actual trajectory
    
    Args:
        prediction_result: Result from predict_trajectory()
        classification: Object classification for title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prediction_result['actual_x'],
        y=prediction_result['actual_y'],
        mode='markers+lines',
        name='Actual Path',
        marker=dict(size=8, color='#00ff00'),
        line=dict(color='#00ff00', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_result['predicted_x'],
        y=prediction_result['predicted_y'],
        mode='lines',
        name='Predicted Path',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    errors = prediction_result['errors']
    for i in range(len(prediction_result['actual_x'])):
        fig.add_trace(go.Scatter(
            x=[prediction_result['actual_x'][i], prediction_result['predicted_x'][i]],
            y=[prediction_result['actual_y'][i], prediction_result['predicted_y'][i]],
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 0.3)', width=1),
            showlegend=False,
            hovertemplate=f'Error: {errors[i]:.2f} px<extra></extra>'
        ))
    
    title = f"Trajectory Prediction - Object #{prediction_result['object_id']}"
    if classification:
        title += f" ({classification})"
    
    fig.update_layout(
        title=title,
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        hovermode='closest',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=500,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed")
    )
    
    return fig

def create_trajectory_error_plot(trajectory_results):
    """
    Create plot showing prediction errors across all objects
    
    Args:
        trajectory_results: List of prediction results
        
    Returns:
        Plotly figure
    """
    if not trajectory_results:
        return None
    
    object_ids = [r['object_id'] for r in trajectory_results]
    mean_errors = [r['mean_error'] for r in trajectory_results]
    max_errors = [r['max_error'] for r in trajectory_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=object_ids,
        y=mean_errors,
        name='Mean Error',
        marker_color='#4ecdc4'
    ))
    
    fig.add_trace(go.Bar(
        x=object_ids,
        y=max_errors,
        name='Max Error',
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="Trajectory Prediction Errors by Object",
        xaxis_title="Object ID",
        yaxis_title="Error (pixels)",
        barmode='group',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def get_trajectory_summary_stats(trajectory_results):
    """
    Calculate summary statistics for all trajectory predictions
    
    Args:
        trajectory_results: List of prediction results
        
    Returns:
        dict with summary statistics
    """
    if not trajectory_results:
        return None
    
    all_mean_errors = [r['mean_error'] for r in trajectory_results]
    all_r2_x = [r['r2_x'] for r in trajectory_results]
    all_r2_y = [r['r2_y'] for r in trajectory_results]
    
    return {
        'total_objects': len(trajectory_results),
        'avg_mean_error': np.mean(all_mean_errors),
        'median_mean_error': np.median(all_mean_errors),
        'avg_r2_x': np.mean(all_r2_x),
        'avg_r2_y': np.mean(all_r2_y),
        'best_r2_x': np.max(all_r2_x),
        'worst_r2_x': np.min(all_r2_x),
        'highly_predictable': sum(1 for r in trajectory_results if r['r2_x'] > 0.95 and r['r2_y'] > 0.95)
    }
