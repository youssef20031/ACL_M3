"""
Adaptation Curve Features for Promoted Teams

Models how promoted players adapt to EPL over first season:
- GW 1-6: Steep learning curve (struggling phase)
- GW 7-15: Adaptation phase (improving)
- GW 16+: Stabilized performance

Features:
- adaptation_progress: Sigmoid curve 0→1 over season
- adaptation_velocity: Rate of improvement
- experience_weight: Blending factor for Championship vs EPL data
"""
import numpy as np
import pandas as pd
from typing import Dict


def sigmoid_adaptation_curve(gw: int, 
                             inflection_point: int = 12,
                             steepness: float = 0.3) -> float:
    """
    Sigmoid adaptation curve for promoted players.
    
    Args:
        gw: Current gameweek
        inflection_point: GW where curve is at 50%
        steepness: How quickly adaptation happens
        
    Returns:
        Adaptation progress [0, 1]
    """
    # Sigmoid: 1 / (1 + exp(-k * (x - x0)))
    progress = 1.0 / (1.0 + np.exp(-steepness * (gw - inflection_point)))
    return float(progress)


def adaptation_velocity(gw: int,
                       inflection_point: int = 12,
                       steepness: float = 0.3) -> float:
    """
    Rate of adaptation (derivative of sigmoid).
    
    Peak velocity occurs at inflection point.
    
    Args:
        gw: Current gameweek
        inflection_point: GW of peak adaptation
        steepness: Curve steepness
        
    Returns:
        Velocity [0, ~0.1]
    """
    x = gw - inflection_point
    exp_term = np.exp(-steepness * x)
    velocity = (steepness * exp_term) / ((1 + exp_term) ** 2)
    return float(velocity)


def experience_blend_weight(gw: int,
                            min_weight: float = 0.1,
                            max_weight: float = 0.9) -> float:
    """
    Calculate blending weight for Championship vs EPL data.
    
    Early season: Rely more on Championship data
    Late season: Rely more on EPL data
    
    Args:
        gw: Current gameweek
        min_weight: Minimum EPL data weight (early season)
        max_weight: Maximum EPL data weight (late season)
        
    Returns:
        EPL data weight [min_weight, max_weight]
    """
    progress = sigmoid_adaptation_curve(gw)
    weight = min_weight + (max_weight - min_weight) * progress
    return float(weight)


def add_adaptation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add adaptation curve features to dataframe.
    
    Features added:
    - adaptation_progress: Sigmoid 0→1
    - adaptation_velocity: Rate of change
    - experience_weight: EPL data blending factor
    - adaptation_phase: Categorical (early/mid/late)
    
    Args:
        df: Dataframe with 'GW' and 'is_promoted' columns
        
    Returns:
        Dataframe with new features
    """
    df = df.copy()
    
    # Only add for promoted players
    promoted_mask = df.get('is_promoted', False)
    
    # Initialize features
    df['adaptation_progress'] = 0.0
    df['adaptation_velocity'] = 0.0
    df['experience_weight'] = 1.0  # Non-promoted players use full EPL weight
    df['adaptation_phase'] = 'established'
    
    # Calculate for promoted players
    if promoted_mask.any():
        for idx, row in df[promoted_mask].iterrows():
            gw = row['GW']
            
            df.at[idx, 'adaptation_progress'] = sigmoid_adaptation_curve(gw)
            df.at[idx, 'adaptation_velocity'] = adaptation_velocity(gw)
            df.at[idx, 'experience_weight'] = experience_blend_weight(gw)
            
            # Categorize phase
            if gw <= 6:
                df.at[idx, 'adaptation_phase'] = 'early'
            elif gw <= 15:
                df.at[idx, 'adaptation_phase'] = 'mid'
            else:
                df.at[idx, 'adaptation_phase'] = 'late'
    
    return df


def get_adaptation_multiplier(gw: int,
                              baseline_multiplier: float = 0.70,
                              final_multiplier: float = 0.95) -> float:
    """
    Get performance multiplier based on adaptation curve.
    
    Used to adjust predicted points:
    - GW 1: 0.70 × prediction (struggling)
    - GW 12: 0.82 × prediction (adapting)
    - GW 30: 0.95 × prediction (adapted)
    
    Args:
        gw: Current gameweek
        baseline_multiplier: Starting multiplier (GW 1)
        final_multiplier: Final multiplier (GW 38)
        
    Returns:
        Multiplier [baseline_multiplier, final_multiplier]
    """
    progress = sigmoid_adaptation_curve(gw)
    multiplier = baseline_multiplier + (final_multiplier - baseline_multiplier) * progress
    return float(multiplier)


# Example usage and testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Generate example data
    gameweeks = np.arange(1, 39)
    
    # Calculate curves
    progress = [sigmoid_adaptation_curve(gw) for gw in gameweeks]
    velocity = [adaptation_velocity(gw) for gw in gameweeks]
    multiplier = [get_adaptation_multiplier(gw) for gw in gameweeks]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(gameweeks, progress, 'b-', linewidth=2)
    axes[0].set_title('Adaptation Progress')
    axes[0].set_xlabel('Gameweek')
    axes[0].set_ylabel('Progress [0, 1]')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=12, color='r', linestyle='--', alpha=0.5, label='Inflection (GW 12)')
    axes[0].legend()
    
    axes[1].plot(gameweeks, velocity, 'g-', linewidth=2)
    axes[1].set_title('Adaptation Velocity')
    axes[1].set_xlabel('Gameweek')
    axes[1].set_ylabel('Velocity')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=12, color='r', linestyle='--', alpha=0.5, label='Peak (GW 12)')
    axes[1].legend()
    
    axes[2].plot(gameweeks, multiplier, 'orange', linewidth=2)
    axes[2].set_title('Performance Multiplier')
    axes[2].set_xlabel('Gameweek')
    axes[2].set_ylabel('Multiplier')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.70, color='r', linestyle='--', alpha=0.3, label='Baseline (70%)')
    axes[2].axhline(y=0.95, color='b', linestyle='--', alpha=0.3, label='Adapted (95%)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('ml/features/adaptation_curves.png', dpi=150)
    print("✅ Saved adaptation curves plot to ml/features/adaptation_curves.png")
    
    # Test with dataframe
    test_df = pd.DataFrame({
        'GW': [1, 6, 12, 20, 38],
        'is_promoted': [True] * 5,
        'name': ['Test Player'] * 5
    })
    
    test_df = add_adaptation_features(test_df)
    print("\nAdaptation features example:")
    print(test_df[['GW', 'adaptation_progress', 'adaptation_velocity', 'experience_weight', 'adaptation_phase']])
