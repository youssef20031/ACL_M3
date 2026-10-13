"""
Test that promoted teams system is integrated with main predictor
"""

print("Testing Promoted Teams Integration...")
print("="*80)

# Test 1: Check if archetype database exists
import os
from pathlib import Path

archetype_v2 = Path('ml/models/promoted_archetypes_v2.pkl')
print(f"\n1. Archetype Database V2: {'✓ EXISTS' if archetype_v2.exists() else '✗ MISSING'}")

if archetype_v2.exists():
    import pickle
    with open(archetype_v2, 'rb') as f:
        archetypes = pickle.load(f)
    print(f"   - {len(archetypes)} player transitions loaded")

# Test 2: Check if translation models exist
print("\n2. Translation Models:")
for pos in ['DEF', 'MID', 'FWD']:
    model_path = Path(f'ml/models/player_translation_{pos}_v1.pkl')
    status = '✓' if model_path.exists() else '✗'
    print(f"   {status} {pos}")

team_model = Path('ml/models/team_translation_v1.pkl')
print(f"   {'✓' if team_model.exists() else '✗'} TEAM")

# Test 3: Initialize predictor
print("\n3. FPLPredictor Initialization:")
try:
    from ml.predictor import FPLPredictor
    predictor = FPLPredictor(enable_promoted_teams_handler=True)
    print("   ✓ Predictor initialized")
    
    if predictor.promoted_teams_handler:
        print("   ✓ Promoted teams handler loaded")
        if hasattr(predictor.promoted_teams_handler, 'archetype_db') and predictor.promoted_teams_handler.archetype_db:
            print(f"   ✓ Archetype DB: {len(predictor.promoted_teams_handler.archetype_db)} players")
        if hasattr(predictor.promoted_teams_handler, 'translation_models'):
            print(f"   ✓ Translation models: {list(predictor.promoted_teams_handler.translation_models.keys())}")
    else:
        print("   ✗ Promoted teams handler not loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Check ML API Integration
print("\n4. ML API Integration:")
try:
    from ml.api_integration import MLAPIIntegration
    print("   ✓ MLAPIIntegration importable")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*80)
print("Integration Test Complete")
print("="*80)
