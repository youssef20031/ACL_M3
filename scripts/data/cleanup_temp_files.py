"""Clean up temporary diagnostic files created during the fixture fix."""
import os

temp_files = [
    "check_fixtures.py",
    "check_teams_players.py",
    "check_relationships.py",
    "check_fixture_structure.py",
    "check_player_properties.py",
    "find_player_team_mapping.py",
    "check_schema.py",
    "check_teams_with_fixtures.py",
    "test_prediction_with_fixture.py",
    "test_fixture_context_working.py",
]

print("Cleaning up temporary diagnostic files...")
print("=" * 60)

for filename in temp_files:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"✅ Deleted: {filename}")
    else:
        print(f"⚠️  Not found: {filename}")

print("\n" + "=" * 60)
print("Cleanup complete!")
print("\nKeeping these important files:")
print("  - create_player_team_relationships.py")
print("  - fix_fixture_team_names.py")
print("  - data/FPL_2026_2027_fixtures_fixed.csv")
print("  - verify_fixture_fix.py")
print("  - FIXTURE_FIX_SUMMARY.md")
