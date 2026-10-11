import pandas as pd

print("Checking FPL_2025_2026.csv after adding GW 30-38...")
print("=" * 60)

df = pd.read_csv('FPL_2025_2026.csv')

print(f"Total rows: {len(df):,}")
print(f"Gameweeks: {sorted(df['GW'].unique())}")
print(f"Latest GW: {df['GW'].max()}")

gw_counts = df.groupby('GW').size()
print(f"\nRows per GW (last 10 GWs):")
print(gw_counts.tail(10))

print(f"\nUnique players: {df['name'].nunique()}")
print(f"Date range: {df['kickoff_time'].min()} to {df['kickoff_time'].max()}")
