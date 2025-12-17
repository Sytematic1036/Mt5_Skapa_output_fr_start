import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

print("=" * 70)
print("VIX VALIDERING: MIN KÄLLA vs YAHOO FINANCE")
print("=" * 70)

# 1. Ladda min betrodda fil
trusted = pd.read_excel("KommunikationMH/VIX_6000d_2025-12-04_21-35-38.xlsx")
trusted['Date'] = pd.to_datetime(trusted['Date'])
trusted = trusted.set_index('Date').sort_index()

start_date = trusted.index.min().strftime('%Y-%m-%d')
end_date = (trusted.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"\nMin fil: {start_date} till {trusted.index.max().strftime('%Y-%m-%d')}, {len(trusted)} rader")

# 2. Ladda yfinance
yf_data = yf.download("^VIX", start=start_date, end=end_date, progress=False)
if isinstance(yf_data.columns, pd.MultiIndex):
    yf_data.columns = yf_data.columns.get_level_values(0)
if yf_data.index.tz:
    yf_data.index = yf_data.index.tz_localize(None)

print(f"YFinance: {yf_data.index.min().strftime('%Y-%m-%d')} till {yf_data.index.max().strftime('%Y-%m-%d')}, {len(yf_data)} rader")

# 3. Jämför datum
trusted_dates = set(trusted.index)
yf_dates = set(yf_data.index)
common = trusted_dates & yf_dates
only_trusted = trusted_dates - yf_dates
only_yf = yf_dates - trusted_dates

print(f"\nGemensamma datum: {len(common)}")
print(f"Endast i min fil: {len(only_trusted)}")
print(f"Endast i yfinance: {len(only_yf)}")

# 4. Jämför Close-kurser
common_list = sorted(common)
comp = pd.DataFrame(index=common_list)
comp['Trusted'] = trusted.loc[common_list, 'Close'].values
comp['YFinance'] = yf_data.loc[common_list, 'Close'].values
comp['Diff'] = (comp['Trusted'] - comp['YFinance']).abs()

corr = comp['Trusted'].corr(comp['YFinance'])

print(f"\n{'='*70}")
print("RESULTAT")
print(f"{'='*70}")
print(f"Korrelation: {corr:.8f}")
print(f"Snitt avvikelse: {comp['Diff'].mean():.4f}")
print(f"Max avvikelse: {comp['Diff'].max():.4f}")

exact = len(comp[comp['Diff'] < 0.01])
print(f"Exakt match (<0.01): {exact} av {len(comp)} ({exact/len(comp)*100:.1f}%)")

# 5. Största avvikelserna
print(f"\nTOP 10 STÖRSTA AVVIKELSER:")
print(f"{'Datum':<12} {'Min fil':>10} {'YFinance':>10} {'Diff':>10}")
print("-" * 45)
for date, row in comp.nlargest(10, 'Diff').iterrows():
    print(f"{date.strftime('%Y-%m-%d'):<12} {row['Trusted']:>10.2f} {row['YFinance']:>10.2f} {row['Diff']:>10.2f}")

# 6. Slutsats
print(f"\n{'='*70}")
if corr > 0.9999 and comp['Diff'].mean() < 0.1:
    print("[OK] SLUTSATS: YFinance ar PALITLIG - data matchar din betrodda kalla!")
else:
    print("[!] SLUTSATS: Avvikelser hittade - granska resultaten ovan")
print(f"{'='*70}")

# 7. Spara jämförelse
comp.to_csv("vix_comparison.csv")
print(f"\nDetaljer sparade i: vix_comparison.csv")
