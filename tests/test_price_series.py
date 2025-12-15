from pathlib import Path
from src.price_series import PriceSeries

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def test_load_msft_has_rows():
    ps = PriceSeries.from_csv("MSFT", DATA_DIR / "MSFT.csv")
    assert len(ps.df) > 1000

def test_daily_returns_not_empty():
    ps = PriceSeries.from_csv("AAPL", DATA_DIR / "AAPL.csv")
    r = ps.daily_returns()
    assert len(r) > 100