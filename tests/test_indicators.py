from pathlib import Path
from src.price_series import PriceSeries
from src.portfolio_analyzer import PortfolioAnalyzer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def test_correlation_matrix_shape():
    ps_list = [
        PriceSeries.from_csv("AAPL", DATA_DIR / "AAPL.csv"),
        PriceSeries.from_csv("MSFT", DATA_DIR / "MSFT.csv"),
        PriceSeries.from_csv("TSLA", DATA_DIR / "TSLA.csv"),
    ]
    pa = PortfolioAnalyzer(ps_list, start_date="2015-01-01")
    corr = pa.correlation_matrix()
    assert corr.shape == (3, 3)

def test_equal_weight_returns_not_empty():
    ps_list = [
        PriceSeries.from_csv("AAPL", DATA_DIR / "AAPL.csv"),
        PriceSeries.from_csv("MSFT", DATA_DIR / "MSFT.csv"),
        PriceSeries.from_csv("TSLA", DATA_DIR / "TSLA.csv"),
    ]
    pa = PortfolioAnalyzer(ps_list, start_date="2015-01-01")
    port = pa.equal_weight_portfolio_returns()
    assert len(port) > 100
