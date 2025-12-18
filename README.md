# Stock Market Analysis Project

A Python library for loading, analyzing, and visualizing historical stock market data. This project provides tools for single-stock analysis and multi-stock portfolio analysis with technical indicators.

## Features

- **PriceSeries**: Load and analyze individual stock price data
  - Data validation and cleaning
  - Daily and cumulative returns calculation
  - Moving averages and volatility metrics
  - Max drawdown analysis
  - Technical indicators (RSI, MACD, Bollinger Bands)

- **PortfolioAnalyzer**: Analyze multiple stocks as a portfolio
  - Correlation analysis between stocks
  - Equal-weight portfolio returns
  - Portfolio summary statistics

- **Technical Indicators**: Built-in technical analysis tools
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands

- **I/O Utilities**: Export analysis results
  - Save DataFrames and Series to CSV
  - Export summary statistics to JSON

## Installation

1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt## Quick Start

### Loading Stock Data

from src.price_series import PriceSeries

# Load a stock from CSV
aapl = PriceSeries.from_csv("AAPL", "data/AAPL.csv")
print(aapl)

# Calculate daily returns
returns = aapl.daily_returns()

# Calculate moving average
ma = aapl.moving_average(window=20)

# Calculate technical indicators
rsi = aapl.rsi(window=14)
macd_df = aapl.macd()
bb = aapl.bollinger_bands()### Portfolio Analysis

from src.price_series import PriceSeries
from src.portfolio_analyzer import PortfolioAnalyzer

# Load multiple stocks
tickers = ["AAPL", "MSFT", "TSLA"]
series_list = [PriceSeries.from_csv(t, f"data/{t}.csv") for t in tickers]

# Create portfolio analyzer
pa = PortfolioAnalyzer(series_list, start_date="2015-01-01")

# Get correlation matrix
corr = pa.correlation_matrix()

# Calculate equal-weight portfolio returns
portfolio_returns = pa.equal_weight_portfolio_returns()### Exporting Results

from src.io_utils import save_dataframe_to_csv, save_dict_to_json

# Save correlation matrix
save_dataframe_to_csv(corr, "outputs/correlation_matrix.csv")

# Save summary statistics
summary = {
    "tickers": pa.tickers(),
    "start_date": pa.summary().start_date,
    "end_date": pa.summary().end_date
}
save_dict_to_json(summary, "outputs/portfolio_summary.json")## Data Format

The project expects CSV files in Stooq format with the following columns:
- `<DATE>` (YYYYMMDD format)
- `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`
- `<VOL>` (volume)

The `PriceSeries.from_csv()` method automatically handles the conversion from Stooq format to standard column names.

## Project Structure

```
stock_market_project/
├── data/              # CSV files with historical stock data
├── src/               # Main source code
│   ├── price_series.py        # Single stock analysis
│   ├── portfolio_analyzer.py # Multi-stock portfolio analysis
│   ├── indicators.py          # Technical indicators
│   └── io_utils.py            # I/O utilities
├── notebooks/         # Jupyter notebooks for interactive analysis
├── tests/             # Unit tests
└── outputs/           # Output directory for exported results
```

## Testing

Run tests with pytest:

```bash
pytest
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib: Data visualization
- pytest: Testing framework

## License

This project is for educational purposes.
```

**Commit message:**
```
Add comprehensive README documentation

- Add project overview and features section
- Include installation instructions
- Add quick start examples for common use cases
- Document data format requirements
- Add project structure overview
- Include testing and dependencies information
- All examples are non-breaking and demonstrate existing functionality
```

This adds documentation without changing any code. All 4 commits are complete and non-breaking.