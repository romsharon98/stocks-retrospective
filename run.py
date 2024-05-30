import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# File paths
FIRM_DATES_FILE = "firms_dates_example.csv"
STOCK_DATA_FILE = "results.csv"


def get_stock_data(ticker, start_date, end_date):
    """
    Retrieve historical market data for a given ticker symbol and date range from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol of the stock.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame or None: Historical market data for the specified ticker and date range.
                           Returns None if an error occurs.
    """
    try:
        # Retrieve historical market data for the specified ticker symbol and date range
        stock_data = yf.download(ticker, start=start_date, end=end_date).reset_index()
        return stock_data
    except Exception as e:
        print(f"Error retrieving data for ticker {ticker}: {e}")
        return None


def process_firm_dates_single(ticker, start_date, end_date):
    """
    Process firm dates and retrieve stock data for a single firm.

    Args:
        ticker (str): The ticker symbol of the stock.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame or None: Historical market data for the specified ticker, and date range.
                           Returns empty dataframe if an error occurs.
    """
    stock_data = get_stock_data(ticker, start_date, end_date)
    result_df = pd.DataFrame()
    if stock_data is not None:
        result_df["firm"] = stock_data["Adj Close"]
        result_df["Date"] = stock_data["Date"]
        return result_df
    else:
        return pd.DataFrame()


def process_firm_dates(firm_dates_df):
    """
    Process firm dates and retrieve stock data for multiple firms.

    Args:
        firm_dates_df (DataFrame): DataFrame containing firm names, ticker symbols,
                                    start dates, and end dates.

    Returns:
        DataFrame: Combined historical market data for all firms in the input DataFrame.
    """
    result_df = pd.DataFrame()
    for _, row in firm_dates_df.iterrows():
        firm = row["firm"]
        ticker = row["ticker"]
        start_date = row["start"]
        end_date = row["end"]
        stock_data = process_firm_dates_single(ticker, start_date, end_date)
        stock_data["name"] = ticker
        if not stock_data.empty:
            result_df = pd.concat([result_df, stock_data], ignore_index=True)
    return result_df


def calculate_alpha(firm_returns, market_returns, r_rf, beta):
    """
    Calculate the alpha for a firm using the Capital Asset Pricing Model (CAPM).

    Args:
        firm_returns (Series): Series containing daily returns for the firm.
        market_returns (Series): Series containing daily returns for the market.
        risk_free_rate (float): The average daily risk-free rate.
        beta (float): The beta coefficient for the firm.

    Returns:
        float: Alpha coefficient for the firm.
    """
    return (firm_returns - r_rf - (beta * (market_returns - r_rf))).mean()


def calculate_beta(firm_returns, market_returns):
    """
    Calculate the beta coefficient for a firm using linear regression.

    Args:
        firm_returns (Series): Series containing daily returns for the firm.
        market_returns (Series): Series containing daily returns for the market.

    Returns:
        float: Beta coefficient for the firm.
    """
    X = sm.add_constant(market_returns)
    model = sm.OLS(firm_returns, X, missing="drop").fit()
    return model.params.iloc[1]


def calculate_sharpe_ratio(firm_returns, risk_free_rate):
    """
    Calculate the Sharpe ratio for a firm.

    Args:
        firm_returns (Series): Series containing daily returns for the firm.
        risk_free_rate (float): The annualized risk-free rate.

    Returns:
        float: Sharpe ratio for the firm.
    """
    excess_returns = firm_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio


def calculate_treynor_ratio(firm_returns, risk_free_rate, beta):
    """
    Calculate the Treynor ratio for a firm.

    Args:
        firm_returns (Series): Series containing daily returns for the firm.
        risk_free_rate (float): The annualized risk-free rate.
        beta (float): The beta coefficient for the firm.

    Returns:
        float: Treynor ratio for the firm.
    """
    excess_returns = firm_returns.mean() - risk_free_rate.mean()
    average_excess_return = excess_returns
    treynor_ratio = average_excess_return / beta
    return treynor_ratio


def get_number_of_days(start_date, end_date):
    """
    Calculate the number of days between two dates.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        int: Number of days between the start and end dates.
    """
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Calculate the number of days between start and end dates
    num_days = (end_date - start_date).days

    return num_days


def convert_to_daily(yearly_return, days):
    return (1 + yearly_return) ** (1 / days) - 1


def calculate_total_return(first_price, last_price):
    """
    Calculate the total return based on the first and last firm prices.

    Args:
        firm_prices (Series): Series containing daily firm prices.

    Returns:
        float: Total return as a percentage.
    """
    total_return = (last_price / first_price) - 1
    return total_return


def calculate_yearly_total_return(
    first_price, last_price, start_date=None, end_date=None
):
    """
    Calculate the yearly total return from the total return and the number of days.

    Args:
        total_return (float): Total return for the entire period.
        num_days (int): Number of days in the entire period.

    Returns:
        float: Yearly total return.
    """

    num_days = get_number_of_days(start_date, end_date)
    total_return = calculate_total_return(first_price, last_price)
    yearly_return = (1 + total_return) ** (365 / num_days) - 1
    return yearly_return


def plot_adj_close_prices(stock_data_df, key, y_lable, title, subplot):
    """
    Plot the 'Adj Close' prices of each firm over time.

    Args:
        stock_data_df (DataFrame): DataFrame containing stock data with columns 'Date', 'Adj Close', and 'name'.

    Returns:
        None
    """
    # Convert 'Date' column to datetime type
    stock_data_df["Date"] = pd.to_datetime(stock_data_df["Date"])

    # Plot 'Adj Close' prices for each firm
    subplot.plot(group["Date"], group[key])

    # Add labels and title
    subplot.set_xlabel("Date")
    subplot.set_ylabel(y_lable)
    subplot.set_title(title)

    # Add legend
    subplot.legend()


def plot_stock_returns_histogram(r_firm, subplot):
    """
    Plot a histogram of stock returns (yield) showing the frequency of returns.

    Args:
        r_firm (DataFrame): DataFrame containing stock data with column 'r_firm'.

    Returns:
        None
    """
    # Plot histogram of stock returns
    subplot.hist(r_firm, bins=60, color="skyblue", edgecolor="black")

    # Add labels and title
    subplot.set_xlabel("Returns")
    subplot.set_ylabel("Frequency")
    subplot.set_title("Histogram of Stock Returns: Discover Financial Services")


def plot_regression_with_alpha_beta(stock_data_df, alpha_hat, beta_hat, subplot):
    """
    Plot the regression line for firm returns along with pre-calculated alpha and beta coefficients.

    Args:
        stock_data_df (DataFrame): DataFrame containing stock data with columns 'r_firm', 'r_market', and 'rf'.
        alpha_hat (float): Estimated alpha coefficient.
        beta_hat (float): Estimated beta coefficient.

    Returns:
        None
    """
    # Plot the actual data points
    subplot.scatter(
        (stock_data_df["r_market"] - stock_data_df["r_rf"]),
        (stock_data_df["r_firm"] - stock_data_df["r_rf"]),
        label="Data",
    )

    # Plot the regression line
    x_values = stock_data_df["r_market"] - stock_data_df["r_rf"]
    y_values = alpha_hat + beta_hat * x_values
    subplot.plot(x_values, y_values, color="red", label="Regression Line")

    # Add alpha and beta annotations
    subplot.text(
        0.05, 0.9, f"Alpha: {alpha_hat:.4f}", transform=plt.gca().transAxes, fontsize=12
    )
    subplot.text(
        0.05, 0.85, f"Beta: {beta_hat:.4f}", transform=plt.gca().transAxes, fontsize=12
    )

    # Add labels and title
    subplot.set_xlabel("Adjust market returns")
    subplot.set_ylabel("Adjusted returns")
    subplot.set_title("Returns vs. market returns: Discover Financial Services")

    # Add legend
    subplot.legend()

    # Show plot
    subplot.grid(True)


# Read the firm_dates.csv file
firm_dates_df = pd.read_csv(FIRM_DATES_FILE)

# Process firm dates and retrieve stock data
stock_data_df = process_firm_dates(firm_dates_df)

# Retrieve SPY data for the same date range
spy_data = get_stock_data(
    "SPY", stock_data_df["Date"].min(), stock_data_df["Date"].max()
)

# Merge data with main dataframe
if not spy_data.empty:
    # Merge SPY data with the main DataFrame based on the date column
    stock_data_df = pd.merge(
        stock_data_df,
        spy_data[["Date", "Adj Close"]],
        on="Date",
        how="left",
        suffixes=("", "_spy"),
    )
    # Rename the merged column
    stock_data_df.rename(columns={"Adj Close": "market"}, inplace=True)

# Retrieve IRX data for the same date range
irx_data = get_stock_data(
    "^irx", stock_data_df["Date"].min(), stock_data_df["Date"].max()
)

if not irx_data.empty:
    # Merge IRX data with the main DataFrame based on the date column
    stock_data_df = pd.merge(
        stock_data_df,
        irx_data[["Date", "Adj Close"]],
        on="Date",
        how="left",
        suffixes=("", "_irx"),
    )
    stock_data_df.rename(columns={"Adj Close": "rf"}, inplace=True)

print(stock_data_df.head(5))

# Calculate daily return for each firm
stock_data_df["r_firm"] = stock_data_df.groupby("name")["firm"].pct_change(
    fill_method=None
)

# Calculate daily return for market
stock_data_df["r_market"] = stock_data_df.groupby("name")["market"].pct_change(
    fill_method=None
)

# Calculate daily return for rf
days = get_number_of_days(stock_data_df["Date"].min(), stock_data_df["Date"].max())
stock_data_df["r_rf"] = stock_data_df["rf"].apply(convert_to_daily, days=13 * days)

stock_data_df = stock_data_df.dropna().reset_index()
print(stock_data_df[stock_data_df["name"] == "DFS"].head(5))

# Calculate Alpha, Beta, Sharp, Treynor, Anual returns
result = pd.read_csv(FIRM_DATES_FILE)
beta_dict = {}
alpha_dict = {}
sharp_ratio = {}
treynor_ratio = {}
yearly_total_return = {}

for name, group in stock_data_df.groupby("name"):
    r_firm = group["r_firm"]
    r_market = group["r_market"]
    r_rf = group["r_rf"]
    firm = group["firm"]
    date = group["Date"]
    beta_dict[name] = calculate_beta(r_firm, r_market)
    alpha_dict[name] = calculate_alpha(r_firm, r_market, r_rf, beta_dict[name])
    sharp_ratio[name] = calculate_sharpe_ratio(r_firm, r_rf)
    treynor_ratio[name] = calculate_treynor_ratio(r_firm, r_rf, beta_dict[name])
    yearly_total_return[name] = calculate_yearly_total_return(
        firm.iloc[0],
        firm.iloc[-1],
        start_date=date.min(),
        end_date=date.max(),
    )

# Map beta values to corresponding firm names
result["Alpha"] = result["ticker"].map(alpha_dict)
result["Beta"] = result["ticker"].map(beta_dict)
result["Sharp"] = result["ticker"].map(sharp_ratio)
result["Treynor"] = result["ticker"].map(treynor_ratio)
result["Annual returns"] = result["ticker"].map(yearly_total_return)

# Plot graphs
for name, group in stock_data_df.groupby("name"):
    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(f"Stock: {name}", fontsize=14)
    plot_adj_close_prices(
        group, "firm", "Price ($)", "Stock prices: Discover Financial Services", fig.add_subplot(2,2, 1)
    )
    plot_adj_close_prices(
        group,
        "r_firm",
        "Returns (%)",
        "Stock returns: Discover Financial Services",
        fig.add_subplot(2,2, 2)
    )
    plot_stock_returns_histogram(group["r_firm"], fig.add_subplot(2,2, 3))
    plot_regression_with_alpha_beta(
        group,
        result.loc[result["ticker"] == name, "Alpha"].values[0],
        result.loc[result["ticker"] == name, "Beta"].values[0],
        fig.add_subplot(2,2, 4)
    )
    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()
# Requirments adjustment
result.rename(columns={"firm": "Firm"}, inplace=True)
result.rename(columns={"start": "Start date"}, inplace=True)
result.rename(columns={"end": "End date"}, inplace=True)
result.drop(columns=["ticker"], inplace=True)
print(result.head(5))

result.to_csv(STOCK_DATA_FILE)  # Save the result DataFrame to a CSV file
print("Stock data saved successfully.")
