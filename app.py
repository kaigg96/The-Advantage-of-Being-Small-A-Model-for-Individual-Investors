# 1. Imports
from flask import Flask, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

# 2. create an instance of the Flask class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asecretkey'

# 3. define a prediction function
# 3a. load price data
start_date = datetime.date(2023,1,1)
end_date = datetime.date.today()
tickers = pd.read_csv('tickers.csv')
tickers = tickers.rename(columns = {"A": "Ticker"})
tickers = tickers["Ticker"]

closing_price_data = pd.DataFrame()
high_price_data = pd.DataFrame()
low_price_data = pd.DataFrame()

downloaded_tickers = []

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
    closing_price_data = pd.concat([closing_price_data, data['Close']], axis=1)
    high_price_data = pd.concat([high_price_data, data['High']], axis=1)
    low_price_data = pd.concat([low_price_data, data['Low']], axis=1)
    
closing_price_data.columns = tickers.values
high_price_data.columns = tickers.values
low_price_data.columns = tickers.values

closing_price_data.to_csv('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/' + 
                          'Equity-market-analysis/closing_price_data.csv', index=True, mode='w')

high_price_data.to_csv('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/' + 
                          'Equity-market-analysis/high_price_data.csv', index=True, mode='w')

low_price_data.to_csv('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/' + 
                          'Equity-market-analysis/low_price_data.csv', index=True, mode='w')

#since csv can't store axis, need to reset it

def load_price_csvs(path):
    """
    Load and set index of the given CSV.
    Meant for use on closing, high, and low price data files.
    """

    price_data = pd.read_csv(path)
    price_data = price_data.set_index("Unnamed: 0")
    price_data = price_data.rename_axis("Date")
    
    return price_data

high_price_data = load_price_csvs('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/Equity-ma' \
                      'rket-analysis/high_price_data.csv')

low_price_data = load_price_csvs('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/Equity-ma' \
                      'rket-analysis/low_price_data.csv')

closing_price_data = load_price_csvs('/Users/kaigroden-gilchrist/Downloads/Personal_Project/SuccessiveSmallGains/Equity-ma' \
                      'rket-analysis/closing_price_data.csv')

market_cap_dict = {}

for ticker in tickers:
    try:
        mcap = yf.Ticker(ticker).fast_info['marketCap']
        market_cap_dict.update({ticker:mcap})
    except: KeyError #ignore tickers with no market cap available

# create dicts of tickers for each market cap
small_caps_dict = {k: v for k, v in market_cap_dict.items() if v < 2_000_000_000}

# retrieve the tickers (keys) for each dict
small_cap_tickers = small_caps_dict.keys()

# filter the original price data for only the tickers with that market cap
small_cap_price_data = closing_price_data[small_cap_tickers]

def min_var_filter(price_data, model_start_date, var=0.1, period=200):
    """
    Return dict of {tickers:price adjusted stdev} for stocks whose price stays within var% of of the mean price
        in period days prior to the model_start_days.

    If max/min price too high/low set var to -1.
    Else find var to make sure price isn't TOO stable.

    """
    # set date range
    start_date = str(model_start_date - datetime.timedelta(period))
    end_date = str(model_start_date)

    # retrieve data and drop nan
    stock_data = closing_price_data.loc[start_date:end_date]
    nan_cols = stock_data.columns[stock_data.iloc[0].isna()]
    stock_data = stock_data.drop(columns=nan_cols)

    var_dict = {}

    def calc_var(closing_price_data, column):
        """
        If the given ticker's max and min price is within var% of the mean, return it and the price adjusted
            deviation. 
        Else return the ticker and -1. 
        """
        ticker = column.name

        avg_close = np.mean(closing_price_data)
        max_close = np.max(closing_price_data)
        min_close = np.min(closing_price_data)

        upper_bound = (1+var)*avg_close
        lower_bound = (1-var)*avg_close

        # check prices
        # if price stays within bounds, calculate variability
        if (max_close > upper_bound) or (min_close < lower_bound):
            pv = -1
        else:         
            closing_std_dev = np.std(closing_price_data) 
            pv = closing_std_dev / avg_close #price adjusted deviation

        return (ticker, pv)

    # use dictionary comprehension to filter out negative values
    var_dict = {key: val for key, val in [calc_var(stock_data[col_name], stock_data[col_name]) 
                                          for col_name in stock_data.columns] if val >= 0}

    return var_dict

def back_calculate_returns(list_of_stocks, start_date, end_date):
    """
    Find and print the optimal buy and sell price of each stock in the given list over the given range
    Also gives total % returns over that period.

    Initially tried vectorized buy and sell functions, but actually increased runtime so reverted to for-loops.
    """

    def retrieve_stock_data(stock, start_date, end_date):
        """Return high and low price data for the given stock in the given date range"""
        high_data = high_price_data.loc[start_date:end_date][stock].round(2)
        low_data = low_price_data.loc[start_date:end_date][stock].round(2)

        stock_data = pd.concat([high_data, low_data], axis=1)
        stock_data.columns = ["High", "Low"]

        return stock_data

    def init(stock_data):
        min_low = np.min(stock_data["Low"])
        max_high = np.max(stock_data["High"])
        increment = (min_low / 100) + 0.01

        return min_low, max_high, increment


    def find_best_buy_sell(stock_data, min_low, max_high, increment, 
                           owned=False, max_profit=0, best_buy=0, best_sell=0):
        """
        Find the best buy and sell prices for a given stock using the given price data and price range
        """
        buy_prices = np.arange(min_low, max_high, increment)

        for buy_price in buy_prices:
            sell_prices = np.arange(buy_price, max_high, increment)

            for sell_price in sell_prices:
                profit = 0.
                owned = False

                for index, row in stock_data.iterrows():
                    if owned:
                        if (row["High"] >= sell_price):
                            profit += (sell_price - buy_price)
                            owned = False
                    else:
                        if (row["Low"] <= buy_price):
                            owned = True

                if (profit > max_profit):
                    max_profit = round(profit, 2)
                    best_sell = round(sell_price, 2)
                    best_buy = round(buy_price, 2)

        return max_profit, best_sell, best_buy

    returns = {}

    for stock in list_of_stocks:
        stock_data = retrieve_stock_data(stock, start_date, end_date)
        min_low, max_high, increment = init(stock_data)
        max_profit, best_sell, best_buy = find_best_buy_sell(stock_data, min_low, max_high, increment)
        percent_return = max_profit / best_buy * 100 if best_buy != 0 else 0
        num_buys_sells = round(max_profit / (best_sell - best_buy)) if best_buy != 0 else 0

        returns.update({stock : {"buy_price" : best_buy, "sell_price" : best_sell, 
                "min_low" : min_low, "max_high" : max_high, 'backtest_returns' : percent_return, 
                       'num_buys_sells' : num_buys_sells}})


    return returns

def front_calculate_returns(dict_of_stocks, start_date, end_date):
    """
    Calculate returns over given interval for each stock using buy/sell prices within the given dict. 

    Sell out if price goes below min low (also within the given dict).
    """

    def retrieve_stock_data(stock, start_date, end_date):
        """Return high and low price data for the given stock in the given date range"""
        high_data = high_price_data.loc[start_date:end_date][stock].round(2)
        low_data = low_price_data.loc[start_date:end_date][stock].round(2)

        stock_data = pd.concat([high_data, low_data], axis=1)
        stock_data.columns = ["High", "Low"]

        return stock_data

    def calculate_returns(stock_data, buy_price, sell_price, min_low):
        owned = False
        profit = 0.

        for index, row in stock_data.iterrows():
            #break if you leave the lower bound of the backtest data
            if (row["Low"] < min_low):
                if (owned):
                    profit += row["Low"]
                break
            elif (owned):
                if (row["High"] >= sell_price):
                    profit += sell_price
                    owned = False
                # sell at the low price (to be conservative) on end date if still owned
                elif (index == stock_data.index[-1]):
                    profit += row["Low"]
                    owned = False
            else:
                if (row["Low"] <= buy_price):
                    profit -= row["Low"]
                    owned = True

        return profit

    fwd_returns_dict = {}  

    for ticker in dict_of_stocks:

        prices = dict_of_stocks[ticker]
        buy_price = prices["buy_price"]
        sell_price = prices["sell_price"]
        min_low = prices["min_low"]
        max_high = prices["max_high"]

        stock_data = retrieve_stock_data(ticker, start_date, end_date)
        stock_data = stock_data.round(2)

        profit = calculate_returns(stock_data, buy_price, sell_price, min_low)

        try:
            percent_return = profit / buy_price * 100
        except ZeroDivisionError:
            percent_return = 0

        fwd_returns_dict.update({ticker:percent_return})

    return fwd_returns_dict

def list_of_returns_calc(price_data, var, dates, back_days, front_days):
    """
    Uses the given price data to:
        1. Filter stocks which kept within var percent of their mean price over the back_days backtest period.
        2. On each filtered stock, performs the backtest using the back_calculate_returns function defined in 
            2.b. 
        3. Retrieve price data for each backtest period, normalized as a percent return from the start date.
        4. Uses the backtest output to calculate forward returns using the front_calculate_returns function 
            defined in 2.c. 
    Returns the output of each of the above as a separate dataframe for ease of validation/processing downstream.
    """

    var_ret = pd.DataFrame()
    back_ret = pd.DataFrame()
    price_movement = pd.DataFrame()
    fwd_ret = pd.DataFrame()

    for date in dates:
        extracted_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        model_start_date = datetime.date(extracted_date.year, extracted_date.month, extracted_date.day)

        var_dict = min_var_filter(price_data, var=var, model_start_date=model_start_date, period=back_days)
        ticker_list = list(var_dict.keys())
        var_ret = pd.concat([var_ret, pd.DataFrame.from_dict(var_dict, orient='index', columns=['Var']).T], axis=1)

        back_start_date = str(model_start_date - datetime.timedelta(back_days))
        back_end_date = str(model_start_date)
        back_returns = back_calculate_returns(ticker_list, back_start_date, back_end_date)
        back_ret = pd.concat([back_ret, pd.DataFrame(back_returns)],axis=1)

        temp = closing_price_data.loc[back_start_date:back_end_date, ticker_list].apply(
                 lambda x: x.div(x.iloc[0]).subtract(1).mul(100)).T
        temp.columns = np.arange(1, len(temp.columns) + 1)
        price_movement = pd.concat([price_movement, temp.T], axis=1)

        fwd_start_date = str(model_start_date) 
        fwd_end_date = str(model_start_date + datetime.timedelta(front_days))
        fwd_returns = front_calculate_returns(back_returns, fwd_start_date, fwd_end_date)
        fwd_ret = pd.concat([fwd_ret, pd.DataFrame.from_dict(fwd_returns, orient='index', columns=['Fwd rets']).T], axis=1)

    return var_ret, back_ret, fwd_ret, price_movement

def return_prediction(deployment_test_dates, model):
    """
    Full deployment function.
    """
  
    def acquire_data(deployment_test_dates):
        deployment_var_df, deployment_back_df, deployment_fwd_df, deployment_price_df = list_of_returns_calc(
            small_cap_price_data, 0.1, deployment_test_dates, 150, 90
        )

        deployment_combined_df = pd.concat([deployment_var_df.T, deployment_back_df.T, 
                                           deployment_fwd_df.T, deployment_price_df.T], axis=1)

        price_cols = deployment_combined_df.loc[:,1:]

        # add mean price feature
        mean_prices = price_cols.mean(axis=1)
        deployment_combined_df['mean_price'] = mean_prices

        # add median price feature
        median_prices = price_cols.median(axis=1)
        deployment_combined_df['median_price'] = median_prices
        
        return deployment_combined_df
    
    def clean_data(deployment_combined_df):
        # drop cols not multiples of 10
        keep_cols = np.arange(1,106,10)
        all_cols = np.arange(1,106)
        drop_cols = np.setdiff1d(all_cols, keep_cols)
        cleaned_df = deployment_combined_df.drop(columns=drop_cols, errors='ignore')

        cleaned_df.columns = cleaned_df.columns.astype(str)
        cleaned_df.dropna(axis=0, inplace=True)
        
        return cleaned_df
    
    deployment_combined_df = acquire_data(deployment_test_dates)
    cleaned_df = clean_data(deployment_combined_df)
    X = cleaned_df.drop(columns=['Fwd rets'])

    return X[model.predict(X)==1]

# 4. load our moment predictor model
model = joblib.load('stock_predictor.joblib')

# 5. create a WTForm Class
class PredictForm(FlaskForm):
    print('predictform')
    text = StringField("Stock")
    submit = SubmitField("Predict")

# 6. set up our home page
@app.route("/", methods=["GET", "POST"])
def index():
    # Create instance of the form
    form = PredictForm()

    # Validate the form
    if form.validate_on_submit():
        session['Stock'] = form.text.data
        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

# 7. define a new "prediction" route that processes form input and returns a model prediction
@app.route('/prediction')
def prediction():
    content = {}
    content['text'] = str(session['Stock'])
    results = return_prediction(model, content['text'])
    return render_template('prediction.html', results=results)

# 8. allows us to run flask using $ python app.py
if __name__ == '__main__':
    app.run()
