#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from plotly import graph_objs as go
from datetime import datetime
import datetime as dt
import urllib, json

yf.pdr_override()

import streamlit as st
import tweepy
import re
from wordcloud import WordCloud, STOPWORDS
from os import environ
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import nltk

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

import pandas_ta as ta

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

from pandas_datareader._utils import RemoteDataError
from matplotlib.backends.backend_agg import RendererAgg


matplotlib.use("agg")

_lock = RendererAgg.lock

st.set_page_config(page_icon="⍶", layout="wide", initial_sidebar_state="expanded")


option = st.sidebar.selectbox(
    "Dashboard?",
    (
        "Home-Page",
        "Sentiment Analysis",
        "Portfolio Analysis",
        "Technical Analysis",
        "Forecast",
    ),
    0,
)

st.header(option)


if option == "Home-Page":
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
        (0.1, 2, 0.2, 1, 0.1)
    )

    row0_1.title("Alpha App")

    with row0_2:
        st.write("")

    row0_2.subheader(
        "A Web App made with ❤️ by [Utkarsh Singhal](https://www.linkedin.com/in/utkarsh-singhal-584770181)"
    )

    row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((0.1, 3.2, 0.1))

    with row1_1:
        About = st.beta_expander("About ℹ️")
        with About:
            st.markdown(
                """ 
            Hey there! Welcome to Utkarsh's Alpha App.
            This app serves four purposes - 
            * **Sentiment Analysis:** This scrapes Tweets for a **Ticker** (from Twitter) and provides the user with a sentiment score based upon the tweets, including some nice visulizations so that you never miss the next 'Game Stonk'. It can also retrieve Top headlines for a particluar **stock** and can beautifully summarize them and again can provide you with sentiment analysis with some nice visulizations.
            * **Portfolio Analysis:** This Analyzes the stocks in your portfolio(what's better than creating your own portfolio?) on the basis of **Risk and Return** and provides you with the weights to be assigned to a particular asset including some gratifying analysis and the number of shares to be purchased.
            * **Technical indicators:** In order to generate alpha, timing is the only thing that matters the most and for that simple reason, This caters various technical indicators with **buy and sell** signals accompanied by Most, if not all, Candle-Stick Patterns so that you never lag behind!
            * **Forecast:** Won't you like to know whether the **winter is coming** or not?, History is fascinating, although sometimes it repeats itself in a way that can be modelled so this models the Close price of the security based on its past values with forecast components.
            * *One last thing*, if you're viewing it on a mobile device, switch over to landscape mode for viewing ease, go for it!"""
            )

        Disclaimer = st.beta_expander("Disclaimer 👉")
        with Disclaimer:
            st.markdown(
                """The content of this web app is for educational purposes only.
                        It should not be relied upon as financial advice. 
                        It is very important to do your own research before forming an opinion or taking an investment decision or entering into trade. 
                        You should not engage in trading unless you fully understand the nature of transactions 
                        that you're entering into and the extent of your exposure to loss. 
                        All the information that has been presented are from personal research and experience.
                        Although best efforts are made to ensure that all the information is accurate 
                        and up to date, occasional errors are deeply regretted.  """
            )

        st.info(
            "Please scroll through different dashboards using the navigation tab on the upper left corner."
        )


stocksymbols = [
    "Select a Ticker",
    "^NSEI",
    "^NSEBANK",
    "^BSESN",
    "ADANIENT.NS",
    "ADANIPOWER.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "BHARTIARTL.NS",
    "TATAMOTORS.NS",
    "DABUR.NS",
    "ICICIBANK.NS",
    "GAIL.NS",
    "MARUTI.NS",
    "COALINDIA.NS",
    "SHREECEM.NS",
    "IOC.NS",
    "HCLTECH.NS",
    "BRITANNIA.NS",
    "ASIANPAINT.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "EICHERMOT.NS",
    "TCS.NS",
    "BPCL.NS",
    "LT.NS",
    "INDUSINDBK.NS",
    "IRCTC.NS",
    "IGL.NS",
    "INFY.NS",
    "HEROMOTOCO.NS",
    "TITAN.NS",
    "TATAPOWER.NS",
    "TATASTEEL.NS",
    "M&M.NS",
    "GRASIM.NS",
    "ONGC.NS",
    "SBILIFE.NS",
    "SUNPHARMA.NS",
    "SPICEJET.NS",
    "SCI.NS",
    "BAJAJFINSV.NS",
    "RELIANCE.NS",
    "HINDALCO.NS",
    "HDFCLIFE.NS",
    "NESTLEIND.NS",
    "BAJAJ-AUTO.NS",
    "JSWSTEEL.NS",
    "HDFC.NS",
    "SUNPHARMA.NS",
    "NTPC.NS",
    "NACLIND.NS",
    "BAJFINANCE.NS",
    "POWERGRID.NS",
    "HINDUNILVR.NS",
    "HDFCBANK.NS",
    "CIPLA.NS",
    "ITC.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "DRREDDY.NS",
    "KOTAKBANK.NS",
    "DIVISLAB.NS",
    "IONEXCHANG.BO",
    "PITTIENG.NS",
    "DAAWAT.NS",
    "SURYAROSNI.NS",
    "UFLEX.NS",
    "NACLIND.NS",
    "SOBHA.NS",
    "AKCAPIT.BO",
    "AXISBANK.NS",
    "TATASTEEL.NS",
    "SBIN.NS",
    "PNB.NS",
    "INR=X",
]

today = datetime.today().strftime("%Y-%m-%d")


start_date = st.sidebar.text_input("Start Date(YYYY-MM-DD)", "2022-06-01")
end_date = st.sidebar.text_input("End Date(YYYY-MM-DD)", f"{today}")
st.sidebar.header("Contact 📝")
st.sidebar.warning(
    """
           Alpha app is created and maintained by
           **Utkarsh Singhal**. Creating this app involved a lot of data cleaning, pre-processing and error resolving as well. It took around 40 cups of coffee ☕️ to build this app!
           If you like this app then please
         share it and provide your valuable feedback on this, Feel free to connect in case you have any questions regarding this project or require any further information, if you want to reach out, you can connect with me via [**Mail**](mailto:utkarshsinghal06@gmail.com) or you can find me on [**Linkedin**](https://www.linkedin.com/in/utkarsh-singhal-584770181)
             """
)
selected_stocks = stocksymbols

def getMyPortfolio(stocks=selected_stocks, start=start_date, end=end_date):
    return web.get_data_yahoo(stocks, data_source="yahoo", start=start, end=end)


try:
    if option == "Sentiment Analysis":

        st.subheader("Twitter sentiment analysis tool")
        st.markdown("**To begin, please enter a query** 👇")

        # Get user input
        query = st.text_input("Query:", "#")

        # As long as the query is valid (not empty or equal to '#')...
        if query != "" and query != "#":
            noOfTweet = st.text_input("Enter number of tweets you want to Analyze", 100)
            if noOfTweet != "":
                noOfDays = st.text_input(
                    "Enter number of days you want to Scrape Twitter for", 2
                )
                if noOfDays != "":
                    with st.spinner(
                        f"Searching for and analyzing {query}, Please be patient, it might take a while..."
                    ):
                        consumer_key = environ["consumer_key"]
                        consumer_secret = environ["consumer_secret"]
                        accessToken = environ["accessToken"]
                        accessTokenSecret = environ["accessTokenSecret"]
                        authenticate = tweepy.OAuthHandler(
                            consumer_key, consumer_secret
                        )
                        authenticate.set_access_token(accessToken, accessTokenSecret)
                        api = tweepy.API(authenticate, wait_on_rate_limit=True)
                        # Creating list to append tweet data to
                        tweets_list1 = []
                        now = dt.date.today()
                        now = now.strftime("%Y-%m-%d")
                        date_since = dt.date.today() - dt.timedelta(days=int(noOfDays))
                        date_since = date_since.strftime("%Y-%m-%d")
                        tweets = tweepy.Cursor(
                            api.search, q=query, lang="en", since=date_since
                        ).items(int(noOfTweet))

                        for tweet in tweets:
                            tweets_list1.append(
                                [
                                    tweet.created_at,
                                    tweet.user.location,
                                    tweet.favorite_count,
                                    tweet.user.screen_name,
                                    tweet.text,
                                ]
                            )

                        # Create a function to clean the tweets
                        def cleanTxt(text):
                            text = re.sub(
                                "@[A-Za-z0–9]+", "", text
                            )  # Removing @mentions
                            text = re.sub("#", "", text)  # Removing '#' hash tag
                            text = re.sub("RT[\s]+", "", text)  # Removing RT
                            text = re.sub(
                                "https?:\/\/\S+", "", text
                            )  # Removing hyperlink
                            return text

                        # Creating a dataframe from the tweets list above
                        tw_df = pd.DataFrame(
                            tweets_list1,
                            columns=["Date", "Location", "Likes", "Username", "Text"],
                        )

                        tw_df["Text"].drop_duplicates(inplace=True)
                        df = pd.DataFrame()
                        df["Text"] = tw_df["Text"]
                        df["Text"].apply(cleanTxt)
                        st.write(tw_df.sort_values(by="Likes", ascending=False))

                        # Sentiment Analysis

                        analyzer = SentimentIntensityAnalyzer()

                        # Creating sentiment scores columns
                        df["compound"] = [
                            analyzer.polarity_scores(x)["compound"] for x in df["Text"]
                        ]
                        df["neg"] = [
                            analyzer.polarity_scores(x)["neg"] for x in df["Text"]
                        ]
                        df["neu"] = [
                            analyzer.polarity_scores(x)["neu"] for x in df["Text"]
                        ]
                        df["pos"] = [
                            analyzer.polarity_scores(x)["pos"] for x in df["Text"]
                        ]

                        # Taking averages of sentiment score columns
                        avg_compound = np.average(df["compound"])
                        avg_neg = np.average(df["neg"])
                        avg_neu = np.average(df["neu"])
                        avg_pos = np.average(df["pos"])

                        # Counting number of tweets
                        count = len(df.index)

                        # Print Statements
                        st.write(
                            "Since " + noOfDays + " days, there have been",
                            count,
                            "tweets on " + query,
                            end="\n*",
                        )
                        st.write("Positive Sentiment:", "%.2f" % avg_pos, end="\n*")
                        st.write("Neutral Sentiment:", "%.2f" % avg_neu, end="\n*")
                        st.write("Negative Sentiment:", "%.2f" % avg_neg, end="\n*")
                        st.write(
                            "**Compound Sentiment:**", "%.2f" % avg_compound, end="\n"
                        )

                        # Creating PieCart
                        labels = [
                            "Positive [" + str(avg_pos) + "%]",
                            "Neutral [" + str(avg_neu) + "%]",
                            "Negative [" + str(avg_neg) + "%]",
                        ]
                        sizes = [avg_pos, avg_neu, avg_neg]
                        colors = ["yellowgreen", "blue", "red"]
                        fig, ax = plt.subplots()
                        patches, texts = ax.pie(sizes, colors=colors, startangle=90)
                        plt.style.use("default")
                        ax.legend(labels)
                        ax.set_title(
                            "Sentiment Analysis Result for keyword= " + query + ""
                        )
                        ax.axis("equal")
                        st.pyplot(fig)

                        # word cloud visualization
                        def word_cloud(text):
                            stopwords = set(STOPWORDS)
                            allWords = " ".join(list(text))
                            wordCloud = WordCloud(
                                background_color="black",
                                width=1600,
                                height=800,
                                stopwords=stopwords,
                                min_font_size=20,
                                max_font_size=150,
                                colormap="prism",
                            ).generate(allWords)
                            fig, ax = plt.subplots(figsize=(20, 10), facecolor="k")
                            plt.imshow(wordCloud)
                            ax.axis("off")
                            fig.tight_layout(pad=0)
                            st.pyplot(fig)

                        st.write("Wordcloud for " + query)
                        word_cloud(df["Text"].values)

        st.subheader("NEWS Analysis tool")

        now = dt.date.today()
        now = now.strftime("%m-%d-%Y")
        yesterday = dt.date.today() - dt.timedelta(days=1)
        yesterday = yesterday.strftime("%m-%d-%Y")

        nltk.download("punkt")
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        config = Config()
        config.browser_user_agent = user_agent

        # stored queries in a list
        query_list = ["News", "Share price forecast", "Fundamental Analysis"]
        # save the company name in a variable
        company_name = st.text_input(
            "Please provide the name of the Company or a Ticker:"
        )
        if company_name != "":
            with st.spinner(
                f"Searching for and analyzing {company_name}, Please be patient, it might take a while..."
            ):
                googlenews = GoogleNews(start=yesterday, end=now)
                googlenews.search(company_name)
                result = googlenews.result()
                df = pd.DataFrame(result)
                try:
                    list = []
                    for ind in df.index:
                        dict = {}
                        article = Article(df["link"][ind], config=config)
                        article.download()
                        article.parse()
                        article.nlp()
                        dict["Date"] = df["date"][ind]
                        dict["Media"] = df["media"][ind]
                        dict["Title"] = article.title
                        dict["Article"] = article.text
                        dict["Summary"] = article.summary
                        dict["Key_words"] = article.keywords
                        list.append(dict)
                    news_df = pd.DataFrame(list)
                    st.write(news_df)

                    analyzer = SentimentIntensityAnalyzer()

                    news_df["compound"] = [
                        analyzer.polarity_scores(x)["compound"]
                        for x in news_df["Summary"]
                    ]
                    news_df["neg"] = [
                        analyzer.polarity_scores(x)["neg"] for x in news_df["Summary"]
                    ]
                    news_df["neu"] = [
                        analyzer.polarity_scores(x)["neu"] for x in news_df["Summary"]
                    ]
                    news_df["pos"] = [
                        analyzer.polarity_scores(x)["pos"] for x in news_df["Summary"]
                    ]

                    # Taking averages of sentiment score columns
                    N_avg_compound = np.average(news_df["compound"])
                    N_avg_neg = np.average(news_df["neg"])
                    N_avg_neu = np.average(news_df["neu"])
                    N_avg_pos = np.average(news_df["pos"])

                    st.write("Positive Sentiment:", "%.2f" % N_avg_pos, end="\n*")
                    st.write("Neutral Sentiment:", "%.2f" % N_avg_neu, end="\n*")
                    st.write("Negative Sentiment:", "%.2f" % N_avg_neg, end="\n*")
                    st.write(
                        "**Compound Sentiment:**:", "%.2f" % N_avg_compound, end="\n"
                    )

                    # Creating PieCart
                    labels = [
                        "Positive [" + str(N_avg_pos) + "%]",
                        "Neutral [" + str(N_avg_neu) + "%]",
                        "Negative [" + str(N_avg_neg) + "%]",
                    ]
                    sizes = [N_avg_pos, N_avg_neu, N_avg_neg]
                    colors = ["yellowgreen", "blue", "red"]
                    fig, ax = plt.subplots()
                    patches, texts = ax.pie(sizes, colors=colors, startangle=90)
                    plt.style.use("default")
                    ax.legend(labels)
                    ax.set_title(
                        "NEWS Sentiment Analysis Result for = " + company_name + ""
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                    # word cloud visualization
                    def news_cloud(text):
                        stopwords = set(STOPWORDS)
                        N_allWords = ' '.join([nws for nws in text])
                        newsCloud = WordCloud(
                            background_color="black",
                            width=1600,
                            height=800,
                            stopwords=stopwords,
                            min_font_size=20,
                            max_font_size=150,
                            colormap="hsv",
                        ).generate(N_allWords)

                        fig, ax = plt.subplots(figsize=(20, 10), facecolor="k")
                        plt.imshow(newsCloud)
                        ax.axis("off")
                        fig.tight_layout(pad=0)
                        st.pyplot(fig)

                    st.write("Wordcloud for " + company_name)
                    news_cloud(news_df["Summary"].values)

                except Exception as e:
                    st.write("Error:" + str(e))
                    st.write(
                        "Seems like there is some problem in fetching  the data, Please try after some time or try with a different ticker."
                    )

except Exception as e:
    st.write("Error:" + str(e))
    st.write(
        "Looks like, There is some error in retrieving the data, Please try again!"
    )

try:
    if option == "Portfolio Analysis":
        st.subheader("Portfolio Analysis")
        portfolio = st.multiselect(
            "Please select Stocks to create a portfolio",
            stocksymbols,
            default=["TATAMOTORS.NS"],
        )
        try:
            df = getMyPortfolio(portfolio)
            df_main = df["Adj Close"]
            numAssets = len(portfolio)
            portfolio_amount = st.text_input(
                "Enter the amount you want to invest", int(50000)
            )
            with st.spinner(
                f"You have {numAssets} assets in your porfolio, Do you want to add more? If not then click on Analyze."
            ):
                pass
        except RemoteDataError:
            pass

        if st.button("Analyze"):
            with st.spinner(
                f"Searching for and analyzing your portfolio, Please be patient, it might take a while..."
            ):
                st.write(df_main)
                fig, ax = plt.subplots(figsize=(15, 8))

                for i in df_main.columns.values:
                    ax.plot(df_main[i], label=i)
                ax.set_title("Portfolio Close Price History")
                ax.set_xlabel("Date", fontsize=18)
                ax.set_ylabel("Close Price INR (₨)", fontsize=18)
                ax.legend(df_main.columns.values, loc="upper left")
                st.pyplot(fig)

                corr = df_main.corr(method="pearson")
                corr

                fig1 = plt.figure()
                sb.heatmap(
                    corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    cmap="YlGnBu",
                    annot=True,
                    linewidth=0.5,
                )
                st.subheader("Correlation between Stocks in your portfolio")
                st.pyplot(fig1)

                daily_simple_return = df_main.pct_change(1)
                st.write(daily_simple_return)
                st.write(daily_simple_return.corr())

                fig2 = plt.figure()
                sb.heatmap(
                    daily_simple_return.corr(),
                    xticklabels=daily_simple_return.corr().columns,
                    yticklabels=daily_simple_return.corr().columns,
                    cmap="cubehelix",
                    annot=True,
                    linewidth=0.5,
                )
                st.subheader(
                    "Correlation between daily simple returns of stocks in your portfolio"
                )
                st.pyplot(fig2)

                # covariance matrix for simple return
                st.write(daily_simple_return.cov())

                st.markdown(
                    "**Covariance** is a measurement of the spread between numbers in a dataset, i.e. it measures how far each number in the data set is from the mean. The higher the variance of an asset price means the higher the **risk** asset bears along with a higher **return** and a higher volatility. The most commonly-used risk model is the covariance matrix, which describes asset volatilities and their co-dependence. This is important because one of the principles of diversification is that risk can be reduced by making many uncorrelated bets (Correlation is just normalised covariance.)"
                )

                fig3 = plt.figure()
                sb.heatmap(
                    daily_simple_return.cov(),
                    xticklabels=daily_simple_return.cov().columns,
                    yticklabels=daily_simple_return.cov().columns,
                    cmap="RdBu_r",
                    annot=True,
                    linewidth=0.5,
                )
                st.subheader(
                    "Covariance between daily simple returns of stocks in your portfolio"
                )
                st.pyplot(fig3)

                st.subheader(
                    "Volatality(%) of individual stocks in your portfolio on the basis of daily simple returns."
                )
                st.write(daily_simple_return.std() * 100)

                # visualize the stock daily simple return
                st.write("Volatility- Daily simple returns")
                fig, ax = plt.subplots(figsize=(15, 8))

                for i in daily_simple_return.columns.values:
                    ax.plot(daily_simple_return[i], lw=2, label=i)

                ax.legend(loc="upper right", fontsize=10)
                ax.set_title("Volatility")
                ax.set_xlabel("Date")
                ax.set_ylabel("Dailty simple returns")
                st.pyplot(fig)

                st.write("Average Daily returns(%) of stocks in your portfolio")
                Avg_daily = daily_simple_return.mean()
                st.write(Avg_daily * 100)

                daily_cummulative_simple_return = (daily_simple_return + 1).cumprod()
                daily_cummulative_simple_return

                # visualize the daily cummulative simple return
                st.write("Cummulative Returns")
                fig, ax = plt.subplots(figsize=(18, 8))

                for i in daily_cummulative_simple_return.columns.values:
                    ax.plot(daily_cummulative_simple_return[i], lw=2, label=i)

                ax.legend(loc="upper left", fontsize=10)
                ax.set_title("Daily Cummulative Simple returns/growth of investment")
                ax.set_xlabel("Date")
                ax.set_ylabel("Growth of ₨ 1 investment")
                st.pyplot(fig)

                # Optimization
                # optimize for maximal sharpe ratio
                # calculating expected annual return and annualized sample covariance matrix of daily assets returns

                mean = expected_returns.mean_historical_return(df_main)

                S = risk_models.sample_cov(df_main)  # for sample covariance matrix

                # sharpe ratio describes that how much excess return you receive for the extra volatily you endure for holding a risky asset
                ef = EfficientFrontier(mean, S)
                weights = ef.max_sharpe()  # for maximizing the sharpe ratio
                cleaned_weights = (
                    ef.clean_weights()
                )  # to clean the raw weights, %setting any weights whose absolute value are below the *cutoff* to 0%, and rounding the rest
                # Get the Keys and store them in a list
                labels = list(cleaned_weights.keys())

                # Get the Values and store them in a list
                values = list(cleaned_weights.values())
                fig, ax = plt.subplots()
                ax.pie(values, labels=labels, autopct="%1.0f%%")
                st.subheader("Portfolio Allocation")
                st.pyplot(fig)

                aList = list(ef.portfolio_performance())
                st.write("Expected annual return:", aList[0] * 100, str("%"))
                st.write("Annual volatility:", aList[1] * 100, str("%"))
                st.write("Sharpe Ratio:", aList[2])

                if portfolio_amount != "":
                    # Get discrete allocation of each share per stock

                    latest_prices = get_latest_prices(df_main)
                    weights = cleaned_weights
                    discrete_allocation = DiscreteAllocation(
                        weights,
                        latest_prices,
                        total_portfolio_value=int(portfolio_amount),
                    )
                    allocation, leftover = discrete_allocation.lp_portfolio()

                    # function to get the co.s name

                    def get_company_name(symbol):
                        response = urllib.request.urlopen(
                            f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
                        )
                        content = response.read()
                        return json.loads(content.decode("utf8"))["quotes"][0][
                            "shortname"
                        ]

                    company_name = []
                    discrete_allocation_list = []

                    for symbol in allocation:
                        company_name.append(get_company_name(symbol))
                        discrete_allocation_list.append(allocation.get(symbol))

                    portfolio_df = pd.DataFrame(
                        columns=["Company name", "Ticker", "Number of stocks to buy"]
                    )

                    portfolio_df["Company name"] = company_name
                    portfolio_df["Ticker"] = allocation
                    portfolio_df["Number of stocks to buy"] = discrete_allocation_list
                    st.subheader(
                        "Number of stocks to buy with the amount of ₨ "
                        + str(portfolio_amount)
                    )
                    st.write(portfolio_df)
                    st.write("Funds remaining with you will be: ₨", leftover)
except Exception as e:
    st.write("Error:" + str(e))
    st.write("Looks like, There is some error in analyzing the data, Please try again!")

try:
    if option == "Technical Analysis":
        selected_stocks = st.selectbox("Select dataset", stocksymbols)
        if selected_stocks == "Select a Ticker":
            pass
        else:
            data_load_state = st.text("Loading data...😃")
            data = getMyPortfolio(selected_stocks)
            data_load_state.text("Done!💯")
            # Adjusted Close Price
            st.header("Close Price")
            st.line_chart(data["Adj Close"])

            # SMA and EMA
            # MA with 30 day window
            st.markdown("Indicators with buy and sell signal")

            data["SMA 30"] = ta.sma(data["Adj Close"], length=30)

            data["SMA 100"] = ta.sma(data["Adj Close"], length=100)

            # SMA BUY SELL
            # Function for buy and sell signal
            def buy_sell(data):
                signalBuy = []
                signalSell = []
                flag = -1

                for i in range(len(data)):
                    if data["SMA 30"][i] > data["SMA 100"][i]:
                        if flag != 1:
                            signalBuy.append(data["Adj Close"][i])
                            flag = 1
                        else:
                            signalBuy.append(np.nan)
                        signalSell.append(np.nan)
                    elif data["SMA 30"][i] < data["SMA 100"][i]:
                        if flag != 0:
                            signalSell.append(data["Adj Close"][i])
                            flag = 0  # To indicate that I actually went there
                        else:
                            signalSell.append(np.nan)
                        signalBuy.append(np.nan)
                    else:
                        signalBuy.append(np.nan)
                        signalSell.append(np.nan)
                return (signalBuy, signalSell)

            # storing the function
            buy_sell = buy_sell(data)
            # for invoking the indicators into dataframe
            data["Buy_Signal_price"] = buy_sell[0]
            data["Sell_Signal_price"] = buy_sell[1]

            st.header("Buy and Sell startegy as per Simple moving Averages")
            st.subheader(
                "Disclamer- This is just for educational purposes not an investment advice"
            )
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(
                data["Adj Close"],
                label=selected_stocks,
                linewidth=0.5,
                color="blue",
                alpha=0.9,
            )
            ax.plot(data["SMA 30"], label="SMA30", alpha=0.85)
            ax.plot(data["SMA 100"], label="SMA100", alpha=0.85)
            ax.scatter(
                data.index,
                data["Buy_Signal_price"],
                label="Buy",
                marker="^",
                color="green",
                alpha=1,
            )
            ax.scatter(
                data.index,
                data["Sell_Signal_price"],
                label="Sell",
                marker="v",
                color="red",
                alpha=1,
            )
            ax.set_title(
                selected_stocks + " Price History with buy and sell signals",
                fontsize=10,
                backgroundcolor="blue",
                color="white",
            )
            ax.set_xlabel(start_date + " - " + end_date, fontsize=18)
            ax.set_ylabel("Close Price INR (₨)", fontsize=18)
            legend = ax.legend()
            ax.grid()
            plt.tight_layout()
            st.pyplot(fig)

            st.write(data["Buy_Signal_price"].tail())
            st.write(data["Sell_Signal_price"].tail())

            # heiken-ashi
            def heikin_ashi(df):
                heikin_ashi_df = pd.DataFrame(
                    index=df.index.values,
                    columns=["HA_open", "HA_high", "HA_low", "HA_close"],
                )

                heikin_ashi_df["HA_close"] = (
                    df["Open"] + df["High"] + df["Low"] + df["Adj Close"]
                ) / 4

                for i in range(len(df)):
                    if i == 0:
                        heikin_ashi_df.iat[0, 0] = (
                            df["Open"].iloc[0] + df["Adj Close"].iloc[0]
                        ) / 2
                    else:
                        heikin_ashi_df.iat[i, 0] = (
                            heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[i - 1, 3]
                        ) / 2

                heikin_ashi_df["HA_high"] = (
                    heikin_ashi_df.loc[:, ["HA_open", "HA_close"]]
                    .join(df["High"])
                    .max(axis=1)
                )

                heikin_ashi_df["HA_low"] = (
                    heikin_ashi_df.loc[:, ["HA_open", "HA_close"]]
                    .join(df["Low"])
                    .min(axis=1)
                )

                return heikin_ashi_df

            HA = heikin_ashi(data)

            # Heiken-Ashi Strategy
            def Heiken_Strategy(data, data1):
                buy_price = []
                sell_price = []
                flag = False

                for i in range(len(data)):
                    if (
                        data["HA_close"][i] > data["HA_open"][i]
                        and data["HA_close"][i - 1] < data["HA_open"][i - 1]
                        and data["HA_close"][i - 2] < data["HA_open"][i - 2]
                    ):
                        if flag == False:
                            buy_price.append(data1["Adj Close"][i])
                            flag = True
                        else:
                            buy_price.append(np.nan)
                        sell_price.append(np.nan)

                    elif (
                        data["HA_close"][i] < data["HA_open"][i]
                        and data["HA_close"][i - 1] < data["HA_open"][i - 1]
                        and data["HA_close"][i - 2] > data["HA_open"][i - 2]
                    ):
                        if flag == True:
                            sell_price.append(data1["Adj Close"][i])
                            flag = False
                        else:
                            sell_price.append(np.nan)
                        buy_price.append(np.nan)
                    else:
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)

                data["buy here"] = buy_price
                data["sell here"] = sell_price

            Heiken_Strategy(HA, data)

            st.header("Buy and Sell startegy as per Heiken-Ashi")
            st.subheader(
                "Disclamer- This is just for educational purposes not an investment advice!"
            )
            HA["DEMA_short"] = ta.dema(HA["HA_close"], length=25)
            data["DEMA_long"] = ta.dema(data["Adj Close"], length=52)
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(
                data["Adj Close"],
                label=selected_stocks,
                linewidth=0.5,
                color="blue",
                alpha=1,
            )
            ax.scatter(
                HA.index,
                HA["buy here"],
                label="Buy",
                marker="^",
                color="green",
                alpha=1,
            )
            ax.scatter(
                HA.index,
                HA["sell here"],
                label="Sell",
                marker="v",
                color="red",
                alpha=1,
            )
            ax.set_title(
                selected_stocks
                + " price History with buy and sell signals as per Heiken Ashi strategy",
                fontsize=10,
                backgroundcolor="blue",
                color="white",
            )
            ax.set_xlabel(start_date + " - " + end_date, fontsize=18)
            ax.set_ylabel("Close Price INR (₨)", fontsize=18)
            legend = ax.legend()
            ax.grid()
            plt.tight_layout()
            st.pyplot(fig)

            st.write(HA["buy here"].tail())
            st.write(HA["sell here"].tail())

            # DEMA Startegy
            def DEMA_strategy(data):
                DEMABuy = []
                DEMASell = []
                flag = False

                for i in range(len(data)):
                    if data["DEMA_long"][i] < data["Adj Close"][i]:
                        if flag == False:
                            DEMABuy.append(data["Adj Close"][i])
                            flag = True
                        else:
                            DEMABuy.append(np.nan)
                        DEMASell.append(np.nan)
                    elif data["DEMA_long"][i] > data["Adj Close"][i]:
                        if flag == True:
                            DEMASell.append(data["Adj Close"][i])
                            flag = False
                        else:
                            DEMASell.append(np.nan)
                        DEMABuy.append(np.nan)
                    else:
                        DEMABuy.append(np.nan)
                        DEMASell.append(np.nan)

                data["DEMA_Buy_Signal_price"] = DEMABuy
                data["DEMA_Sell_Signal_price"] = DEMASell

            # storing the function
            DEMA_strategy = DEMA_strategy(data)

            st.header("Buy and Sell startegy as per Double Exponential Moving Average")
            st.subheader(
                "Disclamer- This is just for educational purposes not an investment advice"
            )

            # DEMA plot
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(
                data["Adj Close"],
                label=selected_stocks,
                linewidth=0.5,
                color="blue",
                alpha=1,
            )
            ax.plot(HA["DEMA_short"], label="DEMA", alpha=0.75)
            data["DEMA_long"].plot(figsize=(14, 8), alpha=0.75)
            ax.scatter(
                data.index,
                data["DEMA_Buy_Signal_price"],
                label="Buy",
                marker="^",
                color="green",
                alpha=1,
            )
            ax.scatter(
                data.index,
                data["DEMA_Sell_Signal_price"],
                label="Sell",
                marker="v",
                color="red",
                alpha=1,
            )
            ax.set_title(
                selected_stocks
                + " Price History with buy and sell signals as per Double Exponential Moving Average",
                fontsize=10,
                backgroundcolor="blue",
                color="white",
            )
            ax.set_xlabel(start_date + " - " + end_date, fontsize=18)
            ax.set_ylabel("Close Price INR (₨)", fontsize=18)
            legend = ax.legend()
            ax.grid()
            plt.tight_layout()
            st.pyplot(fig)

            st.write(data["DEMA_Buy_Signal_price"].tail())
            st.write(data["DEMA_Sell_Signal_price"].tail())

            # BUy and sell as per MACD
            def MACD(per1, per2, per3):
                short = data["Adj Close"].ewm(span=per1, adjust=False).mean()
                long = data["Adj Close"].ewm(span=per2, adjust=False).mean()
                MACD = short - long
                signal = MACD.ewm(span=per3, adjust=False).mean()
                return [MACD, signal]

            def MACD_bar(data):
                return [data["MACD"][i] - data["signal"][i] for i in range(len(data))]

            def MACD_color(data):
                MACD_color = []
                for i in range(len(data)):
                    if data["MACD_bar"][i] > data["MACD_bar"][i - 1]:
                        MACD_color.append(1)
                    else:
                        MACD_color.append(-1)
                return MACD_color

            data["MACD"] = MACD(12, 26, 9)[0]
            data["signal"] = MACD(12, 26, 9)[1]
            data["MACD_bar"] = MACD_bar(data)
            data["MACD_color"] = MACD_color(data)
            data["positive"] = data["MACD_color"] > 0

            def MACD_Strategy(df, risk):
                MACD_Buy = []
                MACD_Sell = []
                flag = False

                for i in range(len(df)):
                    if df["MACD"][i] > df["signal"][i]:
                        MACD_Sell.append(np.nan)
                        if flag == False:
                            MACD_Buy.append(df["Adj Close"][i])
                            flag = True
                        else:
                            MACD_Buy.append(np.nan)
                    elif df["MACD"][i] < df["signal"][i]:
                        MACD_Buy.append(np.nan)
                        if flag == True:
                            MACD_Sell.append(df["Adj Close"][i])
                            flag = False
                        else:
                            MACD_Sell.append(np.nan)
                    elif flag == True and df["Adj Close"][i] < MACD_Buy[-1] * (
                        1 - risk
                    ):
                        MACD_Sell.append(df["Adj Close"][i])
                        MACD_Buy.append(np.nan)
                        flag = False
                    elif flag == True and df["Adj Close"][i] < df["Adj Close"][
                        i - 1
                    ] * (1 - risk):
                        MACD_Sell.append(df["Adj Close"][i])
                        MACD_Buy.append(np.nan)
                        flag = False
                    else:
                        MACD_Buy.append(np.nan)
                        MACD_Sell.append(np.nan)

                data["MACD_Buy_Signal_price"] = MACD_Buy
                data["MACD_Sell_Signal_price"] = MACD_Sell

            # storing the function
            st.header(
                "Buy and Sell startegy as per Moving Average Convergence and Divergence"
            )
            st.subheader(
                "Disclamer- This is just for educational purposes not an investment advice"
            )

            MACD_strategy = MACD_Strategy(data, 0.025)  # df en riskpercentage
            data["Date"] = pd.to_datetime(data.index)

            plt.rcParams.update({"font.size": 10})
            fig, ax1 = plt.subplots(figsize=(14, 8))
            fig.suptitle(
                selected_stocks, fontsize=10, backgroundcolor="blue", color="white"
            )
            ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
            ax2 = plt.subplot2grid((14, 8), (8, 0), rowspan=6, colspan=14)
            ax1.set_ylabel("Price in ₨")
            ax1.plot(
                "Adj Close", data=data, label="Close Price", linewidth=0.5, color="blue"
            )
            ax1.scatter(
                data.index,
                data["MACD_Buy_Signal_price"],
                color="green",
                marker="^",
                alpha=1,
            )
            ax1.scatter(
                data.index,
                data["MACD_Sell_Signal_price"],
                color="red",
                marker="v",
                alpha=1,
            )
            ax1.legend()
            ax1.grid()
            ax1.set_xlabel("Date", fontsize=8)

            ax2.set_ylabel("MACD", fontsize=8)
            ax2.plot("MACD", data=data, label="MACD", linewidth=0.5, color="blue")
            ax2.plot("signal", data=data, label="signal", linewidth=0.5, color="red")
            ax2.bar(
                "Date",
                "MACD_bar",
                data=data,
                label="Volume",
                color=data.positive.map({True: "g", False: "r"}),
                width=1,
                alpha=0.8,
            )
            ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax2.grid()
            st.pyplot(fig)

            st.write(data["MACD_Buy_Signal_price"].tail())
            st.write(data["MACD_Sell_Signal_price"].tail())

            def bb_strategy(data):
                bbBuy = []
                bbSell = []
                flag = False
                bb = ta.bbands(data["Adj Close"], 20, 2)
                data = pd.concat([data, bb], axis=1).reindex(data.index)

                # Flag is gonna tell us when the 2 SMA are crossing each other, -1 = False

                for i in range(len(data)):
                    if data["Adj Close"][i] < data["BBL_20_2.0"][i]:
                        if flag == False:
                            bbBuy.append(data["Adj Close"][i])
                            flag = True
                        else:
                            bbBuy.append(np.nan)
                        bbSell.append(np.nan)
                    elif data["Adj Close"][i] > data["BBU_20_2.0"][i]:
                        if flag == True:
                            bbSell.append(data["Adj Close"][i])
                            flag = False  # To indicate that I actually went there
                        else:
                            bbSell.append(np.nan)
                        bbBuy.append(np.nan)
                    else:
                        bbBuy.append(np.nan)
                        bbSell.append(np.nan)

                data["bb_Buy_Signal_price"] = bbBuy
                data["bb_Sell_Signal_price"] = bbSell

                return data

            data = bb_strategy(data)

            st.header("Buy and Sell startegy as per Bollinger Bands startegy")
            st.subheader(
                "Disclamer- This is just for educational purposes not an investment advice"
            )
            st.subheader(
                "Bolinger Bands are not for Long term trends, they are suitable for short term purposes mostly!"
            )

            # plot
            fig, ax1 = plt.subplots(figsize=(14, 8))
            fig.suptitle(
                selected_stocks, fontsize=10, backgroundcolor="blue", color="white"
            )
            ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
            ax2 = plt.subplot2grid((14, 8), (8, 0), rowspan=6, colspan=14)
            ax1.set_ylabel("Price in ₨")
            ax1.plot(
                data["Adj Close"], label="Close Price", linewidth=0.5, color="blue"
            )
            ax1.scatter(
                data.index,
                data["bb_Buy_Signal_price"],
                color="green",
                marker="^",
                alpha=1,
            )
            ax1.scatter(
                data.index,
                data["bb_Sell_Signal_price"],
                color="red",
                marker="v",
                alpha=1,
            )
            ax1.legend()
            ax1.grid()
            ax1.set_xlabel("Date", fontsize=8)

            ax2.plot(data["BBM_20_2.0"], label="Middle", color="blue", alpha=0.35)
            ax2.plot(data["BBU_20_2.0"], label="Upper", color="green", alpha=0.35)
            ax2.plot(data["BBL_20_2.0"], label="Lower", color="red", alpha=0.35)
            ax2.fill_between(
                data.index, data["BBL_20_2.0"], data["BBU_20_2.0"], alpha=0.1
            )
            ax2.legend(loc="upper left")
            ax2.grid()
            st.pyplot(fig)

            # Simple Moving Average
            data["SMA20"] = ta.sma(data["Adj Close"], length=20)

            # Exponential Moving Average
            data["EMA50"] = ta.ema(data["Adj Close"], length=50)

            # Plot
            st.header("Simple Moving Average vs. Exponential Moving Average")
            st.line_chart(data[["Adj Close", "SMA20", "EMA50"]])

            # Bollinger Bands
            # Plot
            st.header("Bollinger Bands")
            st.line_chart(data[["Adj Close", "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"]])

            # ## MACD (Moving Average Convergence Divergence)
            # MACD
            macd = ta.macd(data["Adj Close"], fast=12, slow=26, signal=9)
            data = pd.concat([data, macd], axis=1).reindex(data.index)

            # Plot
            st.header("Moving Average Convergence Divergence")
            st.line_chart(data[["MACD_12_26_9", "MACDs_12_26_9"]])

            ## CCI (Commodity Channel Index)
            cci = ta.cci(
                high=data["High"], low=data["Low"], close=data["Adj Close"], length=14
            )

            # Plot
            st.header("Commodity Channel Index")
            st.line_chart(cci)

            # ## RSI (Relative Strength Index)
            # RSI
            data["RSI"] = ta.rsi(data["Adj Close"], length=14)

            # Plot
            st.header("Relative Strength Index")
            st.line_chart(data["RSI"])

            # ## OBV (On Balance Volume)
            # OBV
            data["OBV"] = ta.obv(data["Adj Close"], data["Volume"]) / 10**6

            # Plot
            st.header("On Balance Volume")
            st.line_chart(data["OBV"])

            # aroon
            aroon = ta.aroon(high=data["High"], low=data["Low"], length=14)
            data = pd.concat([data, aroon], axis=1).reindex(data.index)

            st.header("AROON")
            st.line_chart(data[["AROOND_14", "AROONU_14"]])

            # Ultimate oscilator
            data["Ultimate"] = ta.uo(
                high=data["High"], low=data["Low"], close=data["Adj Close"]
            )

            st.header("Ultimate Oscilator")
            st.line_chart(data["Ultimate"])

            # william r
            data["Will"] = ta.willr(
                high=data["High"], low=data["Low"], close=data["Adj Close"], length=30
            )

            st.header("William %R")
            st.line_chart(data["Will"])

            # Trix
            trix = ta.trix(close=data["Adj Close"], length=30)
            data = pd.concat([data, trix], axis=1).reindex(data.index)

            st.header("TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA")
            st.line_chart(data["TRIX_30_9"])

            # PPO
            ppo = ta.ppo(close=data["Adj Close"], fast=12, slow=26)
            data = pd.concat([data, ppo], axis=1).reindex(data.index)

            st.header("Percentage Price Oscillator")
            st.line_chart(data["PPO_12_26_9"])

            # kama

            data["kama"] = ta.kama(close=data["Adj Close"], length=30)

            st.header("Kaufman Adaptive Moving Average")
            st.line_chart(data["kama"])

            # Candle stick
            candlestick = go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Adj Close"],
            )

            fig = go.Figure(data=[candlestick])
            fig.update_layout(xaxis_rangeslider_visible=True)

            st.header("Candle stick")
            st.plotly_chart(fig)
except Exception as e:
    st.write("Error:" + str(e))
    st.write("Looks like, There is some error, Please try again!")

try:
    if option == "Forecast":
        selected_stocks = st.selectbox("Select dataset", stocksymbols)
        if selected_stocks == "Select a Ticker":
            pass
        else:
            data_load_state = st.text("Loading data...😃")
            data = getMyPortfolio(selected_stocks)
            data_load_state.text("Done!💯")
            data.reset_index(inplace=True)
            n_years = st.slider("Years of Prediction: ", 1, 4)
            period = n_years * 365

            st.subheader("Raw data")
            st.write(data.tail())

            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=data["Date"], y=data["Adj Close"], name="Stock_close")
                )
                fig.layout.update(
                    title_text="Time Series Data", xaxis_rangeslider_visible=True
                )
                st.plotly_chart(fig)

            plot_raw_data()

            # For Forecasting we need to manupilate it for phrophet to use

            df_train = data[["Date", "Adj Close"]]
            df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})

            m = Prophet()
            m.fit(df_train)

            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            st.subheader("Forecasted data")
            st.write(forecast.head())

            st.write("forecast data")
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write("forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)
except Exception as e:
    st.write("Error:" + str(e))
    st.write("Looks like, There is some error, Please try again!")
