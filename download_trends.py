from pytrends.request import TrendReq
from pytrends.dailydata import get_daily_data
from datetime import date, timedelta
import os
import time

def convert_dates_to_timeframe(start: date, stop: date) -> str:
    """Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"



dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'data', 'daily-complete')
os.makedirs(dir_path, exist_ok=True)

day_count = 63

start_year = 2020
start_mon = 1
start_day = 20


# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()


def extract_then_store_daily(current_dir_path, start_date):

    global pytrend
    # Interest by Region
    interest_by_region_df = pytrend.interest_by_region()
    # Related Queries, returns a dictionary of dataframes
    related_queries_dict = pytrend.related_queries()
    related_queries_dict = related_queries_dict[k]
    top_related_queries_df = related_queries_dict['top']
    rising_related_queries_df = related_queries_dict['rising']

    related_topic = pytrend.related_topics()
    related_topic = related_topic[k]
    top_related_topic_df = related_topic['top']
    rising_related_topic_df = related_topic['rising']

    interest_over_time_df = pytrend.interest_over_time()
    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-interest.csv")
    if interest_over_time_df is not None:
        interest_over_time_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-region.csv")
    if interest_by_region_df is not None:
        interest_by_region_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-top-related-queries.csv")
    if top_related_queries_df is not None:
        top_related_queries_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-rising-related-queries.csv")
    if rising_related_queries_df is not None:
        rising_related_queries_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-top-related-topic.csv")
    if top_related_topic_df is not None:
        top_related_topic_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, start_date.strftime('%Y-%m-%d') + "-rising-related-topic.csv")
    if rising_related_topic_df is not None:
        rising_related_topic_df.to_csv(filename, index=True)


def extract_then_store_range(current_dir_path, date_range):

    global pytrend

    # time.sleep(15)
    # Interest by Region
    interest_by_region_df = pytrend.interest_by_region()

    time.sleep(1)
    related_queries_dict = pytrend.related_queries()
    related_queries_dict = related_queries_dict[k]
    top_related_queries_df = related_queries_dict['top']
    rising_related_queries_df = related_queries_dict['rising']

    time.sleep(1)
    related_topic = pytrend.related_topics()
    related_topic = related_topic[k]
    top_related_topic_df = related_topic['top']
    rising_related_topic_df = related_topic['rising']

    time.sleep(1)
    interest_over_time_df = pytrend.interest_over_time()
    filename = os.path.join(current_dir_path, date_range + "-interest.csv")
    if interest_over_time_df is not None:
        interest_over_time_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, date_range + "-region.csv")
    if interest_by_region_df is not None:
        interest_by_region_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, date_range + "-top-related-queries.csv")
    if top_related_queries_df is not None:
        top_related_queries_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, date_range + "-rising-related-queries.csv")
    if rising_related_queries_df is not None:
        rising_related_queries_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, date_range + "-top-related-topic.csv")
    if top_related_topic_df is not None:
        top_related_topic_df.to_csv(filename, index=True)

    filename = os.path.join(current_dir_path, date_range + "-rising-related-topic.csv")
    if rising_related_topic_df is not None:
        rising_related_topic_df.to_csv(filename, index=True)
# keywords1 = ["covid19 cases", "cases of covid19", "covid 19", "covid 19 cases", "coronavirus cases", "coronavirus covid19"]
keywords2 = ["corona", "coronavirus update", "coronavirus symptoms", "coronavirus news"]
# keywords = ["covid19", "covid"]
keywords = keywords2

for k in keywords:
    start_date = date(start_year, start_mon, start_day)
    current_dir_path = os.path.join(dir_path, k.replace(" ", '-', 5))
    os.makedirs(current_dir_path, exist_ok=True)
    end_date = start_date + timedelta(day_count)
    timeframe = convert_dates_to_timeframe(start_date, end_date)

    print("**** Downloading data for ", k, ":", timeframe, "******")
    pytrend.build_payload(kw_list=[k], timeframe=timeframe, geo='')

    if k != "corona":
        extract_then_store_range(current_dir_path=current_dir_path, date_range=timeframe.replace(" ", '-', 5))

    for i in range(day_count):
        next_date = start_date + timedelta(1)
        timeframe = convert_dates_to_timeframe(start_date, next_date)
        if i < 23 and k == 'corona':
            start_date = next_date
            continue
        print("--- data day:", timeframe, "; counter i=", i)
        # Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
        pytrend.build_payload(kw_list=[k], timeframe=timeframe, geo='')

        extract_then_store_daily(current_dir_path=current_dir_path, start_date=start_date)

        time.sleep(1)
        start_date = next_date

    # sleep 3 minutes for next keywords
    time.sleep(60)



