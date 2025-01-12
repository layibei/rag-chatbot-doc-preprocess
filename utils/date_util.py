from datetime import datetime

import pytz


def get_timestamp_in_utc():
    current_timestamp = datetime.now().timestamp()
    # Convert the timestamp to a timezone-aware datetime object in UTC
    utc_datetime = datetime.fromtimestamp(current_timestamp, tz=pytz.utc)

    return utc_datetime


if __name__ == "__main__":
    print(get_timestamp_in_utc())