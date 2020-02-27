import datetime


def get_days_from_beginning():
    current = datetime.date.today()
    return (current - datetime.date(1995, 4, 21)).days + 1

