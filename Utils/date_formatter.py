import datetime


# when obtaining data from numerous sources, we want to standardize communication units.
# in other words, we want object types to be the same. for instance, things like dataframe index 'type' or 'class'
# should be the same

def formate_date(dates):
    yymmdd = list(map(lambda x: int(x), str(dates).split(" ")[0].split("-")))
    # what this does is take a list of dates in [yy--mm--dd {other stuff} format
    # strips away the other stuff , then returns the datetime object
    return datetime.date(yymmdd[0], yymmdd[1], yymmdd[2])
