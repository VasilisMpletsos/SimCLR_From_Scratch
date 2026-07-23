from datetime import datetime
from time import time


# %% Test numpy
def count_eval(func):

    def wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        end = datetime.now()
        print(f"Difference was {end - start}")

    return wrapper


# %% Test numpy
@count_eval
def calculate():
    for i in range(1_000):
        for j in range(1_000):
            sum = i - j


calculate()
