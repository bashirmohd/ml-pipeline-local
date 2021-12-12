import os
import time

import joblib

import pandas as pd

from plotter import plot

import fire


def main(label="SiteA"):
    os.system("bokeh serve plotter.py".format(label))

if __name__ == "__main__":
    fire.Fire(main)
