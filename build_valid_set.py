#!/usr/bin/python2
# This just calls get_random_data to build a validation set of data
import sys

N = int(sys.argv[1]);
print N

from readin_data import get_random_data
get_random_data(n=N).to_csv("data/valid_set.csv",index=False)
