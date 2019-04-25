import pandas as pd
import csv
import re
import random as rd


INPUT = './data/data.csv'

TRAIN_OUTPUT = './data/test.txt'
TEST_OUTPUT = './data/train.txt'

TEST_TRAIN_RATIO = 0.3


input_values = pd.read_csv(INPUT, dtype = str)

processed_rows = []
for index, row in input_values.iterrows():
    processed_rows.append( '__label__' + str(row[1]) + ' ' + re.sub(r"[^a-zA-Z0-9\s]", "", row[0]) + '\n')

test_size = round(len(processed_rows) * TEST_TRAIN_RATIO)

rd.shuffle(processed_rows)
train_rows = processed_rows[test_size:]
test_rows = processed_rows[:test_size]


with open(TEST_OUTPUT, 'w+') as test_file:
    for row in test_rows:
        test_file.write(row)

with open(TRAIN_OUTPUT, 'w+') as train_file:
    for row in train_rows:
        train_file.write(row)

print("Processed {0} rows: {1} test and {2} train.".format(len(processed_rows), len(test_rows), len(train_rows)))