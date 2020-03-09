import sys

from models import LinearRegression

try:
    passes = int(sys.argv[1])
    split = float(sys.argv[2])
except:
    print("Using default number of passes and split ratio")
    passes = 2
    split = 0.72
model = LinearRegression('data/iris.data')
model.fit(passes, split)
