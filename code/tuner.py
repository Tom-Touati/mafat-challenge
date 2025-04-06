import sys
sys.path.append("/home/tom.touati/mafat-challenge/code/submission")
import pandas as pd
from model import model
m = model()
m.load("/home/tom.touati/mafat-challenge/code/submission")
df = pd.read_csv("/home/tom.touati/mafat-challenge/code/test_sample.csv")
df = df.drop(columns=["Device_ID"])
print(m.predict(df))