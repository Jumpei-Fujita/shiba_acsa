import pandas as pd
from model import ASAP_SHIBA, train, get_heatmap
train_df = pd.read_csv("datasets/train_japenese.csv")
validation_df = pd.read_csv("datasets/dev_jp - dev_jp.csv")
test_df = pd.read_csv("datasets/test_jp - test_jp.csv")

model = ASAP_SHIBA()
model = model.cuda()


