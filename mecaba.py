# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    filename="ratings_train.txt",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
    filename="ratings_test.txt",
)
train_data = pd.read_table("ratings_train.txt")
test_data = pd.read_table("ratings_test.txt")
# %%
print("훈련용 리뷰 개수 :", len(train_data))  # 훈련용 리뷰 개수 출력

# %%
train_data.drop_duplicates(subset=["document"], inplace=True)
train_data["document"] = train_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data["document"].replace("", np.nan, inplace=True)
train_data = train_data.dropna(how="any")
print("전처리 후 훈련용 샘플의 개수 :", len(train_data))
# %%
test_data.drop_duplicates(subset=["document"], inplace=True)
test_data["document"] = test_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test_data["document"].replace("", np.nan, inplace=True)
test_data = test_data.dropna(how="any")
print("전처리 후 테스트용 샘플의 개수 :", len(test_data))

# %%
