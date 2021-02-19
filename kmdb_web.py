# %% [markdown]
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()

print(X_train[0])
print(y_train[0])
# %% [markdown]
word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value + 3] = key

# %%

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
max_len = 500
X_train = pad_sequences(sequences=X_train, maxlen=max_len)
X_test = pad_sequences(sequences=X_test, maxlen=max_len)
# %% [markdown]
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100))
model.add(GRU(units=128))
model.add(Dense(units=1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "GRU_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2
)

# %%
loaded_model = load_model("GRU_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# %%
def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub("[^0-9a-zA-Z ]", "", new_sentence).lower()

    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():
        # 단어 집합의 크기를 10,000으로 제한.
        try:
            if word_to_index[word] <= 10000:
                encoded.append(word_to_index[word] + 3)
            else:
                # 10,000 이상의 숫자는 <unk> 토큰으로 취급.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 취급.
        except KeyError:
            encoded.append(2)

    pad_new = pad_sequences([encoded], maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))


# %%
a = "This movie was just way too overrated. The fighting was not professional and in slow motion. I was expecting more from a 200 million budget movie. The little sister of T.Challa was just trying too hard to be funny. The story was really dumb as well. Don't watch this movie if you are going because others say its great unless you are a Black Panther fan or Marvels fan."
sentiment_predict(a)
