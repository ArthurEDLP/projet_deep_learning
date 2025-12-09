
# les mécanismes d'attention
# times series et cross validation : time series split

# transformation d'une image en un vecteur
# puis on multiplie le vecteur par son transposé ce qui nous donne une matrice des corrélation
# on multiplie par les keys values
# et la matrice de corrélation va nous dire ce qui va être important de ce qu'il n'est pas
# on obtiens un vecteur pondéré par ce qui est important (pondéré par l'importance des valeurs)

# Auto encoder
# Pour regénerer une image de base


#%%

import pandas as pd
import matplotlib.pyplot as plt


#%%

df = pd.read_parquet("data/finance_ml_dataset.parquet", engine="fastparquet")
print(df.head())
print(df.info())

# %%

plt.figure()
plt.plot(df["date"], df["Volume"])
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Volume dans le temps")
plt.show()

# %%
print(df.head())
# %%

################################
########                ########
##       MECA ATTENTION       ##
########                ########
################################

df["headline_concat"] = df["headline_concat"].astype(str)
df["reddit_concat"] = df["reddit_concat"].astype(str)


#%%

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# on combine tous les textes pour construire un vocabulaire unique
texts = df["headline_concat"].tolist() + df["reddit_concat"].tolist()

tokenizer = Tokenizer(num_words=20000, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)

# transformer en séquences d’indices
headline_seq = tokenizer.texts_to_sequences(df["headline_concat"])
reddit_seq = tokenizer.texts_to_sequences(df["reddit_concat"])

# longueur fixe
headline_seq = pad_sequences(headline_seq, maxlen=50)
reddit_seq = pad_sequences(reddit_seq, maxlen=50)
