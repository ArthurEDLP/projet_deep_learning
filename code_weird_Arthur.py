
# les mécanismes d'attention
# times series et cross validation : time series split

# transformation d'une image en un vecteur
# puis on multiplie le vecteur par son transposé ce qui nous donne une matrice des corrélation
# on multiplie par les keys values
# et la matrice de corrélation va nous dire ce qui va être important de ce qu'il n'est pas
# on obtiens un vecteur pondéré par ce qui est important (pondéré par l'importance des valeurs)

# Auto encoder
# Pour regénerer une image de base

################################
########################
################
########                
#### 
##
# Demander à GPT de sortir les mots clés de headline et reddit
##
####
######## 
################
########################
################################

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

#%%

df = pd.read_parquet("data/finance_ml_dataset_clean.parquet", engine="fastparquet")
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
##         3 horizons         ##
########                ########
################################



#%%

# L’attention de texte

class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = Dense(1) # pour calculer les scores d'attention
        # "A quel point chaque mot est-il important ?"

    def call(self, inputs):
        scores = tf.nn.softmax(self.dense(inputs), axis=1) # chaque mot reçoit un poids relatif
        context = tf.reduce_sum(scores * inputs, axis=1) # (tokens, features) → (features) : C’est le résumé intelligent du texte (l'indice latent)
        # (scores * inputs = importance × contenu)
        # Les mots inutiles deviennent presque nuls
        # Les mots forts dominent

        return context


# %%

################################
########                ########
##       MECA ATTENTION       ##
##          3 modèles         ##
##         3 horizons         ##
########                ########
################################

# Ce que l'on a fait c'était un modèle qui prédit les 3 horizons en même temps
# Maintenant on va faire 3 modèles séparés, un pour chaque horizon

def create_text_attention_model(vocab_size, maxlen):# modèle autonome texte → indice

    inputs = Input(shape=(maxlen,))# Une phrase = une séquence de mots
    
    x = Embedding(vocab_size, 128)(inputs)# Chaque mot devient un vecteur, la phrase devient une matrice (tokens, features)
    # Les mots deviennent du sens

    x = Bidirectional(
    LSTM(64, return_sequences=True))(x)# Sans return_sequences=True → pas d’attention possible, car on perd de l'information
    # La phrase est comprise dans son contexte

    x = Attention()(x) # le modèle apprend. Il sait quels mots sont importants.

    x = Dense(32, activation="tanh")(x)# Transformation non linéaire du sens global
    # Car mon indice est continu, négatif/positif, centré autour de 0. Pq tanh ? Car tanh donne des valeurs entre -1 et 1.

    output = Dense(1, activation="tanh")(x)
    # output: mon indice, de sentiment
    # négatif → pessimiste
    # positif → optimiste
    # proche de 0 → neutre

    model = Model(inputs, output)
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

# La fonction d'attention est la même que précédemment

# %%

# Prétraitement texte : rendre le texte “numérique”

tokenizer_V = Tokenizer(num_words=20000, oov_token="<UNK>") # stratégie d'attributions des mots en vecteurs
tokenizer_V.fit_on_texts(df["headline_concat"])
# Permet de prendre en compte l'ordre des mots

X_text_V = tokenizer_V.texts_to_sequences(df["headline_concat"])
X_text_V = pad_sequences(X_text_V, maxlen=100, padding="post")

#%%

# Création des différents horizons

X_text_V              # texte tokenisé
y_h0 = df["target_updown_plus_1_days"].values

# Retour simple : (P_t+1 - P_t)/P_t
#df["return_plus_1"] = df["Close"].pct_change(periods=1).shift(-1)
#df["return_plus_3"] = df["Close"].pct_change(periods=3).shift(-3)

# Je met sous hastag car on l'a déjà fait plus haut, c'est pour se souvenir

y_h1 = df["Close"].pct_change(periods=1).shift(-1).values
y_h3 = df["Close"].pct_change(periods=3).shift(-3).values

#%% 

# Onenlève les lignes où il y a des NaN 

mask_1 = ~np.isnan(y_h1)
mask_3 = ~np.isnan(y_h3)

mask = mask_1 & mask_3

X_text_h = X_text_V[mask]
y_h0 = y_h0[mask]
y_h1 = y_h1[mask]
y_h3 = y_h3[mask]



#%%

# Séparation des données h0

X_train_h0, X_test_h0, y_train_h0, y_test_h0 = train_test_split(
    X_text_h, y_h0, shuffle=False, test_size=0.2
)

model_h0 = create_text_attention_model(20000, 100)
model_h0.fit(X_train_h0, y_train_h0, epochs=10, batch_size=32, shuffle=False)

# %%

# Séparation des données h1

X_train_h1, X_test_h1, y_train_h1, y_test_h1 = train_test_split(
    X_text_h, y_h1, shuffle=False, test_size=0.2
)

model_h1 = create_text_attention_model(20000, 100)
model_h1.fit(X_train_h1, y_train_h1, epochs=10, batch_size=32, shuffle=False)

# %%

# Séparation des données h3

X_train_h3, X_test_h3, y_train_h3, y_test_h3 = train_test_split(
    X_text_h, y_h3, shuffle=False, test_size=0.2
)

model_h3 = create_text_attention_model(20000, 100)
model_h3.fit(X_train_h3, y_train_h1, epochs=10, batch_size=32, shuffle=False)

#%%

y_hat_h0 = model_h0.predict(X_test_h0)

print(y_hat_h0)

#%%

y_hat_h1 = model_h1.predict(X_test_h1)

print(y_hat_h1)

#%%

y_hat_h3 = model_h3.predict(X_test_h3)

print(y_hat_h3)

#%%

# Moyenne pondérés des 3 horizons

y_horizons = np.column_stack([y_hat_h0, y_hat_h1, y_hat_h3])

weights = np.array([0.5, 0.35, 0.15]) # pondération des horizons, car on considère qu'à CT c'est plus important

# On peut aussi faire une regression pour apprendre les poids optimaux

headline_index = y_horizons @ weights

# %%

# ajout de la colonne headline_index à df

df_merged = df.iloc[-len(headline_index):].copy()  # les dernières lignes correspondent à text_index
df_merged["headline_index"] = headline_index

df_merged[["headline_concat", "headline_index"]].head()
# %%


################################
########                ########
##          PROBLEME          ##
##    X_TEST = 397 lignes     ##
## =>   Time Series Split     ##
########                ########
################################


tscv = TimeSeriesSplit(n_splits=5)

#%%

# Horizon 0

y_hat_h0_full = np.zeros(len(y_h0))  # pour stocker toutes les prédictions

for train_idx, test_idx in tscv.split(X_text_h):
    X_train_fold, X_test_fold = X_text_h[train_idx], X_text_h[test_idx]
    y_train_fold, y_test_fold = y_h0[train_idx], y_h0[test_idx]

    model_h0 = create_text_attention_model(20000, 100)
    model_h0.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, shuffle=False, verbose=0)

    y_hat_h0_full[test_idx] = model_h0.predict(X_test_fold).ravel()


#%%

# Horizon 1

y_hat_h1_full = np.zeros(len(y_h1))  # pour stocker toutes les prédictions

for train_idx, test_idx in tscv.split(X_text_h):
    X_train_fold, X_test_fold = X_text_h[train_idx], X_text_h[test_idx]
    y_train_fold, y_test_fold = y_h1[train_idx], y_h1[test_idx]

    model_h1 = create_text_attention_model(20000, 100)
    model_h1.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, shuffle=False, verbose=0)

    y_hat_h1_full[test_idx] = model_h1.predict(X_test_fold).ravel()


#%%

# Horizon 3

y_hat_h3_full = np.zeros(len(y_h3))  # pour stocker toutes les prédictions

for train_idx, test_idx in tscv.split(X_text_h):
    X_train_fold, X_test_fold = X_text_h[train_idx], X_text_h[test_idx]
    y_train_fold, y_test_fold = y_h3[train_idx], y_h3[test_idx]

    model_h3 = create_text_attention_model(20000, 100)
    model_h3.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, shuffle=False, verbose=0)

    y_hat_h3_full[test_idx] = model_h3.predict(X_test_fold).ravel()

#%% 

# Pondération des horizons avec TSP

y_horizons_full = np.column_stack([y_hat_h0_full, y_hat_h1_full, y_hat_h3_full])
weights = np.array([0.5, 0.35, 0.15])
text_index_full = y_horizons_full @ weights

#%%

# Il y a un problème de longueur : 1986 instead of 1989
# Donc fait en sorte d'avoir la longueur de h3

# On crée un mask sur toute la longueur de df
mask = ~np.isnan(df["Close"].pct_change(periods=1).shift(-1)) & \
       ~np.isnan(df["Close"].pct_change(periods=3).shift(-3))

# Garder uniquement les lignes valides
df_merged = df[mask].copy()

# Ajouter ton indice texte calculé
df_merged["text_index_full"] = text_index_full  # text_index_full doit avoir la même longueur que df_merged

# Vérification
df_merged[["headline_concat", "text_index_full"]].head()

# %%

print(df_merged.info())
# %%
df_merged.to_parquet("data/finance_ml_dataset_indices_headline.parquet", engine="fastparquet")
# %%
