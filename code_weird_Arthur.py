
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
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
##         3 horizons         ##
########                ########
################################

from sklearn.model_selection import train_test_split

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

#%%

def create_text_attention_model_horizons(vocab_size, maxlen, horizons): # modèle autonome texte → indice

    inputs = Input(shape=(maxlen,)) # Une phrase = une séquence de mots
    
    x = Embedding(vocab_size, 128)(inputs) # Chaque mot devient un vecteur, la phrase devient une matrice (tokens, features)
    # Les mots deviennent du sens

    x = Bidirectional(
    LSTM(64, return_sequences=True))(x) # Sans return_sequences=True → pas d’attention possible, car on perd de l'information
    # La phrase est comprise dans son contexte

    x = Attention()(x) # le modèle apprend. Il sait quels mots sont importants.

    x = Dense(32, activation="tanh")(x) # Transformation non linéaire du sens global
    # Car mon indice est continu, négatif/positif, centré autour de 0. Pq tanh ? Car tanh donne des valeurs entre -1 et 1.

    output = Dense(len(horizons), activation="tanh")(x) # len(horizons) pour la prise en compte des différents horizons
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

# %%

# Prétraitement texte : rendre le texte “numérique”

tokenizer_2 = Tokenizer(num_words=20000, oov_token="<UNK>") # stratégie d'attributions des mots en vecteurs
tokenizer_2.fit_on_texts(df["headline_concat"])

X_text_2 = tokenizer_2.texts_to_sequences(df["headline_concat"])
X_text_2 = pad_sequences(X_text_2, maxlen=100, padding="post")

#%%


# Création de plusieurs horizons

horizons = [1, 3, 5]

# Retour simple : (P_t+1 - P_t)/P_t
df["return_plus_1"] = df["Close"].pct_change(periods=1).shift(-1)
df["return_plus_3"] = df["Close"].pct_change(periods=3).shift(-3)
df["return_plus_5"] = df["Close"].pct_change(periods=5).shift(-5)


y1 = df["Close"].pct_change(periods=1).shift(-1).values
y3 = df["Close"].pct_change(periods=3).shift(-3).values
y5 = df["Close"].pct_change(periods=5).shift(-5).values

# Je perd les dernières lignes à cause du shift(-horizon)

y_multi = np.column_stack([y1, y3, y5])

scaler = StandardScaler()
y_multi = scaler.fit_transform(y_multi)

# Les rendements futurs créent des NaN → on coupe proprement les données
# on enlève les lignes où y_multi a des NaN

valid_idx = ~np.isnan(y_multi).any(axis=1)
y_multi = y_multi[valid_idx]
X_text_2 = X_text_2[-y_multi.shape[0]:]

# Séparation des données

X_train, X_test, y_train, y_test = train_test_split(
    X_text_2,
    y_multi,
    shuffle=False, # On ne mélange jamais le futur avec le passé
    test_size=0.2
)


#%%

# Modèle

model = create_text_attention_model_horizons(
    vocab_size=20000,
    maxlen=100,
    horizons=horizons
)


#%%

# Entraînement

model.fit(X_train, y_train,
    epochs=10,
    batch_size=32,
    shuffle=False)

# plus le loss est faible (MSE), plus le modèle est bon

#%%

# Pq pas mettre des hyperparamètres sur les poids des horizons 

y_hat_h = model.predict(X_test)
weights = np.array([0.5, 0.3, 0.2]) # pondération des horizons, car on considère qu'à CT c'est plus important
text_index = y_hat_h @ weights



# %%

print(y_hat_h)

# 3 colonnes : horizon 1, 3, 5

# %%

print(text_index[:10])

# 1 colonne : indice pondéré par le poids des horizons


# %%

# Pour ajouter la colonne text_index à df

# Supposons que tu as créé X_text_2 et y_multi avec valid_idx
# valid_idx correspond aux lignes valides utilisées pour le modèle
# Ici, on prend la même portion de df

df_subset = df.iloc[-len(text_index):].copy()  # les dernières lignes correspondent à text_index
df_subset["text_index"] = text_index

df_subset[["headline_concat", "text_index"]].head()

# %%

print(df_subset.info()) # il me reste que 397 lignes CAR je suis avec X_text


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

def create_text_attention_model(vocab_size, maxlen):

    inputs = Input(shape=(maxlen,))
    
    x = Embedding(vocab_size, 128)(inputs)

    x = Bidirectional(
    LSTM(64, return_sequences=True))(x)

    x = Attention()(x)

    x = Dense(32, activation="tanh")(x)

    output = Dense(1, activation="tanh")(x)
    
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
model_h0.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=False)

# %%

# Séparation des données h1

X_train_h1, X_test_h1, y_train_h1, y_test_h1 = train_test_split(
    X_text_h, y_h1, shuffle=False, test_size=0.2
)

model_h1 = create_text_attention_model(20000, 100)
model_h1.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=False)

# %%

# Séparation des données h3

X_train_h3, X_test_h3, y_train_h3, y_test_h3 = train_test_split(
    X_text_h, y_h3, shuffle=False, test_size=0.2
)

model_h3 = create_text_attention_model(20000, 100)
model_h3.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=False)

#%%

y_hat_h0 = model_h0.predict(X_test)

print(y_hat_h0)

#%%

