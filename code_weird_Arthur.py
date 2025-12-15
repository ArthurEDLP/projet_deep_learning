
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
########                ########
################################



# %%
################################
########                ########
##       MECA ATTENTION       ##
##              2             ##
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

def create_text_attention_model(vocab_size, maxlen, horizons): # modèle autonome texte → indice

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

