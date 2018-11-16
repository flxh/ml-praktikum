import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

csvname = './Data_3D_2classes.csv'

print('Reading "' + csvname + '":')
data = np.loadtxt(csvname, delimiter=';')

x = data[:, 0:3]
y = data[:, 3]

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

k = 10
seed = 1337

kfoldcv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def build_model():
    model = Sequential()

    model.add(Dense(input_dim=3, units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


cvscores = []
for training, test in kfoldcv.split(x, encoded_Y):
    model = build_model()
    model.fit(x[training], encoded_Y[training], epochs=200, batch_size=10, verbose=0)
    cvscore = model.evaluate(x[test], encoded_Y[test])[1]
    cvscores.append(cvscore)

print("Model accuracy: {0}% stdev: {1}%".format(np.mean(cvscores)*100, np.std(cvscores)*100))