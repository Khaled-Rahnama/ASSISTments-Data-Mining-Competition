from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import roc_auc_score, mean_squared_error
from math import sqrt
import Vis

from Preprocessing import Preprocessing


class STEMPrediction:
    # sequence length or maximum number of time-steps for each student
    max_length_seq = 500

    batch_size = 32

    def __get_model(self, n_features):
        # create the LSTM network
        model = Sequential()
        model.add(LSTM(100, input_shape=(self.max_length_seq, n_features)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['roc_auc'])
        return model

    def predict(self, x_train, x_test, y_train, y_test):
        n_features = x_train.shape[2]

        model = self.__get_model(n_features)

        history = model.fit(x_train, y_train,
                            epochs=50,
                            batch_size=self.batch_size,
                            validation_split=0.15)

        y_pred = model.predict(x_test, batch_size=self.batch_size)

        roc_score = roc_auc_score(y_test, y_pred)
        mse_score = mean_squared_error(y_test, y_pred)

        # printing and plotting model and results information
        Vis.plot_loss(history)
        Vis.plot_roc(y_test, y_pred)
        print(model.summary())
        print("Test ROC Score: %f" % roc_score)
        print("Test MSE Score: %f" % mse_score)
        print("Final Competition Score: %f" % (1 - sqrt(mse_score) + roc_score))

        return roc_score


pre = Preprocessing()
x, y = pre.load_data(return_as_df=True)


stem = STEMPrediction()
stem.predict(x_train, x_test, y_train, y_test)