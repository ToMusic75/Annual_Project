import numpy as np
import rustlib
from .core import Base, CLASSIFICATION, REGRESSION


class Linear(Base):
    def init_w(self):
        if isinstance(self.x_train, np.ndarray):
            if self.x_train.any() and self.x_train[0].any():
                self.model = rustlib.linear_utils_create_model(len(self.x_train[0]))
            else:
                self.model = rustlib.linear_utils_create_model(len(self.y_train))
        else:
            if self.x_train and self.x_train[0]:
                self.model = rustlib.linear_utils_create_model(len(self.x_train[0]))
            else:
                self.model = rustlib.linear_utils_create_model(len(self.y_train))

    def fit(self, epochs=1, loss_stop=False, type_=None):
        x_train, y_train = super().fit(type_)

        if self._type == CLASSIFICATION:
            if not self.model:
                self.init_w()

            self.model = rustlib.linear_classification_fit(
                self.model,
                x_train,
                y_train,
                self.alpha,
                epochs,
                loss_stop,
            )
            return self
        elif self._type == REGRESSION:
            self.model = rustlib.linear_regression_fit(
                x_train,
                y_train,
            )
            return self

    def predict(self, to_predict):
        to_predict = super().predict(to_predict)

        if self._type == CLASSIFICATION:
            return rustlib.linear_classification_predict(to_predict, self.model)
        elif self._type == REGRESSION:
            return rustlib.linear_regression_predict(to_predict, self.model)
