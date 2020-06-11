import rustlib
import numpy as np
from .core import Base, CLASSIFICATION, REGRESSION


class Mlp(Base):
    def __init__(self, x_train=None, y_train=None, npl=None, alpha=0.01, type_=CLASSIFICATION):
        super().__init__(x_train=x_train, y_train=y_train, alpha=alpha, type_=type_)
        self._npl = npl

        if not npl:
            raise Exception("nlp can't be empty.")

    def init_w(self):
        self.model = rustlib.mpl_utils_create_model(self.npl)

    @property
    def npl(self):
        return [len(self.x_train[0])] + self._npl

    def fit(self, epochs=1, loss_stop=False, type_=None):
        x_train, y_train = super().fit(type_)

        if not self.model:
            self.init_w()

        if self._type == CLASSIFICATION:
            if not isinstance(y_train[0], list):
                y_train = [[x] for x in y_train]

            self.model = rustlib.mlp_classification_fit(
                self.model,
                x_train,
                y_train,
                self.alpha,
                epochs,
                loss_stop,
                self.npl,
            )
            return self
        elif self._type == REGRESSION:
            if not isinstance(y_train[0], list):
                y_train = [[x] for x in y_train]

            self.model = rustlib.mlp_regression_fit(
                self.model,
                x_train,
                y_train,
                self.alpha,
                epochs,
                loss_stop,
                self.npl,
            )
            return self

    def predict(self, to_predict):
        to_predict = super().predict(to_predict)

        if self._type == CLASSIFICATION:
            r = rustlib.mlp_classification_predict(to_predict, self.model, self.npl)

            if len(r) > 1:
                max_index = r.index(max(r))
                r = [-1 if i != max_index else 1 for i, x in enumerate(r)]

            return r
        elif self._type == REGRESSION:
            return rustlib.mlp_regression_predict(to_predict, self.model, self.npl)
