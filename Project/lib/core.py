import rustlib
import numpy as np


CLASSIFICATION = 1
REGRESSION = 2


class Base:
    def __init__(self, x_train=None, y_train=None, alpha=0.01, type_=CLASSIFICATION):
        self.x_train = x_train
        self.y_train = y_train
        self._alpha = alpha
        self._type = type_
        self.model = None

    def load_from_images(self, path, images_dimensions, y_train_value):
        self.x_train, self.y_train = rustlib.misc_images_load_from_images_path(
            path,
            images_dimensions,
            y_train_value
        )

    def save(self, path):
        if not self.model:
            Exception("Your current instance contains no model.")
        else:
            rustlib.misc_models_save(path, self.model)

    def load(self, path):
        self.model = rustlib.misc_models_load(path)

    @property
    def alpha(self):
        return self._alpha or 0.1

    @alpha.setter
    def alpha(self, value):
        if isinstance(value, float):
            if value <= 0:
                raise Exception("alpha value must be positive.")
            else:
                self._alpha = value
        else:
            raise Exception("alpha value must be a float.")

    def fit(self, type_=None):
        if type_:
            self._type = type_

        if self._type not in (CLASSIFICATION, REGRESSION):
            raise Exception("Fit type must be either CLASSIFICATION or REGRESSION.")

        x_train = self.x_train.tolist() if isinstance(self.x_train, np.ndarray) else self.x_train
        y_train = self.y_train.tolist() if isinstance(self.y_train, np.ndarray) else self.y_train

        if not x_train or not y_train:
            raise Exception("No X Train or Y Train found.")

        if x_train and y_train and len(x_train) != len(y_train):
            raise Exception(
                f"x_train and y_train must have the same length (X = {len(x_train)}, Y = {len(y_train)})"
            )

        return x_train, y_train

    def predict(self, to_predict):
        if self._type not in (CLASSIFICATION, REGRESSION):
            raise Exception("Fit type must be either CLASSIFICATION or REGRESSION.")

        to_predict = to_predict.tolist() if isinstance(to_predict, np.ndarray) else to_predict
        return to_predict
