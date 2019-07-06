import inspect

import pandas as pd


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        if feature_names in globals():
            return globals()[feature_names](**kwargs)
        else:
            raise ValueError("No feature defined named with {}".format(feature_names))

    def feature_list(self):
        lst = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                FeatureFactory,
                Feature,
                BasicFeature,
            ]:
                lst.append(obj.__name__)
        return lst


class Feature:
    categories = None
    dummy = True
    default = 0
    dummy_na = True

    def __init___(self, **kwargs):
        self.name = str(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return self.__class__.__name__

    def apply(self, dataset):
        values = self.extract(dataset)

        if self.categories:
            if self.dummy:
                values = self.convert_into_categories(values)
            else:
                self.default = None

        if self.default is not None:
            values = values.fillna(self.default)

        return values

    def convert_into_categories(self, X):
        index = X.index
        X = pd.Categorical(X, categories=self.categories)
        if self.dummy:
            X = pd.get_dummies(X, dummy_na=self.dummy_na)
        X.columns = self.get_columns()
        X.index = index
        return X

    def get_columns(self):
        if isinstance(self.categories, list) and self.dummy:
            columns = [f"{self}_{cat}" for cat in self.categories]
            if self.dummy_na:
                columns += [f"{self}_none"]
            return columns
        else:
            return [str(self)]

    def extract(self, dataset):
        raise NotImplementedError


class BasicFeature(Feature):
    def __init__(self):
        pass


class Type(BasicFeature):
    column = "type"


class Type0(BasicFeature):
    column = "type_0"


class Type1(BasicFeature):
    column = "type_1"


class Atom0(BasicFeature):
    column = "atom_0"


class Atom1(BasicFeature):
    column = "atom_1"
