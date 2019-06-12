import inspect


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        if feature_names in globals():
            return globals()[feature_names](**kwargs)
        else:
            raise ValueError("No feature defined named with {}".format(feature_names))

    def feature_list(self):
        """特徴量名リストを取得する"""
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
    def __init__(self):
        pass

    def apply(self):
        pass

    def extract(self):
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
