class FeatureFactory:
    def __init__(self, feature_names):
        pass

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