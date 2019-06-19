import inspect

import pandas as pd


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        """特徴量インスタンスを生成する

        Parameters
        ----------
        feature_name : str
            特徴量名

        Returns
        -------
        Feature
            特徴量インスタンス

        """
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
    """特徴量クラス"""

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
        """特徴量を抽出する

        Parameters
        ----------
        dataset : DataSet
            データセット

        Returns
        -------
        pandas.Series
            indexはmemberId
            nameはDataFrameになったときのカラム名

        """
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
        """カテゴリ情報を付与し、dummy=Trueの場合はダミー化する

        Parameters
        ----------
        X : pandas.Series
            入力データ

        Returns
        -------
        pandas.Series or pandas.DataFrame
            カテゴリ情報を付与した特徴量

        """
        index = X.index
        X = pd.Categorical(X, categories=self.categories)
        if self.dummy:
            X = pd.get_dummies(X, dummy_na=self.dummy_na)
        X.columns = self.get_columns()
        X.index = index
        return X

    def get_columns(self):
        """カテゴリ変数のカラム名のリストを取得

        Returns
        -------
        list
            カテゴリ情報を持ったカラム名のリスト
        """
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
