class DataProcessor:
    def __init__(self, features, target_feature=None):
        self.features = features
        self.target_feature = target_feature

    def __call__(self, database):
        self._process(database)

    def _process(self, database):
        X_train = database.train.drop(columns=['id', 'molecule_name', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1'])
        y_train = database.train['scalar_coupling_constant']
        X_test = database.test.drop(columns=['id', 'molecule_name', 'atom_index_0', 'atom_index_1'])
        return X_train, y_train, X_test