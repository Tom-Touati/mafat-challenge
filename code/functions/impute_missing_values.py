from sklearn.impute import KNNImputer# , IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares
def impute_missing_values(final_scores_pivot,train_device_ids):
    # Use SoftImpute to fill missing values
    imputer = KNNImputer(n_neighbors=10)
    imputer.fit(final_scores_pivot.loc[train_device_ids])
    imputed_scores = imputer.transform(final_scores_pivot)
    # imputer = AlternatingLeastSquares(factors=10, regularization=0.01, iterations=10,random_state=0)
    # # imputer = MissForest(max_depth=6,max_features=0.8, random_state=0)
    # imputer.fit( user_items = csr_matrix(final_scores_pivot.loc[train_device_ids].values))
    
    # imputed_scores = imputer.recommend_all(csr_matrix(final_scores_pivot.values))
    return imputed_scores
