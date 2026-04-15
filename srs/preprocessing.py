from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd

class CardioPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def process(self, clinical_df, ehr_df):
        # Apply KNN Imputation to EHR
        ehr_imputed = self.imputer.fit_transform(ehr_df)
        
        # Apply Z-score Normalization (StandardScaler) to Clinical
        clin_scaled = self.scaler.fit_transform(clinical_df)
        
        return pd.DataFrame(clin_scaled, columns=clinical_df.columns), \
               pd.DataFrame(ehr_imputed, columns=ehr_df.columns)
