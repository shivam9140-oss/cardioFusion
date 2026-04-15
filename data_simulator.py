import pandas as pd
import numpy as np

def fuse_modalities(gen_df, ehr_df, clin_df, life_df):
    # Ensure all dataframes are sorted by Patient_ID
    # Feature-level concatenation
    fused_df = pd.concat([gen_df, ehr_df, clin_df, life_df], axis=1)
    return fused_df
