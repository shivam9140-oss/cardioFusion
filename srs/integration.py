import pandas as pd
from sklearn.decomposition import PCA

def fuse_modalities(gen_df, clin_df, ehr_df, life_df, pca_components=5):
    # Stage 3: PCA on high-dimensional Genomic data
    pca = PCA(n_components=pca_components)
    gen_reduced = pca.fit_transform(gen_df)
    gen_cols = [f'Genomic_PC{i}' for i in range(pca_components)]
    df_gen_pca = pd.DataFrame(gen_reduced, columns=gen_cols)

    # Feature-level Concatenation
    fused_df = pd.concat([df_gen_pca, clin_df, ehr_df, life_df], axis=1)
    return fused_df
