import pandas as pd
import numpy as np

def generate_cardio_data(n_samples=1000):
    np.random.seed(42)
    
    # 1. Genomic Data (Binary SNPs)
    genomic = np.random.binomial(1, 0.3, size=(n_samples, 10))
    gen_cols = [f'SNP_{i:02d}' for i in range(10)]
    df_gen = pd.DataFrame(genomic, columns=gen_cols)

    # 2. Clinical Data (Gaussian distributions)
    # Systolic BP (120 +/- 15), Cholesterol (200 +/- 40)
    df_clin = pd.DataFrame({
        'systolic_bp': np.random.normal(120, 15, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'blood_glucose': np.random.normal(100, 20, n_samples),
        'bmi': np.random.normal(25, 5, n_samples)
    })

    # 3. EHR Data (Historical counts)
    df_ehr = pd.DataFrame({
        'prior_events': np.random.poisson(0.5, n_samples),
        'medication_count': np.random.randint(0, 5, n_samples)
    })

    # 4. Lifestyle Data (Ordinal/Binary)
    df_life = pd.DataFrame({
        'smoking_status': np.random.binomial(1, 0.2, n_samples),
        'exercise_freq': np.random.randint(0, 7, n_samples),
        'diet_score': np.random.uniform(0, 10, n_samples)
    })

    # Generate Ground Truth (Risk) based on weights
    # If BP is high and SNP_01 is present, risk increases
    logit = (0.05 * df_clin['systolic_bp'] + 
             0.02 * df_clin['cholesterol'] + 
             1.5 * df_gen['SNP_01'] + 
             2.0 * df_life['smoking_status'] - 12)
    
    prob = 1 / (1 + np.exp(-logit))
    risk_label = (prob > 0.5).astype(int)

    return df_gen, df_clin, df_ehr, df_life, risk_label

if __name__ == "__main__":
    g, c, e, l, r = generate_cardio_data()
    print("Data Simulation Complete. Samples:", len(r))
