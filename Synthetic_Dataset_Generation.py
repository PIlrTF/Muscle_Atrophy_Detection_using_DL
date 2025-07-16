#DATASET GENERATION

import numpy as np
import pandas as pd

class MuscleAtrophyDatasetGenerator:
    def __init__(self, n_samples=10000, seed=42):
        self.n_samples = n_samples
        if seed is not None:
            np.random.seed(seed)

    def generate(self):
        # Basic demographic and anatomical features
        age = np.random.normal(60, 10, self.n_samples)
        sex = np.random.binomial(1, 0.5, self.n_samples)  # 0 = female, 1 = male
        muscle_area = np.random.normal(5000, 800, self.n_samples)
        muscle_density = np.random.normal(0.6, 0.05, self.n_samples)
        muscle_texture = np.random.normal(0.5, 0.1, self.n_samples)
        fiber_integrity = np.random.normal(0.7, 0.1, self.n_samples)

        # Add complications and realistic variability
        area_density_interaction = muscle_area * muscle_density
        age_squared = age ** 2
        density_over_age = muscle_density / (age + 1e-3)

        # Comorbidities and lifestyle covariates
        physical_activity_level = np.random.choice([0, 1, 2], self.n_samples, p=[0.3, 0.5, 0.2])
        smoker = np.random.binomial(1, 0.25, self.n_samples)
        alcohol_use = np.random.normal(2, 1, self.n_samples)

        # Modify features based on lifestyle
        for i in range(self.n_samples):
            if smoker[i] == 1:
                fiber_integrity[i] *= np.random.uniform(0.85, 0.95)
            muscle_density[i] *= 1 + 0.05 * (1 - physical_activity_level[i])

        # Simulate previous measurements
        muscle_area_6mo_ago = muscle_area + np.random.normal(-50, 30, self.n_samples)
        muscle_area_change = muscle_area - muscle_area_6mo_ago

        # Add measurement noise
        muscle_density += np.random.normal(0, 0.02, self.n_samples)
        muscle_density = np.clip(muscle_density, 0, 1)

        # Missing values
        missing_mask = np.random.rand(self.n_samples) < 0.05
        muscle_texture[missing_mask] = np.nan

        # Simulate labels with noise
        label = (muscle_area < 4700).astype(int)
        flip_mask = np.random.rand(self.n_samples) < 0.03
        label[flip_mask] = 1 - label[flip_mask]

        # Build the final dataset
        features = pd.DataFrame({
            'age': age,
            'sex': sex,
            'muscle_area': muscle_area,
            'muscle_density': muscle_density,
            'muscle_texture': muscle_texture,
            'fiber_integrity': fiber_integrity,
            'area_density_interaction': area_density_interaction,
            'age_squared': age_squared,
            'density_over_age': density_over_age,
            'physical_activity_level': physical_activity_level,
            'smoker': smoker,
            'alcohol_use': alcohol_use,
            'muscle_area_6mo_ago': muscle_area_6mo_ago,
            'muscle_area_change': muscle_area_change,
            'label': label
        })

        return features

if __name__ == '__main__':
    generator = MuscleAtrophyDatasetGenerator(n_samples=10000)
    df = generator.generate()
    print(df.head())

    # Save the DataFrame to a CSV file
    df.to_csv('muscle_atrophy_synthetic_data.csv', index=False)
    print("CSV file saved as 'muscle_atrophy_synthetic_data.csv'")
