import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SHOInterventionDataset(Dataset):
    def __init__(self, n_samples=5000, t_points=100, t_max=10.0):
        self.n_samples = n_samples
        self.t = np.linspace(0, t_max, t_points)

        self.data = []

        for i in range(n_samples):
            A = np.random.uniform(0.5, 3.0)
            w = np.random.uniform(0.5, 3.0)
            phi = np.random.uniform(0, 2 * np.pi)

            x_original = A * np.sin(w * self.t + phi)

            intervention_type = np.random.choice(['amplitude', 'frequency', 'phase'])

            if intervention_type == 'amplitude':
                delta_A = np.random.uniform(-0.5, 0.5)
                A_new = np.clip(A + delta_A, 0.5, 3.0)
                x_intervened = A_new * np.sin(w * self.t + phi)
                params_original = [A, w, phi]
                params_intervened = [A_new, w, phi]

            elif intervention_type == 'frequency':
                delta_w = np.random.uniform(-0.3, 0.3)
                w_new = np.clip(w + delta_w, 0.5, 3.0)
                x_intervened = A * np.sin(w_new * self.t + phi)
                params_original = [A, w, phi]
                params_intervened = [A, w_new, phi]

            else:
                delta_phi = np.random.uniform(-np.pi / 2, np.pi / 2)
                phi_new = (phi + delta_phi) % (2 * np.pi)
                x_intervened = A * np.sin(w * self.t + phi_new)
                params_original = [A, w, phi]
                params_intervened = [A, w, phi_new]

            self.data.append({
                'x_original': x_original.astype(np.float32),
                'x_intervened': x_intervened.astype(np.float32),
                'intervention_type': intervention_type,
                'params_original': np.array(params_original, dtype=np.float32),
                'params_intervened': np.array(params_intervened, dtype=np.float32)
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]