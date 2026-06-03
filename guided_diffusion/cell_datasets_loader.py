import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def load_data(
    data_dir,
    batch_size,
    vae_path,
    hidden_dim=128,
    train_vae=False,
    label_col=None,          # None => wie früher: kein "y" im extra-Dict
    shuffle=True,
    drop_last=True,
    # aus Sicherheitsgründen: default KEINE Filter, damit nichts "wegfällt"
    filter_cells_min_genes=None,
    normalize=True,
    log1p=True,
    encode_batch=4096,       # Batchgröße fürs VAE-Encoding
):
    """
    Lädt .h5ad, optional VAE-Encode, baut endlosen Generator.

    - Wenn label_col is None:   yield (batch, {})
    - Wenn label_col gesetzt:   yield (batch, {"y": y})

    Schutzmaßnahmen:
      * Keine Gen-Filterung (n_vars muss exakt zum VAE-Checkpoint passen)
      * Zellen-Filter standardmäßig AUS
      * Explizite Sanity-Checks mit klaren Fehlermeldungen
      * Device-Handling analog scDiffusion (dist_util.dev())

    Rückwärtskompatibel für VAE/Diffusion: dort label_col=None lassen.
    """
    import scanpy as sc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from guided_diffusion.cell_datasets_loader import load_VAE as _load_VAE
    from guided_diffusion import dist_util

    # ---- read ----
    adata = sc.read_h5ad(data_dir)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    if n_cells == 0 or n_genes == 0:
        raise RuntimeError(f"[load_data] Empty AnnData after read: cells={n_cells}, genes={n_genes} | file={data_dir}")

    # ---- optional: nur ZELLEN filtern (KEINE Gene!) ----
    if filter_cells_min_genes is not None and filter_cells_min_genes > 0:
        try:
            sc.pp.filter_cells(adata, min_genes=filter_cells_min_genes)
        except Exception as e:
            print(f"[load_data] filter_cells warning: {e}")
    if adata.n_obs == 0:
        raise RuntimeError(f"[load_data] All cells filtered out by min_genes={filter_cells_min_genes}. Disable filtering or lower threshold.")

    # ---- normalize/log (ändert keine Dimensionen) ----
    try:
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
        if log1p:
            sc.pp.log1p(adata)
    except Exception as e:
        print(f"[load_data] normalize/log warning: {e}")

    # ---- to dense float32 ----
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = X.astype(np.float32)
    if X.shape[0] == 0:
        raise RuntimeError(f"[load_data] X has 0 rows after preprocessing. file={data_dir}")

    # ---- optional Labels aus obs[label_col] -> y in {0,1} ODER Kategorie-Mapping ----
    y = None
    if label_col is not None:
        if label_col not in adata.obs:
            raise ValueError(
                f"[load_data] obs['{label_col}'] not found in {data_dir}. "
                f"Available: {list(adata.obs.columns)}"
            )

        labels_raw = adata.obs[label_col]
        labels_str = labels_raw.astype(str)

        # Unique as strings
        unique_vals = sorted(list(np.unique(labels_str.astype(str))))
        print(f"[load_data] label_col='{label_col}' unique values:", unique_vals)

        # CASE A: Already numeric binary {0,1}
        if set(unique_vals) <= {"0", "1"}:
            print(f"[load_data] detected binary numeric labels for '{label_col}'.")
            y = labels_str.astype(int).to_numpy()

        # CASE B: Any two categories (e.g. high/low, basal/luminal, emt/epi)
        elif len(unique_vals) == 2:
            print(f"[load_data] mapping 2-category label '{label_col}' → {{0,1}}")
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            y = labels_str.map(mapping).astype(int).to_numpy()

        # CASE C: Original tumor/normal handling fallback
        elif any("tumor" in v.lower() for v in unique_vals) or any("normal" in v.lower() for v in unique_vals):
            print(f"[load_data] fallback mapping tumor/normal for '{label_col}'")
            y_map = []
            for s in labels_str.str.lower():
                if "tumor" in s:
                    y_map.append(1)
                elif "healthy" in s or "normal" in s:
                    y_map.append(0)
                else:
                    y_map.append(-1)
            y = np.asarray(y_map, dtype=int)

        # CASE D: Unsupported label
        else:
            raise RuntimeError(
                f"[load_data] label_col='{label_col}' has >2 categories and no tumor/healthy semantics. "
                f"Values: {unique_vals}"
            )

        # FILTER CELLS WHERE y==0/1
        keep = (y == 0) | (y == 1)
        if keep.sum() == 0:
            raise RuntimeError(
                f"[load_data] label_col='{label_col}' removed all cells.\n"
                f"Unique values: {unique_vals}"
            )

        if keep.sum() != len(y):
            print(f"[load_data] removing {(~keep).sum()} cells without valid labels for '{label_col}'")
            X = X[keep]
            y = y[keep]


    # ---- VAE laden & auf korrektes Device bringen ----
    device = dist_util.dev()  # cuda:0 oder cpu
    vae = _load_VAE(vae_path, num_gene=n_genes, hidden_dim=hidden_dim).eval().to(device)

    # ---- batched Encoding auf dem Device, zurück auf CPU sammeln ----
    Z_chunks = []
    with torch.no_grad():
        N = X.shape[0]
        for i in range(0, N, encode_batch):
            sl = slice(i, min(N, i + encode_batch))
            x_t = torch.from_numpy(X[sl]).float().to(device)
            if x_t.shape[0] == 0:
                continue
            z_t = vae(x_t, return_latent=True).detach().cpu().numpy()
            Z_chunks.append(z_t)

    if not Z_chunks:
        raise RuntimeError(
            f"[load_data] No chunks encoded. Debug info: N={X.shape[0]}, encode_batch={encode_batch}, file={data_dir}. "
            "Possible causes: X empty, or encode_batch mis-set."
        )

    import numpy as np
    Z = np.concatenate(Z_chunks, axis=0)
    if Z.shape[0] == 0:
        raise RuntimeError("[load_data] Encoded Z has 0 rows — cannot proceed.")

    # ---- DataLoader bauen (CPU-Tensoren) ----
    import torch
    Z_t = torch.from_numpy(Z)
    if y is None:
        dataset = TensorDataset(Z_t)
    else:
        y_t = torch.from_numpy(y)
        dataset = TensorDataset(Z_t, y_t)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    # ---- endloser Generator (API-kompatibel) ----
    while True:
        if y is None:
            for (z_batch,) in loader:
                yield z_batch, {}
        else:
            for z_batch, y_batch in loader:
                yield z_batch, {"y": y_batch}



class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

