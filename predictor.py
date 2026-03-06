"""
SucProPD Predictor - Protein Succinylation Site Prediction Tool
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from transformers import T5Tokenizer, T5EncoderModel
from threading import Thread
from pathlib import Path
import joblib

# Constants
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VALID_AA_SET = set(AMINO_ACIDS)
PROTT5_DIM = 1024
CKSAAP_DIM = 1200  # 20*20*3
PCA_DIM = 512


# ==================== Model Definitions ====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3, num_classes=1):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output.squeeze()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PDeepPP(nn.Module):
    def __init__(self, input_dim=512, seq_len=100, embed_size=256, heads=8,
                 num_layers=3, forward_expansion=4, dropout=0.2, num_classes=1):
        super(PDeepPP, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cnn_layers = nn.Sequential(
            ResidualBlock(1, 32, kernel_size=7, stride=1, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(32, 64, kernel_size=5, stride=1, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock(64, 128, kernel_size=3, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(25)
        )
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Dropout(dropout)
        )
        self._calculate_cnn_output_dim()
        combined_features = self.cnn_output_dim + embed_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def _calculate_cnn_output_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.seq_len)
            output = self.cnn_layers(dummy_input)
            self.cnn_output_dim = output.view(1, -1).shape[1]

    def forward(self, x):
        batch_size = x.shape[0]
        projected = self.input_projection(x)
        cnn_input = x.view(batch_size, 1, -1)
        if cnn_input.shape[2] < self.seq_len:
            pad_size = self.seq_len - cnn_input.shape[2]
            cnn_input = F.pad(cnn_input, (0, pad_size))
        else:
            cnn_input = cnn_input[:, :, :self.seq_len]
        cnn_features = self.cnn_layers(cnn_input)
        cnn_features = cnn_features.view(batch_size, -1)
        transformer_input = projected.unsqueeze(1).repeat(1, 5, 1)
        transformer_output = transformer_input
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output, transformer_output, transformer_output)
        attention_weights = torch.softmax(self.attention_pool(transformer_output).squeeze(-1), dim=1)
        transformer_features = torch.sum(transformer_output * attention_weights.unsqueeze(-1), dim=1)
        combined_features = torch.cat([cnn_features, transformer_features], dim=1)
        output = self.classifier(combined_features)
        return output.squeeze()


class DeepFRI(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024, 512, 256], dropout=0.3):
        super(DeepFRI, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.Tanh(),
            nn.Linear(prev_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        output = self.classifier(weighted_features)
        return output.squeeze()


class TripleEnsemble(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dims=[512, 256, 128],
                 deepfri_hidden_dims=[1024, 512, 256], dropout=0.3, num_classes=1):
        super(TripleEnsemble, self).__init__()
        self.mlp = MLP(input_dim, mlp_hidden_dims, dropout, num_classes)
        self.pdeeppp = PDeepPP(input_dim=input_dim, dropout=dropout, num_classes=num_classes)
        self.deepfri = DeepFRI(input_dim, deepfri_hidden_dims, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
            nn.Sigmoid()
        )
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        mlp_output = self.mlp(x).unsqueeze(1)
        pdeeppp_output = self.pdeeppp(x).unsqueeze(1)
        deepfri_output = self.deepfri(x).unsqueeze(1)
        combined = torch.cat([mlp_output, pdeeppp_output, deepfri_output], dim=1)
        weighted_combined = combined * F.softmax(self.weights, dim=0)
        ensemble_output = self.classifier(weighted_combined)
        return ensemble_output.squeeze()


# ==================== Feature Extractors ====================
class ProtT5Extractor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading ProtT5 model from {self.model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_path, do_lower_case=False, local_files_only=True
        )
        self.model = T5EncoderModel.from_pretrained(
            self.model_path, torch_dtype=torch.float32, local_files_only=True
        )
        self.model = self.model.to('cpu')
        self.model.eval()
        print("ProtT5 model loaded successfully")

    def extract(self, sequences):
        if self.model is None:
            self._load_model()

        features = []
        for seq in sequences:
            try:
                seq = seq.upper()
                if not all(aa in VALID_AA_SET for aa in seq):
                    features.append(np.zeros(PROTT5_DIM))
                    continue

                seq_processed = " ".join(seq)
                inputs = self.tokenizer(
                    seq_processed,
                    padding=False,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                    add_special_tokens=True
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    if hidden_states.shape[1] > 2:
                        seq_rep = hidden_states[0, 1:-1, :].mean(dim=0)
                    else:
                        seq_rep = hidden_states[0].mean(dim=0)
                    features.append(seq_rep.numpy())
            except:
                features.append(np.zeros(PROTT5_DIM))

        return np.array(features)

    def cleanup(self):
        del self.model, self.tokenizer
        gc.collect()


class CKSAAPExtractor:
    def __init__(self, k=3):
        self.k = k
        self.amino_acids = AMINO_ACIDS
        self.aa_list = list(self.amino_acids)
        self.pair_count = len(self.amino_acids) ** 2

    def extract(self, sequences):
        features = []
        for seq in sequences:
            seq = seq.upper()
            seq_len = len(seq)
            feature_vector = []

            for gap in range(1, self.k + 1):
                total_pairs = seq_len - gap - 1
                if total_pairs <= 0:
                    feature_vector.extend([0] * self.pair_count)
                    continue

                for aa1 in self.aa_list:
                    for aa2 in self.aa_list:
                        count = 0
                        for i in range(seq_len - gap - 1):
                            if seq[i] == aa1 and seq[i + gap + 1] == aa2:
                                count += 1
                        feature_vector.append(count / total_pairs)

            features.append(feature_vector)
        return np.array(features)


class SucProPDPredictor:
    def __init__(self, prott5_path, scaler_path, pca_path, ensemble_path):
        self.prott5 = ProtT5Extractor(prott5_path)
        self.cksaap = CKSAAPExtractor(k=3)

        # Load pre-trained scaler and PCA from training
        print(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print(f"Loading PCA from {pca_path}")
        self.pca = joblib.load(pca_path)

        # Load trained ensemble model
        print(f"Loading ensemble model from {ensemble_path}")
        checkpoint = torch.load(ensemble_path, map_location='cpu')

        # Create model instance and load state dict
        self.model = TripleEnsemble(input_dim=PCA_DIM)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Ensemble model loaded successfully")

    def predict_batch(self, sequences):
        """Batch prediction for multiple sequences"""
        # Step 1: Extract ProtT5 features (1024-dim)
        prott5_feat = self.prott5.extract(sequences)

        # Step 2: Extract CKSAAP features (1200-dim)
        cksaap_feat = self.cksaap.extract(sequences)

        # Step 3: Feature concatenation (1024 + 1200 = 2224-dim)
        fused_features = np.concatenate([prott5_feat, cksaap_feat], axis=1)
        print(f"Fused features shape: {fused_features.shape}")

        # Step 4: Apply pre-trained scaler (using training statistics)
        scaled_features = self.scaler.transform(fused_features)

        # Step 5: Apply pre-trained PCA (using training principal components)
        reduced_features = self.pca.transform(scaled_features)
        print(f"Reduced features shape: {reduced_features.shape}")

        # Step 6: Model prediction
        with torch.no_grad():
            probs = self.model(torch.FloatTensor(reduced_features)).numpy()

        return probs, (probs > 0.5).astype(int)

    def parse_fasta(self, fasta_file):
        """Parse FASTA file and return sequences and IDs"""
        sequences = []
        ids = []
        with open(fasta_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                # This is a header line - extract ID (remove '>')
                current_id = lines[i][1:].strip()
                i += 1

                # Read sequence until next header or end
                current_seq = ""
                while i < len(lines) and not lines[i].startswith('>'):
                    current_seq += lines[i]
                    i += 1

                ids.append(current_id)
                sequences.append(current_seq)
            else:
                # No headers, treat as sequence with default ID
                sequences.append(lines[i])
                ids.append(f'seq_{len(sequences)}')
                i += 1

        print(f"Parsed {len(sequences)} sequences")
        if len(sequences) > 0:
            print(f"First ID: {ids[0]}")
            print(f"First sequence length: {len(sequences[0])}")
            print(f"First sequence preview: {sequences[0][:50]}...")

        return ids, sequences

    def cleanup(self):
        self.prott5.cleanup()
        del self.model
        gc.collect()


class SucProPDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SucProPD - Protein Succinylation Site Predictor")
        self.root.geometry("900x700")

        self.predictor = None
        self.sequences = None
        self.sequence_ids = None
        self.last_results = None
        self.is_initialized = False

        # Setup paths
        base_dir = Path(__file__).parent
        self.prott5_path = r"D:\models\ProtT5"
        self.scaler_path = base_dir / 'combined_features_final' / 'pca_models' / 'standard_scaler.pkl'
        self.pca_path = base_dir / 'combined_features_final' / 'pca_models' / 'pca_model_512.pkl'
        self.model_path = base_dir / 'models' / 'triple_ensemble_independent_test_20251207_194411.pth'

        self.create_widgets()

        # Disable buttons initially
        self.upload_btn.config(state='disabled')
        self.predict_btn.config(state='disabled')
        self.save_btn.config(state='disabled')

        # Start initialization
        self.root.after(100, self.auto_init)

    def create_widgets(self):
        # Upload button
        self.upload_btn = tk.Button(self.root, text="Upload FASTA File",
                                    command=self.upload_file, font=('Arial', 11))
        self.upload_btn.pack(pady=10)

        # File content display
        tk.Label(self.root, text="File Content:", font=('Arial', 10, 'bold')).pack()
        self.file_text = scrolledtext.ScrolledText(self.root, height=8, width=80, font=('Courier', 10))
        self.file_text.pack(pady=5)

        # Prediction button
        self.predict_btn = tk.Button(self.root, text="Start Prediction",
                                     command=self.start_prediction, font=('Arial', 11))
        self.predict_btn.pack(pady=10)

        # Results display
        tk.Label(self.root, text="Prediction Results:", font=('Arial', 10, 'bold')).pack()
        self.result_text = scrolledtext.ScrolledText(self.root, height=15, width=80, font=('Courier', 10))
        self.result_text.pack(pady=5)

        # Save button
        self.save_btn = tk.Button(self.root, text="Save Results as CSV",
                                  command=self.save_results, font=('Arial', 11))
        self.save_btn.pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var,
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def auto_init(self):
        def init_task():
            try:
                print("Starting initialization task...")

                # Check required files
                missing_files = []
                if not os.path.exists(self.model_path):
                    missing_files.append(f"Model: {self.model_path}")
                if not os.path.exists(self.scaler_path):
                    missing_files.append(f"Scaler: {self.scaler_path}")
                if not os.path.exists(self.pca_path):
                    missing_files.append(f"PCA: {self.pca_path}")

                if missing_files:
                    error_msg = "Missing files:\n" + "\n".join(missing_files)
                    print(error_msg)
                    self.root.after(0, lambda: self.status_var.set("Initialization failed: missing files"))
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    return

                print(f"Model file found: {self.model_path}")
                print(f"Scaler file found: {self.scaler_path}")
                print(f"PCA file found: {self.pca_path}")
                print(f"ProtT5 path: {self.prott5_path}")

                # Create predictor with pre-trained scaler and PCA
                self.predictor = SucProPDPredictor(
                    self.prott5_path,
                    str(self.scaler_path),
                    str(self.pca_path),
                    str(self.model_path)
                )

                # Set initialized flag
                self.is_initialized = True
                print("Initialization completed successfully")

                # Enable buttons
                self.root.after(0, lambda: self.upload_btn.config(state='normal'))
                self.root.after(0, lambda: self.predict_btn.config(state='normal'))
                self.root.after(0, lambda: self.save_btn.config(state='normal'))
                self.root.after(0, lambda: self.status_var.set("Ready - Please upload a FASTA file"))

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Initialization failed: {str(e)}")
                print(f"Error details: {error_details}")
                self.root.after(0, lambda: self.status_var.set(f"Init failed: {str(e)}"))

        thread = Thread(target=init_task)
        thread.daemon = True
        thread.start()
        print("Initialization thread started")

    def upload_file(self):
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Please wait for initialization to complete")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("FASTA files", "*.fasta *.fa *.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                self.file_text.delete(1.0, tk.END)
                self.file_text.insert(tk.END, content)

                # Parse sequences
                self.sequence_ids, self.sequences = self.predictor.parse_fasta(file_path)
                self.status_var.set(f"Loaded {len(self.sequences)} sequences. Click 'Start Prediction' to continue.")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def start_prediction(self):
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Please wait for initialization to complete")
            return

        if self.sequences is None:
            messagebox.showwarning("No File", "Please upload a FASTA file first!")
            return

        def predict_task():
            try:
                self.root.after(0, lambda: self.status_var.set("Predicting..."))
                self.root.after(0, lambda: self.predict_btn.config(state='disabled'))

                # Predict
                probs, preds = self.predictor.predict_batch(self.sequences)

                # Save results with correct fields
                self.last_results = []
                result_lines = []

                print("\nPrediction Results:")
                print("-" * 80)
                print(f"{'ID':<10} {'Prediction':<10} {'Probability':<12}")
                print("-" * 80)

                for i, (seq_id, seq) in enumerate(zip(self.sequence_ids, self.sequences)):
                    status = "positive" if preds[i] == 1 else "negative"

                    # Store in last_results as a tuple/list with correct order
                    self.last_results.append([
                        seq_id,  # ID
                        seq,  # Sequence
                        status,  # Prediction
                        probs[i]  # Probability
                    ])

                    # Print to console
                    print(f"{seq_id:<10} {status:<10} {probs[i]:.4f}")

                    # Display format for GUI
                    result_lines.append(f">{seq_id}")
                    result_lines.append(f"Prediction: {status} (probability: {probs[i]:.4f})")
                    result_lines.append(f"Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
                    result_lines.append("")

                # Update display
                self.root.after(0, lambda: self.update_results("\n".join(result_lines)))
                self.root.after(0, lambda: self.status_var.set("Prediction complete. You can save results now."))
                self.root.after(0, lambda: self.predict_btn.config(state='normal'))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.predict_btn.config(state='normal'))

        Thread(target=predict_task, daemon=True).start()

    def update_results(self, text):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

    def save_results(self):
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Please wait for initialization to complete")
            return

        if self.last_results is None:
            messagebox.showwarning("No Results", "No prediction results to save! Please run prediction first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if save_path:
            try:
                # Create DataFrame with correct column names
                df = pd.DataFrame(self.last_results)
                df.columns = ['ID', 'Sequence', 'Prediction', 'Probability']

                # Save to CSV
                df.to_csv(save_path, index=False)

                # Show preview in console
                print("\nSaved results preview:")
                print(df.head())

                messagebox.showinfo("Success", f"Results saved to {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")


def main():
    root = tk.Tk()
    app = SucProPDApp(root)

    def on_closing():
        if app.predictor:
            app.predictor.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()