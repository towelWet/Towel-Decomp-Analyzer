import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class CodeAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Code Functionality Analyzer")
        self.root.geometry("900x700")

        # Settings
        self.use_gpu = tk.BooleanVar(value=True)
        self.notable_chunks = []

        # Define enhanced heuristic patterns for various encryption and security types
        self.patterns = {
            "crypto_functions": re.compile(r'\b(AES|RSA|ECDSA|DES|SHA1|SHA256|CryptGenRandom|BCryptGenRandom|getrandom|arc4random)\b', re.IGNORECASE),
            "key_operations": re.compile(r'\b(keygen|generate_key|generateSecretKey|createKey|init_key)\b', re.IGNORECASE),
            "magic_constants": re.compile(r'(0x[0-9a-fA-F]{8,})'),
            "entropy_sources": re.compile(r'\b(/dev/urandom|CryptAcquireContext|RDRAND|RDSEED)\b', re.IGNORECASE),
            "license_strings": re.compile(r'\b(ProductKey|ActivationCode|LicenseKey|SerialNumber)\b', re.IGNORECASE),
            "encryption_types": re.compile(
                r'\b(ChaCha20|Poly1305|Blowfish|Twofish|Camellia|Whirlpool|SEED|Keccak|SHA-3|AES[-\s]?GCM|ECDH)\b',
                re.IGNORECASE),
            "encryption_algorithms": re.compile(r'\b(RSA2048|AES128|RC4|Serpent|XOR)\b', re.IGNORECASE),
            "hash_functions": re.compile(r'\b(SHA256|SHA1|MD5)\b', re.IGNORECASE),
            "digital_signatures": re.compile(r'\b(DSA1024)\b', re.IGNORECASE),
            "encoding_methods": re.compile(r'\b(BASE64|BASE32|BASE16|BASE10)\b', re.IGNORECASE),
            "web_auth": re.compile(r'\b(WEBAUTH)\b', re.IGNORECASE),
            "composite_techniques": re.compile(
                r'\b(C/R\+ModInv|WEBAUTH_RSA256|WEBAUTH_MD5/XOR|MD5/DSA1024/RC4/BASE32)\b',
                re.IGNORECASE),
            "crc": re.compile(r'\bCRC32\b', re.IGNORECASE)  # Added CRC32 detection
        }

        self.create_widgets()

    def create_widgets(self):
        # Top frame for file selection and GPU toggle
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Decompiled File:").pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(top_frame, text="Use GPU", variable=self.use_gpu).pack(side=tk.LEFT, padx=10)
        self.analyze_btn = ttk.Button(top_frame, text="Analyze Code", command=self.start_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Potential Key Generation / Security Code", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.results_text = tk.Text(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Export button
        ttk.Button(self.root, text="Export Results", command=self.export_results).pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.file_path_var.set(file_path)

    def start_analysis(self):
        file_path = self.file_path_var.get()
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Invalid file path!")
            return

        self.analyze_btn.config(state="disabled")
        self.results_text.delete("1.0", tk.END)
        self.notable_chunks = []
        threading.Thread(target=self.analyze_code, args=(file_path,), daemon=True).start()

    def analyze_code(self, file_path):
        try:
            # Initialize model with GPU support if available
            device = 0 if self.use_gpu.get() and torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModelForTokenClassification.from_pretrained("microsoft/codebert-base")
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

            # Read decompiled file
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Process file in chunks (100 lines per chunk)
            chunk_size = 100
            total = len(lines) // chunk_size + 1
            self.progress.config(maximum=total)

            for i in range(0, len(lines), chunk_size):
                chunk = "".join(lines[i:i+chunk_size])
                score = self.score_chunk(chunk, ner_pipeline)

                if score >= 2:  # Require at least 2 indicators
                    header = f"--- Chunk {i // chunk_size + 1} (Score: {score}) ---\n"
                    self.notable_chunks.append(header + chunk)
                    self.results_text.insert(tk.END, header + chunk + "\n")

                self.progress["value"] = (i // chunk_size) + 1
                self.root.update_idletasks()

            messagebox.showinfo("Done", f"Found {len(self.notable_chunks)} relevant chunks.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.analyze_btn.config(state="normal")

    def score_chunk(self, chunk, ner_pipeline):
        score = 0

        # Heuristic pattern matching: increase score for every pattern match
        for name, pattern in self.patterns.items():
            if pattern.search(chunk):
                score += 1

        # Semantic analysis using CodeBERT's NER (processes token-level entities)
        entities = ner_pipeline(chunk)
        # Expanded list of crypto-related terms, including "crc32"
        crypto_terms = [
            "key", "encrypt", "decrypt", "signature", "random",
            "chacha20", "poly1305", "blowfish", "twofish", "camellia",
            "whirlpool", "seed", "keccak", "sha-3", "aes-gcm", "ecdh",
            "rsa2048", "aes128", "rc4", "serpent", "xor", "sha256", "sha1",
            "md5", "dsa1024", "base64", "base32", "base16", "base10", "webauth",
            "crc32"
        ]
        for entity in entities:
            word = entity.get("word", "").lower()
            if any(term in word for term in crypto_terms):
                score += 0.5

        return score

    def export_results(self):
        if not self.notable_chunks:
            messagebox.showwarning("Warning", "No results to export!")
            return
        # Prompt for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not save_path:
            return  # User cancelled save dialog

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(self.notable_chunks))
            messagebox.showinfo("Success", f"Results exported to {save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeAnalyzerGUI(root)
    root.mainloop()
