from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral.model import SimpleInputMetadata
from pathlib import Path
import torch

tokenizer = Tokenizer("/home/mykolas/Documents/Rust/mistral-burn/models/tokenizer.model")
transformer = Transformer.from_folder(Path("/home/mykolas/Documents/Rust/mistral-burn/models/"), max_batch_size=1, device="cpu")

tokens = torch.tensor(tokenizer.encode("test", bos=True))

input_metadata = SimpleInputMetadata.from_seqlens([2], "cpu")
freqs_cis = transformer.freqs_cis[input_metadata.positions]

h = transformer.tok_embeddings(tokens)

r = transformer.layers["0"].attention.forward(transformer.layers["0"].attention_norm.forward(h), freqs_cis, None)

