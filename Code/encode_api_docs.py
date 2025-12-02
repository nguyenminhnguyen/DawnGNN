import os
import csv
import json
from transformers import AutoTokenizer, AutoModel
import torch


def encode_api_docs(csv_path: str, out_json_path: str, model_name: str = 'bert-base-uncased', max_length: int = 128):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    api2desc = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            api = row['api']
            desc = row['description'].strip()
            api2desc[api] = desc

    api2vec = {}
    with torch.no_grad():
        for api, desc in api2desc.items():
            inputs = tok(desc, return_tensors='pt', truncation=True, max_length=max_length)
            outputs = mdl(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().tolist()
            api2vec[api] = cls

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(api2vec, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode API docs to embeddings using BERT (CLS vector).')
    parser.add_argument('--csv', required=True, help='Path to api_docs.csv (columns: api,description)')
    parser.add_argument('--out', required=True, help='Output path for api_embeddings.json')
    parser.add_argument('--model', default='bert-base-uncased', help='HF model name')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    args = parser.parse_args()

    encode_api_docs(args.csv, args.out, args.model, args.max_len)
