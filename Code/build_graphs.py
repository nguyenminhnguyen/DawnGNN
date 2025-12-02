import os
import json


def to_graph(sample):
    apis = sample['apis']
    nodes = sorted(set(apis))
    idx = {a: i for i, a in enumerate(nodes)}
    edges = [[idx[apis[i]], idx[apis[i + 1]]] for i in range(len(apis) - 1)]
    return {'id': sample['id'], 'label': sample['label'], 'nodes': nodes, 'edges': edges}


def build_graphs(jsonl_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            s = json.loads(line)
            g = to_graph(s)
            with open(os.path.join(out_dir, f"{g['id']}.graph.json"), 'w', encoding='utf-8') as fout:
                json.dump(g, fout)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build directed API graphs from samples.jsonl')
    parser.add_argument('--in', dest='inp', required=True, help='Path to samples.jsonl')
    parser.add_argument('--out', required=True, help='Output directory for .graph.json files')
    args = parser.parse_args()

    build_graphs(args.inp, args.out)
