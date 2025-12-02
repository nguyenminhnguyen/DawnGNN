# DawnGNN — Hướng dẫn tái hiện (Reproduction) end-to-end trên Windows PowerShell

Mục tiêu: Bạn có thể chạy toàn bộ pipeline từ dữ liệu mẫu đến mô hình ra kết quả, không cần tra cứu thêm.

## 1) Thiết lập môi trường
- Hệ điều hành: Windows 10/11 (PowerShell)
- Python: 3.8–3.10 (khuyến nghị 3.8/3.9 để tương thích PyTorch 2.0.0)
- GPU: Tùy chọn (nếu có CUDA 11.8). Có thể chạy CPU trước.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch 2.0.0 (CPU)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# NLP & xử lý dữ liệu
pip install transformers==4.28.1 datasets==2.12.0 scikit-learn==1.2.2 numpy==1.24.3 pandas==2.0.1 tqdm==4.65.0

# Crawl & phân tích HTML (nếu tự crawl tài liệu API)
pip install requests==2.31.0 beautifulsoup4==4.12.2 lxml==4.9.2

# Đồ thị; PyG (tuỳ chọn). Nếu lỗi PyG, có bản TinyGAT không cần PyG bên dưới
pip install networkx==3.1
pip install torch-geometric==2.3.1
```

GPU (CUDA 11.8), thay thế dòng cài torch CPU:
```powershell
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

Kiểm tra nhanh:
```powershell
python - << 'PY'
import torch, transformers
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
PY
```

## 2) Cấu trúc dữ liệu tối thiểu
Tạo cấu trúc thư mục và dữ liệu mẫu (2 bản ghi) để chạy demo nhanh:

```powershell
New-Item -ItemType Directory -Path data/raw/dataset1 -Force | Out-Null
New-Item -ItemType Directory -Path data/raw/apidocs -Force | Out-Null

@'{
"id": "toy_1", "apis": ["CreateFileW","WriteFile","RegSetValueW","InternetConnectA"], "label": 1
}
{
"id": "toy_2", "apis": ["CreateFileW","ReadFile","CloseHandle"], "label": 0
}
'@ | Set-Content data/raw/dataset1/samples.jsonl -Encoding UTF8

@'api,description
CreateFileW,Creates or opens a file or I/O device.
WriteFile,Writes data to the specified file or input/output (I/O) device.
RegSetValueW,Sets the data for the specified value in the registry.
InternetConnectA,Opens an FTP or HTTP session for a given site.
ReadFile,Reads data from a file, starting at the position indicated by the file pointer.
CloseHandle,Closes an open object handle.
'@ | Set-Content data/raw/apidocs/api_docs.csv -Encoding UTF8
```

- `samples.jsonl`: mỗi dòng là một JSON {id, apis: [..], label: 0/1}.
- `api_docs.csv`: hai cột `api,description` (mô tả chức năng súc tích).

Mẹo: chưa crawl được mô tả chính thức? Dùng mô tả ngắn gọn để chạy thử, rồi thay bằng mô tả chuẩn để tăng chất lượng.

## 3) Sinh API embeddings bằng BERT (nhanh)
Không fine-tune, dùng `bert-base-uncased` để có embedding 768 chiều cho mỗi API từ mô tả.

```powershell
python - << 'PY'
from transformers import AutoTokenizer, AutoModel
import torch, csv, json, os

model_name = 'bert-base-uncased'
out_dir = 'data/processed/embeddings'
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModel.from_pretrained(model_name)
mdl.eval()

api2desc = {}
with open('data/raw/apidocs/api_docs.csv', encoding='utf-8') as f:
	for api, desc in csv.DictReader(f):
		api2desc[api] = desc.strip()

api2vec = {}
with torch.no_grad():
	for api, desc in api2desc.items():
		inputs = tok(desc, return_tensors='pt', truncation=True, max_length=128)
		outputs = mdl(**inputs)
		cls = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().tolist()
		api2vec[api] = cls

with open(os.path.join(out_dir, 'api_embeddings.json'), 'w', encoding='utf-8') as f:
	json.dump(api2vec, f)

print('Saved embeddings for', len(api2vec), 'APIs')
PY
```

## 4) Xây đồ thị từ chuỗi API
Mỗi mẫu → nút là API, cạnh có hướng giữa API kế tiếp.

```powershell
python - << 'PY'
import json, os
in_path = 'data/raw/dataset1/samples.jsonl'
out_dir = 'data/processed/graphs'
os.makedirs(out_dir, exist_ok=True)

def to_graph(sample):
	apis = sample['apis']
	nodes = sorted(set(apis))
	idx = {a:i for i,a in enumerate(nodes)}
	edges = [[idx[apis[i]], idx[apis[i+1]]] for i in range(len(apis)-1)]
	return { 'id': sample['id'], 'label': sample['label'], 'nodes': nodes, 'edges': edges }

with open(in_path, 'r', encoding='utf-8') as fin:
	for line in fin:
		g = to_graph(json.loads(line))
		json.dump(g, open(os.path.join(out_dir, f"{g['id']}.graph.json"), 'w', encoding='utf-8'))

print('Graphs saved to', out_dir)
PY
```

## 5) Ghép embedding vào nút (tạo tensor)

```powershell
python - << 'PY'
import os, json
indir = 'data/processed/graphs'
outdir = 'data/processed/graphs_tensor'
os.makedirs(outdir, exist_ok=True)
emb = json.load(open('data/processed/embeddings/api_embeddings.json', 'r', encoding='utf-8'))
for fn in os.listdir(indir):
	if not fn.endswith('.graph.json'): continue
	g = json.load(open(os.path.join(indir, fn),'r',encoding='utf-8'))
	g['X'] = [emb.get(api, [0.0]*768) for api in g['nodes']]  # gán vector 0 nếu thiếu mô tả
	json.dump(g, open(os.path.join(outdir, fn.replace('.graph.json','.tensor.json')), 'w', encoding='utf-8'))
print('Tensor graphs saved to', outdir)
PY
```

## 6) Huấn luyện demo (TinyGAT bằng PyTorch thuần)
Không cần PyG. Dùng cho toy demo; với dữ liệu thật, tăng `EPOCHS` lên ~100.

```powershell
python - << 'PY'
import os, json, random
import torch, torch.nn as nn, torch.nn.functional as F
from glob import glob

data_dir = 'data/processed/graphs_tensor'
files = sorted(glob(os.path.join(data_dir, '*.tensor.json')))
random.shuffle(files)
N = len(files)
train, val, test = files[:max(1,int(0.8*N))], files[max(1,int(0.8*N)):max(1,int(0.9*N))], files[max(1,int(0.9*N)):]

def load(fn):
	g = json.load(open(fn,'r',encoding='utf-8'))
	X = torch.tensor(g['X'], dtype=torch.float)
	E = torch.tensor(g['edges'], dtype=torch.long)
	y = torch.tensor([g['label']], dtype=torch.long)
	N = X.size(0)
	A = torch.zeros((N,N))
	if E.numel()>0: A[E[:,0],E[:,1]] = 1.
	return X, A, y

class TinyGAT(nn.Module):
	def __init__(self, din=768, dh=64, dout=64):
		super().__init__(); self.W=nn.Linear(din,dh,False); self.a=nn.Linear(2*dh,1,False); self.W2=nn.Linear(dh,dout)
	def forward(self,X,A):
		H=self.W(X); N=H.size(0)
		Hi=H.unsqueeze(1).expand(N,N,-1); Hj=H.unsqueeze(0).expand(N,N,-1)
		e=self.a(torch.cat([Hi,Hj],-1)).squeeze(-1).masked_fill(A==0,-1e9)
		alpha=F.softmax(e,1); Z=alpha@H; return F.elu(self.W2(Z))
class Cls(nn.Module):
	def __init__(self):
		super().__init__(); self.g1=TinyGAT(); self.g2=TinyGAT(64,32,64); self.mlp=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
	def forward(self,X,A):
		Z=self.g1(X,A); Z=self.g2(Z,A); g=Z.mean(0); return self.mlp(g)

m=Cls(); opt=torch.optim.Adam(m.parameters(), lr=1e-4)

def run(files, train=True):
	tot=0; hit=0; loss_sum=0.
	m.train() if train else m.eval()
	for fn in files:
		X,A,y=load(fn)
		if train: opt.zero_grad()
		logit=m(X,A).unsqueeze(0)
		loss=F.cross_entropy(logit,y)
		if train: loss.backward(); opt.step()
		pred=logit.argmax(-1)
		tot+=1; hit+=int(pred.item()==y.item()); loss_sum+=loss.item()
	return loss_sum/max(1,tot), hit/max(1,tot)

for ep in range(1,6):  # demo 5 epoch
	tl,ta=run(train,True); vl,va=run(val,False)
	print(f'Epoch {ep}: train_loss={tl:.4f} acc={ta:.2f} | val_acc={va:.2f}')
print('Test acc=', run(test,False)[1])
PY
```

## 7) Tham số khuyến nghị (theo paper)
- Epoch: 100
- Batch size: 128
- Learning rate: 1e-4
- Hidden dims: GCN=32, GIN=16, GAT=12 (khi dùng triển khai chuẩn GNN)
- Chia tập: train 80% / val 10% / test 10% (xáo trộn ngẫu nhiên)

## 8) Lỗi thường gặp & cách xử lý nhanh
- Cài `torch-geometric` thất bại: dùng bản TinyGAT (không phụ thuộc PyG) hoặc chạy CPU trước (dễ cài hơn GPU).
- Thiếu mô tả API → embedding trống: tạm gán vector 0 cho API đó; sau bổ sung mô tả chuẩn để cải thiện.
- Mất cân bằng dữ liệu (benign ít): dùng stratified split, reweight loss, hoặc điều chỉnh threshold.
- Thiếu RAM: giảm `max_length` khi encode BERT, giảm hidden size, hoặc dùng batch nhỏ hơn.

## 9) Nâng cấp từ demo lên phiên bản “chuẩn”
- Dùng `torch_geometric.nn.GATConv` thay TinyGAT để tối ưu hiệu năng và mở rộng.
- Fine-tune BERT với nhiệm vụ MLM trên toàn bộ corpus mô tả API (đạt embedding sát ngữ nghĩa hơn).
- Thêm thuộc tính nút khác (tham số API, vị trí, metadata) và thử các kiến trúc GNN khác (GIN/GCN) để so sánh.

