import torch
import torch.nn.functional as F
from dgl.data import CoraGraphDataset,FraudDataset
from model import DeepWalk
import numpy as np
from torch.optim import SparseAdam
import dgl
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

@torch.no_grad()
def evaluate(model,y,mask):
    
    X = model.node_embed.weight.detach()
    clf = LogisticRegression().fit(X[mask].cpu().numpy(), y[mask].cpu().numpy())
    acc=clf.score(X[mask].cpu().numpy(), y[mask].cpu().numpy())
    return acc

# dataset = CoraGraphDataset()
# g = dataset[0]

dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]
num_classes = dataset.num_classes
node_labels = hete_g.ndata['label']
fake_node_labels=torch.hstack([node_labels,(torch.zeros(1,2)).squeeze(0).bool()]).cuda()
fake_node_labels[-1]=1


train_mask = hete_g.ndata['train_mask']
fake_train_mask=torch.hstack([train_mask,torch.ones(1,2).squeeze(0)]).bool().cuda()
valid_mask = hete_g.ndata['val_mask']
valid_mask=torch.hstack([valid_mask,torch.zeros(1,2).squeeze(0)]).bool().cuda()
test_mask = hete_g.ndata['test_mask']
test_mask=torch.hstack([test_mask,torch.zeros(1,2).squeeze(0)]).bool().cuda()


out_dim=32
EPOCH = 5
interval=1


graph = dgl.to_homogeneous(hete_g)
label_idx_start=graph.num_nodes()
normal_id=graph.num_nodes()
fraud_id=graph.num_nodes()+1
starter=node_labels
starter[starter==0]=normal_id
starter[starter==1]=fraud_id
distination=torch.tensor(list(range(graph.num_nodes()))).long()
graph.add_nodes(2) 
graph.add_edges(starter,distination)




cos_dists,euclid_dists,dot_dists=[],[],[]

for i in range(50):
    print(f'training {i}')
    model = DeepWalk(graph,emb_dim=out_dim,walk_length=out_dim).cuda()
    dataloader = DataLoader(torch.arange(graph.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample,drop_last=True)
    optimizer = SparseAdam(model.parameters(), lr=0.01)


    for epoch in range(EPOCH):
        for batch,batch_walk in enumerate(dataloader):
            if batch_walk.min()==-1:
                batch_walk[batch_walk==-1]=0
            loss = model(batch_walk.cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%interval==0:
            acc=evaluate(model,fake_node_labels,valid_mask)
            print(f'epoch {epoch}, loss {loss.item():.4f}, acc {acc:.4f}')



    accs=[]
    for _ in range(10):
        acc=evaluate(model,fake_node_labels,test_mask)    
        accs.append(acc)
    print(f'final test: {np.mean(accs):.4f}, std {np.std(accs):.4f}')

    pos_rep = model.node_embed.weight.detach()[-2]
    neg_rep = model.node_embed.weight.detach()[-1]

    euclid_dists.append(torch.dist(pos_rep,neg_rep,p=2).item())
    cos_dists.append(F.cosine_similarity(pos_rep.unsqueeze(0),neg_rep.unsqueeze(0)).item())
    dot_dists.append((pos_rep@neg_rep).item())

print(f"euclid: mean {np.mean(euclid_dists)}({np.std(euclid_dists)})")
print(f"cos: mean {np.mean(cos_dists)}({np.std(cos_dists)})")
print(f"dot: mean {np.mean(dot_dists)}({np.std(dot_dists)})")
