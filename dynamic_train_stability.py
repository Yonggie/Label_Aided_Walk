import dgl
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dgl.data import FraudDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DeepWalk


def pprint(text):
    print(f"\033[031m{text}\033[0m \n")

@torch.no_grad()
def evaluate(model,y,mask):
    
    X = model.node_embed.weight.detach()
    clf = LogisticRegression().fit(X[mask].cpu().numpy(), y[mask].cpu().numpy())
    acc=clf.score(X[mask].cpu().numpy(), y[mask].cpu().numpy())
    return acc


def LR_eval(logistic,model,graph,feat,labels):
    score2=logistic.score(model.embed(graph,feat)[:-2].numpy(),labels)
    return score2





dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]
num_classes = dataset.num_classes
node_labels = hete_g.ndata['label']
fake_node_labels=torch.hstack([node_labels,(torch.zeros(1,2)).squeeze(0).bool()]).cuda()
fake_node_labels[-1]=1

def part_mask(original_mask, ratio):
    assert ratio<=1 and ratio>=0
    p=ratio*torch.ones(original_mask.shape)
    mask_ratio=torch.bernoulli(p).bool()
    new_mask=mask_ratio&original_mask

    return new_mask


first_ratio=0.5


train_mask = hete_g.ndata['train_mask'].bool()
train_mask_1st=part_mask(train_mask,first_ratio)
train_mask_1st=torch.hstack([train_mask_1st,torch.ones(1,2).squeeze(0).bool()]).cuda()
train_mask_2nd=torch.hstack([train_mask,torch.ones(1,2).squeeze(0).bool()]).cuda()


valid_mask = hete_g.ndata['val_mask'].bool()
valid_mask=torch.hstack([valid_mask,torch.zeros(1,2).squeeze(0).bool()]).cuda()
test_mask = hete_g.ndata['test_mask'].bool()
test_mask=torch.hstack([test_mask,torch.zeros(1,2).squeeze(0).bool()]).cuda()


out_dim=32
EPOCH=100
interval=5


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


gaps=[]
for i in range(30):
    pprint(f'exp{i+1}')
    model = DeepWalk(graph,emb_dim=out_dim,walk_length=out_dim).cuda()
    idxs=torch.arange(graph.num_nodes()).cuda()
    dataloader = DataLoader(idxs[train_mask_1st], batch_size=128,
                            shuffle=True, collate_fn=model.sample,drop_last=True)
    optimizer = SparseAdam(model.parameters(), lr=0.01)


    for epoch in range(5):
        for batch,batch_walk in enumerate(dataloader):
            if batch_walk.min()==-1:
                batch_walk[batch_walk==-1]=0
            loss = model(batch_walk.cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%interval==0:
            acc=evaluate(model,fake_node_labels,valid_mask)
            print(f'epoch {epoch}, loss {loss.item():.4f}, valid acc {acc:.4f}')



    # train Logistic regression
    X = model.node_embed.weight.detach()
    origin_rep=X
    y=fake_node_labels
    LR = LogisticRegression().fit(X[test_mask].cpu().numpy(), y[test_mask].cpu().numpy())
    score1=LR.score(X[test_mask].cpu().numpy(), y[test_mask].cpu().numpy())

    print(f'1st training LR score:{score1}')

    # 2nd training
    model2 = DeepWalk(graph,emb_dim=out_dim,walk_length=out_dim,sparse=False).cuda()
    frozen_label=origin_rep[-2:].detach().requires_grad_(False)
    # model2.node_embed.weight[-1]=frozen_label[-1]
    # model2.node_embed.weight[-2]=frozen_label[-2]

    optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
    dataloader = DataLoader(idxs[train_mask_2nd], batch_size=128,
                            shuffle=True, collate_fn=model.sample,drop_last=True)
    

    print('start training...')
    for epoch in range(EPOCH):
        for batch,batch_walk in enumerate(dataloader):
            if batch_walk.min()==-1:
                batch_walk[batch_walk==-1]=0
            loss1 = model2(batch_walk.cuda())
            
            pos_mask=torch.zeros(1,graph.num_nodes()).squeeze(0).bool()
            pos_mask[fake_node_labels==0]=True
            pos_mask[-1]=False
            pos_mask[-2]=False

            neg_mask=torch.zeros(1,graph.num_nodes()).squeeze(0).bool()
            neg_mask[fake_node_labels==1]=True
            neg_mask[-1]=False
            neg_mask[-2]=False
            
            reps=model2.node_embed.weight
            pos_nodes=reps[pos_mask]
            pos_label=frozen_label[0].expand_as(pos_nodes)

            neg_nodes=reps[neg_mask]
            neg_label=frozen_label[1].expand_as(neg_nodes)
            # loss2 = -(F.cosine_similarity(pos_label,pos_nodes).mean()+F.cosine_similarity(neg_label,neg_nodes).mean()).mean()
            loss2 = -(torch.dist(pos_label,pos_nodes).mean()+torch.dist(neg_label,neg_nodes).mean()).mean()
            loss=0.5*loss1+0.5*loss2
            
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%interval==0:
            acc=evaluate(model,fake_node_labels,valid_mask)
            print(f'epoch {epoch}, loss {loss.item():.4f}, valid acc {acc:.4f}')

                    


    # print(frozen_label)

    reps2=model2.node_embed.weight.detach()
    score2=LR.score(reps2[test_mask].cpu().numpy(),y[test_mask].cpu().numpy())
    pprint(f'LR score: {score1:.4f},{score2:.4f}')
    gap=abs(score1-score2)
    gaps.append(gap)


print('+'*40)
print(f'final: gap {np.mean(gaps):.4f}({np.std(gaps):.4f})')