from sklearn.model_selection import train_test_split
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import networkx as nx



def create_graph_data(corrected_cov):
    # Convert domain names to numerical indices
    unique_domains = pd.concat([corrected_cov['source'], corrected_cov['target']]).unique()
    domain_to_idx = {domain: idx for idx, domain in enumerate(unique_domains)}
    
    # Create edge index and edge weights
    edge_index = torch.tensor([
        [domain_to_idx[s] for s in corrected_cov['source']],
        [domain_to_idx[t] for t in corrected_cov['target']]
    ], dtype=torch.long)
    
    edge_weight = torch.tensor(corrected_cov['corrected_cov'].values, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        edge_index=edge_index,
        edge_attr=edge_weight,
        num_nodes=len(unique_domains)
    )
    return data, domain_to_idx

def train_node2vec(data, device='cuda', epochs=100):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        p=1,
        q=1,
        sparse=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02d}, Loss: {loss:.4f}')
            
    return model
@ray.remote
def _compute_device_embedding(device_id, group, embeddings, domain_mapping):
    valid_embeddings = []
    valid_weights = []
    
    for _, row in group.iterrows():
        domain = row['Domain_Name']
        if domain in domain_mapping:
            idx = domain_mapping[domain]
            valid_embeddings.append(embeddings[idx])
            valid_weights.append(row['fraction'])
    
    if valid_embeddings:
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        device_embedding = np.average(valid_embeddings, weights=valid_weights, axis=0)
        return device_id, device_embedding
    return device_id, None

def compute_device_embeddings(device_fractions, embeddings, domain_mapping):
    # Group device fractions by Device_ID
    grouped_fractions = device_fractions.groupby('Device_ID')
    
    # Create remote tasks
    futures = [
        _compute_device_embedding.remote(device_id, group, embeddings, domain_mapping)
        for device_id, group in grouped_fractions
    ]
    
    # Collect results
    results = ray.get(futures)
    
    # Convert results to dictionary
    device_embeddings = {
        device_id: embedding 
        for device_id, embedding in results 
        if embedding is not None
    }
    
    return device_embeddings

# Main flow
