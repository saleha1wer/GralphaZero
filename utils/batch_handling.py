import torch
import numpy as np

def split_by_graph(tensor, graph_edge_indices):
    try:
        num_graphs = graph_edge_indices.max().item() + 1
    except:
        print('Set num graphs to 1 because graph_edge_indices is -->')
        print(graph_edge_indices)
        num_graphs = 1
    # Split the input tensor into separate tensors for each graph
    split_tensors = [tensor[graph_edge_indices == i] for i in range(num_graphs)]
    return split_tensors

def softmax_each_group(policy, graph_edge_indices,split=True):
    try:
        num_graphs = graph_edge_indices.max().item() + 1
    except:
        print('Set num graphs to 1 because graph_edge_indices is -->')
        print(graph_edge_indices)
        num_graphs = 1
        graph_edge_indices = torch.zeros(policy.size()[0],dtype=torch.long)
    # Create a temporary tensor to store the results of the softmax
    temp_policy = torch.zeros(num_graphs, *policy.size()[1:], device=policy.device)

    # Use scatter_add to sum the exponentials of each edge's values based on their graph index
    temp_policy.scatter_add_(0, graph_edge_indices.unsqueeze(1).expand_as(policy), torch.exp(policy))

    # Normalize the softmax by dividing the original exponentials by the sum
    policy_softmax = torch.gather(temp_policy, 0, graph_edge_indices.unsqueeze(1).expand_as(policy))
    policy_softmax = torch.exp(policy) / policy_softmax
    policy_softmax = split_by_graph(policy_softmax, graph_edge_indices) if split else policy_softmax
    return policy_softmax
    
def get_edge_indices_for_all_graphs(edge_index, batch):
    # Find the graph index for the source node of each edge
    graph_indices = batch[edge_index[0]]
    return graph_indices

def combine_edge_and_graph_embeddings(edge_emb, graph_emb, edge_graph_indices):
    # Expand the graph_emb tensor to have the same number of rows as the edge_emb tensor
    expanded_graph_emb = graph_emb[edge_graph_indices]
    # Combine the edge embeddings with the corresponding graph embeddings
    combined_emb = torch.cat([edge_emb, expanded_graph_emb], dim=1)
    return combined_emb
