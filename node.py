import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DynamicCache
from collections import deque

def aggregate_attention_matrix(attention_matricies):
    matricies = []
    for layer in attention_matricies:
        matricies.append(layer.cpu().numpy())
    return np.stack(matricies, axis=0).squeeze()

class ICLNode:
    def __init__(self, content, parents=None, id=None, 
                 start_pos_id=None, end_pos_id=None):
        self.content = content
        self.parents = parents if parents is not None else []
        if (start_pos_id is not None) and (end_pos_id is not None):
            raise ValueError('Only one of start_pos_id and end_pos_id should be given')
        self.start_pos_id = start_pos_id
        self.end_pos_id = end_pos_id
        self.position_ids = None
        self.token_ids = None
        if id is None:
            self.id = self.content[:10]
        else:
            self.id = id

    def tokenize(self, tokenizer=None, remove_bos_token=True):
        '''
        Tokenize the content of this node
        remove_bos_token: Remove the begin_of_sentence token, since we don't want 
        multiple BOS tokens in the input, and we may want to manually have a node called BOS.
        '''
        if self.token_ids is not None:
            return self.token_ids
        if tokenizer is None:
            raise ValueError("No tokenizer provided nor token ids available")
        self.token_ids = tokenizer.encode(self.content)
        if remove_bos_token:
            if self.token_ids[0] == tokenizer.bos_token_id:
                if len(self.token_ids) > 1:
                    self.token_ids = self.token_ids[1:]
        self.token_ids = torch.tensor(self.token_ids)
        token_len = len(self.token_ids)
        self.position_ids = torch.arange(token_len)
        if self.start_pos_id is not None:
            self.position_ids += self.start_pos_id
        elif self.end_pos_id is not None:
            self.position_ids += self.end_pos_id - token_len
        return self.token_ids
    
    def get_parents_input_position_ids(self):
        '''
        Get the input_ids of the parent nodes concatenated with the current node
        '''
        if not self.parents:
            return None
        input_ids = torch.concat(
            [node.token_ids for node in self.parents] + [self.token_ids]
        ).unsqueeze(0)
        position_ids = torch.concat(
            [node.position_ids for node in self.parents]
        ).unsqueeze(0)
        return input_ids, position_ids

    def __str__(self):
        return f'{self.id}'

def build_attention_matrix(leaf_node, tokenizer=None, materialize_attention_matrix=True):
    """
    Tokenize and build the attention matrix for the DAG rooted at the leaf node
    """
    all_nodes = upstream_nodes(leaf_node)
    node_to_idx = {}
    for i, node in enumerate(all_nodes):
        node_to_idx[node] = i
    token_ids = [node.tokenize(tokenizer) for node in all_nodes]
    seq_lens = [len(ids) for ids in token_ids]
    N = np.sum(seq_lens) # Total number of context tokens
    start_end_pos = list(zip(np.cumsum([0] + seq_lens[:-1]), np.cumsum(seq_lens)))
    names = [str(node) for node in all_nodes]
    if not materialize_attention_matrix:
        return None, all_nodes, start_end_pos, names
    attention_matrix = np.zeros((N, N))
    # Fill in each diagonal with a lower triangular attention matrix
    for i, node in enumerate(all_nodes):
        start, end = start_end_pos[i]
        attention_matrix[start:end, start:end] = np.triu(np.ones((end - start, end - start))).T
    # Connect the nodes in the atteinion matrix
    for node in all_nodes:
        for parent in node.parents:
            attention_matrix[start_end_pos[node_to_idx[node]][0]:start_end_pos[node_to_idx[node]][1], 
                                start_end_pos[node_to_idx[parent]][0]:start_end_pos[node_to_idx[parent]][1]] = 1
    return attention_matrix.astype('int'), all_nodes, start_end_pos, names

def prepare_inputs(nodes):
    assert isinstance(nodes, list)
    # Concatenate the token_ids and position_ids of all nodes
    input_ids = torch.cat([node.token_ids for node in nodes])
    position_ids = torch.cat([node.position_ids for node in nodes])
    return input_ids.unsqueeze(0), position_ids.unsqueeze(0)

def upstream_nodes(leaf_node):
    # Collect all nodes by traversing upwards from the leaf
    all_nodes = set()
    queue = deque([leaf_node])
    
    while queue:
        node = queue.popleft()
        if node not in all_nodes:
            all_nodes.add(node)
            queue.extend(node.parents)
    
    # Convert set to list for easier handling
    all_nodes = list(all_nodes)
    
    # Create a dictionary to store in-degrees for each node
    in_degree = {node: len(node.parents) for node in all_nodes}
    
    # Create a queue and add all nodes with no parents (in-degree 0)
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # For each child of the current node
        for node in all_nodes:
            if current in node.parents:
                # Decrease the in-degree of the child
                in_degree[node] -= 1
                # If the child has no more parents, add it to the queue
                if in_degree[node] == 0:
                    queue.append(node)
    
    # Check if all nodes were visited (to detect cycles)
    if len(result) != len(all_nodes):
        raise ValueError("Graph contains a cycle")
    
    return result


def is_in(node, nodes):
    for p in nodes:
        if node == p:
            return True
    return False


def print_dag(leaf_node, output_character_limit=10):
    def build_tree(node, prefix="", is_last=True):
        lines = []
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + str(node)[:output_character_limit])
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        for i, parent in enumerate(node.parents):
            lines.extend(build_tree(parent, new_prefix, i == len(node.parents) - 1))
        
        return lines

    tree_lines = build_tree(leaf_node)
    return "\n".join(tree_lines)

# Utils for flex attention support

def build_adjacency_matrix(nodes):
    node_to_index = {node.id: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # Initialize adjacency matrix with zeros
    adj_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        adj_matrix[i][i] = 1
    # Fill the adjacency matrix
    for node in nodes:
        current_idx = node_to_index[node.id]
        for parent in node.parents:
            parent_idx = node_to_index[parent.id]
            # Mark edge from parent to current node
            adj_matrix[current_idx][parent_idx] = 1
    
    return adj_matrix


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def generate_flex_attention_mask_mod(leaf_node, tokenizer):
    _, all_nodes, start_end_pos, names = build_attention_matrix(leaf_node, tokenizer)
    node_id = _offsets_to_doc_ids_tensor(torch.tensor([0] + [end for (start, end) in start_end_pos]))
    adjancey_matrix = torch.tensor(build_adjacency_matrix(all_nodes), dtype=torch.bool)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        node_mask = adjancey_matrix[node_id[q_idx], node_id[kv_idx]]
        return causal_mask & node_mask

    return document_causal_mask