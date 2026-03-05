import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from joinminer.utils.array import grouped_arange
from ..loader.base import LoaderBatch


class BilinkBatch:
    """
    Bilink batch result with device transfer support.
    """

    def __init__(
        self,
        pair_id_df: Optional[pd.DataFrame],
        label: Optional[torch.FloatTensor],
        features: Dict[str, Dict[str, Optional[torch.FloatTensor]]],
        token_to_path_group: Dict[str, torch.LongTensor],
        path_group_to_sample: torch.LongTensor,
        completed_file: Optional[str] = None
    ):
        self.pair_id_df = pair_id_df
        self.label = label
        self.features = features  # {'node': {'type': tensor/None}, 'edge': {...}}
        self.token_to_path_group = token_to_path_group  # {'embed_id': ..., 'pos_id': ...}
        self.path_group_to_sample = path_group_to_sample
        self.completed_file = completed_file

    def to_device(self, device: str) -> 'BilinkBatch':
        """
        Transfer batch to device in-place.

        Args:
            device: Device string (e.g., 'cuda:0', 'npu:1', 'cpu')

        Returns:
            self for chaining
        """

        # Transfer label if exists
        if self.label is not None:
            self.label = self.label.to(device)

        # Transfer features (skip None)
        for element_type in ['node', 'edge']:
            for type_name, tensor in self.features[element_type].items():
                if tensor is not None:
                    self.features[element_type][type_name] = tensor.to(device)

        # Transfer token_to_path_group dict
        self.token_to_path_group['embed_id'] = self.token_to_path_group['embed_id'].to(device)
        self.token_to_path_group['pos_id'] = self.token_to_path_group['pos_id'].to(device)

        # Transfer path_group_to_sample
        self.path_group_to_sample = self.path_group_to_sample.to(device)

        return self


class BilinkConverter:
    """
    Bilink batch converter.

    Converts bipath collection data into path group token format
    for BiLink model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize converter with config.

        Args:
            config: Dataset configuration containing:
                - require_labels: Whether to include labels
                - require_ids: Whether to include sample IDs
                - pair_id_mapping: ID column mapping
                - feature_embed_map: Type ID to embedding ID mapping
                - path_group_len: Path group length
                - max_path_group: Max path groups per type
                - pair: Node type configuration
                - path_hops: Hop configuration per path type
                - feature_types: Feature type configuration
        """
        # Extract common config
        self.require_ids = config["require_ids"]
        self.require_labels = config["require_labels"]
        self.graph = config["graph"]
        self.pair = config["pair"]
        self.path = config["path"]
        self.path_group_len = config["path_group_len"]
        self.max_path_group = config["max_path_group"]

    def convert(self, loader_batch: LoaderBatch) -> BilinkBatch:
        """
        Convert LoaderBatch into model input.

        Features are now embedded directly in batch_df with columns:
        - node_{type_name}_feat_id_array
        - node_{type_name}_feat_vector_array
        - edge_{type_name}_feat_id_array
        - edge_{type_name}_feat_vector_array

        Args:
            loader_batch: LoaderBatch from batch loader

        Returns:
            BilinkBatch object with to_device() method
        """
        batch_df = loader_batch.data

        # Stage 1: Extract pair_id_df
        pair_id_df = None
        if self.require_ids:
            id_columns = ['u_node_id', 'v_node_id'] + self.graph.partition_columns
            pair_id_df = batch_df[id_columns]

        # Stage 2: Extract label
        label = None
        if self.require_labels:
            label = torch.FloatTensor(np.vstack(batch_df['label'].to_numpy()))

        # Stage 3: Extract unique features for each element type
        features_map = {'node': {}, 'edge': {}}
        features = {'node': {}, 'edge': {}}
        embed_offset = 4  # First 4 are special tokens (padding, cls_forward, cls_backward, cls_bipath)

        for element_type in ['node', 'edge']:
            elements = self.graph.nodes if element_type == 'node' else self.graph.edges
            for type_name in self.path[element_type]:
                type_info = elements[type_name]
                if type_info['feature_count'] > 0:
                    feat_ids, feat_vectors = self._extract_unique_features(
                        batch_df, element_type, type_name
                    )
                    features_map[element_type][type_name] = {
                        'offset': embed_offset,
                        'ids': feat_ids
                    }
                    features[element_type][type_name] = torch.FloatTensor(feat_vectors)
                    embed_offset += len(feat_ids)
                else:
                    # feature_count == 0: only set offset as placeholder
                    features_map[element_type][type_name] = {
                        'offset': embed_offset
                    }
                    features[element_type][type_name] = None
                    embed_offset += 1

        # Stage 4: Get paths with path_group assignment
        path_df, path_group_df, batch_path_group_len = self._map_path_to_path_group(batch_df)

        # Stage 5: Map tokens to path groups
        token_to_path_group, total_path_groups = self._map_token_to_path_group(
            batch_df, path_df, path_group_df, batch_path_group_len, features_map
        )

        # Stage 6: Map path groups to samples
        path_group_to_sample = self._map_path_group_to_sample(
            batch_df, path_group_df, total_path_groups
        )

        return BilinkBatch(
            pair_id_df=pair_id_df,
            label=label,
            features=features,
            token_to_path_group=token_to_path_group,
            path_group_to_sample=path_group_to_sample,
            completed_file=loader_batch.completed_file
        )

    def _explode_paths(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode batch_df to get individual paths.

        Args:
            batch_df: Batch DataFrame with array columns

        Returns:
            path_df with columns:
            - sample_id: Original sample index
            - path_type: 0=forward, 1=backward, 2=bipath
            - hop_k: Number of hops
            - node_type_array: Node types in path
            - node_feat_id_array: Node feature IDs
            - edge_type_array: Edge types in path
            - u_index_array: Edge u indices
            - v_index_array: Edge v indices
            - edge_feat_id_array: Edge feature IDs
        """
        # Array columns to explode
        array_columns = [
            'path_type_array', 'hop_k_array',
            'paths_node_type_array', 'paths_node_feat_id_array',
            'paths_edge_type_array', 'paths_u_index_array',
            'paths_v_index_array', 'paths_edge_feat_id_array'
        ]
        path_columns = ['path_count'] + array_columns

        # Select and copy
        df = batch_df[path_columns].copy()
        df['sample_id'] = np.arange(len(df))

        # Filter and drop path_count
        df = df[df['path_count'] > 0]
        df = df.drop(columns=['path_count'])

        # Explode all array columns together
        df = df.explode(array_columns)

        # Rename columns (remove prefixes and _array suffix for scalar columns)
        rename_map = {
            'path_type_array': 'path_type',
            'hop_k_array': 'hop_k',
            'paths_node_type_array': 'node_type_array',
            'paths_node_feat_id_array': 'node_feat_id_array',
            'paths_edge_type_array': 'edge_type_array',
            'paths_u_index_array': 'u_index_array',
            'paths_v_index_array': 'v_index_array',
            'paths_edge_feat_id_array': 'edge_feat_id_array'
        }
        df = df.rename(columns=rename_map)

        df['path_type'] = df['path_type'].astype(np.int64)
        df['hop_k'] = df['hop_k'].astype(np.int64)

        return df.reset_index(drop=True)

    def _map_path_to_path_group(self, batch_df: pd.DataFrame) -> tuple:
        """
        Map paths to path groups.

        Args:
            batch_df: Batch DataFrame with array columns

        Returns:
            (path_df, path_group_df):
            - path_df: DataFrame with columns sample_id, path_type, hop_k, path_group_id,
              token_len, token_offset, path_pos, and array columns
            - path_group_df: DataFrame with columns sample_id, path_type, path_group_count
        """
        # Get individual paths from batch_df
        path_df = self._explode_paths(batch_df)

        # Compute token_len for each path
        # forward/backward: 2 * hop_k, bipath: 2 * hop_k - 1
        path_df['token_len'] = np.where(
            path_df['path_type'] == 2,
            2 * path_df['hop_k'] - 1,
            2 * path_df['hop_k']
        )

        # Sort by sample_id, path_type, token_len
        path_df = path_df.sort_values(['sample_id', 'path_type', 'token_len'])

        # Group by sample_id, path_type (reuse grouped object)
        by_path_type = path_df.groupby(['sample_id', 'path_type'])
        path_df['path_group_id'] = by_path_type.cumcount()
        path_df['type_group_idx'] = by_path_type.ngroup()  # for global path_group_id calculation
        path_group_df = by_path_type.size().reset_index(name='path_count')

        # Convert max_path_group config
        max_path_group_by_type = {
            0: self.max_path_group['forward'],
            1: self.max_path_group['backward'],
            2: self.max_path_group['bipath']
        }

        # Apply modulo to limit paths per type
        path_df['path_group_id'] = path_df['path_group_id'] % path_df['path_type'].map(max_path_group_by_type)

        # Calculate actual path_group_count per (sample_id, path_type)
        max_path_group_series = path_group_df['path_type'].map(max_path_group_by_type)
        path_group_df['path_group_count'] = np.minimum(path_group_df['path_count'], max_path_group_series)
        path_group_df = path_group_df.drop(columns=['path_count'])

        # Calculate token end position (for filtering)
        # Base: bipath=3 (cls + u_node + v_node), forward/backward=2 (cls + start_node)
        path_df['token_end_pos'] = np.where(path_df['path_type'] == 2, 3, 2)

        # Group by sample_id, path_type, path_group_id (reuse grouped object)
        by_path_group = path_df.groupby(['sample_id', 'path_type', 'path_group_id'])
        path_df['token_end_pos'] += by_path_group['token_len'].cumsum()
        path_df['path_pos'] = by_path_group.cumcount() + 1

        # Filter paths that exceed path_group_len
        path_df = path_df[path_df['token_end_pos'] <= self.path_group_len].copy()

        # Get actual max token length for this batch
        batch_path_group_len = path_df['token_end_pos'].max()

        # Calculate token_offset and drop token_end_pos
        path_df['token_offset'] = path_df['token_end_pos'] - path_df['token_len']
        path_df = path_df.drop(columns=['token_end_pos'])

        # Convert path_group_id to global index using ngroup() result
        path_group_offsets = np.concatenate([[0], path_group_df['path_group_count'].values.cumsum()[:-1]])
        path_df['path_group_id'] += path_group_offsets[path_df['type_group_idx']]
        path_df = path_df.drop(columns=['type_group_idx'])

        return path_df, path_group_df, batch_path_group_len

    def _map_token_to_path_group(
        self,
        batch_df: pd.DataFrame,
        path_df: pd.DataFrame,
        path_group_df: pd.DataFrame,
        batch_path_group_len: int,
        features_map: dict
    ):
        """
        Map all tokens to path groups.

        Args:
            batch_df: Batch DataFrame with array columns
            path_df: Path DataFrame from _map_path_to_path_group
            path_group_df: Path group DataFrame from _map_path_to_path_group
            batch_path_group_len: Actual max token length for this batch
            features_map: Feature map with offset, ids per type (for embed_id conversion)
        """
        # Initialize token_to_path_group as dict
        # embed_id: (total_path_groups, batch_path_group_len) - token embedding IDs
        # pos_id: (total_path_groups, batch_path_group_len, 4) - [path_pos, node_pos, u_node_pos, v_node_pos]
        total_path_groups = path_group_df['path_group_count'].sum()
        token_to_path_group = {
            'embed_id': np.zeros((total_path_groups, batch_path_group_len), dtype=np.int64),
            'pos_id': np.zeros((total_path_groups, batch_path_group_len, 4), dtype=np.int64)
        }

        # Add cls tokens to first position (column 0)
        # cls token value = path_type + 1, pos_id stays all zeros
        cls_tokens = np.repeat(path_group_df['path_type'].values + 1, path_group_df['path_group_count'].values)
        token_to_path_group['embed_id'][:, 0] = cls_tokens

        # Fill pair node tokens (columns 1 and 2)
        self._fill_pair_token(token_to_path_group, batch_df, path_group_df, features_map)

        # Fill path node tokens
        self._fill_path_node_token(token_to_path_group, path_df, features_map)

        # Fill path edge tokens
        self._fill_path_edge_token(token_to_path_group, path_df, features_map)

        # Convert to tensors
        token_to_path_group = {
            'embed_id': torch.LongTensor(token_to_path_group['embed_id']),
            'pos_id': torch.LongTensor(token_to_path_group['pos_id'])
        }
        return token_to_path_group, total_path_groups

    def _feat_id_to_embed_id(
        self,
        feat_ids: np.ndarray,
        element_type: str,
        type_name: str,
        features_map: dict
    ) -> np.ndarray:
        """
        Convert feat_id to embed_id.

        Args:
            feat_ids: Array of feature IDs
            element_type: 'node' or 'edge'
            type_name: Type name (e.g., 'author', 'work')
            features_map: Feature map dictionary with offset and ids

        Returns:
            Array of embed_ids
        """
        type_features = features_map[element_type][type_name]
        if 'ids' in type_features:
            return type_features['offset'] + np.searchsorted(type_features['ids'], feat_ids)
        else:
            return np.full(len(feat_ids), type_features['offset'], dtype=np.int64)

    def _fill_pair_token(
        self,
        token_to_path_group: dict,
        batch_df: pd.DataFrame,
        path_group_df: pd.DataFrame,
        features_map: dict
    ):
        """
        Fill pair node embed_id and pos_id to token_to_path_group columns 1 and 2.

        - Column 1: u_node for forward/bipath, v_node for backward
        - Column 2: v_node for bipath only
        """
        # Extract and expand to token level (same length as token_to_path_group rows)
        path_group_counts = path_group_df['path_group_count'].values
        sample_ids = np.repeat(path_group_df['sample_id'].values, path_group_counts)
        path_types = np.repeat(path_group_df['path_type'].values, path_group_counts)

        u_node_type = self.pair['u_node_type']
        v_node_type = self.pair['v_node_type']

        # Get feat_id and convert to embed_id
        u_feat_ids = batch_df['u_node_feat_id'].values
        v_feat_ids = batch_df['v_node_feat_id'].values
        u_embed_ids = self._feat_id_to_embed_id(u_feat_ids, 'node', u_node_type, features_map)
        v_embed_ids = self._feat_id_to_embed_id(v_feat_ids, 'node', v_node_type, features_map)

        # Column 1: u_node for forward/bipath, v_node for backward
        token_to_path_group['embed_id'][:, 1] = np.where(
            path_types == 1,
            v_embed_ids[sample_ids],
            u_embed_ids[sample_ids]
        )
        token_to_path_group['pos_id'][:, 1, 1] = 1  # node_pos=1

        # Column 2: v_node for bipath only
        bipath_mask = path_types == 2
        token_to_path_group['embed_id'][bipath_mask, 2] = v_embed_ids[sample_ids[bipath_mask]]
        token_to_path_group['pos_id'][bipath_mask, 2, 1] = 2  # node_pos=2

    def _fill_path_node_token(
        self,
        token_to_path_group: dict,
        path_df: pd.DataFrame,
        features_map: dict
    ):
        """
        Fill path node embed_id and pos_id to token_to_path_group.

        For each path, fills node tokens starting from token_offset position.
        """
        # Select necessary columns (include hop_k for node_counts calculation)
        node_df = path_df[['path_group_id', 'path_type', 'path_pos', 'token_offset',
                          'hop_k', 'node_type_array', 'node_feat_id_array']].copy()

        # Calculate node_counts and node_token_offset before explode
        # node count: forward/backward = hop_k, bipath = hop_k - 1
        node_counts = np.where(
            node_df['path_type'].values == 2,
            node_df['hop_k'].values - 1,
            node_df['hop_k'].values
        )
        node_token_offset = grouped_arange(node_counts, start=0)

        # Explode node_type_array and node_feat_id_array together
        node_df = node_df.explode(['node_type_array', 'node_feat_id_array'])

        # Rename columns and reset index
        node_df = node_df.rename(columns={
            'node_type_array': 'node_type',
            'node_feat_id_array': 'node_feat_id'
        })
        node_df = node_df.reset_index(drop=True)

        # Convert node_feat_id to embed_id by node_type
        node_df['embed_id'] = 0  # Initialize column
        for type_name in self.path['node']:
            type_index = self.graph.nodes[type_name]['type_index']
            mask = node_df['node_type'] == type_index
            node_df.loc[mask, 'embed_id'] = self._feat_id_to_embed_id(
                node_df.loc[mask, 'node_feat_id'].values,
                'node', type_name, features_map
            )

        # Calculate target position in token_to_path_group
        token_col = node_df['token_offset'].values + node_token_offset
        path_group_ids = node_df['path_group_id'].values

        # Fill embed_id
        token_to_path_group['embed_id'][path_group_ids, token_col] = node_df['embed_id'].values

        # Calculate node_pos: path_type 0/1 -> offset+2, path_type 2 -> offset+3
        node_pos = np.where(
            node_df['path_type'].values == 2,
            node_token_offset + 3,
            node_token_offset + 2
        )

        # Fill pos_id: [path_pos, node_pos, 0, 0]
        token_to_path_group['pos_id'][path_group_ids, token_col, 0] = node_df['path_pos'].values
        token_to_path_group['pos_id'][path_group_ids, token_col, 1] = node_pos

    def _fill_path_edge_token(
        self,
        token_to_path_group: dict,
        path_df: pd.DataFrame,
        features_map: dict
    ):
        """
        Fill path edge embed_id and pos_id to token_to_path_group.

        For each path, fills edge tokens after node tokens.
        """
        # Select necessary columns (keep hop_k for token_col calculation)
        edge_df = path_df[['path_group_id', 'path_type', 'path_pos', 'token_offset', 'hop_k',
                           'edge_type_array', 'edge_feat_id_array',
                           'u_index_array', 'v_index_array']].copy()

        # Calculate edge_counts and edge_token_offset before explode
        # edge count = hop_k for all path types
        edge_counts = edge_df['hop_k'].values
        edge_token_offset = grouped_arange(edge_counts, start=0)

        # Explode all edge arrays together
        edge_df = edge_df.explode(['edge_type_array', 'edge_feat_id_array',
                                   'u_index_array', 'v_index_array'])

        # Rename columns
        edge_df = edge_df.rename(columns={
            'edge_type_array': 'edge_type',
            'edge_feat_id_array': 'edge_feat_id',
            'u_index_array': 'u_index',
            'v_index_array': 'v_index'
        })
        edge_df = edge_df.reset_index(drop=True)

        # Convert edge_feat_id to embed_id by edge_type
        edge_df['embed_id'] = 0  # Initialize column
        for type_name in self.path['edge']:
            type_index = self.graph.edges[type_name]['type_index']
            mask = edge_df['edge_type'] == type_index
            edge_df.loc[mask, 'embed_id'] = self._feat_id_to_embed_id(
                edge_df.loc[mask, 'edge_feat_id'].values,
                'edge', type_name, features_map
            )

        # Calculate node_token_count: path_type 0/1 -> hop_k, path_type 2 -> hop_k - 1
        node_token_count = np.where(
            edge_df['path_type'].values == 2,
            edge_df['hop_k'].values - 1,
            edge_df['hop_k'].values
        )

        # Calculate target position in token_to_path_group
        # token_col = token_offset + node_token_count + edge_token_offset
        token_col = edge_df['token_offset'].values + node_token_count + edge_token_offset
        path_group_ids = edge_df['path_group_id'].values

        # Fill embed_id
        token_to_path_group['embed_id'][path_group_ids, token_col] = edge_df['embed_id'].values

        # Calculate u_node_pos and v_node_pos
        # First, add 1 to u_index and v_index
        u_pos = edge_df['u_index'].values + 1
        v_pos = edge_df['v_index'].values + 1

        # For bipath (path_type 2), special handling:
        # - index 1 stays unchanged (becomes 2 after +1)
        # - index 2 to hop_k-1: add 1 more
        # - index hop_k: becomes 2
        bipath_mask = edge_df['path_type'].values == 2
        hop_k = edge_df['hop_k'].values

        # u_pos adjustments
        u_mid_mask = bipath_mask & (u_pos >= 3) & (u_pos <= hop_k)  # original index 2 to hop_k-1
        u_end_mask = bipath_mask & (u_pos == hop_k + 1)  # original index hop_k
        u_pos = np.where(u_mid_mask, u_pos + 1, u_pos)
        u_pos = np.where(u_end_mask, 2, u_pos)

        # v_pos adjustments (same logic)
        v_mid_mask = bipath_mask & (v_pos >= 3) & (v_pos <= hop_k)
        v_end_mask = bipath_mask & (v_pos == hop_k + 1)
        v_pos = np.where(v_mid_mask, v_pos + 1, v_pos)
        v_pos = np.where(v_end_mask, 2, v_pos)

        # Fill pos_id: [path_pos, 0, u_node_pos, v_node_pos]
        token_to_path_group['pos_id'][path_group_ids, token_col, 0] = edge_df['path_pos'].values
        token_to_path_group['pos_id'][path_group_ids, token_col, 2] = u_pos
        token_to_path_group['pos_id'][path_group_ids, token_col, 3] = v_pos

    def _extract_unique_features(
        self,
        batch_df: pd.DataFrame,
        element_type: str,
        type_name: str
    ) -> tuple:
        """
        Extract unique features for a single element type.

        Args:
            batch_df: Batch DataFrame
            element_type: 'node' or 'edge'
            type_name: Element type name (e.g., 'author', 'work_author')

        Returns:
            (feat_ids, feat_vectors): numpy arrays, sorted by feat_id ascending
        """
        count_col = f'{element_type}_{type_name}_count'
        feat_id_col = f'{element_type}_{type_name}_feat_id_array'
        feat_vector_col = f'{element_type}_{type_name}_feat_vector_array'

        # Filter rows with count > 0
        valid_df = batch_df[batch_df[count_col] > 0]

        if len(valid_df) == 0:
            return np.array([], dtype=np.int64), np.array([]).reshape(0, -1)

        # Expand feat_id and feat_vector arrays (synchronized)
        all_feat_ids = np.concatenate(valid_df[feat_id_col].values)
        all_feat_vectors = np.concatenate([np.vstack(row) for row in valid_df[feat_vector_col].values])

        # Get unique feat_ids and first occurrence indices (np.unique returns sorted)
        unique_ids, first_indices = np.unique(all_feat_ids, return_index=True)
        unique_vectors = all_feat_vectors[first_indices]

        return unique_ids, unique_vectors

    def _map_path_group_to_sample(
        self,
        batch_df: pd.DataFrame,
        path_group_df: pd.DataFrame,
        total_path_groups: int
    ) -> torch.LongTensor:
        """
        Map path groups to samples.

        Creates a matrix where each row is a sample, and columns contain
        path_group indices for that sample.

        Args:
            batch_df: Batch DataFrame
            path_group_df: Path group DataFrame with sample_id and path_group_count
            total_path_groups: Total number of path groups (from _map_token_to_path_group)

        Returns:
            path_group_to_sample: (sample_count, batch_path_group_count + 1) matrix
                Column 0 is cls (value=1), remaining columns are path_group indices
        """
        # Calculate path_group count per sample
        sample_path_group_count = path_group_df.groupby('sample_id')['path_group_count'].sum()

        # Get max as batch_path_group_count
        batch_path_group_count = sample_path_group_count.max()

        # Initialize matrix (sample_count, batch_path_group_count + 1)
        # +1 for cls column
        sample_count = len(batch_df)
        path_group_to_sample = np.zeros((sample_count, batch_path_group_count + 1), dtype=np.int64)

        # Column 0 is cls (value=1)
        path_group_to_sample[:, 0] = 1

        # Generate path_group global indices
        path_group_ids = 2 + np.arange(total_path_groups)

        # Expand sample_ids
        sample_ids = np.repeat(
            sample_path_group_count.index.values,
            sample_path_group_count.values
        )

        # Generate col_indices (1-indexed, skip cls column)
        col_indices = grouped_arange(sample_path_group_count.values, start=1)

        # Fill matrix: row=sample_id, col=col_indices, value=path_group_ids
        path_group_to_sample[sample_ids, col_indices] = path_group_ids

        return torch.LongTensor(path_group_to_sample)
