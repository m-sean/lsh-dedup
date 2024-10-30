from typing import Dict, List, Optional

class MinHash:
    """A MinHash instance. Can only be generated within a MinHashLSH instance."""

    def jaccard_similarity(self, other: MinHash) -> float:
        pass

class MinHashLSH:
    """A locality sensitive hashing instance using min-wise independent permutations."""

    def __init__(self, records: List[str], num_perm: int, num_bands: int):
        pass

    def get_minhash_index(self) -> Dict[int, MinHash]:
        pass

    def query(self, minhash: MinHash, threshold: Optional[float]) -> List[int]:
        pass

class DeduplicationIndex:
    """A clustering manager for MinHashLSH. Queries the hashtables for all records at a given 
    threshold to form groups of duplicate documents' indices.
    """

    def __init__(self, lsh: MinHashLSH, threshold: Optional[float]):
        pass

    def grouped_indices(self) -> List[List[int]]:
        pass