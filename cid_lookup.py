# cid_lookup.py
import json
import os
from pubchempy import get_compounds

CACHE_FILE = "smiles_cache.json"

# Load existing cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        SMILES_CACHE = json.load(f)
else:
    SMILES_CACHE = {}

def cid_to_smiles(cid):
    """Convert PubChem CID → SMILES with local caching."""
    cid = str(cid)

    # Return from cache if exists
    if cid in SMILES_CACHE:
        return SMILES_CACHE[cid]

    # Query PubChem
    try:
        cmpd = get_compounds(cid, "cid")
        if len(cmpd) == 0:
            return None
        smiles = cmpd[0].canonical_smiles
    except Exception:
        return None

    # Save to cache
    SMILES_CACHE[cid] = smiles
    with open(CACHE_FILE, "w") as f:
        json.dump(SMILES_CACHE, f, indent=2)

    return smiles