import json
import pickle as pkl
from random import choice
from itertools import permutations, chain
from termcolor import colored
from tqdm.auto import tqdm
import os

class KG():
    def __init__(self, kg):
        super().__init__()
        self.kg = kg
        
    def search(self, ents, rels):
        connected = list()
        walkable = list()
        seen = dict()
        
        for e in ents:
            if e in rels:
                for path in rels[e]:
                    leaf = ents[:]
                    leaf.remove(e)
                    result = self.walk(start=e, path=path, ends=leaf)
                    if result != (None, None):
                        if result[1] is not None:
                            query = str(sorted([result[1][0], result[1][-1]]))
                            if query not in seen:
                                conn_with_rel = result[1][:1]+list(chain(*[[r, e] for r, e in zip(path, result[1][1:])]))
                                connected.append(conn_with_rel)
                                seen[query] = None
                        if result[0][0] != result[0][-1]:
                            query = str(sorted([result[0][0], result[0][-1]]))
                            if query not in seen:
                                walk_with_rel = result[0][:1]+list(chain(*[[r, e] for r, e in zip(path, result[0][1:])]))
                                walkable.append(walk_with_rel)
                                seen[query] = None
                
        return {"connected":connected, "walkable":walkable}
                
    def walk(self, start, path, ends=None):
        branches = [[start,],]
        for r in path:
            updated_branches = list()
            for branch in branches:
                h = branch[-1]
                ts = self.get_tail(h, r)
                if (r == path[-1]) and ts:
                    rand_branch = branch+[choice(list(ts.keys())),]
                    for e in ends:
                        if e in ts:
                            return rand_branch, branch+[e,]
                    return rand_branch, None
                else:
                    if ts:
                        for t in ts:
                            updated_branches.append(branch+[t,])
            if len(updated_branches) <= len(branches):
                return None, None
            branches = updated_branches

    def get_tail(self, h, r):
        if h in self.kg:
            if r in self.kg[h]:
                return {x:None for x in self.kg[h][r]}
            else:
                return {}
        else:
            return {}    

def prepare_input(data_path, kg_path):
    with open(kg_path, "rb") as pkf:
        raw_kg = pkl.load(pkf)
    kg = KG(raw_kg)

    search_results = dict()

    with open(os.path.join(data_path, 'factkg_train.pickle'), "rb") as pkf:
        db = pkl.load(pkf)

    for i, (claim, elem) in tqdm(enumerate(db.items()), total=len(db)):
        ents = elem["Entity_set"]
        rels = elem["Evidence"]
        search_results[claim] = kg.search(ents, rels)
            
    assert len(search_results)==len(db)
    with open("./train_candid_paths.bin", "wb") as pkf:
        pkl.dump(search_results, pkf)