import numpy as np
import time
import math
from multiprocessing.pool import ThreadPool
import itertools
import time 
from lolbo.utils.mol_utils.moses_metrics.SA_Score import sascorer 
try: # for tdc docking 
    from tdc import Oracle
except: 
    print("Warning: Failed to import tdc docking oracle, only needed for molecule docking tasks")

from rdkit import Chem
from rdkit.Chem import Crippen
import networkx as nx
from rdkit.Chem import rdmolops
from rdkit.Chem.QED import qed
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import FoldFingerprint
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from guacamol import standard_benchmarks 

med1 = standard_benchmarks.median_camphor_menthol() #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil() #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings() # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO' 
siga = standard_benchmarks.sitagliptin_replacement() #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula() # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop() # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop() # Scaffold Hop'
rano= standard_benchmarks.ranolazine_mpo() #'Ranolazine MPO' 
fexo = standard_benchmarks.hard_fexofenadine() # 'Fexofenadine MPO'... 'make fexofenadine less greasy'

guacamol_objs = {"med1":med1,"pdop":pdop, "adip":adip, "rano":rano, "osmb":osmb,
        "siga":siga, "zale":zale, "valt":valt, "med2":med2,"dhop":dhop, "shop":shop, 
        'fexo':fexo} 
logP_mean = 2.4399606244103639873799239
logP_std = 0.9293197802518905481505840
SA_mean = -2.4485512208785431553792478
SA_std = 0.4603110476923852334429910
cycle_mean = -0.0307270378623088931402396
cycle_std = 0.2163675785228087178335699

GUACAMOL_TASK_NAMES = [
    'med1', 'pdop', 'adip', 'rano', 'osmb', 'siga',
    'zale', 'valt', 'med2', 'dhop', 'shop', 'fexo'
]

def smile_is_valid_mol(smile):
    if smile is None or len(smile)==0:
        return False
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return False
    return True


def smile_to_guacamole_score(obj_func_key, smile):
    if smile is None or len(smile)==0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func.objective.score(smile)
    if score is None:
        return None
    if score < 0:
        return None
    return score 


def smile_to_rdkit_mol(smile): return Chem.MolFromSmiles(smile)
vectorized_smiles_arr_to_mols_arr = np.vectorize(smile_to_rdkit_mol) 


def smile_to_QED(smile):
    """
    Computes RDKit's QED score
    """
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    qed_score = qed(mol)
    return qed_score

def smile_to_sa(smile):
    """Synthetic Accessibility Score (SA): 
    a heuristic estimate of how hard (10) 
    or how easy (1) it is to synthesize a given molecule.""" 
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def smile_to_penalized_logP(smile):
    """ calculate penalized logP for a given smiles string """
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    logp = Crippen.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    cycle_length = _cycle_score(mol)
    
    if logp >= 10:
        return -float("inf")
    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    # score = (
    #         (logp - 2.45777691) / 1.43341767
    #         + (-sa + 3.05352042) / 0.83460587
    #         + (-cycle_length - -0.04861121) / 0.28746695
    # )
    score = (
            (logp - logP_mean) / logP_std
            + (-sa - SA_mean) / SA_std
            + (-cycle_length - cycle_mean) / cycle_std
    )
    return max(score, -float("inf"))

def smiles_ls_to_penalized_logP(smiles_list):
    mol_list = []
    for idx,smile in enumerate(smiles_list):
        if smile is None:
            mol_list.append(None)
            continue
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            mol_list.append(None)
            continue
        else:
            mol_list.append(mol)
    assert len(mol_list) == len(smiles_list)
    logp = np.array([Crippen.MolLogP(mol) if mol is not None else np.nan for mol in mol_list])
    sa = np.array([sascorer.calculateScore(mol) if mol is not None else np.nan for mol in mol_list])
    cycle_length = np.array([_cycle_score(mol) if mol is not None else np.nan for mol in mol_list])
    
    # logp = np.where(logp >= 10, np.nan, logp)
    logp = np.ma.array(logp, mask=np.isnan(logp))
    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    # score = (
    #         (logp - 2.45777691) / 1.43341767
    #         + (-sa + 3.05352042) / 0.83460587
    #         + (-cycle_length - -0.04861121) / 0.28746695
    # )
    score = (
            (logp - logP_mean) / logP_std
            + (-sa - SA_mean) / SA_std
            + (-cycle_length - cycle_mean) / cycle_std
    )
    return score.data

def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def setup_tdc_oracle(protien_name):
    oracle = Oracle(name=protien_name)
    return oracle


def smile_to_tdc_docking_score(smiles_str, tdc_oracle, max_smile_len=600, timeout=600):
    # goal of function:
    #          return docking score (score = tdc_oracle(smiles_str) ) iff it can be computed within timeout seconds
    #           otherwise, return None
    if not smile_is_valid_mol(smiles_str):
        return None
    smiles_str = Chem.CanonSmiles(smiles_str)
    if len(smiles_str) > max_smile_len:
        return None
    start = time.time()

    def get_the_score(smiles_str):
        docking_score = tdc_oracle(smiles_str)
        return docking_score

    pool = ThreadPool(1)

    async_result = pool.apply_async(get_the_score, (smiles_str,))
    from multiprocessing.context import TimeoutError
    try:
        ret_value = async_result.get(timeout=timeout)
    except Exception as e:
        print("Error occurred getting score from smiles str::",smiles_str,  e)
        # print('TimeoutError encountered getting docking score for smiles_str:', smiles_str)
        ret_value = None

    print(f"getting docking score: {ret_value} from protein took {time.time()-start} seconds")
    return ret_value

import sys,os
import pandas as pd
sys.path.append('%s/../../../QSAR-VEGFR2/utils/qsar' % os.path.dirname(os.path.realpath(__file__)))
print('%s/../../../QSAR-VEGFR2/mlqsar' % os.path.dirname(os.path.realpath(__file__)))
import pickle
from Prediction import predict
SAVE_FITTED_PIPELINE = '%s/../../../QSAR-VEGFR2/raw_data_features/VEGFR2/pipeline' % os.path.dirname(os.path.realpath(__file__))

    
def smiles_to_pIC50(smile):
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    
    fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=4096, nBitsPerHash=2)
    
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    
    temp = pd.DataFrame(ar).T
    temp['ID'] = ['temp']
    pred = predict(materials_path = SAVE_FITTED_PIPELINE,
               data = temp, ID = 'ID')
    temp_rp = pred.predict(verbose = False)
    y_pre = temp_rp['Predict'].values[0]
    
    if 0 <= y_pre <= 10:
        return y_pre
    else:
        return None

def smiles_ls_to_pIC50(smiles_list):
    def RDKFp(mol, maxPath=7, fpSize=4096, nBitsPerHash=2):
        fp = Chem.RDKFingerprint(mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    valid_mol_list = []
    valid_idx = []
    result = np.array([np.nan]*len(smiles_list))
    for idx,smile in enumerate(smiles_list):
        if smile is None:
            continue
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        else:
            valid_mol_list.append(mol)
            valid_idx.append(idx)
    if len(valid_mol_list) == 0:
        return result
    
    temp = pd.DataFrame({'ID':valid_idx,'mol':valid_mol_list})
    X = np.stack(temp['mol'].apply(RDKFp).values)
    data_X = pd.concat([temp['ID'],pd.DataFrame(X)], axis = 1)
    
    pred = predict(materials_path = SAVE_FITTED_PIPELINE,
            data = data_X, ID = 'ID')
    temp_rp = pred.predict(verbose = True)
    y_pred = temp_rp['Predict'].values

    y_pred = [y if 0 <= y <= 10 else None for y in y_pred]
    assert len(y_pred) == len(valid_mol_list)
    
    result[valid_idx] = y_pred
    
    return result

def smiles_to_dual(smile):
    pchem = smiles_to_pIC50(smile)
    if (pchem is None) or (not math.isfinite(pchem)):
        return -float("inf")
    penlogp = smile_to_penalized_logP(smile)
    if (penlogp is None) or (not math.isfinite(penlogp)):
        return -float("inf")
    pchem_norm = pchem/10*2-1
    penlogp_norm = (penlogp - (-10))/(10-(-10))*2-1
    dual = penlogp_norm + pchem_norm
    return dual

def smiles_ls_to_dual(smiles_list):
    pchem = smiles_ls_to_pIC50(smiles_list)
    pchem = np.ma.array(pchem, mask=np.isnan(pchem))
    penlogp = smiles_ls_to_penalized_logP(smiles_list)
    penlogp = np.ma.array(penlogp, mask=np.isnan(penlogp))
    pchem_norm = pchem/10*2-1
    penlogp_norm = (penlogp - (-10))/(10-(-10))*2-1
    dual = penlogp_norm + pchem_norm
    return dual.data

def smiles_to_desired_scores(smiles_list, task_id="logp" ):
    scores = [] 
    if task_id == 'pic50':
        score_ = smiles_ls_to_pIC50(smiles_list)
        return np.array(score_) 
    elif task_id == 'dual':
        score_ = smiles_ls_to_dual(smiles_list)
        return np.array(score_) 
    for smiles_str in smiles_list:
        if task_id == "logp":
            score_ = smile_to_penalized_logP(smiles_str)
        elif task_id == "qed":
            score_ = smile_to_QED(smiles_str)
        else: # otherwise, assume it is a guacamol task
            score_ = smile_to_guacamole_score(task_id, smiles_str)
        if (score_ is not None) and (math.isfinite(score_) ):
            scores.append(score_) 
        else:
            scores.append(np.nan)

    return np.array(scores) 


def get_fingerprint_similarity(smile1, smile2): 
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    if (mol1 is None) or (mol2 is None): 
        print("one of the input smiles is not a valid molecule!") 
        return None
    fp1 = FingerprintMols.FingerprintMol(mol1) 
    fp2 = FingerprintMols.FingerprintMol(mol2)
    fps = DataStructs.FingerprintSimilarity(fp1,fp2) 
    return fps
    # TIMING get_fingerprint_similarity: 
    # time to convert each smile to fp: 0.0023517608642578125 (SMILE --> MOL --> FP)
    # time to compute FPS from fp1, fp2: 0.000148  (fps = DataStructs.FingerprintSimilarity(fp1,fp2) )
    # Percentage of Time Used to convert each smile to fp: 0.9408622663105685
    # Percentage of Time Used Compute FPS:0.05913773368943152
    # total time: 0.0024995803833007812


def get_fp_and_fpNbits_from_smile(smile1):
    mol1 = Chem.MolFromSmiles(smile1)
    if mol1 is None:
        return (None, None) 
    fp1 = FingerprintMols.FingerprintMol(mol1) 
    sz1 = fp1.GetNumBits() 
    return (fp1, sz1) 


def get_fps_efficient(fp1, sz1, fp2, sz2): 
    # https://github.com/rdkit/rdkit/blob/master/rdkit/DataStructs/__init__.py 
    if sz1 < sz2:
        fp2 = FoldFingerprint(fp2, sz2 // sz1)
    elif sz2 < sz1:
        fp1 = FoldFingerprint(fp1, sz1 // sz2)
    return TanimotoSimilarity(fp1, fp2) # FPS! 


def get_fps_to_list_of_targets(fp1, sz1, target_fps, target_szs):
    return np.array([get_fps_efficient(fp1, sz1, target_fps[ix], target_szs[ix]) for ix in range(len(target_fps))])
    # TEST SCRIPT: 
        # target_fps = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smile1)) for smile1 in train_smiles[0:100_000] ]
        # target_szs = [fp.GetNumBits() for fp in target_fps]  
        # et_fps_to_list_of_targets(fp1, sz1, target_fps, target_szs) 

def efficient_get_all_pairwise_fps(fp_nbits_list):
    all_pairs = list(itertools.combinations(fp_nbits_list, r=2)) 
    fpss = [] 
    for fps_nbits_pair in all_pairs: 
        fp1_nb1, fp2_nb2 = fps_nbits_pair[0], fps_nbits_pair[1]
        fpss.append(get_fps_efficient(fp1_nb1[0], fp1_nb1[1], fp2_nb2[0], fp2_nb2[1]) )
    return np.array(fpss) 



def get_all_pairwise_fps(smiles_list, return_fp_nbits=False): 
    mol_list = [Chem.MolFromSmiles(smile1) for smile1 in smiles_list] 
    if None in mol_list: # if one of smiles can't convert to mol
        if return_fp_nbits:
            return None, None
        return None
    fp_list = [FingerprintMols.FingerprintMol(mol1) for mol1 in mol_list] 
    sz_list = [fp1.GetNumBits() for fp1 in fp_list] 
    if return_fp_nbits:
        fps_nbits = []
        for fp, sz in zip(fp_list, sz_list):
            fps_nbits.append((fp, sz))
    all_idxs = [i for i in range(len(smiles_list))]
    all_pairs_idxs = list(itertools.combinations(all_idxs, r=2)) 
    fpss = [] 
    for ix_pair in all_pairs_idxs: 
        ix1, ix2 = ix_pair[0], ix_pair[1]
        fps = get_fps_efficient(fp_list[ix1], sz_list[ix1], fp_list[ix2], sz_list[ix2]) 
        if fps is not None: 
            fpss.append(fps) 
    if return_fp_nbits:
        return np.array(fpss), fps_nbits
    return np.array(fpss) 


def get_pairwise_edit_distances(smiles_list1, smiles_list2):
    edit_dists = []
    for s1, s2 in zip(smiles_list1, smiles_list2):
        edit_dists.append(smiles_edit_distance(s1, s2)) 
    return np.array(edit_dists)


def get_all_pairwise_edit_dists(smiles_list): 
    all_idxs = [i for i in range(len(smiles_list))]
    all_pairs_idxs = list(itertools.combinations(all_idxs, r=2)) 
    dists = [] 
    for ix_pair in all_pairs_idxs: 
        ix1, ix2 = ix_pair[0], ix_pair[1]
        edit_dist = smiles_edit_distance(smiles_list[ix1], smiles_list[ix2])
        dists.append(edit_dist) 
    return np.array(dists)  


def smiles_edit_distance(s1, s2): 
    ''' Returns Levenshtein Edit Distance btwn two Smiles strings'''

    # Must first Canonize so they are comparable 
    s1 = Chem.CanonSmiles(s1)
    s2 = Chem.CanonSmiles(s2)

    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1 
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j] 


def check_smiles_equivalence(smile1, smile2):
    if smile1 is None or smile2 is None:
        print("one of smiles strings is NONE")
        return False
    smile1 = Chem.CanonSmiles(smile1)
    smile2 = Chem.CanonSmiles(smile2)
    return smile1 == smile2
