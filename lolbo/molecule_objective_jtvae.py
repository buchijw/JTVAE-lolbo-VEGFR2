import numpy as np
import torch 
# import selfies as sf 
from lolbo.utils.mol_utils.mol_utils import smiles_to_desired_scores
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from lolbo.utils.mol_utils.selfies_vae.data import collate_fn
from lolbo.latent_space_objective import LatentSpaceObjective
from lolbo.utils.mol_utils.mol_utils import GUACAMOL_TASK_NAMES
import rdkit,sys,os
from rdkit import Chem,DataStructs
from rdkit.Chem import MolFromSmiles, MolToSmiles,AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops, RDConfig
from tqdm.auto import tqdm
sys.path.append('%s/../JTVAE-GA/fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtprop_vae import JTPropVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import tensorize_prop,smiles_to_moltree
import func_timeout
# import pkg_resources
# # make sure molecule software versions are correct: 
# assert pkg_resources.get_distribution("selfies").version == '2.0.0'
# assert pkg_resources.get_distribution("rdkit-pypi").version == '2022.3.1'
# assert pkg_resources.get_distribution("molsets").version == '0.3.1'


class MoleculeObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        task_id='pdop',
        path_to_vae_statedict='model',
        xs_to_scores_dict={},
        num_calls=0,
        decoded_smiles = [],
        vae_hyper={},
        beta:float=0.2
    ):
        assert task_id in GUACAMOL_TASK_NAMES + ["logp","pic50","dual"]

        self.dim                    = 56
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        # self.max_string_length      = max_string_length # max string length that VAE can generate
        # self.smiles_to_selfies      = smiles_to_selfies # dict to hold computed mappings form smiles to selfies strings
        self.decoded_smiles = decoded_smiles
        self.vae_hyper=vae_hyper
        self.beta=beta
        self.vocab = self.vae_hyper['vocab']
        self.vocab_set = set(self.vocab.vocab)

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )
    
    def check_vocab(self, smiles: str):
        cset = set()
        mol = MolTree(smiles)
        for c in mol.nodes:
            cset.add(c.smiles)
        return cset.issubset(self.vocab_set)

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float().reshape((-1,self.dim))
        z = z.cuda()
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # sample molecular string form VAE decoder
        # sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        decoded_smiles = []
        print('Decoding...')
        for i in tqdm(range(z.shape[0])):
            all_vec = z[i].reshape((1,-1))
            tree_vec,mol_vec = torch.hsplit(all_vec, 2)
            tree_vec = create_var(tree_vec.float())
            mol_vec = create_var(mol_vec.float())
            try:
                s = func_timeout.func_timeout(600, self.vae.decode, args=(tree_vec, mol_vec), kwargs={'prob_decode':False})
                if s is not None:
                #     decoded_smiles.append(s)
                    # if any(nitro in s for nitro in ['n','N']) and len(s) <= 100:
                    if (len(s) <= 100) and self.check_vocab(s):
                        decoded_smiles.append(s)
                    else:
                        # decoded_smiles.append('c1c2')
                        decoded_smiles.append(None)
            except:
                decoded_smiles.append(None)
                print("timed out")

        return decoded_smiles


    def query_oracle(self, x_ls):
        ''' Input: 
                x list
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assumes x is a single smiles string
        score = smiles_to_desired_scores(x_ls, self.task_id)

        return score


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        # self.dataobj = SELFIESDataset()
        self.vae = JTPropVAE(**self.vae_hyper)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
                state_dict = torch.load(self.path_to_vae_statedict) 
                self.vae.load_state_dict(state_dict, strict=True)
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        # self.vae.max_string_length = self.max_string_length


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        X_list = []
        # prop_list = []
        prop_list = list(self.query_oracle(xs_batch))
        for smile in xs_batch:
            # prop_list.append(self.query_oracle(smile))
            X_list.append(smiles_to_moltree(smile))
        x_batch = tensorize_prop((X_list,prop_list),self.vocab, assm=True)
        z,loss = self.vae.forwar_to_z_loss(x_batch,self.beta)
        loss = loss.item()
        z = z.reshape(-1,self.dim)

        return z, loss


if __name__ == "__main__":
    # testing molecule objective
    obj1 = MoleculeObjective(task_id='pdop' ) 
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,256))
    print(dict1['scores'], obj1.num_calls)
