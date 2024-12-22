import torch
import pickle
import os
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          BertTokenizer,
                          BertModel
                         )

'''
Here intorduces some availabel pretrained language model, including
ClinicalBERT —— Nature Medicine 2023
bioLongformer——AMIA 2023
bioRoberta——ACL2020
BioBert —— NAACL Workshop 2019
Bert —— NAACL 2019
'''

def loadBert(args, device):
    print(args.language_model, 'is used as the language model.')
    assert args.language_model != None, "args.language_model is None."
    if args.language_model == 'BioBert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        BioBert=AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif args.language_model =="bioRoberta":
        config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
        tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
    elif  args.language_model == "Bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BioBert = BertModel.from_pretrained("bert-base-uncased")
    elif args.language_model == "bioLongformer":
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        BioBert= AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
    elif args.language_model == "ClinicalBERT":
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        BioBert = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    else:
        raise ValueError("language_model should be BioBert, bioRoberta, bioLongformer, ClinicalBERT or Bert")
    
    for param in BioBert.parameters():
        param.requires_grad = False
    
    BioBert = BioBert.to(device)
    return BioBert, tokenizer



'''Defining Dataloader'''

def Multimodal_EHRs(args, mode, tokenizer = None):
    task_name = args.task
    language_model = args.language_model
    dataset = Loading_Data(args, mode, task_name, language_model, tokenizer)
    if mode == 'train':
        sampler = RandomSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=DataPadding, drop_last=True)
    elif mode in ['val', 'test']:
        sampler = SequentialSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=DataPadding, drop_last=True)
    return dataset, dataloader

class Loading_Data(Dataset):
    def __init__(self, args, mode, task_name, language_model, tokenizer, data = None):
        if data != None:
            self.data = data
        else:
            self.data = load_dataset(args, mode, debug=False)

        self.task_name = task_name
        self.language_model = language_model
        self.dim_labtest = args.dim_labtest
        self.dim_medications = args.dim_medications
        self.dim_diagnoses = args.dim_diagnoses
        self.dim_notes = args.dim_notes
        self.tokenizer = tokenizer
        self.num_of_notes = args.num_of_notes
        self.num_of_labtests = args.num_of_labtests
        
        if args.task == 'hospitalization':
            self.templates = pickle.load(open('./config/task1_template.pkl', "rb"))[0].cpu()
        elif args.task == 'critical_outcome':
            self.templates = pickle.load(open('./config/task2_template.pkl', "rb"))[0].cpu()
        self.mode = mode
            

    def __getitem__(self, idx):
        data_details = self.data[idx]
        stay_id = data_details['stay_id']

        '''Read triage variables'''
        triage_variables = data_details['triage_variables']

        '''Read ICU metrics in lab test'''
        if 'labtest' in data_details.keys():
            labtest = data_details['labtest'].astype(float)
        else:
            labtest = np.zeros((12, self.dim_labtest))
        
        while labtest.shape[0] < self.num_of_labtests:
            labtest = np.concatenate((labtest, np.zeros((1, self.dim_labtest))), axis=0)

        if 'medications' in data_details.keys():
            medication = data_details['medications']
            medication_one_hot_tensors = np.zeros((medication.shape[0], self.dim_medications))
            for j in range(medication.shape[0]):
                med_index = medication[j, 0]
                medication_one_hot_tensors[j, med_index] = 1
        else:
            medication_one_hot_tensors = np.zeros((1, self.dim_medications))

        if 'diagnoses' in data_details.keys():
            diagnoses = data_details['diagnoses']
            diagnoses_one_hot_tensors = np.zeros((diagnoses.shape[0], self.dim_diagnoses))
            for j in range(diagnoses.shape[0]):
                diag_index = diagnoses[j, 0]
                diagnoses_one_hot_tensors[j, diag_index] = 1
        else:
            diagnoses_one_hot_tensors = np.zeros((1, self.dim_diagnoses))

        '''
        Read text representations from the frozen pretrained language model.
        '''
        if 'notes' in data_details.keys():
            text_representations = data_details['notes'].squeeze()
        else:
            text_representations = np.zeros((5, self.dim_notes))        
        
        '''Read labels'''
        if self.task_name == 'hospitalization':
            label = data_details['task1_hospitalization_label']
        elif self.task_name == 'inhospital_mortality':
            label = data_details['task2_inhospital_mortality_label']
        elif self.task_name == 'icu_transfer_12h':
            label = data_details['task3_icu_transfer_12h_label']
        elif self.task_name == 'critical_outcome':
            label = data_details['task4_outcome_critical_label']
        

        task_template = self.templates[:2]
        mask_template = self.templates[1]

        '''Setting tensors'''        
        triage_variables = torch.tensor(triage_variables, dtype=torch.float)
        labtest = torch.tensor(labtest, dtype=torch.float)
        medication_one_hot_tensors = torch.tensor(medication_one_hot_tensors, dtype=torch.float)
        diagnoses_one_hot_tensors = torch.tensor(diagnoses_one_hot_tensors, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return {'stay_id': stay_id, 
                'triage_variables': triage_variables, 
                'labtest': labtest,
                'medication': medication_one_hot_tensors,
                'diagnoses':diagnoses_one_hot_tensors,
                'notes': text_representations,
                'task_template': task_template,
                'mask_template': mask_template, 
                'label': label}

    def __len__(self):
        return len(self.data)

    
def load_dataset(args, mode, debug=False):
    dataPath = os.path.join(args.data_path, mode + '.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data=data[:100]
    return data


def DataPadding(batch):
    batch = list(filter(lambda x: x is not None, batch))
    triage_variables=torch.stack([example['triage_variables'] for example in batch])
    labtest = pad_sequence([example['labtest'] for example in batch], batch_first=True, padding_value=0)
    medication_one_hot_tensors = pad_sequence([example['medication'] for example in batch], batch_first=True, padding_value=0)
    diagnoses_one_hot_tensors = pad_sequence([example['diagnoses'] for example in batch], batch_first=True, padding_value=0)
    text_representations=torch.stack([example['notes'] for example in batch])
    task_template=torch.stack([example['task_template'] for example in batch])
    mask_template=torch.stack([example['mask_template'] for example in batch])
    
    label=torch.stack([example["label"] for example in batch])

    return triage_variables, labtest, medication_one_hot_tensors, diagnoses_one_hot_tensors, text_representations, \
           task_template, mask_template, label
