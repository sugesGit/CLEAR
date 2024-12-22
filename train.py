'''Loading packages'''
import os, sys
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

'''Loading customized packages'''
sys.path.append('../code')
from utils import metrics
from utils.utils import parse_args
from data.data import Multimodal_EHRs
from utils.get_loss import get_loss
from model.CLEAR import CLEAR



'''Parameter Configuration'''
args = parse_args()
alpha = args.alpha
print('args:', args)

'''
---Loading dataset---
Following previous works on multimodal EHR, split the dataset into training, validation, and test sets with a ratio of 8: 1: 1
'''
train_dataset, train_dataloader = Multimodal_EHRs(args, 'train')
val_dataset, val_dataloader = Multimodal_EHRs(args, 'val')
test_dataset, test_dataloader = Multimodal_EHRs(args, 'test')
num_train = len(train_dataset)
num_val = len(val_dataset)
num_test  = len(test_dataset)
print('The number of training set:', num_train)
print('The number of validation set:', num_val)
print('The number of test set:', num_val)
print('The ratio of training, validation and test set:', \
      num_train/(num_train + num_val + num_val), num_val/(num_train + num_val + num_val), num_val/(num_train + num_val + num_val))


'''Checking if GPU is available'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("available device: {}".format(device))


'''Training Process'''
'''Loading Model'''
RANDOM_SEED = args.seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True


'''Defining Model'''
embed_dim = 128
rnn_dim = int(embed_dim/2)
triage_dim = args.dim_triage
numerical_sequence_parameters = {"input_dim": args.dim_labtest, "rnn_dim": rnn_dim}
category_sequence_parameters1 = {'input_dim': args.dim_medications, 'embed_dim': embed_dim, 'rnn_dim': rnn_dim}
category_sequence_parameters2 = {'input_dim': args.dim_diagnoses, 'embed_dim': embed_dim, 'rnn_dim': rnn_dim}
textembed_dim = 768
num_class = 2
hidden_dim = embed_dim * args.num_modalities * num_class
model = CLEAR(args, triage_dim, numerical_sequence_parameters, category_sequence_parameters1, category_sequence_parameters2, \
              textembed_dim, embed_dim, hidden_dim, device, output_dim = num_class)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

'''Defining optimizer'''
print('Training modalities', args.modalities, len(args.modalities))
if not 'notes' in args.modalities:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
                ], lr=args.learning_rate, weight_decay = args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
print(optimizer)

start_epoch = 0
load_checkpoint = False
if load_checkpoint:
    load_path = 'path to weight'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['net'], strict=False)
    start_epoch = checkpoint['epoch'] + 1


'''Start training'''
epochs = args.num_train_epochs
savemodel_name = args.output_path

fold_count = 0
fold_train_loss = []
fold_valid_loss = []


best_auroc = 0
best_auprc = 0
best_f1 = 0
best_recall = 0
best_precision = 0
    
_best_auroc = 0
_best_auprc = 0
_best_f1 = 0
_best_recall = 0
_best_precision = 0


for each_epoch in tqdm(range(start_epoch, start_epoch + epochs)):
    epoch_loss = []
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        triage_variables, labtest, medication_one_hot_tensors, diagnoses_one_hot_tensors, text_representations, \
        task_template, mask_template, label = batch
        optimizer.zero_grad()

        triage_variables = triage_variables.clone().detach().to(device, dtype=torch.float32)
        labtest = labtest.clone().detach().to(device, dtype=torch.float32)
        medication_one_hot_tensors = medication_one_hot_tensors.clone().detach().to(device, dtype=torch.float32)
        diagnoses_one_hot_tensors = diagnoses_one_hot_tensors.clone().detach().to(device, dtype=torch.float32)
        text_representations = text_representations.clone().detach().to(device, dtype=torch.float32)
        task_template = task_template.clone().detach().to(device, dtype=torch.float32)
        mask_template = mask_template.clone().detach().to(device, dtype=torch.float32)
        label = label.clone().detach().to(device, dtype=torch.long).squeeze(-1)

        output, dispairty, POP_output = model(triage_variables = triage_variables, \
                        labtest = labtest, \
                        medication_one_hot_tensors = medication_one_hot_tensors,\
                        diagnoses_one_hot_tensors = diagnoses_one_hot_tensors,\
                        text_representations = text_representations, \
                        task_template = task_template, \
                        mask_template = mask_template, \
                        tao = args.tao, \
                        mode = 'train')      
        
        loss_cl = get_loss(output, label.long())
        
        loss_cf = get_loss(POP_output, label.long())
        
        loss_d = torch.mean(dispairty)
        loss = (1-alpha)*loss_cl + alpha*(loss_d + loss_cf)

        epoch_loss.append(loss.cpu().detach().numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        optimizer.step()
        model.zero_grad()

    epoch_loss = np.mean(epoch_loss)
    fold_train_loss.append(epoch_loss)

    '''Validation'''
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            triage_variables, labtest, medication_one_hot_tensors, diagnoses_one_hot_tensors, text_representations, \
            task_template, mask_template, label = batch

            triage_variables = triage_variables.clone().detach().to(device, dtype=torch.float32)
            labtest = labtest.clone().detach().to(device, dtype=torch.float32)
            medication_one_hot_tensors = medication_one_hot_tensors.clone().detach().to(device, dtype=torch.float32)
            diagnoses_one_hot_tensors = diagnoses_one_hot_tensors.clone().detach().to(device, dtype=torch.float32)
            text_representations = text_representations.clone().detach().to(device, dtype=torch.float32)
            task_template = task_template.clone().detach().to(device, dtype=torch.float32)
            mask_template = mask_template.clone().detach().to(device, dtype=torch.float32)
            label = label.clone().detach().to(device, dtype=torch.long).squeeze(-1)

            output = model(triage_variables = triage_variables, \
                            labtest = labtest, \
                            text_representations = text_representations, \
                            medication_one_hot_tensors = medication_one_hot_tensors,\
                            diagnoses_one_hot_tensors = diagnoses_one_hot_tensors,\
                            task_template = task_template, \
                            mask_template = mask_template, \
                            tao = args.tao, \
                            mode = 'val')
                
            y_pred += list(output.cpu().detach().numpy())
            y_true += list(label.cpu().numpy())

        ret = metrics.print_metrics_binary(y_true, y_pred, verbose = 0)

        cur_aupr = ret['auprc']
        if cur_aupr > best_auprc:                
            best_auroc = ret['auroc']
            best_auprc = ret['auprc']
            best_f1 = ret['f1_score']
            best_recall = ret['rec1']
            best_precision = ret['prec1']

            state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': each_epoch
                        }
                
            torch.save(state, savemodel_name+'_aupr_'+str(fold_count))

            print('------------ Save best model - auprc: %.4f ------------'%cur_aupr)
            print("precision class 1 = {}".format(best_precision))
            print("recall class 1 = {}".format(best_recall))
            print("AUC of ROC = {}".format(best_auroc))
            print("AUC of PRC = {}".format(best_auprc))
            print("f1_score = {}".format(best_f1))

                        
        _cur_f1 = ret['f1_score']
        if _cur_f1 > _best_f1:             
            _best_auroc = ret['auroc']
            _best_auprc = ret['auprc']
            _best_f1 = _cur_f1
            _best_recall = ret['rec1']
            _best_precision = ret['prec1']

            state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': each_epoch
                        }
                
            torch.save(state, savemodel_name+'_f1_'+str(fold_count))

            print('------------ Save best model - F1: %.4f ------------'%_cur_f1)
            print("precision class 1 = {}".format(_best_precision))
            print("recall class 1 = {}".format(_best_recall))
            print("AUC of ROC = {}".format(_best_auroc))
            print("AUC of PRC = {}".format(_best_auprc))
            print("f1_score = {}".format(_best_f1))

    scheduler.step(ret['auprc'])
    print('Fold %d, Epoch %d, roc = %.4f, prc = %.4f, f1 = %.4f, recall = %.4f, precision = %.4f' \
        %(fold_count, each_epoch,  ret['auroc'], ret['auprc'], ret['f1_score'], ret['rec1'], ret['prec1']))


print('best_auroc %.4f'%(best_auroc))
print('best_auprc %.4f'%(best_auprc))
print('best_f1 %.4f'%(best_f1))
print('best_recall %.4f'%(best_recall))
print('best_precision %.4f'%(best_precision))
    
print("==========================================================")

print('_best_auroc %.4f'%(_best_auroc))
print('_best_auprc %.4f'%(_best_auprc))
print('_best_f1 %.4f'%(_best_f1))
print('_best_recall %.4f'%(_best_recall))
print('_best_precision %.4f'%(_best_precision))