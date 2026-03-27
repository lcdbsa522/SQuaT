from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pickle
import copy
import collections
import math
import secrets
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset

from torch.nn import CrossEntropyLoss, MSELoss

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer.modeling_quant import get_student_quant_params
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from transformer import QuantizeLinear, BertSelfAttention, FP_BertSelfAttention
from utils_glue import *

from tqdm import tqdm

import torch.nn.functional as F
import time

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0 
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[index] = token
            index += 1
    return vocab

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)


    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels, teacher_model=None):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in eval_dataloader:
        batch_ = tuple(t.to(device) for t in batch_)
        
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _, _, _ = model(input_ids, segment_ids, input_mask)
        
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data', type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default='models', type=str, help="The model dir.")
    parser.add_argument("--teacher_model", default=None, type=str, help="The models directory.")
    parser.add_argument("--student_model", default=None, type=str, help="Student model directory.")
    parser.add_argument("--task_name", default='sst-2', type=str, help="The name of the task to train.")
    parser.add_argument("--output_dir", default='output', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=None, help="random seed for initialization (if not provided, a random seed will be generated)")
    parser.add_argument('--save_fp_model', action='store_true', help="Whether to save fp32 model")
    parser.add_argument('--save_quantized_model', default=False, type=str2bool, help="Whether to save quantized model")
    parser.add_argument("--weight_bits",default=2, type=int, choices=[1,2,3,4,8], help="Quantization bits for weight.")
    parser.add_argument("--input_bits", default=8, type=int, help="Quantization bits for activation.")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--clip_val", default=2.5, type=float, help="Initial clip value.")
    parser.add_argument('--act_quant', default=True, type=str2bool, help="Whether to quantize activation")
    parser.add_argument('--weight_quant', default=True, type=str2bool, help="Whether to quantize weight")
    parser.add_argument('--neptune', default=True, type=str2bool, help="neptune logging option")
    parser.add_argument('--aug_train', default=False, type=str2bool, help="Whether to use augmented data or not")
    parser.add_argument("--aug_N", default=30, type=int, help="Data Augmentation N Number")
    parser.add_argument("--exp_name", default="", type=str, help="Output Directory Name")

    # Distillation Arguments
    parser.add_argument('--pred_distill', default=False, type=str2bool, help="prediction distill option")
    parser.add_argument('--attn_distill', default=True, type=str2bool, help="attention Score Distill Option")
    parser.add_argument('--rep_distill', default=True, type=str2bool, help="Transformer Layer output Distill Option")
    parser.add_argument('--attnmap_distill', default=True, type=str2bool, help="attention Map Distill Option")
    parser.add_argument('--context_distill', default=True, type=str2bool, help="Context Value Distill Option")
    parser.add_argument('--output_distill', default=False, type=str2bool, help="Context Value Distill Option")
    parser.add_argument('--sa_output_distill', default=False, type=str2bool, help="MSA output Distill Option")
    parser.add_argument('--gt_loss', default=True, type=str2bool, help="Ground Truth Option")
    parser.add_argument('--bert', default="base", type=str, help="which bert model to be use (base, large)")
    parser.add_argument("--map_coeff", default=1, type=float, help="Attention Map Loss Coefficient")
    parser.add_argument("--output_coeff", default=1, type=float, help="Attention Output Loss Coefficient")
    
    # SQuaT Arguments
    parser.add_argument('--squat_distill', default=False, type=str2bool, help="SQuaT Feature Distillation Option")
    parser.add_argument("--squat_coeff", default=1.0, type=float, help="SQuaT Loss Coefficient")
    parser.add_argument('--use_adaptor', default=False, type=str2bool, help="Use adaptor for feature matching")
    parser.add_argument('--use_adaptor_bn', default=False, type=str2bool, help="Use batch norm in adaptor")
    parser.add_argument("--squat_layer", default=-1, type=int, help="Which layer to use for SQuaT (-1: last layer, -2: all layers)")
    parser.add_argument("--squat_token", default='cls', type=str, choices=['cls', 'all'], help="Which token to use for SQuaT (cls or all)")

    args = parser.parse_args() 
    
    start_time = time.time()
    start_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    
    if args.seed is None:
        args.seed = secrets.randbelow(2**32)  
        logger.info(f"Random seed generated: {args.seed}")
    
    if args.neptune:
        import neptune.new as neptune        
        run = neptune.init(project='' + args.task_name.upper(),
                    api_token='')
    else:
        run = None

    exp_name = args.exp_name 

    exp_name += f"_{args.bert}"

    if args.attn_distill:
        exp_name += "_S"
    if args.attnmap_distill:
        exp_name += "_M"
    if args.context_distill:
        exp_name += "_C"
    if args.output_distill:
        exp_name += "_O"
    if args.sa_output_distill:
        exp_name += "_SA"
    if args.squat_distill:
        exp_name += "_SQuaT"
    exp_name += f"_{args.seed}"
    exp_name += f"_{start_time_str}"
    
    args.exp_name = exp_name
    
    if args.aug_train:
        logger.info(f'DA QAT')        
        
    logger.info(f'EXP SET: {exp_name}')
    logger.info(f'TASK: {args.task_name}')
    logger.info(f'MAP-COEFF: {args.map_coeff}')
    logger.info(f'OUTPUT-COEFF: {args.output_coeff}')
    logger.info(f"SIZE: {args.bert}")
    logger.info(f"SEED: {args.seed}")
    logger.info(f'EPOCH: {args.num_train_epochs}')
    
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir,task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir, exist_ok=True)
    
    if args.bert == "large":
        args.model_dir = os.path.join(args.model_dir, "BERT_large")
        args.output_dir = os.path.join(args.output_dir, "BERT_large")
    
    output_dir = os.path.join(args.output_dir,task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_quant_dir = None
    if args.save_quantized_model:
        output_quant_dir = os.path.join(output_dir, 'exploration', args.exp_name)
        os.makedirs(output_quant_dir, exist_ok=True)

    def _resolve_model_source(user_value, default_dir):
        source = user_value if user_value is not None else default_dir
        if source is None:
            return None
        if os.path.exists(source):
            return source
        if os.path.sep not in source:
            return source
        return "bert-large-uncased" if args.bert == "large" else "bert-base-uncased"

    default_model_dir = os.path.join(args.model_dir, task_name)
    args.student_model = _resolve_model_source(args.student_model, default_model_dir)
    args.teacher_model = _resolve_model_source(args.teacher_model, default_model_dir)
    
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    default_params = {
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step": 400 if args.aug_train else 50},
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":8000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":100},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":100},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":100 if args.aug_train else 20},
        "wnli": {"max_seq_length": 128,"batch_size":32,"eval_step":50}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", "wnli"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = args.gpus
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size*n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]
    else:
        logger.warning(f"Task {task_name} not found in default_params. Using default values.")
        args.batch_size = 32
        args.max_seq_length = 128
        args.eval_step = 100
    
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    # Save vocabulary to output directory instead of current directory
    # vocab_output_dir = os.path.join(output_dir, "vocab")
    # os.makedirs(vocab_output_dir, exist_ok=True)
    # tokenizer.save_vocabulary(vocab_output_dir)

    if args.aug_train: 
        try:
            train_file = os.path.join(processed_data_dir,f'aug_data_{args.aug_N}.pkl')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_aug_examples(data_dir, args.aug_N)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            train_file = os.path.join(processed_data_dir,'data.pkl')
            train_features = pickle.load(open(train_file,'rb'))
            
        except:
            train_examples = processor.get_train_examples(data_dir)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    num_train_epochs = args.num_train_epochs 
    num_train_optimization_steps = math.ceil(len(train_features) / args.batch_size) * num_train_epochs
        
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    try:
        test_file = os.path.join(processed_data_dir,'test.pkl')
        test_features = pickle.load(open(test_file,'rb'))
    except:
        test_examples = processor.get_test_examples(data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        with open(test_file, 'wb') as f:
                pickle.dump(test_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        dev_file = os.path.join(processed_data_dir,'dev.pkl')
        eval_features = pickle.load(open(dev_file,'rb'))
    except:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        with open(dev_file, 'wb') as f:
                pickle.dump(eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    if task_name == "mnli":
        mm_processor = processors["mnli-mm"]()
        try:
            dev_mm_file = os.path.join(processed_data_dir,'dev-mm_data.pkl')
            mm_eval_features = pickle.load(open(dev_mm_file,'rb'))
        except:
            mm_eval_examples = mm_processor.get_dev_examples(data_dir)
            mm_eval_features = convert_examples_to_features(
                mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            with open(dev_mm_file, 'wb') as f:
                pickle.dump(mm_eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                        batch_size=args.batch_size)

    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    
    if args.squat_distill:
        from transformer.feature_quant_module import FeatureQuantizerBERT
        class ArgsForFeatureQuantizer:
            def __init__(self, args):
                self.input_bits = args.input_bits
                self.squat_token = args.squat_token
        feature_quant_args = ArgsForFeatureQuantizer(args)
        teacher_model.feature_quantizer = FeatureQuantizerBERT(feature_quant_args)
        logger.info("Added Teacher FeatureQuantizer for SQuaT")
    
    teacher_model.to(device)
    teacher_model.eval()
    
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    
    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels)

    fp32_performance = ""
    fp32_score = 0.0
    
    if task_name in acc_tasks:
        if task_name in ['sst-2','mnli','qnli','rte','wnli']:
            fp32_performance = f"acc:{result['acc']}"
            fp32_score = result['acc']
        elif task_name in ['mrpc','qqp']:
            fp32_performance = f"f1/acc:{result['f1']}/{result['acc']} avg : {(result['f1'] + result['acc'])*50}"
            fp32_score = (result['f1'] + result['acc'])*50
    elif task_name in corr_tasks:
        fp32_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']} corr:{result['corr']}"
        fp32_score = result['corr']*100
    elif task_name in mcc_tasks:
        fp32_performance = f"mcc:{result['mcc']}"
        fp32_score = result['mcc']

    if task_name == "mnli":
        result_mm = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                            device, output_mode, mm_eval_labels, num_labels)
        fp32_performance = f"matched-acc:{result['acc']}  mismatch-acc:{result_mm['acc']}"
        fp32_score = result['acc']
    
    fp32_performance = task_name +' fp32   ' + fp32_performance
    logger.info(fp32_performance)

    student_config = BertConfig.from_pretrained(args.student_model, 
                                                quantize_act=args.act_quant,
                                                quantize_weight=args.weight_quant,
                                                weight_bits = args.weight_bits,
                                                input_bits = args.input_bits,
                                                clip_val = args.clip_val,
                                                )
    
    if args.squat_distill:
        student_config.squat_enable = True
        student_config.squat_layer = args.squat_layer
    else:
        student_config.squat_enable = False
        student_config.squat_layer = -1
    
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    
    student_model.to(device)
    
    for name, module in student_model.named_modules():
        if isinstance(module, QuantizeLinear):
            module.act_flag = args.act_quant
            module.weight_flag = args.weight_quant      
    
    if args.squat_distill and args.use_adaptor:
        from transformer.adaptor import AdaptorBERT
        if hasattr(student_model, 'bert') and hasattr(student_model.bert, 'encoder'):
            hidden_size = student_model.bert.encoder.layer[0].attention.output.config.hidden_size
        else:
            hidden_size = student_config.hidden_size
        student_model.adaptor = AdaptorBERT(hidden_size, use_bn=args.use_adaptor_bn)
        student_model.adaptor.to(device)
        logger.info("Added Adaptor to Student model for SQuaT")
    else:
        student_model.adaptor = None      

    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    param_optimizer = list(student_model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)
    
    loss_mse = MSELoss()
    
    global_step = 0
    best_dev_score = 0.0
    previous_best = None
    best_mismatched_acc = None 

    l_gt_loss = AverageMeter()
    l_attmap_loss = AverageMeter()
    l_att_loss = AverageMeter()
    l_rep_loss = AverageMeter()
    l_cls_loss = AverageMeter()
    l_output_loss = AverageMeter()
    l_sa_output_loss = AverageMeter()
    l_context_loss = AverageMeter()
    l_squat_loss = AverageMeter()  
    l_loss = AverageMeter()
    
    for epoch_ in range(int(num_train_epochs)):
    
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_+1}/{int(num_train_epochs)}", mininterval=1, ascii=True, leave=True)
        for step, batch in enumerate(pbar):

            student_model.train()
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            
            att_loss = torch.tensor(0.0, device=device, requires_grad=False)
            attmap_loss = torch.tensor(0.0, device=device, requires_grad=False)
            rep_loss = torch.tensor(0.0, device=device, requires_grad=False)
            cls_loss = torch.tensor(0.0, device=device, requires_grad=False)
            attscore_loss = torch.tensor(0.0, device=device, requires_grad=False)
            output_loss = torch.tensor(0.0, device=device, requires_grad=False)
            sa_output_loss = torch.tensor(0.0, device=device, requires_grad=False)
            context_loss = torch.tensor(0.0, device=device, requires_grad=False)
            squat_loss = torch.tensor(0.0, device=device, requires_grad=False)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_attn_blocks = teacher_model(input_ids, segment_ids, input_mask)
            
            student_logits, student_atts, student_reps, student_probs, student_attn_blocks = student_model(input_ids, segment_ids, input_mask)

            if args.gt_loss:
                if output_mode == "classification":
                    lprobs = torch.nn.functional.log_softmax(student_logits, dim=-1)
                    loss = torch.nn.functional.nll_loss(lprobs, label_ids, reduction='sum')
                elif output_mode == "regression":
                    loss = loss_mse(student_logits, teacher_logits)
                l_gt_loss.update(loss.item())
                
            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                elif output_mode == "regression":
                    cls_loss = MSELoss()(student_logits, teacher_logits)
                l_cls_loss.update(cls_loss.item())

            if args.context_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[0], teacher_attn_block[0]) 
                    context_loss += tmp_loss
                l_context_loss.update(context_loss.item())
            
            if args.output_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[1], teacher_attn_block[1]) 
                    output_loss += tmp_loss
                l_output_loss.update(output_loss.item())
            
            if args.sa_output_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[2], teacher_attn_block[2]) 
                    sa_output_loss += tmp_loss
                l_sa_output_loss.update(sa_output_loss.item())
            
            if args.attn_distill:
                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):    
                            
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                teacher_att)

                    tmp_loss = MSELoss()(student_att, teacher_att)
                    attscore_loss += tmp_loss
                l_att_loss.update(attscore_loss.item())

            if args.attnmap_distill:
            
                BATCH_SIZE = student_probs[0].shape[0]
                NUM_HEADS = student_probs[0].shape[1]
                MAX_SEQ = student_probs[0].shape[2]
                
                mask = torch.zeros(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ, dtype=torch.float32)
                
                for sent in range(BATCH_SIZE):
                    s = seq_lengths[sent]
                    mask[sent, :, :s, :s] = 1.0
                
                mask = mask.to(device)

                for i, (student_prob, teacher_prob) in enumerate(zip(student_probs, teacher_probs)):            
  
                    student = torch.clamp_min(student_prob, 1e-8)
                    teacher = torch.clamp_min(teacher_prob, 1e-8)

                    neg_cross_entropy = teacher * torch.log(student) * mask
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) 
                    neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) / seq_lengths.view(-1, 1) 

                    neg_entropy = teacher * torch.log(teacher) * mask
                    neg_entropy = torch.sum(neg_entropy, dim=-1)  
                    neg_entropy = torch.sum(neg_entropy, dim=-1) / seq_lengths.view(-1, 1) 

                    kld_loss = neg_entropy - neg_cross_entropy
                    
                    kld_loss_mean = torch.mean(kld_loss)
                            
                    attmap_loss += kld_loss_mean
                
                l_attmap_loss.update(attmap_loss.item())
            
            if args.rep_distill:
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                    tmp_loss = MSELoss()(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                l_rep_loss.update(rep_loss.item())

            if args.squat_distill:
                squat_loss_val = 0.0
                squat_loss_count = 0

                if args.squat_layer == -2:
                    layer_indices = list(range(len(student_attn_blocks)))
                elif args.squat_layer == -1:
                    layer_indices = [len(student_attn_blocks) - 1]
                else:
                    layer_indices = [args.squat_layer]

                actual_student = student_model.module if hasattr(student_model, 'module') else student_model
                actual_teacher = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model

                for li in layer_indices:
                    fd_map_s = None
                    fd_map_t = None

                    try:
                        s_block = student_attn_blocks[li]
                        if isinstance(s_block, (tuple, list)) and len(s_block) >= 1:
                            student_qact = s_block[-1]
                        else:
                            student_qact = None

                        if student_qact is not None:
                            if args.squat_token == 'cls' and student_qact.dim() == 3:
                                student_qact = student_qact[:, 0, :]

                            if hasattr(actual_student, 'adaptor') and actual_student.adaptor is not None:
                                fd_map_s = actual_student.adaptor(student_qact)
                            else:
                                fd_map_s = student_qact
                    except Exception as e:
                        logger.warning(f"SQuaT: Failed to extract student FFN-fc1 q_input (layer {li}): {e}")

                    try:
                        t_block = teacher_attn_blocks[li]
                        teacher_feat = None

                        if isinstance(t_block, (tuple, list)):
                            if len(t_block) > 1 and isinstance(t_block[1], torch.Tensor):
                                teacher_feat = t_block[1]
                            else:
                                for item in t_block:
                                    if isinstance(item, torch.Tensor):
                                        teacher_feat = item
                                        break
                        elif isinstance(t_block, torch.Tensor):
                            teacher_feat = t_block

                        if teacher_feat is None:
                            raise ValueError("Teacher feature not found in attention block")

                        if args.squat_token == 'cls' and teacher_feat.dim() == 3:
                            teacher_feat = teacher_feat[:, 0, :]

                        quant_params = get_student_quant_params(actual_student, layer_idx=li)
                        if hasattr(actual_teacher, 'feature_quantizer'):
                            fd_map_t = actual_teacher.feature_quantizer(teacher_feat, quant_params=quant_params)
                        else:
                            fd_map_t = teacher_feat
                    except Exception as e:
                        logger.warning(f"SQuaT: Failed to extract teacher FFN-fc1 input (layer {li}): {e}")

                    if fd_map_s is None or fd_map_t is None:
                        continue

                    if fd_map_s.shape != fd_map_t.shape:
                        if args.squat_token == 'cls' and fd_map_s.dim() == 2 and fd_map_t.dim() == 3:
                            fd_map_t = fd_map_t[:, 0, :]
                        else:
                            logger.warning(
                                f"SQuaT: Shape mismatch (layer {li}): student {tuple(fd_map_s.shape)} vs teacher {tuple(fd_map_t.shape)}; skipping"
                            )
                            continue

                    try:
                        _loss = MSELoss()(fd_map_s, fd_map_t)
                        squat_loss_val += _loss
                        squat_loss_count += 1
                    except Exception as e:
                        logger.warning(f"SQuaT: Loss computation failed (layer {li}): {e}")

                if squat_loss_count > 0:
                    squat_loss = squat_loss_val / float(squat_loss_count)
                    l_squat_loss.update(squat_loss.item())
                else:
                    squat_loss = torch.tensor(0.0, device=device)

            total_distill_loss = cls_loss + rep_loss + (attmap_loss * args.map_coeff) + (output_loss * args.output_coeff) + sa_output_loss + attscore_loss + context_loss + (squat_loss * args.squat_coeff)
            
            if args.gt_loss:
                loss = loss + total_distill_loss
            else:
                loss = total_distill_loss
            
            if not (args.gt_loss or args.pred_distill or args.context_distill or 
                    args.output_distill or args.sa_output_distill or args.attn_distill or 
                    args.attnmap_distill or args.rep_distill or args.squat_distill):
                raise ValueError("At least one loss component must be enabled!")
            
            l_loss.update(loss.item())
            
            if step % 10 == 0:
                logs = {'loss': f"{l_loss.avg:.4f}"}
                if args.squat_distill:
                    logs['sq_loss'] = f"{l_squat_loss.avg:.4f}"
                if args.pred_distill:
                    logs['cls_loss'] = f"{l_cls_loss.avg:.4f}"
                pbar.set_postfix(logs)

            if n_gpu > 1:
                loss = loss.mean()           
                
            if global_step == 0: 
                if run is not None:           
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    if args.squat_distill:
                        run["loss/squat_loss_loss"].log(value=l_squat_loss.avg, step=global_step)
                    
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)
 
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            global_step += 1

            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1: 
                student_model.eval()
                result = do_eval(student_model, task_name, eval_dataloader,
                                    device, output_mode, eval_labels, num_labels)
            
                result['global_step'] = global_step
                result['cls_loss'] = l_cls_loss.avg
                result['att_loss'] = l_att_loss.avg
                result['rep_loss'] = l_rep_loss.avg
                if args.squat_distill:
                    result['squat_loss'] = l_squat_loss.avg
                result['loss'] = l_loss.avg
                
                if task_name == "mnli":
                    result_mm = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                        device, output_mode, mm_eval_labels, num_labels)
                    result['acc_matched'] = result['acc']
                    result['acc_mismatched'] = result_mm['acc']
                    logger.info(f"Step {global_step} - MNLI matched-acc: {result['acc']:.4f}, mismatch-acc: {result_mm['acc']:.4f}")
                
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    run["loss/sa_output_loss_loss"].log(value=l_sa_output_loss.avg, step=global_step)
                    run["loss/attmap_loss_loss"].log(value=l_attmap_loss.avg, step=global_step)
                    if args.squat_distill:
                        run["loss/squat_loss_loss"].log(value=l_squat_loss.avg, step=global_step)
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                eval_result = None
                eval_score = 0.0
                
                if task_name=='cola':
                    eval_score = result["mcc"]
                    if run is not None:
                        run["metrics/mcc"].log(value=result['mcc'], step=global_step)
                    eval_result = result["mcc"]  
                elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                    eval_score = result["acc"]
                    if run is not None:
                        run["metrics/acc"].log(value=result['acc'], step=global_step)
                        if task_name == "mnli":
                            run["metrics/acc_matched"].log(value=result['acc'], step=global_step)
                            run["metrics/acc_mismatched"].log(value=result['acc_mismatched'], step=global_step)
                    eval_result = result["acc"]
                elif task_name in ['mrpc','qqp']:
                    eval_score = result["acc_and_f1"]
                    if run is not None:
                        run["metrics/acc_and_f1"].log(value=result['acc_and_f1'],step=global_step)
                    eval_result = result["acc_and_f1"]
                elif task_name in corr_tasks:
                    eval_score = result["corr"]
                    if run is not None:
                        run["metrics/corr"].log(value=result['corr'],step=global_step)
                    eval_result = result["corr"]
                else:
                    logger.warning(f"Unknown task_name: {task_name}, using default eval_result")
                    eval_result = result.get('acc', 0.0)
                    eval_score = eval_result
                
                save_model = False

                if task_name in acc_tasks and result['acc'] > best_dev_score:
                    if task_name == "mnli":
                        if "acc_mismatched" in result:
                            previous_best = f"matched:{result['acc']*100:.2f} mismatch:{result['acc_mismatched']*100:.2f}"
                            best_mismatched_acc = result['acc_mismatched']
                        else:
                            result_mm = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                                device, output_mode, mm_eval_labels, num_labels)
                            previous_best = f"matched:{result['acc']*100:.2f} mismatch:{result_mm['acc']*100:.2f}"
                            best_mismatched_acc = result_mm['acc']
                    elif task_name in ['sst-2','qnli','rte','wnli']:
                        previous_best = f"{result['acc']*100:.2f}"
                    elif task_name in ['mrpc','qqp']:
                        previous_best = f"{(result['f1'] + result['acc'])*50:.2f}"
                    best_dev_score = result['acc']
                    save_model = True

                if task_name in corr_tasks and result['corr'] > best_dev_score:
                    previous_best = f"{result['corr']*100:.2f}"
                    best_dev_score = result['corr']
                    save_model = True

                if task_name in mcc_tasks and result['mcc'] > best_dev_score:
                    previous_best = f"{result['mcc']:.4f}"
                    best_dev_score = result['mcc']
                    save_model = True

                    if args.save_fp_model:
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                    
                    if args.save_quantized_model and output_quant_dir is not None:
                        os.makedirs(output_quant_dir, exist_ok=True)
                        
        
        if epoch_ < int(num_train_epochs) - 1: 
            student_model.eval()
            result = do_eval(student_model, task_name, eval_dataloader,
                            device, output_mode, eval_labels, num_labels)
            
            result['global_step'] = global_step
            result['cls_loss'] = l_cls_loss.avg
            result['att_loss'] = l_att_loss.avg
            result['rep_loss'] = l_rep_loss.avg
            if args.squat_distill:
                result['squat_loss'] = l_squat_loss.avg
            result['loss'] = l_loss.avg
            
            if task_name == "mnli":
                result_mm = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                    device, output_mode, mm_eval_labels, num_labels)
                result['acc_matched'] = result['acc']
                result['acc_mismatched'] = result_mm['acc']
            
            if task_name in acc_tasks:
                if task_name in ['sst-2','qnli','rte','wnli']:
                    logger.info(f"Epoch {epoch_+1}/{int(num_train_epochs)} - Accuracy: {result['acc']:.4f}")
                elif task_name == "mnli":
                    logger.info(f"Epoch {epoch_+1}/{int(num_train_epochs)} - MNLI matched-acc: {result['acc']:.4f}, mismatch-acc: {result['acc_mismatched']:.4f}")
                elif task_name in ['mrpc','qqp']:
                    logger.info(f"Epoch {epoch_+1}/{int(num_train_epochs)} - F1: {result['f1']:.4f}, Acc: {result['acc']:.4f}")
            elif task_name in corr_tasks:
                logger.info(f"Epoch {epoch_+1}/{int(num_train_epochs)} - Correlation: {result['corr']:.4f}")
            elif task_name in mcc_tasks:
                logger.info(f"Epoch {epoch_+1}/{int(num_train_epochs)} - MCC: {result['mcc']:.4f}")
                        
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    logger.info(f"==> Previous Best = {previous_best}")
    logger.info(f"==> Last Result = {result}")
    logger.info(f"==> Elapsed Time = {time_str} ({elapsed_time:.2f} seconds)")
    
    if args.save_quantized_model and output_quant_dir is not None:
        best_txt = os.path.join(output_quant_dir, "best_info.txt")
        last_txt = os.path.join(output_quant_dir, "last_info.txt")
        
        with open(best_txt, "w") as f_w:
            if previous_best:
                f_w.write(previous_best)
            else:
                f_w.write("N/A")
        
        with open(last_txt, "w") as f_w:
            if task_name == "mnli":
                if "acc_mismatched" in result:
                    f_w.write(f"matched:{result['acc']*100:.2f} mismatch:{result['acc_mismatched']*100:.2f}")
                else:
                    result_mm = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                        device, output_mode, mm_eval_labels, num_labels)
                    f_w.write(f"matched:{result['acc']*100:.2f} mismatch:{result_mm['acc']*100:.2f}")
            elif eval_result is not None:
                f_w.write(f"{eval_result*100:.2f}")
            else:
                f_w.write("N/A")

if __name__ == "__main__":
    main()
