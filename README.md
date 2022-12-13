# Machine_Learning_Group_TLA

Liên kết google colab với Drive cá nhân để thuận tiện cho việc đọc các file dữ liệu:
```
from google.colab import drive
import os

drive.mount('/content/drive/')
os.chdir('/content/drive/My Drive/BERT/')
```

Cài đặt các thư viện transformers, fastBPE, fairseq và vncorenlp:
```
!pip install transformers
!pip install fastBPE
!pip install fairseq
!pip install vncorenlp

!mkdir -p vncorenlp/models/wordsegmenter
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

Load model và bpe của PhoBERT:
```
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/content/drive/MyDrive/BERT/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("/content/drive/MyDrive/BERT/PhoBERT_base_transformers/dict.txt")
```

Tiến hành tách từ, thay thế những cụm từ viết tắt ngắn gọn bằng cụm từ đầy đủ:
```
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/content/drive/MyDrive/BERT/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

def checkWords(argument):
    switcher = {
        "k": "không",
        "ko": "không",
        "j":"gì",
        "đc":"được",
        "dc":"được",
        "ntn":"như thê nào",
        "ok":"ổn",
        "ncl":"nói chung là",
        "mn":"mọi người",
        "mng":"mọi người",
        "vs":"với",
        "cx":"cũng",
        "bt":"bình thường",
        "bth":"bình thường",
        "nv":"nhân viên",
        "recommend":"gợi ý tốt",
        "mk":"mình",
    }

    return switcher.get(argument, argument)

def standardizeData(comment):
    comment = rdrsegmenter.tokenize(comment)

    for sentence in comment:
        for i in range(0,len(sentence)):
            sentence[i]=sentence[i].lower()
            sentence[i]=checkWords(sentence[i])
    
    comment = ' '.join([' '.join(sentence) for sentence in comment])

    comment = comment.replace(",", "").replace(".", "") \
    .replace(";", "").replace("“", "") \
    .replace(":))", "cười").replace("”", "") \
    .replace('"', "").replace("'", "") \
    .replace("!", "").replace("?", "") \
    .replace("-", "").replace(":","")
    return comment
```

Bắt đầu đọc các file .csv, tách dữ liệu từ các cột về thành các mảng 1 chiều chứa id, comment và label
```
import csv
# import re

train_path = '/content/drive/MyDrive/BERT/full_train.csv'
test_path = '/content/drive/MyDrive/BERT/test.csv'

train_id, train_text, train_labels = [], [], []
test_id, test_text = [], []


with open(train_path) as f:
    reader = csv.reader(f)

    for row in reader:
        if row[3] == 'Comment':
            continue
        id = row[1]

        comment = standardizeData(row[3])
  
        label = int(row[5])
        train_id.append(id)
        train_text.append(comment)
        train_labels.append(label)

with open(test_path) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[3] == 'Comment':
            continue
        id = row[1]
        comment = standardizeData(row[3])

        test_id.append(id)
        test_text.append(comment)
```

Tách dữ liệu thành 2 tập train và validation theo tỉ lệ 9:1
```
from sklearn.model_selection import train_test_split

train_sents, val_sents, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=0.1)
```

Tiếp theo, từ dữ liệu thô này, chúng ta sử dụng bpe đã load ở trên để đưa text đầu vào dưới dạng subword và ánh xạ các subword này về dạng index trong từ điển:
```
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 256

train_ids = []
for sent in train_sents:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    
    train_ids.append(encoded_sent)

val_ids = []
for sent in val_sents:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    
    val_ids.append(encoded_sent)

train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
```

Tiếp theo, tạo một mask gồm các giá trị 0, 1 để làm đầu vào cho thư viện transformers:
```
train_masks = []
for sent in train_ids:
    mask = [int(token_id > 0) for token_id in sent]
    train_masks.append(mask)

val_masks = []
for sent in val_ids:
    mask = [int(token_id > 0) for token_id in sent]
    val_masks.append(mask)
```

Chuyển dữ liệu về tensor và sử dụng DataLoader của torch để tạo dataloader
```
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

train_inputs = torch.tensor(train_ids)
val_inputs = torch.tensor(val_ids)
train_labels = torch.tensor(train_labels)

val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)
```

Bắt đầu load model của PhoBert
```
from torch import nn
from transformers import RobertaModel,RobertaForSequenceClassification, RobertaConfig, AdamW

config = RobertaConfig.from_pretrained(
    "/content/drive/MyDrive/BERT/PhoBERT_base_transformers/config.json", from_tf=False, num_labels = 2, output_hidden_states=False,
)

model = RobertaForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/BERT/PhoBERT_base_transformers/model.bin",
    config=config
)

model.cuda()
```

Định nghĩa hàm flat_accuracy:
```
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def flat_accuracy(preds, labels):
    #red_flat = np.argmax(preds, axis=1).flatten()
    #labels_flat = labels.flatten()
    
    F1_score = f1_score(pred_flat, labels_flat, average='macro')
    
    return accuracy_score(pred_flat, labels_flat), F1_score
```

Và bắt đầu training mô hình:
```
import random
from tqdm import tqdm_notebook
from sklearn import metrics

device = 'cuda'
epochs = 3

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    model.train()
    train_accuracy = 0
    train_auc =0
    nb_train_steps = 0
    train_f1 = 0
    
    for step, batch in tqdm_notebook(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(pred_flat, labels_flat)
        
        train_accuracy += tmp_train_accuracy
        train_f1 += tmp_train_f1
        nb_train_steps += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" F1 score: {0:.4f}".format(train_f1/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))

    print("Running Validation...")
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    eval_auc=0
    for batch in tqdm_notebook(val_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            
            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)
            
            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    print("Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print("F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))

print("Training complete!")
```

Thực hiện tương tự các bước tiền xử lý trên với bộ test:
```
test_ids = []
test_labels = []
test_masks = []

for sent in test_text:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    test_ids.append(encoded_sent)

test_ids = pad_sequences(test_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

for sent in test_ids:
    mask = [int(token_id > 0) for token_id in sent]
    test_masks.append(mask)

for i in range(0,len(test_masks)):
  test_labels.append(0)

test_inputs = torch.tensor(test_ids)
test_masks = torch.tensor(test_masks)
test_labels = torch.tensor(test_labels)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
```

Bắt đầu sử dụng model để tính kết quả bộ test, kết quả trả về được lưu lại dưới dạng file csv:
```
output_path = '/content/drive/MyDrive/BERT/test_output_3_epochs_1e-5.csv'

output_file = [["RevID","Rating"]]
id = 0;

print("Running Test...")
model.eval()
for batch in tqdm_notebook(test_dataloader):

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, 
        token_type_ids=None, 
        attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        
        pred_flat = np.argmax(logits, axis=1).flatten()

        for i in pred_flat:
          #result = f'{id} {i}'
          #print(result)
          output_file.append([test_id[id],i])
          id+=1

with open(output_path, 'w') as f:
    file_writer = csv.writer(f) # create csv writer
    file_writer.writerows(output_file) 
    f.close() # close file 

print("Testing complete!")
```

Và cuối cùng, trích xuất model:
```
torch.save(model, "/content/drive/MyDrive/BERT/model.pth")
```
