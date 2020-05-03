import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn as nn
from models.selfAttention import SelfAttention
from tqdm import tqdm

#load surrogate attention masks
generated_attention_maps=np.load('question_attention_masks.npy')
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()


# This section aligns training examples, validation examples and surrogate attention maps
train =[]
val=[]
for i in tqdm(generated_attention_maps):
    flag=0

    for j in train_iter:
        if(np.array_equal(i[0],j.text[0].numpy())==True):
            train.append(j)
            flag=1
            break
    if(flag==0):
        for j in valid_iter:
            if(np.array_equal(i[0],j.text[0].numpy())==True):
                train.append(j)
                flag=1
                break
print("length of training set is ",len(train))

for i in tqdm(train_iter):
    val.append(i)
for i in tqdm(valid_iter):
    val.append(i)

    
for i in tqdm(train): 
    for j in val:
        if(np.array_equal(i.text[0].numpy(),j.text[0].numpy())==True):
            val.remove(j)
            break
    

print("length of valisation set is ",len(val))

#############################################################################################




#generates text attention visualisation
def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k].split(" ");
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    console.log(intensity[i])
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    console.log(intensity[i])
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    console.log(intensity[i])
    if (k%2 == 0) {
    var heat_text = "<p><br><b>Example:</b><br>";
    } else {
    var heat_text = "<b>Example:</b><br>";
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
  
    return


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/dim
    return torch.exp(-kernel_input) # (x_size, y_size)

#computes mmd loss
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(tqdm(train)):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        optim.zero_grad()
        prediction,q_att = model(text)
        loss = loss_fn(prediction, target)
        #supervision loss
        loss2=compute_mmd(q_att.type(torch.DoubleTensor).cpu(),torch.from_numpy(generated_attention_maps[idx][1]).type(torch.DoubleTensor).cpu())

        loss=loss.type(torch.DoubleTensor).cpu()+10000*loss2
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss=loss.type(torch.FloatTensor)
        loss.backward()

        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        

        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train), total_epoch_acc/len(train)

def eval_model(model, val):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val)):
            text = batch.text[0]

            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction,q_att = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val), total_epoch_acc/len(val)
	

learning_rate = 2e-5
batch_size = 1
output_size = 6
hidden_size = 256
embedding_length = 300
model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy
max_accuracy=0
for epoch in range(10):
    train_loss, train_acc = train_model(model, train, epoch)
    val_loss, val_acc = eval_model(model, val)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    if(val_acc>max_accuracy):
        max_accuracy=val_acc
        torch.save(model.state_dict(),'supervision_attention_TREC.pkl')

test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

