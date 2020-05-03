import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.selfAttention import SelfAttention
from tqdm import tqdm

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    

def generate_masks(model, train_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    to_write_all=[]
    max_length=0
    print("length of training set if",len(val_iter))
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            text = batch.text[0]
            q=text[0]       
            text_numpy=text.numpy()
            
            n=batch.text[1][0].item()
            mq=[]
            q_len=text[0].size()[0]
            mq.append(q.clone())
            mask = torch.ones(n)
            mask = mask.long()
            for i in range(n):
                mask[i] = 0
                for j in range(i+1,n):
                    mask[j] = 0
                    m = torch.mul(mask,q[0:n])
                    temp = []
                    temp.append(m.clone())
                    temp2 = torch.zeros(q_len-n)
                    temp.append(temp2.long())
                    temp = torch.cat(temp)
                    mq.append(temp.clone())
                    mask[j] = 1
                mask[i] = 1

            mq=torch.stack(mq)
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            
            attMax = torch.ones(1,mq.size()[1])
            max_prediction=-100000
            for txt in mq:      
                txt=txt.view(1,-1)
                prediction ,q_att= model(txt)
                if(prediction[0][target[0].item()].item()>max_prediction):
                    max_prediction=prediction[0][target[0].item()].item()
                    attMax=q_att


            attMax_numpy=attMax.numpy()
            to_write=[]
            to_write.append(text_numpy)
            to_write.append(attMax_numpy)
            to_write = np.asarray(to_write)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            to_write_all.append(to_write)

    to_write_all=np.array(to_write_all)
    np.save("question_attention_masks",to_write_all)
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	

learning_rate = 2e-5
batch_size = 1
output_size = 6
hidden_size = 256
embedding_length = 300
model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy
max_accuracy=0

# load weights of the original self-attention model 
model.load_state_dict(torch.load('self_attention_TRECpkl'))

#generate masks for training dataset
loss, acc =generate_masks(model, train_iter)

print("WE HAVE CREATED THE MASKS SUCCESSFULLY !!!!")
