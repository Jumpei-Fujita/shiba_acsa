from shiba import Shiba, CodepointTokenizer, get_pretrained_state_dict
tokenizer = CodepointTokenizer()

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import itertools
import random
from IPython.display import display, HTML


aspects = ['Location#Transportation', 'Location#Downtown',
       'Location#Easy_to_find', 'Service#Queue', 'Service#Hospitality',
       'Service#Parking', 'Service#Timely', 'Price#Level',
       'Price#Cost_effective', 'Price#Discount', 'Ambience#Decoration',
       'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary', 'Food#Portion',
       'Food#Taste', 'Food#Appearance', 'Food#Recommend']


def get_label_number(number):
  if number == 1:
    l = 0
  elif number == 0:
    l = 1
  elif number == -1:
    l = 2
  elif number == -2:
    l = 3
  return l 

def get_label(dataset, batch, gpu):
  star_tensors = []
  aspects_tensors = []
  for i in batch:
    aspect = []
    star_tensors.append(dataset['star'][i])
    for j in aspects:
      aspect.append(get_label_number(dataset[j][i]))
    aspects_tensors.append(aspect)
  star_tensors = torch.FloatTensor(star_tensors)
  aspects_tensors = torch.LongTensor(aspects_tensors)
  if gpu:
    star_tensors = star_tensors.cuda()
    aspects_tensors = aspects_tensors.cuda()

  return {'star' : star_tensors, 'aspect' : aspects_tensors}

def get_input(dataset, batch_size, max_input_tokens, gpu):
  batch = np.random.choice(dataset.index.size, size=(batch_size, ), replace=False)
  review_array = []
  for i in batch:
    review_array.append(dataset['Japanese'][i])
  review_tensors = tokenizer(review_array, truncation=True, padding=True, max_length=max_input_tokens, return_tensors='pt')
  attention_mask = review_tensors['attention_mask']
  review_tensor = review_tensors['input_ids']
  if gpu:
    review_tensor = review_tensor.cuda()
    attention_mask = attention_mask.cuda()
  label = get_label(dataset, batch, gpu)

  return review_tensor, attention_mask, label

class Aspect_attention(nn.Module):
    def __init__(self):
        super(Aspect_attention, self).__init__()
        self.w_in = nn.Linear(768, 768, bias=False)
        self.w_i = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(768, 4)

    def forward(self, input, attention_mask=None):
        #[b,z,d]
        a = self.tanh(self.w_in(input))
        #[b,z,d]
        a = self.w_i(a)
        #[b,z,1]
        if attention_mask is not None:
            a =a.masked_fill(attention_mask == 0, float('-inf'))
        a = self.softmax(a)

        context = torch.matmul(input.transpose(1,2), a).transpose(1, 2)
        #context:[b, 1, d]
        output = self.out(context)
        return output, a.transpose(1, 2) #a:[b, 1, z]

class Rating_Prediction(nn.Module):
    def __init__(self):
        super(Rating_Prediction, self).__init__()
        self.star_w = nn.Linear(768, 768)
        self.star_beta = nn.Linear(768, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input):
        #input : [b, 768]
        star = self.tanh(self.star_w(input))
        #star = self.dropout(star)
        star = self.star_beta(star)
        return star

class ASAP_SHIBA(nn.Module):
    def __init__(self):
        super(ASAP_SHIBA, self).__init__()
        shiba_model = Shiba()
        shiba_model.load_state_dict(get_pretrained_state_dict())
        self.bert = shiba_model
        self.dropout = nn.Dropout(0.1)
        self.LT_attention = Aspect_attention()
        self.LD_attention = Aspect_attention()
        self.LE_attention = Aspect_attention()
        self.SQ_attention = Aspect_attention()
        self.SH_attention = Aspect_attention()
        self.SP_attention = Aspect_attention()
        self.ST_attention = Aspect_attention()
        self.PL_attention = Aspect_attention()
        self.PC_attention = Aspect_attention()
        self.PD_attention = Aspect_attention()
        self.AD_attention = Aspect_attention()
        self.AN_attention = Aspect_attention()
        self.ASpace_attention = Aspect_attention()
        self.ASanitary_attention = Aspect_attention()
        self.FP_attention = Aspect_attention()
        self.FT_attention = Aspect_attention()
        self.FA_attention = Aspect_attention()
        self.FR_attention = Aspect_attention()
        self.rating_prediction = Rating_Prediction()
        
    
    def forward(self, input, attention_mask=None):
        bert_out = self.bert(input, attention_mask=attention_mask)
        bert_out_att = bert_out[0]
        bert_out_star = bert_out[0][:, 0, :]

        bert_out_star = self.dropout(bert_out_star)
        bert_out_att = self.dropout(bert_out_att)

        star = self.rating_prediction(bert_out_star)
        attention_mask = attention_mask.unsqueeze(-1)

        aspects_prediction = []
        aspects_prediction_attention = []
        aspects_attentions = [
                              self.LT_attention, 
                              self.LD_attention,
                              self.LE_attention,
                              self.SQ_attention,
                              self.SH_attention,
                              self.SP_attention,
                              self.ST_attention,
                              self.PL_attention,
                              self.PC_attention,
                              self.PD_attention,
                              self.AD_attention,
                              self.AN_attention,
                              self.ASpace_attention,
                              self.ASanitary_attention,
                              self.FP_attention,
                              self.FT_attention,
                              self.FA_attention,
                              self.FR_attention
                              ]
        for i in aspects_attentions:
          p, a = i(bert_out_att, attention_mask)
          aspects_prediction.append(p)
          aspects_prediction_attention.append(a)


        #[b, 18, 4]
        output = torch.cat(aspects_prediction, dim=1)
        #[b, 18, 1]
        attentions = torch.cat(aspects_prediction_attention, dim=1)

        return star, output, attentions 

def train_i(batch_size, max_length, train_dataset, optimizer, model, gpu):
    criterion_acsa = nn.CrossEntropyLoss()
    criterion_rp = nn.MSELoss()
    batch = np.random.choice(train_dataset.index.size, (batch_size, ))
    model.train()
    optimizer.zero_grad()
    review_tensor, attention_mask, label = get_input(train_dataset, batch_size, max_length, gpu)
    star, output, attentions = model(review_tensor, attention_mask)
    output = output.view(batch_size*18, -1).contiguous()
    label_output = label['aspect'].view(batch_size*18, ).contiguous()
    loss_acsa = criterion_acsa(output, label_output)
    loss_rp = criterion_rp(star.squeeze(1), label['star'])**0.5
    loss = loss_acsa + loss_rp
    loss.backward()
    optimizer.step()
    return [loss.item(), loss_acsa.item(), loss_rp.item()]
    del loss, output

def validation_i(batch_size, max_length, validation_dataset, model, gpu):
    with torch.no_grad():
      criterion_acsa = nn.CrossEntropyLoss()
      criterion_rp = nn.MSELoss()
      batch = np.random.choice(validation_dataset.index.size, (batch_size, ))
      model.eval()
      review_tensor, attention_mask, label = get_input(validation_dataset, batch_size, max_length, gpu)
      star, output, attentions = model(review_tensor, attention_mask)
      output = output.view(batch_size*18, -1).contiguous()
      label_output = label['aspect'].view(batch_size*18, ).contiguous()
      loss_acsa = criterion_acsa(output, label_output)
      loss_rp = criterion_rp(star.squeeze(1), label['star'])**0.5
      loss = loss_acsa + loss_rp
    return [loss.item(), loss_acsa.item(), loss_rp.item()]
    del loss, output

def train(model, epochs, show_count, batch_size, max_length, train_dataset, validation_dataset, gpu,model_path):
    train_history_label = ['train_loss', 'train_acsa', 'train_rp']
    validation_history_label = ['validation_loss', 'validation_acsa', 'validation_rp']

    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    train_history = []
    validation_history = []
    acc = []
    star_error = []
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        loss_train = train_i(batch_size, max_length, train_dataset, optimizer, model, gpu)
        train_history.append(loss_train)
        loss_validation = validation_i(batch_size, max_length, validation_dataset, model, gpu)
        validation_history.append(loss_validation)

        if epoch % show_count == 0:
            elapsed_time = time.time() - start_time
            print("epoch : ", epoch, "loss_train : ", loss_train[0], " loss_validation : ", loss_validation[0])
            print("elapsed_time : ", elapsed_time)
            #if epoch != 0:
                #validation_get_heatmap(validation_dataset, model)
            rp, acsa = test_loop(validation_dataset, gpu, model, 200, max_length)
            acc.append(acsa)
            star_error.append(rp)
            
            print("===========")
        if (epoch % 10000 == 0) and (epoch != 0):
            torch.save(model.state_dict(), model_path)
            print("saved : ", model_path)
    print("trainning finished")
    torch.save(model.state_dict(), model_path)
    print("saved : ", model_path)
    train_history = np.array(train_history)
    validation_history = np.array(validation_history)
    plots_n(train_history, train_history_label)
    plots_n(validation_history, validation_history_label)
    acc = np.concatenate(acc)
    print(acc.shape)
    plots_n(acc[:, :9], aspects[:9])
    plots_n(acc[:, 9:], aspects[9:])
    plots(star_error, "star", 5)
    plots_n_conv(train_history, train_history_label, 1000)
    plots_n_conv(validation_history, validation_history_label, 1000)
    plots_n_conv(acc[:, :9], aspects[:9], 5)
    plots_n_conv(acc[:, 9:], aspects[9:], 5)
    return model, train_history, train_history_label

def test_loop(dataset, gpu, model, batch_size, max_length):
  with torch.no_grad():
    batch = np.random.choice(dataset.index.size, (batch_size, ), replace=False)
    model.eval()
    review_tensor, attention_mask, label = get_input(dataset, batch_size, max_length, gpu)
    rps = []
    accses = []
    for i in range(batch_size):
      star, output, attentions = model(review_tensor[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
      #star
      rp_mean_error = torch.abs(label['star'][i].unsqueeze(0) - star.squeeze(-1))
      #aspects
      accs = output.argmax(dim=2) == label['aspect'][0].unsqueeze(0)
      rps.append(rp_mean_error.unsqueeze(0))
      accses.append(accs)
      del star, output, attentions, rp_mean_error, accs
    rps = torch.cat(rps, dim=0).sum(0).item()/batch_size
    accses = torch.cat(accses, dim=0).sum(0)/batch_size
    
    return rps, accses.cpu().unsqueeze(0).numpy()

def plots_n_conv(history, aspects_a, num):
    fig = plt.figure(figsize=(9.0, 6.0))
    plt.rcParams["font.size"] = 18
    print("conv, num=", num)
    b=np.ones(num)/num
    for i in range(len(aspects_a)):
        plt.plot(np.convolve(history[:, i], b, mode='same'), label=aspects_a[i])
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5,), borderaxespad=0)
    plt.show() 

def plots_n(history, aspects_a):
    fig = plt.figure(figsize=(9.0, 6.0))
    plt.rcParams["font.size"] = 18
    for i in range(len(aspects_a)):
        plt.plot(history[:, i], label=aspects_a[i])
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5,), borderaxespad=0)
    plt.show() 

def plots(history1, history1_name, num, history2=None, history2_name=None):
    print("conv, num=", num)
    fig = plt.figure(figsize=(9.0, 6.0))
    plt.rcParams["font.size"] = 18
    b=np.ones(num)/num
    plt.plot(np.convolve(history1, b, mode='same'), label=history1_name)
    if history2 is not None:
        plt.plot(history2, label=history2_name)
    plt.grid()
    plt.xlabel('epochs')
    plt.legend()
    plt.show()  

sentiment = ["positive", "neutral", "negative", "not_mentioned"]

def get_heatmap(text, model):
  with torch.no_grad():
    input = tokenizer(text, return_tensors='pt')
    rp, acsa, a = model(input['input_ids'].cuda(), input['attention_mask'].cuda())
    print("rating_prediction : ", rp[0][0].cpu().numpy())
    a = a.cpu().numpy()
    r = tokenizer.convert_ids_to_tokens(input['input_ids'][0])
    s = []
    for i in range(len(aspects)):
      plt.figure(figsize=(2*len(r), 2))
      print(aspects[i], " : ", sentiment[acsa.argmax(-1)[0][i]])
      sns.heatmap(a[:, i], fmt='', cmap='Blues', annot=np.array([r]))
      plt.show()
      s.append(sentiment[acsa.argmax(-1)[0][i]])
    return s

def validation_get_heatmap(validation_df, model):
    random_datanum = np.random.choice(len(validation_df))
    print("^^^^^^^^^^^^^^^^^^")
    print("text : ", validation_df['Japanese'][random_datanum])
    print("star : ", validation_df['star'][random_datanum])
    for i in aspects:
        print(i, " : ", sentiment[get_label_number(validation_df[i][random_datanum])])
    print("prediction")
    s = get_heatmap(validation_df['Japanese'][random_datanum], model)
    for i in range(len(aspects)):
        print(aspects[i], " : ",s[i])
    print("^^^^^^^^^^^^^^^^^^")
        

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'

def represent_sentiment(asp, s):
    if s == "positive":
        print(asp , " : ", pycolor.BLUE, s, pycolor.END)
    elif s == "neutral":
        print(asp, " : ", pycolor.GREEN, s, pycolor.END)
    elif s == "negative":
        print(asp, " : ", pycolor.RED, s, pycolor.END)
    else:    
        print(asp, " : ", pycolor.BLACK, s, pycolor.END)

def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(sentence, attns):
    html = ""
    for word, attn in zip(sentence, attns):
        html += ' ' + highlight(
            word,
            attn
        )
    return html
    
def get_attention_map(text, model):
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        r_tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])
        rp, acsa, a = model(input['input_ids'].cuda(), input['attention_mask'].cuda())
        a = a.cpu().numpy()
        print("rating_prediction : ", rp.cpu().numpy()[0][0])
        sentiments = []
        for i in range(len(aspects)):
            s = sentiment[acsa.argmax(-1)[0][i]]
            asp = aspects[i]
            represent_sentiment(asp, s)
            display(HTML(mk_html(r_tokens, a[0][i])))
        sentiments.append(s)

def get_attention_map2(text, model):
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        r_tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])
        rp, acsa, a = model(input['input_ids'].cuda(), input['attention_mask'].cuda())
        a = a.cpu().numpy()
        print("rating_prediction : ", rp.cpu().numpy()[0][0])
        sentiments = []
        for i in range(len(aspects)):
            s = sentiment[acsa.argmax(-1)[0][i]]
            asp = aspects[i]
            represent_sentiment(asp, s)
            display(HTML(mk_html(r_tokens, a[0][i])))
            sentiments.append(s)
            
    return sentiments

def validation_get_attention_map(validation_df, model):
    random_datanum = np.random.choice(len(validation_df))
    print("^^^^^^^^^^^^^^^^^^")
    print("text : ", validation_df['Japanese'][random_datanum])
    print("star : ", validation_df['star'][random_datanum])
    for i in aspects:
        print(i, " : ", sentiment[get_label_number(validation_df[i][random_datanum])])
    print("prediction")
    get_attention_map(validation_df['Japanese'][random_datanum], model)
    print("^^^^^^^^^^^^^^^^^^")

