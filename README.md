# Clean Language Models
This repository contains the implementation of several character-level language models for text generation. The work aims at clarity in the implementation and modularity.  

Currently, the following models are available:

## **Bigram Model**

Evaluation loss after a single epoch:
<div align="center">
  <img src="https://github.com/meraccos/nizami/blob/main/losses/bigram.svg" alt="Bigram loss" width="400" height="400">
</div>

Example text generation after a single epoch:

<span style="background-color: lightgray; padding: 10px; display: block; text-align: center;">
  <i>
    
    QUSCUSTAUS:
    And aye a wo ache moove remewes firavoorenong qurell b  
    SPO akid it eengrd me, way, dreanthimalleat tesil ' thacisssenedat  
    DLI wanas, chof oicrlds n nt panefatyould Serde! pr t ak bun wone crs:  
    KEWer l ISiwe t ty:  
    Whe vecaldeedes,  
    Bo pe;
  </i>
</span>

## **LSTM Model**

Evaluation loss after a single epoch:
<div align="center">
  <img src="https://github.com/meraccos/nizami/blob/main/losses/lstm.svg" alt="LSTM loss" width="400" height="400">
</div>

Example text generation after a single epoch:

<span style="background-color: lightgray; padding: 10px; display: block; text-align: center;">
  <i>
    
    Do knight!  
      
    WARWICK:  
    Whom good all fair east for a stristalle some their wind;  
    She lords, Kings me? Nap I will is Warwer.  
    Let you, grow gencuatchs of born tears?  
    Whearor proed of France God iond and some none  
    Which thou melmel buinty than and abters   
  </i>
</span>

### Datasets
The dataset to be trained can be put into the datasets folder. Originally, there are two datasets available:
* **nizami.txt**: concatenated text of five books of Nizami Ganjavi, in Azerbaijani. **(~1.6M characters)**
* **tinyshakespeare.txt**: concatenated text of all the famous works of William Shakespeare, in English. **(~ 1M characters)**


### Training
To train each model, run:
```python
python3 <model name>.py
```

To generate text from the latest saved model, run: (eg. LSTM, model lstm_3.pt, 250 characters)
```python
python3 generate.py --model_name=lstm --model_idx=3 --num_chars=250
```
