# Clean Language Models
This repository contains the implementation of several character-level language models for text generation. The work aims at clarity in the implementation and modularity.
Currently, the following models are available:

## **Bigram Model**

Evaluation loss after a single epoch:
<div align="center">
  <img src="https://github.com/meraccos/nizami/blob/main/losses/bigram.svg" alt="Bigram loss Text" width="400" height="400">
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

To train each model, run:
```python
python3 <model name>.py
```

To generate text from the latest saved model, run:
```python
python3 generate.py --model_name=<model name> --model_idx=<model_idx> --num_chars=<num_chars>
```
