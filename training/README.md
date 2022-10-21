This repo requires the use of Brians conda environment /dccstor/bmbelgod1/environments/moltran 
the files of focus are the run_pubchem_light.sh and train_pubchem_light.py

If you want to run an experiment on just pubchem then just set the train_load to 'pubchem', if you just want to train on zinc then set 
that train_load to 'zinc', if you wamt to train on the concat of both then set the flag train_load to 'both'
TO DO
    1)better faster tokenizer (right now we are using the huggingface datasets package. Still using datasets but now with a bucketing method in each minibatch 
    2) erase data chache created by datastes package at end of traning and switching machines. this is done, if debug flag is used the cache will not be removed at the end of the program but if debug flag is not used the cache will be removed (done)
    3) better checkpointing, (have 2 methods of checkponiting, one that saves at the end of each epoch, this is also done
        with no overwrites and another that saves every x iterations with overwrites just so we don't have
        to restart if something goes wrong (method 2 is done).
    4) save/load seeds (done) 
    5) initialize seeds (done)
    6) hyper parametr tunning need to do still

