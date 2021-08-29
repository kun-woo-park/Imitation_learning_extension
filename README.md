# Imitation_learning_extension
## Implementation
``` bash
# data generation
# you can modify the velocity of aircraft by following variables in data_gen.py. Vm_start, Vm_end, Vt_start, Vt_end 
python3 data_gen.py --num_data <Number of data>
```
``` bash
# train model
python3 colision_avoidance_net.py
```

You can test model with colavoid_extention_test.ipynb
