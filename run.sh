#!/bin/bash

# Running multiple Python scripts in parallel with logging
nohup python Fed.py --method FedAvg > FedAvg.log 2>&1 &
nohup python Fed.py --method FedProx > FedProx.log 2>&1 &
nohup python Fed.py --method Stochastic > Stochastic.log 2>&1 &
nohup python Fed.py --method Stochastic_each_epoch > Stochastic_each_epoch.log 2>&1 &
# ... add as many scripts as you need

# Wait for all of them to finish
wait

echo "All scripts have completed."
