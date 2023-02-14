#!/bin/bash

#python3 ../src/testbed_STDN_32.py --alpha 0.45 --voting True --postfix alpha_0.45_voting_True
#python3 ../src/testbed_STDN_32.py --alpha 0.50 --voting False --postfix alpha_0.50_voting_False

python3 ../src/testbed_STDN_32.py --alpha 0.25 --voting False --postfix alpha_0.40_voting_False
python3 ../src/testbed_STDN_32.py --alpha 0.30 --voting False --postfix alpha_0.40_voting_False
python3 ../src/testbed_STDN_32.py --alpha 0.35 --voting True --postfix alpha_0.40_voting_True
python3 ../src/testbed_STDN_32.py --alpha 0.40 --voting False --postfix alpha_0.50_voting_False
python3 ../src/testbed_STDN_32.py --alpha 0.45 --voting False --postfix alpha_0.40_voting_False
