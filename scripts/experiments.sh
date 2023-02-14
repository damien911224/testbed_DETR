#!/bin/bash

# testbed_Search52.py: default no MHP
# testbed_Search53.py: default MHP (not separate)
# testbed_Search54.py: MHP with classification and actionness (not feature separate)
# testbed_Search55.py: MHP with valley classification
# testbed_Search57.py: MHP with feature separate

#python3 ../src/testbed_Search57_00.py --postfix HG_MHP_base_C256_S2_L4_I512_CH_02_AH_03_subset_20
#python3 ../src/testbed_Search57_01.py --postfix HG_MHP_base_C256_S2_L4_I512_CH_02_AH_03_subset_20_all_stages

python3 ../src/testbed_Search57_00.py
python3 ../src/testbed_Search57_01.py
python3 ../src/testbed_Search57_02.py
python3 ../src/testbed_Search57_03.py

