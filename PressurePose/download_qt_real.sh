#!/bin/bash
mkdir -p ../data_BR/real/S103

wget -O ../data_BR/real/S103/participant_info_red.p S103 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/HFW8JZ
wget -O ../data_BR/real/S103/prescribed.p S103 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/1X6YXX
wget -O ../data_BR/real/S103/p_select.p S103 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/SABD5Q
