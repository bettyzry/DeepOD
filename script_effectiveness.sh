flag=.effectivenss
entity_combined=0

for model in DIF TranAD USAD COUTA AnomalyTransformer TimesNet;do
    echo $model
    dataset=SMD,MSL,SMAP,SWaT,Epilepsy,DSADS
    python -u testbed/testbed_unsupervised_tsad.py \
    --dataset $dataset \
    --model $model \
    --runs 5 --entity_combined $entity_combined \
    --flag $flag \
    > /dev/null &
done

