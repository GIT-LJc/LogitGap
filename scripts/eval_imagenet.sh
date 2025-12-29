GPU=0
ood_dataset='near_semantic'  # 'covariate'

# Zero-Shot Evaluation
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet --batch_size=256 --n_shot 0 --method zsclip --score logitgap_topN --ood_dataset ${ood_dataset}  --MCM_topN 88

score='MCM' # 'energy' 'max-logit' 
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet --batch_size=256 --n_shot 0 --method zsclip --score ${score} --ood_dataset ${ood_dataset} 

# Few-Shot Evaluation
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet --ood_dataset ${ood_dataset} --batch_size=1 --n_shot 4 --method coop --n_epoch 3 --score MCM 
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet --ood_dataset ${ood_dataset} --batch_size=1 --n_shot 4 --method coop --n_epoch 3 --score logitgap_topN --MCM_topN 88

