GPU=0

# Zero-Shot Evaluation
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet10 --batch_size=256 --n_shot 0 --method zsclip --score logitgap_topN --ood_dataset ImageNet20  --MCM_topN 5
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet20 --batch_size=256 --n_shot 0 --method zsclip --score logitgap_topN --ood_dataset ImageNet10  --MCM_topN 10


score='MCM' # 'energy' 'max-logit' 
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet10 --batch_size=256 --n_shot 0 --method zsclip --score ${score} --ood_dataset ImageNet20
CUDA_VISIBLE_DEVICES=${GPU} python eval_ood_detection.py --in_dataset ImageNet20 --batch_size=256 --n_shot 0 --method zsclip --score ${score} --ood_dataset ImageNet10


