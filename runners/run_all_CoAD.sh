# run the UCR(KDD21) dataset
# freq_tf_v2 IS the CoAD model

current_time=$(date +"%H-%M-%S")

for g in $(seq 1 250)
do
  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2 --timestamp $current_time --device "cuda:0" --plot True --num_epochs 300 --batch_size 64 --window_multiple 4 --lr 2e-3 --patch_size 8 --omega 10
done

# run the TSB-AD dataset
#for g in $(seq 225 236)
#do
# python train_classify_mask.py --dataset_name TSB-AD  --group_name $g --model_name freq_tf_v2 --timestamp $current_time --device "cuda:0" --plot True --num_epochs 300 --batch_size 128 --window_multiple 4 --lr 2e-3 --patch_size 8 --omega 10
#done
#
#for g in $(seq 237 256)
#do
# python train_classify_mask.py --dataset_name TSB-AD  --group_name $g --model_name freq_tf_v2 --timestamp $current_time --device "cuda:0" --plot True --num_epochs 300 --batch_size 128 --window_multiple 4 --lr 2e-3 --patch_size 8 --omega 10
#done
#
#for g in $(seq 260 276)
#do
# python train_classify_mask.py --dataset_name TSB-AD  --group_name $g --model_name freq_tf_v2 --timestamp $current_time --device "cuda:0" --plot True --num_epochs 300 --batch_size 128 --window_multiple 4 --lr 2e-3 --patch_size 8 --omega 10
#done
#
#for g in $(seq 287 301)
#do
# python train_classify_mask.py --dataset_name TSB-AD  --group_name $g --model_name freq_tf_v2 --timestamp $current_time --device "cuda:0" --plot True --num_epochs 300 --batch_size 128 --window_multiple 4 --lr 2e-3 --patch_size 8 --omega 10
#done
