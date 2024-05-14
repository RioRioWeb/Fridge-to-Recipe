results='./classifier/results/veg_dataset'
data='./classifier/data/veg_dataset/'
results='./classifier/results/veg_and_fruit'
data='./classifier/data/veg_and_fruits_dataset/'
model='resnet18-fine'
save_dir=$results/$model
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=0 nohup python classifier/train.py \
    --data_dir $data \
    --results_dir $results \
    --model $model > $save_dir/train.log &



results='./classifier/results/veg_dataset'
data='./classifier/data/veg_dataset/'
model='resnet50-fine'
save_dir=$results/$model
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=2 nohup python classifier/train.py \
    --data_dir $data \
    --results_dir $results \
    --model $model > $save_dir/train.log &



results='./classifier/results/veg_dataset'
data='./classifier/data/veg_dataset/'
model='resnet18'
save_dir=$results/$model
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=3 nohup python classifier/train.py \
    --data_dir $data \
    --results_dir $results \
    --model $model > $save_dir/train.log &



results='./classifier/results/veg_dataset'
data='./classifier/data/veg_dataset/'
model='resnet50'
save_dir=$results/$model
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=4 nohup python classifier/train.py \
    --data_dir $data \
    --results_dir $results \
    --model $model > $save_dir/train.log &