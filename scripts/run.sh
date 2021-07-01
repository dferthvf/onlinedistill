# ======================
# exampler commands on miniImageNet
# ======================

# online distillation
python train_od.py --trial od_999_200 --model_path /path/to/save/your/model --tb_path /path/to/save/your/tensorboard_log --data_root /path/to/your/data --cosine

python eval_fewshot.py --model_path checkpoint/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_cosine_trial_od_999/ckpt_epoch_200.pth --data_root /home/yuewang/hdd/byol/tieredImageNet
