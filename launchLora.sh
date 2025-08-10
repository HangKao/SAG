# Train the SAG model ()

unset TMPDIR

rm log.txt

python mainSAMlora.py -sam_ckpt ./pretrain/sam_vit_b_01ec64.pth \
                    -gpu_device 0,1 \
                    -b 2 \
                    -image_size 1024 \
                    -val_freq 1 \
                    -epoch 200 \
                    -vis 40 \
                    -lr 1e-5 \
                    -exp_name $1 

