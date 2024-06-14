cd projects/StyTR-2/;

CUDA_VISIBLE_DEVICES=0
python main.py --style_dir /home/filippo/datasets/wikiart --content_dir /home/filippo/datasets/train2014  \
                --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth \
                --decoder_path checkpoints/mamba/cross_selective_scan_c_input_rand/decoder_iter_160000.pth --embedding_path checkpoints/mamba/cross_selective_scan_c_input_rand/embedding_iter_160000.pth \
                --Trans_path checkpoints/mamba/cross_selective_scan_c_input_rand/transformer_iter_160000.pth \
                --model_name c_input_rand --output eval/cross_selective_scan_c_input_rand --seed 123456 --mode eval \
                --use_mamba_enc  --use_mamba_dec --c_input  --rnd_style --img_size 512 #--use_conv