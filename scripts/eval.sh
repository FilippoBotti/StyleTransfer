cd projects/StyTR-2/;

CUDA_VISIBLE_DEVICES=0
python main.py --style_dir /home/filippo/datasets/wikiart --content_dir /home/filippo/datasets/train2014  \
                --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth \
                --decoder_path checkpoints/mamba/c_style_skip_all_layer_channel_att/decoder_iter_160000.pth --embedding_path checkpoints/mamba/c_style_skip_all_layer_channel_att/embedding_iter_160000.pth \
                --Trans_path checkpoints/mamba/c_style_skip_all_layer_channel_att/transformer_iter_160000.pth \
                --model_name c_style_skip_all_layer_channel_att --output outputs/mamba/c_style_skip_all_layer_channel_att/eval_256 --seed 123456 --mode eval \
                --use_mamba_enc  --use_mamba_dec --c_style --rnd_style --img_size 256 --d_state 16  #--use_conv