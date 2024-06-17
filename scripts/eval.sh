cd projects/StyTR-2/;

CUDA_VISIBLE_DEVICES=0
python main.py --style_dir /home/filippo/datasets/wikiart --content_dir /home/filippo/datasets/train2014  \
                --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth \
                --decoder_path checkpoints/mamba/c_style_no_conv_skip_conn_all_layer/decoder_iter_160000.pth --embedding_path checkpoints/mamba/c_style_no_conv_skip_conn_all_layer/embedding_iter_160000.pth \
                --Trans_path checkpoints/mamba/c_style_no_conv_skip_conn_all_layer/transformer_iter_160000.pth \
                --model_name c_style_no_conv_skip_conn_all_layer --output eval/c_style_no_conv_skip_conn_all_layer_256 --seed 123456 --mode eval \
                --use_mamba_enc  --use_mamba_dec --c_style --rnd_style --img_size 256 #--use_conv