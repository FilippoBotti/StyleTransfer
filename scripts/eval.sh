cd projects/StyTR-2/;

CUDA_VISIBLE_DEVICES=0
python main.py --style_dir /home/filippo/datasets/wikiart --content_dir /home/filippo/datasets/train2014     \
                --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth \
                --decoder_path checkpoints/hidden_cross/c_input/decoder_iter_160000.pth --embedding_path checkpoints/hidden_cross/c_input/embedding_iter_160000.pth \
                --Trans_path checkpoints/hidden_cross/c_input/transformer_iter_160000.pth \
                --model_name c_input_with_pos_embed --output results/hidden_cross_pos_embed/evaluation_256 --seed 123456 --mode eval \
                --use_mamba_enc  --use_mamba_dec --c_input --rnd_style --img_size 256 --d_state 16 