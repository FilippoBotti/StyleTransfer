cd projects/StyTR-2/;

CUDA_VISIBLE_DEVICES=0

python main.py --model_name c_input_with_pos_embed --style_dir /home/filippo/datasets/wikiart/ --content_dir /home/filippo/datasets/train2014  \
                --style_test_dir /home/filippo/datasets/sty_tr_test/style --content_test_dir /home/filippo/datasets/sty_tr_test/content  \
                --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth --print_every 1000 --eval_every 320000 \
                --checkpoints_dir checkpoints/hidden_cross/c_input_with_pos_embed --results_dir outputs/hidden_cross/c_input_with_pos_embed  \
                --batch_size 4 --mode train  --use_mamba_enc --use_mamba_dec --d_state 16 --c_input --lr 1e-4 --rnd_style --use_pos_embed
                #   --decoder_path checkpoints/mamba/c_input_512/decoder_iter_160000.pth --embedding_path checkpoints/mamba/c_input_512/embedding_iter_160000.pth \
                # --Trans_path checkpoints/mamba/c_input_512/transformer_iter_160000.pth --resume_iter 160000 --continue_train --max_iter 320000