# art-fid
cd projects/StyTR-2/evaluation;

CUDA_VISIBLE_DEVICES=0
python eval_artfid.py --sty /home/filippo/datasets/sty_tr_test_256/sty_eval --cnt /home/filippo/datasets/sty_tr_test_256/cnt_eval --tar /home/filippo/projects/StyTR-2/results/hidden_cross_pos_embed/evaluation_256

# histo loss
python eval_histogan.py --sty /home/filippo/datasets/sty_tr_test_256/sty_eval --tar /home/filippo/projects/StyTR-2/results/hidden_cross_pos_embed/evaluation_256