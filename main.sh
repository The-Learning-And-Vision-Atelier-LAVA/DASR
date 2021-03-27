# noise-free degradations with isotropic Gaussian blurs
python main.py --dir_data='D:/LongguangWang/Data' \
               --model='blindsr' \
               --scale='2' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0


# general degradations with anisotropic Gaussian blurs and noises
python main.py --dir_data='D:/LongguangWang/Data' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='aniso_gaussian' \
               --noise=25.0 \
               --lambda_min=0.2 \
               --lambda_max=4.0

cmd /k
