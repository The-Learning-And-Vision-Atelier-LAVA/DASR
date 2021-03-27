# noise-free degradations with isotropic Gaussian blurs
python test.py --test_only \
               --dir_data='D:/LongguangWang/Data' \
               --data_test='Set14' \
               --model='blindsr' \
               --scale='2' \
               --resume=600 \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig=1.2


# general degradations with anisotropic Gaussian blurs and noises
python test.py --test_only \
               --dir_data='D:/LongguangWang/Data' \
               --data_test='Set14' \
               --model='blindsr' \
               --scale='4' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --noise=10.0 \
               --theta=0.0 \
               --lambda_1=0.2 \
               --lambda_2=4.0

cmd /k