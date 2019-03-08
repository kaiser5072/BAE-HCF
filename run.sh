# TRAINING
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 0 -r 0.5
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 1 -r 0.5
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 2 -r 0.5
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 3 -r 0.5
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 4 -r 0.5

# EVALUATION
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 0 -m predict
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 1 -m predict
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 2 -m predict
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 3 -m predict
CUDA_VISIBLE_DEVICES=0 python main.py --dims 500 200 --data_dir 'Input/tmp' -b 512 -e 1400 --display_every 4000 --l2_lambda 0.00004 -r 0.0 -f 4 -m predict
