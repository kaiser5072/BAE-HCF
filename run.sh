# TRAINING
for i in 0 1 2 3 4
do
    python main.py --dims 500 200 --data_dir 'Input/citeulike-t/warm' -b 512 -e 500 --display_every 4000 --l2_lambda 0.00004 -r 0.5 -f $i
done

for i in 0 1 2 3 4
do
    python main.py --dims 500 200 --data_dir 'Input/citeulike-t/warm' -b 512 -e 500 --display_every 4000 --l2_lambda 0.00004 -r 0.5 -f $i -m predict
done

python ./utils/average_n_fold.py