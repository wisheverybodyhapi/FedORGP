# 后台运行此脚本指令。如果一次性准备跑大量实验，记得sleep！！
# nohup sh total.sh > /dev/null 2>&1 &


# test command (3 channels image like Cifar100)
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth -clam 1 -slam 1 -gam 0.1 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGH -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedTGP -lam 0.1 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo LG-FedAvg -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FML -al 0.5 -bt 0.5 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedProto -lam 1 -sr 0 -er 1


# test command (grey image like EMNIST)
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth -clam 1 -slam 1 -gam 0.1 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGH -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedTGP -lam 0.1 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo LG-FedAvg -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 1 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FML -al 0.5 -bt 0.5 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 -sr 0 -er 1
# python -u main.py -t 1 -ab 1 -jr 0.05 -lbs 64 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedProto -lam 1 -sr 0 -er 1

# 1. find best hyper-param
# find best hyper-param combination of FedOrth when batchsize=32,dataset=Cifar100 (done)
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 0.1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 0.1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 0.1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 0.1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 0.1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 10 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 10 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 0.1 -slam 10 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 0.1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 10 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 10 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 1 -slam 10 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 0.1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 0.1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 0.1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 1 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 1 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 10 -gam 0.1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 10 -gam 1 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 32 -clam 10 -slam 10 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &


# find best server epochs when clam 10 -slam 1 -gam 10, dataset=Cifar100, batchsize=16 (deleted)
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 16 -clam 10 -slam 1 -gam 10 -se 1 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=16_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 16 -clam 10 -slam 1 -gam 10 -se 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=16_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 16 -clam 10 -slam 1 -gam 10 -se 3 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=16_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 16 -clam 10 -slam 1 -gam 10 -se 5 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=16_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -jr 1 -lbs 16 -clam 10 -slam 1 -gam 10 -se 7 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedOrth --save_folder_name results/find_best_hyper-param_when_batchsize=16_dataset=Cifar100 > /dev/null 2>&1 &


# 2. test all 
# test all batchsize=32 dataset=Cifar10 2 classes per client (done)
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedProto --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGH -se 1 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedTGP -lam 0.1 -se 100 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo LG-FedAvg --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedGen -nd 32 -glr 0.0003 -se 50 -hd 512 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FML -al 0.5 -bt 0.5 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedOrth -clam 10 -slam 1 -gam 10 -se 1 --save_folder_name results/test_all_batchsize=32_dataset=Cifar10 > /dev/null 2>&1 &


# test all when batchsize=32 dataset=Cifar100 10 classes per client (done)
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedProto -lam 1 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGH -se 1 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 2 -sr 0 -er 100 -algo FedTGP -lam 0.1 -se 100 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo LG-FedAvg --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 7 -sr 0 -er 100 -algo FedGen -nd 32 -glr 0.0003 -se 50 -hd 512 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FML -al 0.5 -bt 0.5 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth -clam 10 -slam 1 -gam 10 -se 1 --save_folder_name results/test_all_batchsize=32_dataset=Cifar100 > /dev/null 2>&1 &


# test all when batchsize=32 dataset=EMNIST 7 classes per client (done)
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 4 -sr 0 -er 100 -algo FedProto -lam 1 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGH --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FedTGP -lam 0.1 -se 100 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 5 -sr 0 -er 100 -algo LG-FedAvg --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 1 -sr 0 -er 100 -algo FedGen -nd 32 -glr 0.0003 -se 50 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 0 -sr 0 -er 100 -algo FML -al 0.5 -bt 0.5 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 6 -sr 0 -er 100 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &
# nohup python -u main.py -t 3 -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 47 -data EMNIST -m HtFE8 -fd 512 -did 3 -sr 0 -er 100 -algo FedOrth -clam 10 -slam 1 -gam 10 -se 1 --save_folder_name results/test_all_batchsize=32_dataset=EMNIST > /dev/null 2>&1 &


