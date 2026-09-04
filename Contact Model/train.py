import torch.nn as nn
from options import TrainOptions
from utils import evaluate, save_model
import torch
import data
import utils
from test import test_
from torch.nn import DataParallel
from models import BiRNN, IKeyboard, SANCD, ShortTermMLP, BERT, BiRNNLinearBert


if __name__ == "__main__":
    args = TrainOptions().parse()
    args = utils.bashRun(args)
    torch.manual_seed(args.seed)                            # for reproducibility
    torch.backends.cudnn.deterministic = True       # for reproducibility
    torch.backends.cudnn.benchmark = False          # for reproducibility

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bigru:
        predictor = BiRNN.BidirectionalRNN(char_embed_size=args.char_embed_size, nhid=args.nhid, nlayer=args.nlayers,
                                           rnn_type='GRU').to(device)
        model_name = 'BiRNN'

        train_data ='./data/data_normalized/data_train.csv'
        val_data ='./data/data_normalized/data_val.csv'
        test_data ='./data/data_normalized/data_test.csv'

    elif args.bert:
        predictor = BERT.BERT(args=args).to(device)

        train_data = r"C:\Users\mno64\Datasets\1-billion-word-benchmark\training-monolingual.tokenized.shuffled\processed.csv"
        val_data ='./data/data_normalized/data_val.csv'
        test_data = val_data

    elif args.sa_ncd:
        predictor = SANCD.SANCD(args=args).to(device)
        model_name = 'SANCD'

        # TODO: data paths

    save_path = '{}_h{}_n{}_{}.pth'.format(model_name, str(args.nhid), args.nlayers, 'MLM' if args.masked_LM else '')

    if args.multi_gpu:
        predictor = DataParallel(predictor, output_device=1)
    else:
        predictor = predictor.to(device)


    dataloader_train = data.get_dataloader(args.train_data, batch_size=args.batch_size, masked_LM=args.masked_LM)
    dataloader_val = data.get_dataloader(args.val_data, batch_size=args.batch_size, test=True, masked_LM=args.masked_LM)
    dataloader_test = data.get_dataloader(args.test_data, batch_size=args.batch_size, test=True, masked_LM=args.masked_LM)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_model = getattr(eval(model_name), 'train_')(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path=save_path)

    val_loss, val_acc, val_wer = test_(args, predictor, best_model, dataloader_val, print_count=False)
    test_loss, test_acc, test_wer = test_(args, predictor, best_model, dataloader_test, print_count=False)

    print('val ({:5.4f} | {:2.2f}%) | Test ({:5.4f} | {:2.2f}%)'.format(val_loss, val_acc, test_loss, test_acc))

    print("WER : val {:2.2f}% test {:2.2f}%".format(val_wer, test_wer))

    print('-' * 80)

