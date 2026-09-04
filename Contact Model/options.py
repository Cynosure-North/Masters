import argparse


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='TMI-Keyboard',
                            help='name of the experiment. It decides where to store samples and models')

        self.parser = parser
        self.arg_parsed = False

    def parse(self):
        # get the basic options
        if not self.arg_parsed:
            opt = self.parser.parse_args()
            self.opt = opt
            self.arg_parsed = True
        self.print_options(self.opt)

        return self.opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        '''
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        '''


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        # Training options
        self.parser.add_argument('--seed', default=0, help="fix randomness")
        self.parser.add_argument('--test_seed', default=1, help="fix randomness")
        self.parser.add_argument('--epoch', default=30, type=int, help="number of training epochs")
        self.parser.add_argument('--batch_size', default=64)
        self.parser.add_argument('--batch_norm', default=False)
        self.parser.add_argument('--optimizer', default='ADAM', help='SGD or ADAM or RMS')
        self.parser.add_argument('--lr', default=0.0001,
                                 help="learning rate : 0.0001 for BERT, 3.0 for BiGRU")
        self.parser.add_argument('--gru_lr_down', default=0.5)
        self.parser.add_argument('--augment', default=True)
        self.parser.add_argument('--multi_gpu', default=False)

        self.parser.add_argument('--step_lr', default=10, type=int)
        self.parser.add_argument('--early_stop', default=10)


        # Embedding size
        self.parser.add_argument('--char_embed_size', default=0, help="char embedding dimension")
        self.parser.add_argument('--feat_size', default=4,
                                 help="final feature size of seperate path before softmax decoding")

        # Model Selection
        self.parser.add_argument('--bigru', default=False, action='store_true', help="use Bi-directional GRU as a decoding model")
        self.parser.add_argument('--bert', default=False, action='store_true', help="use Bert as a decoding model")
        self.parser.add_argument('--sa_ncd', default=False, action='store_true', help="use TMIkeyboard")


        # Options of Transformer(BERT) & BidirectionalRNN
        self.parser.add_argument('--nhid', default=512,
                                 help="number of nodes in hidden layer in Decoder")
        self.parser.add_argument('--nlayers', default=12,
                                 help="number of encoder block in Transformer(BERT) or number of layers in Bi-GRU")
        self.parser.add_argument('--nhead', default=8,
                                 help="the number of heads in the multi-head attention models")
        self.parser.add_argument('--intermediate_size', default=1024,
                                 help="Intermediate size of BERT")
        self.parser.add_argument('--excessive_output', default=True, help="Excessive output for BERT (only uses first 31 dimension of 256 output")
        self.parser.add_argument('--intermediate_loss', default=False, action=argparse.BooleanOptionalAction, help="")
        self.parser.add_argument('--intermediate_stop', default=30, type=int, help="")
        self.parser.add_argument('--cm_threshold', default=0.45, type=float, help="")


        self.parser.add_argument('--geometric_decoder_path', default='BiRNN_h512_n12_.pth', help="location of pre-trained Geometric Decoder")

        self.parser.add_argument('--semantic_decoder_path', default='BERT_h512_n12_MLM_epoch2.pth',
                                 help="location of pre-trained Semantic Decoder (Transformer Encoder trained as Masked Character Language Model)")








