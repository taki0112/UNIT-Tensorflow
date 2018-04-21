from UNIT import UNIT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of UNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GAN_weight', type=float, default=10.0, help='Weight about GAN, lambda0')
    parser.add_argument('--KL_weight', type=float, default=0.1, help='Weight about VAE, lambda1')
    parser.add_argument('--L1_weight', type=float, default=100.0, help='Weight about VAE, lambda2' )
    parser.add_argument('--KL_cycle_weight', type=float, default=0.1, help='Weight about VAE Cycle, lambda3')
    parser.add_argument('--L1_cycle_weight', type=float, default=100.0, help='Weight about VAE Cycle, lambda4')

    parser.add_argument('--gan_type', type=str, default='gan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_encoder', type=int, default=3, help='The number of encoder')
    parser.add_argument('--n_enc_resblock', type=int, default=3, help='The number of encoder_resblock')
    parser.add_argument('--n_enc_share', type=int, default=1, help='The number of share_encoder')
    parser.add_argument('--n_gen_share', type=int, default=1, help='The number of share_generator')
    parser.add_argument('--n_gen_resblock', type=int, default=3, help='The number of generator_resblock')
    parser.add_argument('--n_gen_decoder', type=int, default=3, help='The number of generator_decoder')

    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = UNIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()