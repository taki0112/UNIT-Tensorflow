from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder

class UNIT(object) :
    def __init__(self, sess, args):
        self.model_name = 'UNIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight about VAE """
        self.KL_weight = args.KL_weight # lambda 1
        self.L1_weight = args.L1_weight # lambda 2

        """ Weight about VAE Cycle"""
        self.KL_cycle_weight = args.KL_cycle_weight # lambda 3
        self.L1_cycle_weight = args.L1_cycle_weight # lambda 4

        """ Weight about GAN """
        self.GAN_weight = args.GAN_weight # lambda 0

        """ Encoder """
        self.n_encoder = args.n_encoder
        self.n_enc_resblock = args.n_enc_resblock
        self.n_enc_share = args.n_enc_share

        """ Generator """
        self.n_gen_share = args.n_gen_share
        self.n_gen_resblock = args.n_gen_resblock
        self.n_gen_decoder = args.n_gen_decoder

        """ Discriminator """
        self.n_dis = args.n_dis

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Encoder #####")
        print("# encoder blocks : ", self.n_encoder)
        print("# encoder resblock : ", self.n_enc_resblock)
        print("# encoder share : ", self.n_enc_share)

        print()

        print("##### Decoder #####")
        print("# decoder share : ", self.n_gen_share)
        print("# decoder resblock : ", self.n_gen_resblock)
        print("# decoder blocks : ", self.n_gen_decoder)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)

        ##############################################################################
        # BEGIN of ENCODERS

    def encoder(self, x, reuse=False, scope="encoder"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=7, stride=1, pad=3, scope='conv_0')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_encoder):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, scope='conv_' + str(i))
                x = lrelu(x, 0.01)
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock):
                x = resblock(x, channel, scope='resblock_'+str(i))

            return x
        # END of ENCODERS
        ##############################################################################

        ##############################################################################
        # BEGIN of SHARED LAYERS
        # Shared residual-blocks

    def share_encoder(self, x, reuse=False, scope="share_encoder"):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(0, self.n_enc_share):
                x = resblock(x, channel, scope='resblock_' + str(i))

            x = gaussian_noise_layer(x)

            return x

    def share_generator(self, x, reuse=False, scope="share_generator"):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(0, self.n_gen_share):
                x = resblock(x, channel, scope='resblock_' + str(i))

        return x
        # END of SHARED LAYERS
        ##############################################################################

        ##############################################################################
        # BEGIN of DECODERS

    def generator(self, x, reuse=False, scope="generator"):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(0, self.n_gen_resblock):
                x = resblock(x, channel, scope='resblock_' + str(i))

            for i in range(0, self.n_gen_decoder - 1):
                x = deconv(x, channel // 2, kernel=3, stride=2, scope='deconv_' + str(i))
                x = lrelu(x, 0.01)
                channel = channel // 2

            x = deconv(x, channels=3, kernel=1, stride=1, scope='G_logit')
            x = tanh(x)

            return x
        # END of DECODERS
        ##############################################################################

        ##############################################################################
        # BEGIN of DISCRIMINATORS

    def discriminator(self, x, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=3, stride=2, pad=1, scope='conv_0')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, scope='conv_' + str(i))
                x = lrelu(x, 0.01)
                channel *= 2

            x = conv(x, channels=1, kernel=1, stride=1, scope='D_logit')

            return x

    # END of DISCRIMINATORS
    ##############################################################################

    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, scope="encoder_A"), self.encoder(x_B, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out)
        out = self.share_generator(shared)

        out_A = self.generator(out, scope="generator_A")
        out_B = self.generator(out, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A):
        out = self.encoder(x_A, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, reuse=True)
        out = self.share_generator(shared, reuse=True)
        out = self.generator(out, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B):
        out = self.encoder(x_B, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, reuse=True)
        out = self.share_generator(shared, reuse=True)
        out = self.generator(out, reuse=True, scope="generator_A")

        return out, shared

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        trainB = trainB.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()


        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()


        """ Define Encoder, Generator, Discriminator """
        x_aa, x_ba, x_ab, x_bb, shared = self.translation(self.domain_A, self.domain_B)
        x_bab, shared_bab = self.generate_a2b(x_ba)
        x_aba, shared_aba = self.generate_b2a(x_ab)

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)


        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        enc_loss = KL_divergence(shared)
        enc_bab_loss = KL_divergence(shared_bab)
        enc_aba_loss = KL_divergence(shared_aba)

        l1_loss_a = L1_loss(x_aa, self.domain_A)  # identity
        l1_loss_b = L1_loss(x_bb, self.domain_B)  # identity
        l1_loss_aba = L1_loss(x_aba, self.domain_A)  # reconstruction
        l1_loss_bab = L1_loss(x_bab, self.domain_B)  # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_image')

        self.test_fake_A, _ = self.generate_b2a(self.test_image)
        self.test_fake_B, _ = self.generate_a2b(self.test_image)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict={self.test_image: sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        index.close()
