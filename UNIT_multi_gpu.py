from ops import *
from utils import *
from glob import glob
import time

class UNIT(object):
    def __init__(self, sess, args):
        self.model_name = 'UNIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch # 100000
        self.batch_size_per_gpu = args.batch_size
        self.batch_size = args.batch_size * args.gpu_num
        self.gpu_num = args.gpu_num

        self.lr = args.lr # 0.0001
        """ Weight about VAE """
        self.KL_weight = args.KL_weight # lambda 1
        self.L1_weight = args.L1_weight # lambda 2

        """ Weight about VAE Cycle"""
        self.KL_cycle_weight = args.KL_cycle_weight # lambda 3
        self.L1_cycle_weight = args.L1_cycle_weight # lambda 4

        """ Weight about GAN """
        self.GAN_weight = args.GAN_weight # lambda 0


        """ Encoder """
        self.ch = args.ch # base channel number per layer
        self.n_encoder = args.n_encoder
        self.n_enc_resblock = args.n_enc_resblock
        self.n_enc_share = args.n_enc_share

        """ Generator """
        self.n_gen_share = args.n_gen_share
        self.n_gen_resblock = args.n_gen_resblock
        self.n_gen_decoder = args.n_gen_decoder

        """ Discriminator """
        self.n_dis = args.n_dis # + 2

        self.res_dropout = args.res_dropout
        self.smoothing = args.smoothing
        self.lsgan = args.lsgan
        self.norm = args.norm
        self.replay_memory = args.replay_memory
        self.pool_size = args.pool_size
        self.img_size = args.img_size
        self.channel = args.img_ch
        self.augment_flag = args.augment_flag
        self.augment_size = self.img_size + (30 if self.img_size == 256 else 15)
        self.normal_weight_init = args.normal_weight_init

        self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name, size=self.img_size)
        self.num_batches = max(len(self.trainA) // self.batch_size, len(self.trainB) // self.batch_size)

    ##############################################################################
    # BEGIN of ENCODERS
    def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            return x
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    def share_encoder(self, x, is_training=True, reuse=False, scope="share_encoder"):
        channel = self.ch * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_enc_share) :
                x = resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            x = gaussian_noise_layer(x)

            return x

    def share_generator(self, x, is_training=True, reuse=False, scope="share_generator"):
        channel = self.ch * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_share) :
                x = resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

        return x
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, reuse=False, scope="generator"):
        channel = self.ch * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                x = resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = deconv(x, self.channel, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DISCRIMINATORS
    def discriminator(self, x, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_dis) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            x = conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')

            return x
    # END of DISCRIMINATORS
    ##############################################################################

    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)

        out_A = self.generator(out, self.is_training, scope="generator_A")
        out_B = self.generator(out, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_A")

        return out, shared

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def discriminate_fake_pool(self, x_ba, x_ab):
        fake_A_pool_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
        fake_B_pool_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory

        return fake_A_pool_logit, fake_B_pool_logit

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.prob = tf.placeholder(tf.float32)
        self.condition = tf.logical_and(tf.greater(self.prob, tf.constant(0.5)), self.is_training)

        """ Input Image"""
        domain_A = self.domain_A = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.channel], name='domain_A') # real A
        domain_B = self.domain_B = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.channel], name='domain_B') # real B

        self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.channel], name='test_domain_A')
        self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.channel], name='test_domain_B')

        if self.augment_flag :
            """ Augmentation """
            domain_A = tf.cond(
                self.condition,
                lambda : augmentation(domain_A, self.augment_size),
                lambda : domain_A
            )

            domain_B = tf.cond(
                self.condition,
                lambda : augmentation(domain_B, self.augment_size),
                lambda : domain_B
            )

        domain_A = tf.split(domain_A, self.gpu_num)
        domain_B = tf.split(domain_B, self.gpu_num)

        G_A_losses= []
        G_B_losses = []
        D_A_losses = []
        D_B_losses = []

        G_losses = []
        D_losses = []

        self.fake_A = []
        self.fake_B = []
        for gpu_id in range(self.gpu_num) :
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)) :
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)) :
                    """ Define Encoder, Generator, Discriminator """
                    x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A[gpu_id], domain_B[gpu_id])
                    x_bab, shared_bab = self.generate_a2b(x_ba)
                    x_aba, shared_aba = self.generate_b2a(x_ab)

                    real_A_logit, real_B_logit = self.discriminate_real(domain_A[gpu_id], domain_B[gpu_id])

                    if self.replay_memory :
                        self.fake_A_pool = ImagePool(self.pool_size)  # pool of generated A
                        self.fake_B_pool = ImagePool(self.pool_size)  # pool of generated B
                        fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                    else :
                        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)



                    """ Define Loss """
                    G_ad_loss_a = generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.lsgan)
                    G_ad_loss_b = generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.lsgan)

                    D_ad_loss_a = discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.lsgan)
                    D_ad_loss_b = discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.lsgan)

                    enc_loss = KL_divergence(shared)
                    enc_bab_loss = KL_divergence(shared_bab)
                    enc_aba_loss = KL_divergence(shared_aba)

                    l1_loss_a = L1_loss(x_aa, domain_A[gpu_id]) # identity
                    l1_loss_b = L1_loss(x_bb, domain_B[gpu_id]) # identity
                    l1_loss_aba = L1_loss(x_aba, domain_A[gpu_id]) # reconstruction
                    l1_loss_bab = L1_loss(x_bab, domain_B[gpu_id]) # reconstruction

                    Generator_A_loss_split = self.GAN_weight * G_ad_loss_a + \
                                       self.L1_weight * l1_loss_a + \
                                       self.L1_cycle_weight * l1_loss_aba + \
                                       self.KL_weight * enc_loss + \
                                       self.KL_cycle_weight * enc_bab_loss

                    Generator_B_loss_split = self.GAN_weight * G_ad_loss_b + \
                                       self.L1_weight * l1_loss_b + \
                                       self.L1_cycle_weight * l1_loss_bab + \
                                       self.KL_weight * enc_loss + \
                                       self.KL_cycle_weight * enc_aba_loss

                    Discriminator_A_loss_split = self.GAN_weight * D_ad_loss_a
                    Discriminator_B_loss_split = self.GAN_weight * D_ad_loss_b

                    Generator_loss_split = Generator_A_loss_split + Generator_B_loss_split
                    Discriminator_loss_split = Discriminator_A_loss_split + Discriminator_B_loss_split

                    """ Generated Image """
                    fake_B, _ = self.generate_a2b(domain_A[gpu_id])  # for test
                    fake_A, _ = self.generate_b2a(domain_B[gpu_id])  # for test

                    G_A_losses.append(Generator_A_loss_split)
                    G_B_losses.append(Generator_B_loss_split)
                    D_A_losses.append(Discriminator_A_loss_split)
                    D_B_losses.append(Discriminator_B_loss_split)

                    G_losses.append(Generator_loss_split)
                    D_losses.append(Discriminator_loss_split)

                    self.fake_A.append(fake_A)
                    self.fake_B.append(fake_B)

        Generator_A_loss = tf.reduce_mean(G_A_losses)
        Generator_B_loss = tf.reduce_mean(G_B_losses)
        Discriminator_A_loss = tf.reduce_mean(D_A_losses)
        Discriminator_B_loss = tf.reduce_mean(D_B_losses)

        self.Generator_loss = tf.reduce_mean(G_losses)
        self.Discriminator_loss = tf.reduce_mean(D_losses)

        self.fake_A = tf.concat(self.fake_A, axis=0)
        self.fake_B = tf.concat(self.fake_B, axis=0)

        self.test_fake_B = self.generate_a2b(self.test_domain_A)
        self.test_fake_A = self.generate_b2a(self.test_domain_B)

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, colocate_gradients_with_ops=True, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, colocate_gradients_with_ops=True, var_list=D_vars)
        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                random_index_A = np.random.choice(len(self.trainA), size=self.batch_size, replace=False)
                random_index_B = np.random.choice(len(self.trainB), size=self.batch_size, replace=False)
                batch_A_images = self.trainA[random_index_A]
                batch_B_images = self.trainB[random_index_B]
                p = np.random.uniform(low=0.0, high=1.0)


                train_feed_dict = {
                    self.domain_A : batch_A_images,
                    self.domain_B : batch_B_images,
                    self.prob : p,
                    self.is_training : True
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                if np.mod(counter, 10) == 0 :
                    batch_A_images = np.split(batch_A_images, self.gpu_num)
                    batch_B_images = np.split(batch_B_images, self.gpu_num)
                    fake_A = np.split(fake_A, self.gpu_num)
                    fake_B = np.split(fake_B, self.gpu_num)

                    for gpu_id in range(self.gpu_num) :
                        save_images(batch_A_images[gpu_id], [self.batch_size_per_gpu, 1],
                                    './{}/real_A_{}_{:02d}_{:04d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+2))
                        save_images(batch_B_images[gpu_id], [self.batch_size_per_gpu, 1],
                                    './{}/real_B_{}_{:02d}_{:04d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+2))

                        save_images(fake_A[gpu_id], [self.batch_size_per_gpu, 1],
                                    './{}/fake_A_{}_{:02d}_{:04d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+2))
                        save_images(fake_B[gpu_id], [self.batch_size_per_gpu, 1],
                                    './{}/fake_B_{}_{:02d}_{:04d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+2))

                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
            self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name, self.norm)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

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

        """
        testA, testB = test_data(dataset_name=self.dataset_name, size=self.img_size)
        test_A_images = testA[:]
        test_B_images = testB[:]
        """
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

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
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image, self.is_training : False})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_A, feed_dict = {self.test_domain_B : sample_image, self.is_training : False})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")
        index.close()