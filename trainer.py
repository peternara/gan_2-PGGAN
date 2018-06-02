import numpy as np
import math
import glob
import utils
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
import cv2

initial_size = 4
final_size = 512
target_size = int(sys.argv[1])

z_dim = 128
gin_ch1 = [z_dim,z_dim,z_dim,z_dim,z_dim//2,z_dim//4,z_dim//8,z_dim//16]
gout_ch1 = [z_dim,z_dim,z_dim,z_dim//2,z_dim//4,z_dim//8,z_dim//16,z_dim//32]
ksz = 3

def actf(x):
    return tf.maximum(x, 0.2*x)
    
def weight(shape, name, gain=2.0):
    coef = np.float32(np.sqrt( gain / np.prod(shape[:-1])))
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.initializers.random_normal())
    return coef * v
    
def bias(shape, name):
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        b = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer)
    return b

def norm_scale(z):
    m = tf.rsqrt(1e-8+tf.reduce_mean(tf.square(z),axis=0,keepdims=True))
    return m * z
    
def upsample(z):#, size):
    hh = 2 * z.get_shape()[1]
    ww = 2 * z.get_shape()[2]
    return tf.image.resize_nearest_neighbor(z, [hh, ww])
        
def downsample(z):
    return tf.nn.avg_pool(z, [1,2,2,1], [1,2,2,1], 'VALID')
    
def down_up(z):
    return upsample(downsample(z))

def gen_first(z):
    nch = int(z.get_shape()[1])
    g1_1 = weight([nch,nch*initial_size*initial_size], 'g1_1')#, 1/8.0)
    g1_1b = bias([nch], 'g1_1b')
    g1_2 = weight([ksz, ksz, nch, nch], 'g1_2')
    g1_2b = bias([nch], 'g1_2b')
    g1 = norm_scale(actf(tf.nn.bias_add(tf.reshape(tf.matmul(tf.reshape(z, [-1, nch]), g1_1), [-1,initial_size, initial_size, nch]), g1_1b)))
    g2 = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(g1, g1_2, [1,1,1,1], 'SAME'), g1_2b)))
    return g2
    
def discr_last(z):
    nch = int(z.get_shape()[3])
    d1_1 = weight([ksz, ksz, 1+nch, nch], 'd1_1')
    d1_1b = bias([nch], 'd1_1b')
    d1_2 = weight([initial_size, initial_size, nch, nch], 'd1_2')
    d1_2b = bias([nch], 'd1_2b')
    d1_3 = weight([nch, nch], 'd1_3')#, 1.0)
    d1_3b = bias([nch], 'd1_3b')
    sdev = tf.reduce_mean(tf.sqrt(tf.subtract(tf.reduce_mean(tf.square(z), axis = 0), tf.square(tf.reduce_mean(z, axis = 0)))))
    dd = tf.fill([int(z.get_shape()[0]), initial_size, initial_size, 1], sdev)
    zz = tf.concat([z, dd], axis=3)
    d1 = actf(tf.nn.bias_add(tf.nn.conv2d(zz, d1_1, [1,1,1,1], 'SAME'), d1_1b))
    d2 = actf(tf.nn.bias_add(tf.nn.conv2d(d1, d1_2, [1,1,1,1], 'VALID'), d1_2b))
    d3 = tf.nn.bias_add(tf.matmul(tf.reshape(d2, [-1, nch]), d1_3), d1_3b)
    return d3

def toRGB(z, tsz):
    grgb = weight([1,1,int(z.get_shape()[3]),3], 'grgb'+str(tsz))#, 1.0)
    grgb_b = bias([3], 'grgb'+str(tsz)+'_b')
    return tf.tanh(tf.nn.bias_add(tf.nn.conv2d(z, grgb, [1,1,1,1], 'SAME'), grgb_b))

def fromRGB(z, sz, tsz):
    drgb = weight([1,1,3,sz], 'drgb'+str(tsz))
    drgb_b = bias([sz], 'drgb'+str(tsz)+'_b')
    return actf(tf.nn.bias_add(tf.nn.conv2d(z, drgb, [1,1,1,1], 'SAME'), drgb_b))

def generator(z, tsz, alpha):
    d = gen_first(z)
    sz = initial_size
    cnt = 1
    while tsz > sz:
        if sz * 2 == tsz :
            dd = (1.-alpha) * upsample(toRGB(d,tsz//2))
        w1 = weight([ksz,ksz,gin_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_2')
        b2 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_2b')
        d = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(upsample(d), w1, [1,1,1,1], 'SAME'), b1)))
        d = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2)))
        cnt = cnt + 1
        sz *= 2
    d = toRGB(d, tsz)
    if cnt > 1:
        d = dd + alpha * d
    return d

def discriminator(real, alpha):
    tsz = int(real.get_shape()[1])
    tmp = tsz
    cnt = 0
    while tmp > initial_size:
        tmp //= 2
        cnt = 1 + cnt
    d = fromRGB(real, gout_ch1[cnt], tsz)
    d2 = fromRGB(downsample(real), gin_ch1[cnt], tsz//2)

    sz = tsz
    while sz > initial_size:
        w1 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'd'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'd'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gin_ch1[cnt]], 'd'+str(1+cnt)+'_2')
        b2 = bias([gin_ch1[cnt]], 'd'+str(1+cnt)+'_2b')
        d = actf(tf.nn.bias_add(tf.nn.conv2d(d, w1, [1,1,1,1], 'SAME'), b1))
        d = actf(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2))
        d = downsample(d)
        if sz == tsz:
            d = (1.-alpha)*d2 + alpha * d
        cnt = cnt - 1
        sz //= 2
    return discr_last(d)

def latent_vec(bn, ndim):
    vec = np.random.randn(bn, ndim)
    for i in range(bn):
        vec[i] /= np.sqrt(vec[i].dot(vec[i]))
    return vec

def build(batch_size):
    gp_lambda = 10
    
    """ graphs """
    with tf.device('/gpu:0'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0, beta2 = 0.99)
        
        ''' graph definition '''
        ''' declare all of variables to be used with dummy '''
        alpha = tf.placeholder(tf.float32, shape=[1])
        real = tf.placeholder(tf.float32, shape=[batch_size, target_size, target_size, 3])
        real = (1.-alpha)*down_up(real) + alpha * real
        z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
        z2 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
        real_dummy = tf.placeholder(tf.float32, shape=[2, final_size, final_size, 3])
        z1_dummy = tf.placeholder(tf.float32, shape=[2, z_dim])
        z2_dummy = tf.placeholder(tf.float32, shape=[2, z_dim])
        
        fake1 = generator(z1, target_size, alpha)
        fake2 = generator(z2, target_size, alpha)
        fake1_dummy = generator(z1_dummy, final_size, alpha)
        fake2_dummy = generator(z2_dummy, final_size, alpha)

        r_logit = discriminator(real, alpha)
        f1_logit = discriminator(fake1, alpha)
        f2_logit = discriminator(fake2, alpha)
        r_logit_dummy = discriminator(real_dummy, alpha)
        f1_logit_dummy = discriminator(fake1_dummy, alpha)
        f2_logit_dummy = discriminator(fake2_dummy, alpha)
        
        # Cramer GAN losses
        def L2(logit):
            ret = tf.sqrt( tf.reduce_sum( tf.square( logit ), axis = 1 ) )
            return ret

        def critic(logit, logitd):
            return tf.reduce_mean(L2( logit - logitd ) - L2( logit ))

        def gradient_penalty(bsz, real, fake, logitd):
            def interpolate(a, b):
                rg = tf.random_uniform(shape=[bsz,1,1,1], minval=0., maxval=1.)
                return rg * a + (1.-rg)*b

            x = interpolate(real, fake)
            pred = discriminator(x, alpha)
            pred = L2( pred - logitd ) - L2( pred )
            grad = tf.reshape(tf.gradients(pred, x)[0], shape=[bsz, -1])
            
            # when P_fake ~= P_real, for almost all x in the distribution has 1 L2-norm
            return tf.reduce_mean((L2(grad)-1)**2.0)
        
        g_loss = tf.reduce_mean(L2(r_logit - f1_logit) + L2( r_logit - f2_logit ) - L2( f1_logit - f2_logit ))
        g_surr = critic(r_logit, f2_logit) - critic( f1_logit, f2_logit )
        gp = gp_lambda * gradient_penalty(batch_size, real, fake1, f2_logit)
        d_loss = -g_surr + gp
        
        g_loss_dummy = tf.reduce_mean(L2(r_logit_dummy - f1_logit_dummy) + L2( r_logit_dummy - f2_logit_dummy ) - L2( f1_logit_dummy - f2_logit_dummy ))
        g_surr_dummy = critic(r_logit_dummy, f2_logit_dummy) - critic( f1_logit_dummy, f2_logit_dummy )
        gp_dummy = gp_lambda * gradient_penalty(2, real_dummy, fake1_dummy, f2_logit_dummy)
        d_loss_dummy = -g_surr_dummy + gp_dummy
        
        g_vars = []
        d_vars = []
        for vari in tf.trainable_variables():
            if vari.name.startswith('x/g'):
                g_vars.append(vari)
            if vari.name.startswith('x/d'):
                d_vars.append(vari)

        # otpims
        g_step = optimizer.minimize(g_loss, var_list=g_vars)
        d_step = optimizer.minimize(d_loss, var_list=d_vars)
        g_step_dummy = optimizer.minimize(g_loss_dummy, var_list=g_vars)
        d_step_dummy = optimizer.minimize(d_loss_dummy, var_list=d_vars)

        # summaries
        g_summary = utils.summary({g_loss: 'g_loss'})
        d_summary = utils.summary({d_loss: 'd_loss', gp : 'grad_penal'})#, wd: 'wd', gp: 'gp', rlm:'rlm'})

        # sample
        f_sample = generator(z1, target_size, alpha)

    return {\
    'dummy' : { 'd' : d_step_dummy, 'g' : g_step_dummy,\
        'input' : { 'real' : real_dummy, 'z1' : z1_dummy, 'z2' : z2_dummy, 'alpha' : alpha }  },\
    'product' : { 'd' : d_step, 'g' : g_step,\
        'input' : { 'real' : real, 'z1' : z1, 'z2' : z2, 'alpha' : alpha } },\
    'sample' : f_sample, 'summaries' : {'d' : d_summary, 'g': g_summary } }

def get_ipt(bsz, tsz, alpha, pool, ndim, input):
    z1_sample = latent_vec(bsz, ndim)
    z2_sample = latent_vec(bsz, ndim)
    real_ipt = pool.batch()
    real_ipt = np.reshape(real_ipt,(-1,tsz,tsz,3))
    return { input['real'] : real_ipt, input['z1'] : z1_sample, input['z2'] : z2_sample, input['alpha'] : [alpha] }

def get_ipt_for_sample(bsz, ndim, input):
    z1_sample = latent_vec(bsz, ndim)
    return { input['z1'] : z1_sample, input['alpha'] : [1.0] }

        
def train():
    alpha_span = 800000
    batch_size = 32
    ckpt_dir = './checkpoints/wgp'
    n_gen = 1
    n_critic = 1
    it_start = 0
    #epoch = 20*(alpha_span * 2 // (2*4936)) # 4936 is number of images
    
    def preprocess_fn(img):
        img = tf.image.resize_images(img, [target_size, target_size], method=tf.image.ResizeMethod.AREA) / 127.5 -1
        return img

    def preprocess_fn_dummy(img):
        img = tf.image.resize_images(img, [final_size, final_size], method=tf.image.ResizeMethod.AREA) / 127.5 -1
        return img
    
    # dataset
    img_paths = glob.glob('./imgs/faces/*.png')
    data_pool = utils.DiskImageData(5, img_paths, batch_size//2, shape=[640, 640, 3], preprocess_fn=preprocess_fn)
    data_pool_dummy = utils.DiskImageData(7, img_paths, 1, shape=[640, 640, 3], preprocess_fn=preprocess_fn_dummy)    
    batch_epoch = len(data_pool) // (batch_size * 1)#n_critic

    # build graph
    print('Building a graph ...')
    nodes = build(batch_size)
    # session
    sess = utils.session()
    saver = tf.train.Saver()
    # summary
    summary_writer = tf.summary.FileWriter('./summaries/wgp/', sess.graph)
    utils.mkdir(ckpt_dir + '/')

    print('Initializing all variables ...')
    sess.run(tf.global_variables_initializer())
    
    # run final size session for storing all variables to be used into the optimizer
    print('Running final size dummy session ...')
    #if target_size == initial_size and len(sys.argv) <= 3:
    #    _ = sess.run([nodes['dummy']['d']], feed_dict=get_ipt(2, final_size, 1.0, data_pool_dummy ,z_dim, nodes['dummy']['input'] ))
    #    _ = sess.run([nodes['dummy']['g']], feed_dict=get_ipt(2, final_size, 1.0, data_pool_dummy ,z_dim, nodes['dummy']['input'] ))
        
    # load checkpoint
    if len(sys.argv)>3 and sys.argv[2]=='resume':
        print ('Loading the checkpoint ...')
        saver.restore(sess, ckpt_dir+'/model.ckpt')
        it_start = 1 + int(sys.argv[3])
    last_saved_iter = it_start - 1

    ''' train '''
    for it in range(it_start, 9999999999):
        # fade alpha
        alpha_ipt = it / (alpha_span / batch_size)
        if alpha_ipt > 1 or target_size == initial_size:
            alpha_ipt = 1.0
        print('Alpha : %f' % alpha_ipt)
        alpha_ipt = 1.0
        
        # train D
        for i in range(n_critic):
            d_summary_opt, _ = sess.run([nodes['summaries']['d'], nodes['product']['d']],\
                feed_dict=get_ipt(batch_size, target_size, alpha_ipt, data_pool, z_dim, nodes['product']['input']))
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        for i in range(n_gen):
            g_summary_opt, _ = sess.run([nodes['summaries']['g'], nodes['product']['g']],\
                feed_dict=get_ipt(batch_size, target_size, alpha_ipt, data_pool, z_dim, nodes['product']['input']))
        summary_writer.add_summary(g_summary_opt, it)
        
        # display
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1
        if it % 1 == 0:
            print("iter : %8d, epoch : (%3d) (%5d/%5d) _ resume point : %d" % (it, epoch, it_epoch, batch_epoch,last_saved_iter))

        # sample
        if (it + 1) % batch_epoch == 0:
            f_sample_opt = sess.run(nodes['sample'], feed_dict=get_ipt_for_sample(batch_size, z_dim, nodes['product']['input']))
            f_sample_opt = np.clip(f_sample_opt, -1, 1)
            save_dir = './sample_images_while_training/wgp/'
            utils.mkdir(save_dir + '/')
            osz = int(math.sqrt(batch_size))+1
            utils.imwrite(utils.immerge(f_sample_opt, osz, osz), '%s/iter_(%d).png' % (save_dir, it))
            
        # save
        if (it + 1) % batch_epoch == 0:
            last_saved_iter = it
            save_path = saver.save(sess, '%s/model.ckpt' % (ckpt_dir))
            print('Model saved in file: %s' % save_path)
            
train()
