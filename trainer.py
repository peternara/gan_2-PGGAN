import numpy as np
import math
import glob
import utils
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
import cv2

""" param """
alpha_span = 200000
# 4-32 : 800000
# 64- : 1200000
# 128 : 1600000
# 256 :1600000
batch_size = 16
# -64:32? 128:24 256:16 512:8
epoch = 20*(alpha_span * 2 // (2*4936)) # 4936 is number of images
g_lr = 0.0005
d_lr = 0.0005+1e-7
z_dim = 128
gpu_id = 0
n_critic = 5
ksz = 3
gin_ch1 = [z_dim,z_dim,z_dim,z_dim,z_dim//2,z_dim//4,z_dim//8,z_dim//16]
gout_ch1 = [z_dim,z_dim,z_dim,z_dim//2,z_dim//4,z_dim//8,z_dim//16,z_dim//32]
target_size = int(sys.argv[1])
mean_img = np.zeros([640,640,3], dtype='float32')
initial_size = 4

def actf(x):
    return tf.maximum(x, 0.2*x)
    
path = './imgs/faces/'
def preprocess_fn(img):
    img = tf.image.resize_images(img, [target_size, target_size], method=tf.image.ResizeMethod.AREA) / 127.5 -1
    return img

img_paths = glob.glob(path+'*.png')
data_pool = utils.DiskImageData(img_paths, batch_size//2, shape=[640, 640, 3], preprocess_fn=preprocess_fn)

def weight(shape, name, gain=2.0):
    coef = np.float32(np.sqrt( gain / np.prod(shape[:-1])))
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.initializers.random_normal())
        v = coef * v
    return v
    
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

def gen_5(z):
    g1_1 = weight([z_dim,z_dim*initial_size*initial_size], 'g1_1', 1/8.0)
    g1_1b = bias([z_dim], 'g1_1b')
    g1_2 = weight([ksz, ksz, z_dim, z_dim], 'g1_2')
    g1_2b = bias([z_dim], 'g1_2b')
    g1 = norm_scale(actf(tf.nn.bias_add(tf.reshape(tf.matmul(tf.reshape(z, [-1, z_dim]), g1_1), [-1,initial_size, initial_size, z_dim]), g1_1b)))
    g2 = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(g1, g1_2, [1,1,1,1], 'SAME'), g1_2b)))
    return g2
    
def discr_5(z):
    # todo : minibatch norm
    d1_1 = weight([ksz, ksz, z_dim + 1, z_dim], 'd1_1')
    d1_1b = bias([z_dim], 'd1_1b')
    d1_2 = weight([initial_size, initial_size, z_dim, z_dim], 'd1_2')
    d1_2b = bias([z_dim], 'd1_2b')
    d1_3 = weight([z_dim, z_dim], 'd1_3', 1.0)
    d1_3b = bias([z_dim], 'd1_3b')
    d1_4 = weight([z_dim, z_dim], 'd1_4')
    d1_4b = bias([z_dim], 'd1_4b')
    sdev = tf.reduce_mean(tf.sqrt(tf.subtract(tf.reduce_mean(tf.square(z), axis = 0), tf.square(tf.reduce_mean(z, axis = 0)))))
    dd = tf.fill([batch_size, initial_size, initial_size, 1], sdev)
    zz = tf.concat([z, dd], axis=3)
    d1 = actf(tf.nn.bias_add(tf.nn.conv2d(zz, d1_1, [1,1,1,1], 'SAME'), d1_1b))
    d2 = actf(tf.nn.bias_add(tf.nn.conv2d(d1, d1_2, [1,1,1,1], 'VALID'), d1_2b))
    d3 = actf(tf.nn.bias_add(tf.matmul(tf.reshape(d2, [-1, z_dim]), d1_3), d1_3b))
    d4 = tf.nn.bias_add(tf.matmul(tf.reshape(d3, [-1, z_dim]), d1_4), d1_4b)
    return d4

def toRGB(z, tsz):
    grgb = weight([1,1,int(z.get_shape()[3]),3], 'grgb'+str(tsz), 1.0)
    grgb_b = bias([3], 'grgb'+str(tsz)+'_b')
    return tf.tanh(tf.nn.bias_add(tf.nn.conv2d(z, grgb, [1,1,1,1], 'SAME'), grgb_b))

def fromRGB(z, sz, tsz):
    drgb = weight([1,1,3,sz], 'drgb'+str(tsz))
    drgb_b = bias([sz], 'drgb'+str(tsz)+'_b')
    return actf(tf.nn.bias_add(tf.nn.conv2d(z, drgb, [1,1,1,1], 'SAME'), drgb_b))

def generator(z, alpha):
    d = gen_5(z)
    sz = initial_size
    cnt = 1
    while target_size > sz:
        if sz * 2 == target_size :
            dd = (1.-alpha) * upsample(toRGB(d,target_size//2))
        w1 = weight([ksz,ksz,gin_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_2')
        b2 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_2b')
        d = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(upsample(d), w1, [1,1,1,1], 'SAME'), b1)))
        d = norm_scale(actf(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2)))
        cnt = cnt + 1
        sz *= 2
    d = toRGB(d, target_size)
    if cnt > 1:
        d = dd + alpha * d
    return d

def discriminator(real, alpha):
    sz = target_size
    tsz = sz
    cnt = 0
    while tsz > initial_size:
        tsz //= 2
        cnt = 1 + cnt
    
    d = fromRGB(real, gout_ch1[cnt], target_size)
    d2 = fromRGB(downsample(real), gin_ch1[cnt], target_size//2)
    while sz > initial_size:
        w1 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'd'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'd'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gin_ch1[cnt]], 'd'+str(1+cnt)+'_2')
        b2 = bias([gin_ch1[cnt]], 'd'+str(1+cnt)+'_2b')
        d = actf(tf.nn.bias_add(tf.nn.conv2d(d, w1, [1,1,1,1], 'SAME'), b1))
        d = actf(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2))
        d = downsample(d)
        if sz == target_size:
            d = (1.-alpha)*d2 + alpha * d
        cnt = cnt - 1
        sz //= 2
    return discr_5(d)

print ('prepare')

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' graph '''
    # inputs
    # Cramer GAN
    
    real = tf.placeholder(tf.float32, shape=[batch_size, target_size, target_size, 3])
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
    z2 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
    alpha = tf.placeholder(tf.float32)
    
    if target_size > 4:
        real = (1.-alpha)*down_up(real) + alpha * real

    fake1 = generator(z1, alpha)
    fake2 = generator(z2, alpha)

    r_logit = discriminator(real, alpha)
    f1_logit = discriminator(fake1, alpha)
    f2_logit = discriminator(fake2, alpha)

    print ('r_logit shape : ' + repr(r_logit.get_shape()) )
    print ('f1_logit shape : ' + repr(f1_logit.get_shape()) )
    print ('f2_logit shape : ' + repr(f2_logit.get_shape()) )

    def L2(logit):
        return tf.reduce_mean( tf.sqrt( tf.reduce_sum( tf.square( logit ), axis = 1 ) ) )

    def critic(logit):
        return L2( logit - f2_logit ) - L2( logit )
    
    # losses
    def gradient_penalty(real, fake):
        def interpolate(a, b):
            rg = tf.random_uniform(shape=[batch_size,1,1,1], minval=0., maxval=1.)
            rg = tf.image.resize_nearest_neighbor(rg, [initial_size, initial_size])
            rg = tf.concat([rg,rg,rg], axis=3)
            return rg * a + (1.-rg)*b

        x = interpolate(real, fake)
        pred = discriminator(x, alpha)
        gradients = tf.reshape(tf.gradients(pred, x)[0], shape=[batch_size, -1])
        print ('grad shape : ' + repr(gradients.get_shape()) )
        return tf.square(L2( gradients ) - 1.)
    
    g_loss = L2(r_logit - f1_logit) + L2( r_logit - f2_logit ) - L2( f1_logit - f2_logit )
    s_loss = critic( r_logit ) - critic( f1_logit )
    d_loss = -s_loss + 10.0 * gradient_penalty(real, fake1)

    # otpims
    g_vars = []
    d_vars = []
    for vari in tf.trainable_variables():
        if vari.name.startswith('x/g'):
            g_vars.append(vari)
        if vari.name.startswith('x/d'):
            d_vars.append(vari)
    d_step = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=0, beta2 = 0.9).minimize(d_loss, var_list=d_vars)
    g_step = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=0, beta2 = 0.9).minimize(g_loss, var_list=g_vars)
    
    # summaries
    d_summary = utils.summary({d_loss: 'd_loss'})#, wd: 'wd', gp: 'gp', rlm:'rlm'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z1, alpha)


""" init """
# session
sess = utils.session()
# saver
saver = tf.train.Saver(max_to_keep=1)

tsz=target_size
tcnt=1
while tsz > initial_size:
    tcnt=1+tcnt
    tsz//=2

loadvars=[]
for vari in tf.trainable_variables():
    for ttcnt in range(1,tcnt):
        if vari.name.startswith('x/g'+str(ttcnt)) or vari.name.startswith('x/d'+str(ttcnt)):
            loadvars.append(vari)

# loader
if target_size > initial_size:
    loader = tf.train.Saver(var_list=loadvars)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/wgp/%dx%d' % (target_size, target_size), sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/wgp'
utils.mkdir(ckpt_dir + '/')
sess.run(tf.global_variables_initializer())
if len(sys.argv)>3 and sys.argv[2]=='resume':
    saver.restore(sess, ckpt_dir+'/'+str(target_size)+'.ckpt')
elif target_size > initial_size:
    loader.restore(sess, ckpt_dir+'/'+str(target_size//2)+'.ckpt')

def get_input():
    vec = np.random.randn(batch_size, z_dim)
    for i in range(batch_size):
        vec[i] /= np.sqrt(vec[i].dot(vec[i]))
    return vec
    
''' train '''
batch_epoch = len(data_pool) // (batch_size * 1)#n_critic
max_it = epoch * batch_epoch
alpha_ipt = 0.0

it_start = 0
if len(sys.argv)>3 and sys.argv[2]=='resume':
    it_start = int(sys.argv[3])

for it in range(it_start, max_it):
    if target_size > initial_size:
        alpha_ipt = it / (alpha_span / batch_size)
    if alpha_ipt > 1:
        alpha_ipt = 1.0
    print(alpha_ipt)

    # which epoch
    epoch = it // batch_epoch
    it_epoch = it % batch_epoch + 1

    def get_ipt():
        z1_sample = get_input()
        z2_sample = get_input()
        real_ipt = data_pool.batch()
        real_ipt = np.reshape(real_ipt,(-1,target_size,target_size,3))
        return {real: real_ipt, z1: z1_sample, z2 : z2_sample, alpha: alpha_ipt}
        
    # train D
    for i in range(n_critic):
        d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict=get_ipt())
    summary_writer.add_summary(d_summary_opt, it)

    # train G
    g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict=get_ipt())
    summary_writer.add_summary(g_summary_opt, it)
    
    # display
    if it % 1 == 0:
        print("Iter: %6d, Epoch: (%3d/%3d) (%5d/%5d)" % (it,epoch, max_it // batch_epoch, it_epoch, batch_epoch))

    # sample
    if (it + 1) % batch_epoch == 0:
        f_sample_opt = sess.run(f_sample, feed_dict={z1: get_input(), alpha : alpha_ipt})
        f_sample_opt = np.clip(f_sample_opt, -1, 1)
        save_dir = './sample_images_while_training/wgp/%dx%d' % (target_size, target_size)
        utils.mkdir(save_dir + '/')
        osz = int(math.sqrt(batch_size))+1
        utils.imwrite(utils.immerge(f_sample_opt, osz, osz), '%s/Epoch_(%d)_(%dof%d).png' % (save_dir, epoch, it_epoch, batch_epoch))
        
    # save
    if (it + 1) % batch_epoch == 0:
        save_path = saver.save(sess, '%s/%d.ckpt' % (ckpt_dir, target_size))
        print('Model saved in file: %s' % save_path)