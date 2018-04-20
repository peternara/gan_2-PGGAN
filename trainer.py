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
alpha_span = 800000
batch_size = 14 # 256:14 512:6 640:3
epoch = (alpha_span * 2 // 4936) # 4936 is number of images
lr = 0.001
z_dim = 512
gpu_id = 0
n_critic = 1
ksz = 3
gin_ch1 = [512,512,512,512,256,128,64,32]
gout_ch1 = [512,512,512,256,128,64,32,16]
target_size = 256
mean_img = np.zeros([640,640,3], dtype='float32')
initial_size = 4

path = './imgs/faces/'
def preprocess_fn(img):
    #img = tf.subtract(tf.to_float(img) / 255.0, tf.constant(mean_img))# 127.5 - 1
    img = tf.image.resize_images(img, [target_size, target_size], method=tf.image.ResizeMethod.AREA) / 127.5 -1
    return img

img_paths = glob.glob(path+'*.png')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[640, 640, 3], preprocess_fn=preprocess_fn)

def weight(shape, name):
    if len(shape) == 4:
        coef = math.sqrt( 2.0 / (shape[0] * shape[1] * shape[2]) )
    elif len(shape) == 2:
        coef = math.sqrt( 2.0 / shape[0] )
    print(name + ' ' + str(coef))
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer()) #tf.contrib.layers.xavier_initializer())#tf.contrib.layers.variance_scaling_initializer())
        v = tf.multiply(tf.constant(coef, shape=shape), v)# coef * v
    return v
    
def bias(shape, name):
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        b = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer)
    return b

def norm_scale(z, shape):
    m = tf.add(tf.constant(1e-8, shape=shape), tf.sqrt(tf.reduce_mean(tf.square(z),axis=0,keepdims=True)))
    return tf.div(z, m)#tf.constant(1.0 / m + 1e-8, shape=shape))

def upsample(z, size):
    return tf.image.resize_nearest_neighbor(z, [size, size])
    
def downsample(z):
    return tf.nn.avg_pool(z, [1,2,2,1], [1,2,2,1], 'SAME')
    
def gen_5(z):
    base_channel = 512
    g1_1 = weight([initial_size, initial_size, base_channel, z_dim], 'g1_1')# deconv filter : [fh, fw, out_ch, in_ch]
    g1_1b = bias([base_channel], 'g1_1b')
    g1_2 = weight([ksz, ksz, base_channel, base_channel], 'g1_2')# conv filter : [fh, fw, in_ch, out_ch]
    g1_2b = bias([base_channel], 'g1_2b') # same as out_ch
    a = tf.reshape(z, [-1, 1, 1, z_dim])
    # tensor, filter, output_size, stride
    outsize = [batch_size, initial_size, initial_size, base_channel]
    m = tf.nn.bias_add(tf.nn.conv2d_transpose(a, g1_1, outsize, [1,initial_size,initial_size,1]), g1_1b)
    b = tf.nn.leaky_relu(norm_scale(m, outsize))
    # tensor, filter, stride, padding
    bb = tf.nn.bias_add(tf.nn.conv2d(b, g1_2, [1,1,1,1], 'SAME'), g1_2b)
    return tf.nn.leaky_relu(norm_scale(bb, outsize))
    
def discr_5(z):
    # todo : minibatch norm
    base_channel = 512
    d1_1 = weight([ksz, ksz, base_channel + 1, base_channel], 'd1_1')
    d1_1b = bias([base_channel], 'd1_1b')
    d1_2 = weight([initial_size, initial_size, base_channel, base_channel], 'd1_2')
    d1_2b = bias([base_channel], 'd1_2b')
    d1_3 = weight([base_channel, 1], 'd1_3') # fullcon : flatten after batch axis and matmul
    d1_3b = bias([1], 'd1_3b')
    n = tf.constant(1.0 / batch_size, shape=[1, initial_size, initial_size, base_channel])
    m = tf.multiply(tf.reduce_sum(z, axis = 0), n)
    s = tf.multiply(tf.reduce_sum(tf.square(z), axis = 0), n)
    sdev = tf.reduce_mean(tf.sqrt(tf.subtract(s, tf.multiply(m,m))))
    dd = tf.fill([batch_size, initial_size, initial_size, 1], sdev)
    zz = tf.concat([z, dd], axis=3)
    d0 = tf.nn.bias_add(tf.nn.conv2d(zz, d1_1, [1,1,1,1], 'SAME'), d1_1b)
    d1 = tf.nn.leaky_relu(d0)
    d2 = tf.nn.bias_add(tf.nn.conv2d(d1, d1_2, [1,1,1,1], 'VALID'), d1_2b)
    d3 = tf.nn.leaky_relu(d2)
    d4 = tf.nn.bias_add(tf.matmul(tf.reshape(d3, [-1, base_channel]), d1_3), d1_3b)
    return d4

def toRGB(z, sz, tsz):
    grgb = weight([1,1,sz,3], 'grgb'+str(tsz))
    return tf.nn.conv2d(z, grgb, [1,1,1,1], 'SAME')

def fromRGB(z, sz, tsz):
    drgb = weight([1,1,3,sz], 'drgb'+str(tsz))
    return tf.nn.leaky_relu(tf.nn.conv2d(z, drgb, [1,1,1,1], 'SAME'))

def generator(z, alpha):
    d = gen_5(z)
    sz = initial_size
    cnt = 1
    while target_size > sz:
        p = tf.identity(d)
        w1 = weight([ksz,ksz,gin_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'g'+str(1+cnt)+'_2')
        b2 = bias([gout_ch1[cnt]], 'g'+str(1+cnt)+'_2b')
        d = tf.nn.leaky_relu(norm_scale(tf.nn.bias_add(tf.nn.conv2d(upsample(d, 2*sz), w1, [1,1,1,1], 'SAME'), b1), [batch_size, 2*sz, 2*sz, gout_ch1[cnt]]))
        d = tf.nn.leaky_relu(norm_scale(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2), [batch_size, 2*sz, 2*sz, gout_ch1[cnt]]))
        if sz * 2 == target_size :
            c = tf.subtract(tf.constant(1.0, shape=[batch_size, target_size, target_size, 3]), tf.fill([batch_size, target_size, target_size, 3], alpha))
            dd = tf.multiply(c, toRGB(upsample(p, target_size), gout_ch1[cnt-1],target_size//2))
        cnt = cnt + 1
        sz *= 2
    d = toRGB(d, gout_ch1[cnt-1],target_size)
    if cnt > 1:
        d = tf.add(tf.multiply(tf.fill([batch_size, sz, sz, 3], alpha), d), dd)
    return d

def discriminator(real, alpha):
    sz = target_size
    tsz = sz
    cnt = 0
    while tsz > initial_size:
        tsz //= 2
        cnt = 1 + cnt
    
    d = fromRGB(real, gout_ch1[cnt],target_size)
    d2 = fromRGB(downsample(real), gin_ch1[cnt],target_size//2)
    while sz > initial_size:
        w1 = weight([ksz,ksz,gout_ch1[cnt],gout_ch1[cnt]], 'd'+str(1+cnt)+'_1')
        b1 = bias([gout_ch1[cnt]], 'd'+str(1+cnt)+'_1b')
        w2 = weight([ksz,ksz,gout_ch1[cnt],gin_ch1[cnt]], 'd'+str(1+cnt)+'_2')
        b2 = bias([gin_ch1[cnt]], 'd'+str(1+cnt)+'_2b')
        d = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(d, w1, [1,1,1,1], 'SAME'), b1))
        d = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2))
        d = downsample(d)
        # downsample
        if sz == target_size:
            c = tf.subtract(tf.constant(1.0, shape=[batch_size, target_size//2, target_size//2, gin_ch1[cnt]]), tf.fill([batch_size, target_size//2, target_size//2, gin_ch1[cnt]], alpha))
            d = tf.add(tf.multiply(c, d2), tf.multiply(tf.fill([batch_size, target_size//2, target_size//2, gin_ch1[cnt]], alpha), d))
        cnt = cnt - 1
        sz //= 2
    return discr_5(d)

print ('prepare')

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    print ('gpu')
    ''' models '''
    #generator = models.generator
    #discriminator = models.discriminator_wgan_gp

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[batch_size, target_size, target_size, 3])
    z = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
    alpha = tf.placeholder(tf.float32)
    
    # generate
    fake = generator(z, alpha)

    # dicriminate
    r_logit = discriminator(real, alpha)
    f_logit = discriminator(fake, alpha)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha_ = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha_ * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x, alpha)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))#, axis=range(1, x.shape.ndims)))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    rmean = tf.reduce_mean(tf.square(r_logit))
    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0 + (0.001*rmean)
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    #d_var = utils.trainable_variables('discriminator')
    #g_var = utils.trainable_variables('generator')
    g_vars = []
    d_vars = []
    for vari in tf.trainable_variables():
        if vari.name.startswith('x/g'):
            g_vars.append(vari)
        if vari.name.startswith('x/d'):
            d_vars.append(vari)
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0, beta2 = 0.99).minimize(d_loss, var_list=d_vars)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0, beta2 = 0.99).minimize(g_loss, var_list=g_vars)
    
    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, alpha)


""" init """
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter(init=0)
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
summary_writer = tf.summary.FileWriter('./summaries/wgp', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/wgp'
utils.mkdir(ckpt_dir + '/')
sess.run(tf.global_variables_initializer())
if len(sys.argv)>1 and sys.argv[0]=='resume':
    saver.restore(sess, ckpt_dir+'/'+str(target_size)+'.ckpt')
elif target_size > initial_size:
    loader.restore(sess, ckpt_dir+'/'+str(target_size//2)+'.ckpt')

    
def get_input():
    r = 2 * np.random.random(batch_size*z_dim).reshape(batch_size,z_dim) - 1
    for i in range(batch_size):
        r[i]/=math.sqrt(np.add.reduce(np.square(r[i])))
    return r
    
''' train '''
batch_epoch = len(data_pool) // (batch_size * n_critic)
max_it = epoch * batch_epoch
alpha_ipt = 0.0

for it in range(sess.run(it_cnt), max_it):
    sess.run(update_cnt)
    z_ipt_sample = get_input()
    
    if target_size > initial_size:
        alpha_ipt = it / (alpha_span / batch_size)
    if alpha_ipt > 1:
        alpha_ipt = 1.0
    print(alpha_ipt)

    # which epoch
    epoch = it // batch_epoch
    it_epoch = it % batch_epoch + 1

    # train D
    for i in range(n_critic):
        # batch data
        real_ipt = data_pool.batch()
        z_ipt = get_input()# np.random.normal(size=[batch_size, z_dim])
        d_summary_opt, _, lossd = sess.run([d_summary, d_step, d_loss], feed_dict={real: real_ipt, z: z_ipt, alpha: alpha_ipt})
        print ('loss_d = ' + repr(lossd))
    summary_writer.add_summary(d_summary_opt, it)

    # train G
    z_ipt = get_input()# np.random.normal(size=[batch_size, z_dim])
    g_summary_opt, _, lossg = sess.run([g_summary, g_step, g_loss], feed_dict={z: z_ipt, alpha : alpha_ipt})
    summary_writer.add_summary(g_summary_opt, it)
    print ('loss_g = ' + repr(lossg))

    # display
    if it % 1 == 0:
        print("Epoch: (%3d/%3d) (%5d/%5d)" % (epoch, max_it // batch_epoch, it_epoch, batch_epoch))

    # save
    if (it + 1) % batch_epoch == 0:
        save_path = saver.save(sess, '%s/%d.ckpt' % (ckpt_dir, target_size))
        print('Model saved in file: %s' % save_path)

    # sample
    if (it + 1) % batch_epoch == 0:
        f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample, alpha : alpha_ipt})
        f_sample_opt = np.clip(f_sample_opt, -1, 1)
        save_dir = './sample_images_while_training/wgp'
        utils.mkdir(save_dir + '/')
        utils.imwrite(utils.immerge(f_sample_opt, 4, 4), '%s/Epoch_(%d)_(%dof%d).png' % (save_dir, epoch, it_epoch, batch_epoch))
