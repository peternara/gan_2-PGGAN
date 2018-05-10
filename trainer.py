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
alpha_span = 1600000
# 4-32 : 800000
# 64- : 1200000
# 128 : 1600000
# 256 :1600000
batch_size = 16
# -64:32? 128:24 256:16 512:8
epoch = (alpha_span * 2 // (2*4936)) # 4936 is number of images
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
data_pool = utils.DiskImageData(img_paths, batch_size//2, shape=[640, 640, 3], preprocess_fn=preprocess_fn)

def weight(shape, name, gain=2.0):
    coef = np.float32(np.sqrt( gain / np.prod(shape[:-1])))
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.initializers.random_normal())
        v = tf.multiply(tf.constant(coef, shape=shape), v)
    return v
    
def bias(shape, name):
    with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
        b = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer)
    return b

def norm_scale(z):
    m = tf.sqrt(tf.add(tf.constant(1e-8, shape=z.get_shape()), tf.reduce_mean(tf.square(z),axis=0,keepdims=True)))
    #return tf.nn.lrn(z,depth_radius=0,bias=1e-8,alpha=(1.0/ch),beta=0.5)
    return tf.div(z, m)
    
def upsample(z, size):
    return tf.image.resize_nearest_neighbor(z, [size, size])
    
def down_up(z):
    hh = z.get_shape()[1]
    ww = z.get_shape()[2]
    return tf.image.resize_nearest_neighbor(tf.image.resize_images(z, [hh//2,ww//2], method=tf.image.ResizeMethod.AREA), [hh,ww])
    
def downsample(z):
    return tf.nn.avg_pool(z, [1,2,2,1], [1,2,2,1], 'VALID')
    
def gen_5(z):
    base_channel = 512
    #g1_1 = weight([initial_size, initial_size, base_channel, z_dim], 'g1_1')# deconv filter : [fh, fw, out_ch, in_ch]
    #g1_1b = bias([base_channel], 'g1_1b')
    g1_1 = weight([z_dim,base_channel*initial_size*initial_size], 'g1_1', 1/8.0)
    g1_1b = bias([base_channel], 'g1_1b')
    g1_2 = weight([ksz, ksz, base_channel, base_channel], 'g1_2')# conv filter : [fh, fw, in_ch, out_ch]
    g1_2b = bias([base_channel], 'g1_2b') # same as out_ch
    #outsize = [batch_size, initial_size, initial_size, base_channel] # tensor, filter, output_size, stride
    #a = tf.reshape(z, [-1, 1, 1, z_dim])
    #use tansposed conv2d m = tf.nn.bias_add(tf.nn.conv2d_transpose(a, g1_1, outsize, [1,initial_size,initial_size,1]), g1_1b)
    g1 = norm_scale(tf.nn.leaky_relu(tf.nn.bias_add(tf.reshape(tf.matmul(tf.reshape(z, [-1, base_channel]), g1_1), [-1,initial_size, initial_size, base_channel]), g1_1b)))
    # tensor, filter, stride, padding
    g2 = norm_scale(tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(g1, g1_2, [1,1,1,1], 'SAME'), g1_2b)))
    return g2
    
def discr_5(z):
    # todo : minibatch norm
    base_channel = 512
    d1_1 = weight([ksz, ksz, base_channel + 1, base_channel], 'd1_1')
    d1_1b = bias([base_channel], 'd1_1b')
    d1_2 = weight([initial_size, initial_size, base_channel, base_channel], 'd1_2')
    d1_2b = bias([base_channel], 'd1_2b')
    # fullcon : flatten after batch axis and matmul
    d1_3 = weight([base_channel, base_channel], 'd1_3', 1.0)
    d1_3b = bias([base_channel], 'd1_3b')
    d1_4 = weight([base_channel, 1], 'd1_4')
    d1_4b = bias([1], 'd1_4b')
    
    if False:
        group_size = tf.minimum(4, batch_size)     # Minibatch must be divisible by (or smaller than) group_size.
        s = z.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(z, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, z.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
        zz = tf.concat([z, y], axis=3)                          # [NCHW]  Append as new fmap.
    else:
        sdev = tf.reduce_mean(tf.sqrt(tf.subtract(tf.reduce_mean(tf.square(z), axis = 0), tf.square(tf.reduce_mean(z, axis = 0)))))
        dd = tf.fill([batch_size, initial_size, initial_size, 1], sdev)
        zz = tf.concat([z, dd], axis=3)
    d1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(zz, d1_1, [1,1,1,1], 'SAME'), d1_1b))
    d2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(d1, d1_2, [1,1,1,1], 'VALID'), d1_2b))
    d3 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(tf.reshape(d2, [-1, base_channel]), d1_3), d1_3b))
    d4 = tf.nn.bias_add(tf.matmul(tf.reshape(d3, [-1, base_channel]), d1_4), d1_4b)
    return d4

def toRGB(z, sz, tsz):
    grgb = weight([1,1,sz,3], 'grgb'+str(tsz), 1.0)
    grgb_b = bias([3], 'grgb'+str(tsz)+'_b')
    return tf.nn.bias_add(tf.nn.conv2d(z, grgb, [1,1,1,1], 'SAME'), grgb_b)

def fromRGB(z, sz, tsz):
    drgb = weight([1,1,3,sz], 'drgb'+str(tsz))
    drgb_b = bias([sz], 'drgb'+str(tsz)+'_b')
    return tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(z, drgb, [1,1,1,1], 'SAME'), drgb_b))

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
        d = norm_scale(tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(upsample(d, 2*sz), w1, [1,1,1,1], 'SAME'), b1)))
        d = norm_scale(tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(d, w2, [1,1,1,1], 'SAME'), b2)))
        if sz * 2 == target_size :
            c = tf.subtract(tf.constant(1.0, shape=[batch_size, target_size, target_size, 3]), tf.fill([batch_size, target_size, target_size, 3], alpha))
            dd = tf.multiply(c, upsample(toRGB(p, gout_ch1[cnt-1],target_size//2), target_size))
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
    
    if target_size > 4:
        coef = tf.subtract(tf.constant(1.0, shape=real.get_shape()), tf.fill(real.get_shape(), alpha))
        real = tf.add(tf.multiply(coef, down_up(real)), tf.multiply(tf.fill(real.get_shape(), alpha), real))

    # generate
    fake = generator(z, alpha)

    # dicriminate
    r_logit = discriminator(real, alpha)
    f_logit = discriminator(fake, alpha)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            #shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            rg = tf.image.resize_nearest_neighbor(tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.), [target_size, target_size])
            rg = tf.concat([rg, rg, rg], axis=3)
            return tf.add(tf.multiply(rg, a), tf.multiply(tf.subtract(tf.constant(1.0, shape=[batch_size, target_size, target_size, 3]), rg), b))
            #inter = a + alpha_ * (b - a)
            #inter.set_shape(a.get_shape().as_list())
            #return inter

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
if len(sys.argv)>2 and sys.argv[1]=='resume':
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

it_start = 0
if len(sys.argv)>2 and sys.argv[1]=='resume':
    it_start = int(sys.argv[2])

for it in range(it_start, max_it):
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
        real_ipt = np.reshape(real_ipt,(-1,target_size,target_size,3))
        
        z_ipt = get_input()# np.random.normal(size=[batch_size, z_dim])
        d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt, alpha: alpha_ipt})
        #print ('loss_d = ' + repr(lossd))
    summary_writer.add_summary(d_summary_opt, it)

    # train G
    z_ipt = get_input()# np.random.normal(size=[batch_size, z_dim])
    g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt, alpha : alpha_ipt})
    summary_writer.add_summary(g_summary_opt, it)
    #print ('loss_g = ' + repr(lossg))

    # display
    if it % 1 == 0:
        print("Iter: %6d, Epoch: (%3d/%3d) (%5d/%5d)" % (it,epoch, max_it // batch_epoch, it_epoch, batch_epoch))

    # save
    if (it + 1) % batch_epoch == 0:
        save_path = saver.save(sess, '%s/%d.ckpt' % (ckpt_dir, target_size))
        print('Model saved in file: %s' % save_path)

    # sample
    if (it + 1) % batch_epoch == 0:
        f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample, alpha : alpha_ipt})
        f_sample_opt = np.clip(f_sample_opt, -1, 1)
        save_dir = './sample_images_while_training/wgp/%dx%d' % (target_size, target_size)
        utils.mkdir(save_dir + '/')
        osz = int(math.sqrt(batch_size))+1
        utils.imwrite(utils.immerge(f_sample_opt, osz, osz), '%s/Epoch_(%d)_(%dof%d).png' % (save_dir, epoch, it_epoch, batch_epoch))
