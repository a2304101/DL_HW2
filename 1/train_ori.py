import dataset2
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
text_file = open("Output.txt", "w")
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

tf.reset_default_graph()
batch_size = 224

#Prepare input data
classes = ['Bread','Dairy product','Dessert','Egg','Fried food','Meat','NoodlesPasta','Rice','Seafood','Soup','VegetableFruit']
#classes = ['dogs','cats']
           
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='Food11/training'
valid_path='Food11/evaluation'
# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset2.read_train_sets(train_path , img_size , classes , validation_size=0)

data_valid = dataset2.read_train_sets(valid_path, img_size, classes, validation_size=0)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    #range(-0.1~0.1)

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    
    layer += biases
    
    ## We shall be using max-pooling.  
    
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    tf.summary.histogram("normal/w123", weights)
    tf.summary.histogram("normal/b123", biases)
    return tf.layers.batch_normalization(layer)

    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer

filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128
layer_conv1 = create_convolutional_layer(input=x,num_input_channels=3,conv_filter_size=3,num_filters=32)
layer_conv1 = tf.nn.max_pool(value=layer_conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
tf.summary.histogram("normal/layer_conv1", layer_conv1)
tf.add_to_collection('activations', layer_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,num_input_channels=32,conv_filter_size=3,num_filters=64)
layer_conv2_1 = create_convolutional_layer(input=layer_conv2,num_input_channels=64,conv_filter_size=3,num_filters=64)
layer_conv2_1 = tf.nn.max_pool(value=layer_conv2_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
tf.summary.histogram("normal/layer_conv2", layer_conv2_1)
tf.add_to_collection('activations', layer_conv2_1)

layer_conv3_1= create_convolutional_layer(input=layer_conv2_1,num_input_channels=64,conv_filter_size=3,num_filters=128)
layer_conv3 = create_convolutional_layer(input=layer_conv3_1,num_input_channels=128,conv_filter_size=3,num_filters=128)
layer_conv3 = tf.nn.max_pool(value=layer_conv3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
tf.summary.histogram("normal/layer_conv3", layer_conv3) 
tf.add_to_collection('activations', layer_conv3)
      
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = ff_1=tf.layers.dense(layer_flat,1024)
layer_fc1 = tf.nn.sigmoid(layer_fc1)
layer_fc1 = tf.layers.batch_normalization(layer_fc1)
layer_fc1 = tf.layers.dropout(layer_fc1,0.1)
layer_fc2 = tf.layers.dense(layer_fc1,num_classes)


y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
cosss=-tf.reduce_sum(y_true*tf.log(y_pred+(1-y_true)*tf.log(1-y_pred)))

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 

writer = tf.summary.FileWriter("histogram_example")
summaries = tf.summary.merge_all()




def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    with open('Output.txt', 'a') as the_file:
        the_file.write(msg.format(epoch + 1, acc, val_acc, val_loss)+'\n')
total_iterations = 0


#saver =tf.train.import_meta_graph('model/dogs-cats-model.meta')
#saver.restore(session , tf.train.latest_checkpoint('model/'))
saver = tf.train.Saver()
def train(num_iteration):
    
    global total_iterations
    yy=10
    while yy>1: 
        for i in range(total_iterations,
                       total_iterations + num_iteration):
    
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data_valid.train.next_batch(batch_size)
    
            
            feed_dict_tr = {x: x_batch,y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch,y_true: y_valid_batch}
    
            session.run(optimizer, feed_dict=feed_dict_tr)
            
            cos,acc = session.run([cost,accuracy], feed_dict=feed_dict_tr)
            cos2,acc2 = session.run([cost,accuracy], feed_dict=feed_dict_val)
            print(cos,acc,cos2,acc2)
            summ = session.run(summaries, feed_dict=feed_dict_tr)
            writer.add_summary(summ, global_step=i)
            if i % int(data.train.num_examples/batch_size) == 0: 
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                #cosss2 = session.run(cosss, feed_dict=feed_dict_val)
                #print(cosss2)
                epoch = int(i / int(data.train.num_examples/batch_size))    
                
                show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(session, '.model/dogs-cats-model') 
    
    
        total_iterations += num_iteration

train(num_iteration=10000)

text_file.close()