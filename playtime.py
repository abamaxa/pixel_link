import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Pure SGD
BATCH_SIZE=1
#Batch Gradient Descent
#BATCH_SIZE=1000
starter_learning_rate=0.001

xdata=np.linspace(0.,2*np.pi,1000)[:,np.newaxis]
ydata=np.sin(xdata)+np.random.normal(0.0,0.05,size=1000)[:,np.newaxis]
plt.scatter(xdata,ydata)

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32, [None,1])
#We define global_step as a variable initialized at 0
global_step=tf.Variable(0,trainable=False)

w1=tf.Variable(0.05*tf.random_normal((1,100)),tf.float32)
w2=tf.Variable(0.05*tf.random_normal((100,1)),tf.float32)

b1=tf.Variable(np.zeros([100]).astype("float32"),tf.float32)
b2=tf.Variable(np.zeros([1]).astype("float32"),tf.float32)

h1=tf.nn.relu(tf.matmul(x,w1)+b1)
y_model=tf.matmul(h1,w2)+b2


L=tf.reduce_mean(tf.square(y_model-y))

#We want to decrease the learning rate after having seen all the data 5 times
NUM_EPOCHS_PER_DECAY=5
LEARNING_RATE_DECAY_FACTOR=0.1
#Since the mechanism of the decay depends on the number of iterations and not epochs we have to connect the number of epochs to the number of iterations

#So if we have batch_size=1 we have to iterate exactly 1000 times to do one epoch so 5*1000=5000 iterations before decaying if the batch_size was 1000 1 iterations=1epoch and thus we decrease it after 5 iterations 
num_batches_per_epoch=int(xdata.shape[0]/float(BATCH_SIZE))
decay_steps=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
decayed_learning_rate=tf.train.exponential_decay(starter_learning_rate, 
                                                     global_step, 
                                                     decay_steps, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)

#So now we have an object that depends on global_step and that will be divided by 10 every decay_steps iterations i.e. when global_step=N*decay_steps with N a non zero integer         

#We now create a train_step to which we pass the learning rate created each time this function is called global_step will be incremented by 1 we are gonna check that it is the case BE CAREFUL WE HAVE TO GIVE IT GLOBAL_STEP AS AN ARGUMENT
train_step=tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(L,global_step=global_step)
sess=tf.Session()
sess.run(tf.initialize_all_variables())
GLOBAL_s=[]
lr_val=[]
COSTS=[]
for i in range(16000):
    #We will do 1600 iterations so as there is a decay every 5000 iterations we will see 3 decays (5000,10000,15000)
    start_data=(i*BATCH_SIZE)%1000
    COSTS.append([sess.run(L, feed_dict={x:xdata,y:ydata})])
    GLOBAL_s.append([sess.run(global_step)])
    lr_val.append([sess.run(decayed_learning_rate)])
    #I see the train_step as implicitely executing sess.run(tf.add(global_step,1))
    sess.run(train_step,feed_dict={x:xdata[start_data:start_data+BATCH_SIZE],y:ydata[start_data:start_data+BATCH_SIZE]})


plt.figure()
plt.subplot(211)
plt.plot(GLOBAL_s,lr_val,"-b")
plt.title("Evolution of learning rate" )
plt.subplot(212)
plt.plot(GLOBAL_s,COSTS,".g")
plt.title("Evolution of cost" )


#notice two things first global_step is actually being incremented and learning rate is actually being decayed
