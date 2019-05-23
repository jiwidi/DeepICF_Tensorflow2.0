import tensorflow as tf

class GetEmbeddings(tf.keras.layers.Layer):
    """Layer to only get embeddings from onehot encoding
    
    Example:
    
    layer = GetEmbeddings()
    layer(tf.constant([[1,0,1,0,1],[1,0,1,0,1]]), tf.constant([[10.0, 10.0, 20.0, 20.0],
                      [11.0, 10.0, 10.0, 30.0],
                      [12.0, 10.0, 10.0, 20.0],
                      [13.0, 10.0, 10.0, 20.0],
                      [14.0, 11.0, 21.0, 31.0],
                      [15.0, 11.0, 11.0, 21.0],
                      [16.0, 11.0, 11.0, 21.0],
                      [17.0, 11.0, 11.0, 21.0]]))
                      
    output:
    <tf.Tensor: id=2764, shape=(2, 5, 4), dtype=float32, numpy=
    array([[[11., 10., 10., 30.],
            [10., 10., 20., 20.],
            [11., 10., 10., 30.],
            [10., 10., 20., 20.],
            [11., 10., 10., 30.]],

           [[11., 10., 10., 30.],
            [10., 10., 20., 20.],
            [11., 10., 10., 30.],
            [10., 10., 20., 20.],
            [11., 10., 10., 30.]]], dtype=float32)>
    
    """
    def __init__(self):
        super(GetEmbeddings, self).__init__()
    
    def build(self,input_shape):
        pass
    
    def call(self,input_tensor,embeddings_tensor):
        input_tensor = tf.expand_dims(input_tensor,tf.size(input_tensor.shape))
        output = tf.gather_nd(embeddings_tensor, input_tensor)
        return output
    
class PairwiseComparison(tf.keras.layers.Layer):
    """
    
    layer = PairwiseComparison()
    layer(tf.constant([[3,0,1],[1,2,3]]),tf.constant([[2,0,1]]))
    
    
    output:
    <tf.Tensor: id=2693, shape=(2, 3), dtype=int32, numpy=
    array([[6, 0, 1],
       [2, 0, 3]], dtype=int32)>
    """
    def __init__(self):
        super(PairwiseComparison, self).__init__()
    
    def build(self,input_shape):
        pass
    
    def call(self,customer_embeddings,item_embedding):
        item_embedding = tf.tile(item_embedding,[customer_embeddings.shape[0],1])        
        output = tf.math.multiply(customer_embeddings,item_embedding)
        return output
    
    
class AveragePooling(tf.keras.layers.Layer):
    """This pooling layer will garante that we allwyas get a constant length output. 
    The length will be the size of the embedding layer. 
    
    layer = AveragePooling()
    layer(tf.constant([[6, 0, 1],
           [2, 0, 3]]))
           
    output:
    <tf.Tensor: id=2727, shape=(3,), dtype=int32, numpy=array([4, 0, 2], dtype=int32)>
    
    """
    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def build(self,input_shape):
        pass
    
    def call(self,input):
        return tf.reduce_mean(input,0)

    
# also use the keras input layer     



def main():
    # ish how it will look I guess. Dont think there is need for more actually ...... 
    # How is the prediction score used .... 
    # KERAS layer input
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/InputLayer
    # First we have the input layer
    # https://www.tensorflow.org/guide/estimators#model_to_estimator  
    input_dim = 100
    # Define the input layers
    user_items = tf.keras.layers.InputLayer(input_shape=(28, 28, 1),name = "user_items")
    current_items = tf.keras.layers.InputLayer(input_shape=(1, 28, 1),name = "current_item")
    # First define the embeddings
    embeddings_user = tf.keras.layers.Embedding(input_dim, 64)(user_items)
    embeddings_items = tf.keras.layers.Embedding(input_dim, 64)(current_items)
    # Then we build the models extracting the embeddings based upon the keys
    layer_one = GetEmbeddings()(user_items,embeddings_user)
    layer_two = GetEmbeddings()(current_items,embeddings_items)
    # Then we do the pair wise comparison of the data
    layer_three = PairwiseComparison()(layer_one,layer_two)
    # Next step is to do the average pooling
    layer_four = AveragePooling()(layer_three)
    # Then it is the top dense layers
    layer_five = tf.keras.layers.Dense(64,32)(layer_four)
    layer_six = tf.keras.layers.Dense(32,16)(layer_five)
    out_put = tf.keras.layers.Dense(2,activation="Softmax")(layer_six)
    out_put.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                            loss='categorical_crossentropy',
                            metric='accuracy')
    

