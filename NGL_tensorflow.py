import tensorflow

class NGL(tensorflow.keras.losses.Loss):
    def __init__(
    	self, 
    	scaling=False,
    	name="ngl_loss"):
        super().__init__(name=name)
        self.name = name
        self.scaling = scaling

    def call(self, y_true, y_pred):
        y_true = tensorflow.cast(y_true, tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, tensorflow.float32)
        if self.scaling == True:
	 	        y_pred = tensorflow.math.sigmoid(y_pred)
        part_1 = tensorflow.math.exp(2.4092 - y_pred - y_pred*y_true)
        part_2 = tensorflow.math.cos(tensorflow.math.cos(tensorflow.math.sin(y_pred)))
        elements = part_1 - part_2
        loss = tensorflow.reduce_mean(elements)
        return loss
