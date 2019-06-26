# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)

from sklearn import datasets
iris = datasets.load_iris()

print(type(iris))
