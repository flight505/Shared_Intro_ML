from loader import reader

task = 'classification'
x_train, y_train, x_test, y_test = reader().get_all(task)

x_train = reader().get_x_train(task)
y_train = reader().get_y_train(task)
x_test = reader().get_x_test(task)
y_test = reader().get_y_test(task)

