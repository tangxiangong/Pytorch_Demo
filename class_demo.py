# class 类名
#     def(self, 参数列表):
#         pass
# 对象变量 = 类名()


class Cat:
    def __init__(self):
        pass

    def eat(self):
        print('小猫爱吃鱼')

    def drink(self):
        print('小猫要喝水')


tom = Cat()
tom.drink()
tom.eat()

