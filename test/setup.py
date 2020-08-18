#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : setup.py
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='ncrelu_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            'ncrelu_cpp', ['ncrelu.cpp']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)