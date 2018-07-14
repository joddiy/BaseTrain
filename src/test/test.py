# -*- coding: utf-8 -*-
# file: test.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:29 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod


class Test(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        """
        :return:
        """
        pass

    @abstractmethod
    def predict(self, model):
        """
        :return:
        """
        pass
