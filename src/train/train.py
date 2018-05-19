# -*- coding: utf-8 -*-
# file: train.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:29 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod


class Train(metaclass=ABCMeta):
    @abstractmethod
    def run(self, *args):
        """
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def get_model(self, *args):
        """
        :param args:
        :return:
        """
        pass
