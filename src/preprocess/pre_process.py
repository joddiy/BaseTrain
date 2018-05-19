# -*- coding: utf-8 -*-
# file: pre_process.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 8:23 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from abc import ABCMeta, abstractmethod


class PreProcess(metaclass=ABCMeta):
    @abstractmethod
    def run(self, *args):
        """
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def read_input(self, *args):
        """
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def feature_engineering(self, *args):
        """
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def save_cleaned_data(self, *args):
        """
        :param args:
        :return:
        """
        pass
