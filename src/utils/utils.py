# -*- coding: utf-8 -*-
# file: utils.py
# author: joddiyzhang@gmail.com
# time: 2018/5/20 12:45 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import pickle


def save(obj, name):
    try:
        filename = open(name + ".pickle", "wb")
        pickle.dump(obj, filename)
        filename.close()
        return True
    except:
        return False


