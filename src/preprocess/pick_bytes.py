# -*- coding: utf-8 -*-
# file: pick_bytes.py
# author: joddiyzhang@gmail.com
# time: 2018/6/2 10:44 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------


max_len = 2048


def crop_exceed_data(data):
    if len(data) <= max_len:
        return data
    return data[0: max_len]


def get_bytes_array(data):
    """
    int to bytes array
    :param data:
    :return:
    """
    bytes_data = bytes(map(int, data.split(",")))
    bytes_data = crop_exceed_data(bytes_data)
    return [int(single_byte) for single_byte in bytes_data]


def reverse_bytes(original):
    """
    make the bytes inverse
    :param original:
    :return:
    """
    return original[::-1]


def convert_int(str_bytes):
    """
    convert bytes to int
    :param str_bytes:
    :return:
    """
    return int.from_bytes(str_bytes, byteorder='big', signed=False)


def decode_rich_sign(rich_sign):
    """
    decode the rich sign, use the last 4 bytes to xor each 4 bytes from the start to end
    :param rich_sign:
    :return:
    """
    key = rich_sign[-4:]
    rich_sign_d = bytearray()
    for i in range(len(rich_sign)):
        rich_sign_d.append(rich_sign[i] ^ key[i % 4])
    return bytes(rich_sign_d)


def get_fixed_head(data):
    """
    select some useful parts from the whole PE head
    :param data:
    :return:
    """
    bytes_data = bytes(map(int, data.split(",")))
    # mz head
    mz_head = bytes_data[0:64]
    # # dos sub
    # ms_dos_sub = bytes_data[64:128]
    # decode rich sign
    rich_sign_end = bytes_data[128:].find(b'\x52\x69\x63\x68') + 136
    rich_sign = decode_rich_sign(bytes_data[128:rich_sign_end])
    # pe head
    pe_head_start = bytes_data[128:].find(b'\x50\x45\x00\x00') + 128
    pe_head = bytes_data[pe_head_start:pe_head_start + 24]
    # there are two types of image optional head, PE 32 and PE 32+
    other_head = bytes_data[pe_head_start + 24:]
    if other_head[0:2] == b'\x0b\x01':
        image_optional_head_end = 96
    else:
        image_optional_head_end = 112
    image_optional_head = other_head[0:image_optional_head_end]
    # data directory
    data_directory = other_head[image_optional_head_end: image_optional_head_end + 128]
    # append all above parts
    # fixed_head = mz_head + ms_dos_sub + rich_sign + pe_head + image_optional_head + data_directory
    fixed_head = mz_head + rich_sign + pe_head + image_optional_head + data_directory
    # for each sections, just get the non-zero value
    number_of_sections = convert_int(reverse_bytes(pe_head[6:8]))
    for offset in range(number_of_sections):
        offset_sections_start = image_optional_head_end + 128 + 40 * offset
        # fixed_head += other_head[offset_sections_start: offset_sections_start + 28] + \
        #               other_head[offset_sections_start + 36:offset_sections_start + 40]
        fixed_head += other_head[offset_sections_start + 36:offset_sections_start + 40]
    return [int(single_byte) for single_byte in fixed_head]
