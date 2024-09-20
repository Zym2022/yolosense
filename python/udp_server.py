#!/usr/bin/python3
# -*- coding: utf-8 -*-
 
"""
udp通信例程：udp client端，修改udp_addr元组里面的ip地址，即可实现与目标机器的通信，
此处以单机通信示例，ip为127.0.0.1，实际多机通信，此处应设置为目标服务端ip地址
"""
 
from time import sleep
import socket
 
pub_r00, pub_r01, pub_r02, pub_t0, pub_r10, pub_r11, pub_r12, pub_t1, pub_r20, pub_r21, pub_r22, pub_t2 \
=-0.702695906162262, 2.0923598640365526e-06, 0.7114903330802917, 0.6234617829322815, \
2.8049548745912034e-06, 1.0, -1.705287218101148e-07, -0.1501055508852005, \
-0.7114903330802917, 1.8758682927000336e-06, -0.702695906162262, 0.35909584164619446

# = -0.4999368190765381, -1.5891189832473174e-05, 0.8660618662834167, 0.7147712707519531,\
# -1.0617844964144751e-05, 1.0, 1.221960974362446e-05, -0.1500995010137558,\
# -0.8660618662834167, -3.086678134422982e-06, -0.4999368190765381, 0.16625721752643585



def main():
    # udp 通信地址，IP+端口号
    udp_addr = ('127.0.0.1', 8888)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
    # 发送数据到指定的ip和端口,每隔1s发送一次，发送10次
    for i in range(10):
        udp_socket.sendto((str(pub_r00) + ' ' + str(pub_r01)+ ' ' + str(pub_r02)+ ' ' + str(pub_t0)+ ' ' + str(pub_r10)+ ' ' + str(pub_r11)+ ' ' + str(pub_r12)+ ' ' + str(pub_t1)+ ' ' + \
                           str(pub_r20) + ' ' + str(pub_r21)+ ' ' + str(pub_r22)+ ' ' + str(pub_t2)+ ' ' + str(0)+ ' ' + str(0)+ ' ' + str(0)+ ' ' + str(1)) .encode('utf-8'), udp_addr)
        print(str(pub_r00) + ' ' + str(pub_r01)+ ' ' + str(pub_r02)+ ' ' + str(pub_t0)+ ' ' + str(pub_r10)+ ' ' + str(pub_r11)+ ' ' + str(pub_r12)+ ' ' + str(pub_t1)+ ' ' + \
              str(pub_r20) + ' ' + str(pub_r21)+ ' ' + str(pub_r22)+ ' ' + str(pub_t2)+ ' ' + str(0)+ ' ' + str(0)+ ' ' + str(0)+ ' ' + str(1) )
        sleep(1)
 
    # 5. 关闭套接字
    udp_socket.close()
 
if __name__ == '__main__':
    print("udp client ")
    main()