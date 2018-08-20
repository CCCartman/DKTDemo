# -*- coding: utf-8 -*-
"""

@author: Rui Wenhao
@Mail:rwhcartman@163.com

"""

import numpy as np
import tensorflow as tf

hello=tf.constant('hhh')
sess=tf.Session()
print(sess.run(hello))

class ListNode:
    def __init__(self,x):
        self.val = x
        self.next = None
        
def reverseListNode(p):
    pReversedHead = None
    pre = None
    cur = p
    while cur:
        pNext = cur.next
        if pNext:
            pReversedHead = cur
        cur.next = pre
        pre = cur
        cur = pNext
    return pReversedHead
        
    
    