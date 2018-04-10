# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 1:43:58 2018

@author: Nikhil Dikshit
"""

import pandas

dataset = pandas.read_csv("C:/Users/Nikhil Dikshit/Desktop/ISTRAC/Dataset/CICIDS2017.csv", engine = 'python', header = None, sep = ',')

dataset.columns = ["ifInOctets11",
"ifOutOctets11" ,
"ifoutDiscards11",
"ifInUcastPkts11",
"ifInNUcastPkts11",
"ifInDiscards11",
"ifOutUcastPkts11",
"ifOutNUcastPkts11",
"tcpOutRsts",
"tcpInSegs",
"tcpOutSegs",
"tcpPassiveOpens",
"tcpRetransSegs",
"tcpCurrEstab",
"tcpEstabResets",
"tcpActiveOpens",
"udpInDatagrams",
"udpOutDatagrams",
"udpInErrors",
"udpNoPorts",
"ipInReceives",
"ipInDelivers",
"ipOutRequests",
"ipOutDiscards",
"ipInDiscards",
"ipForwDatagrams",
"ipOutNoRoutes",
"ipInAddrErrors",
"icmpInMsgs",
"icmpInDestUnreachs",
"icmpOutMsgs",
"icmpOutDestUnreachs",
"icmpInEchos",
"icmpOutEchoReps",
"labels"]

allLayer = dataset[(dataset['labels'].isin(["normal", "icmp-echo", "tcp-syn", "udp-flood", 
											"httpFlood", "slowloris", "slowPost", "bruteForce"]))]

print("\nNumber of possible attacks from the PCAP file:\n")
print(allLayer['labels'].value_counts())

allLayer.to_csv('C:/Users/Nikhil Dikshit/Desktop/ISTRAC/Output Files/Instances.txt', header = None, index = False)

# breakpoint
# print("\nNikhil")