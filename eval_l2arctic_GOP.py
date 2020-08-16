import argparse
import pickle
import numpy as np
import os
from collections import defaultdict

def ReadFile(ref, trans):
    groundturth_dict = {}
    with open(ref, "r") as f:
        for line in f.readlines():
            line_list = line.replace("\n","").split(",")
            n_line_list = []
            for item in line_list[1:]:
                if(item[:3]!="sil" and item[:2]!="sp"):
                    n_line_list.append(item)
            groundturth_dict[line_list[0]] = n_line_list

    predict_dict = {}
    with open(trans, "r") as f:
        for line in f.readlines():
            line = line.replace("\n","")
            predict_dict[line.split(",")[0]] = line.split(",")[1:]
        
    return groundturth_dict, predict_dict

def GetThreshold(ref, trans):
    # print(trans)
    threshold_candidate_dict = {}
    threshold_dict = {}
    for key, value in trans.items():
        for i in range(len(trans[key])):
            ref_item_list = ' '.join(ref[key][i].split()).split(" ")
            trans_item_list = ' '.join(trans[key][i].split()).split(" ")
            if(ref_item_list[0] not in threshold_candidate_dict):
                threshold_candidate_dict[ref_item_list[0]] = [[0,0] for i in range(101)]
            for i in range(101):
                if(ref_item_list[2]=="c" and float(trans_item_list[1])<float(i)):
                    threshold_candidate_dict[ref_item_list[0]][i][0]+=1
                elif(ref_item_list[2]!="c" and float(trans_item_list[1])>float(i)):
                    threshold_candidate_dict[ref_item_list[0]][i][1]+=1
                    
    for key, value in threshold_candidate_dict.items():
        min_count = 999999
        min_index = -1
        for i in range(len(value)):
            if(abs(value[i][0]+value[i][1])<min_count):
                min_index = i
                min_count = abs(value[i][0]-value[i][1])
        threshold_dict[key] = min_index
    return threshold_dict

    

def ConfusionMatrix(ref, trans, threshold_dict):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for key, value in trans.items():
        for i in range(len(trans[key])):
            ref_item_list = ' '.join(ref[key][i].split()).split(" ")
            trans_item_list = ' '.join(trans[key][i].split()).split(" ")
            if(ref_item_list[0] in threshold_dict):
                if(ref_item_list[2]=="c" and float(trans_item_list[1])>=threshold_dict[ref_item_list[0]]):
                    TP += 1
                elif(ref_item_list[2]!="c" and float(trans_item_list[1])<threshold_dict[ref_item_list[0]]):
                    TN += 1
                elif(ref_item_list[2]=="c" and float(trans_item_list[1])<threshold_dict[ref_item_list[0]]):
                    FN += 1
                elif(ref_item_list[2]!="c" and float(trans_item_list[1])>=threshold_dict[ref_item_list[0]]):
                    FP += 1
            else:
                if(ref_item_list[2]=="c" and float(trans_item_list[1])>=100):
                    TP += 1
                elif(ref_item_list[2]!="c" and float(trans_item_list[1])<100):
                    TN += 1
                elif(ref_item_list[2]=="c" and float(trans_item_list[1])<100):
                    FN += 1
                elif(ref_item_list[2]!="c" and float(trans_item_list[1])>=100):
                    FP += 1

    return TP, FN, FP, TN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ref_file", type=str, required=True)
    parser.add_argument("--train_trans_file", type=str, required=True)
    parser.add_argument("--test_ref_file", type=str, required=True)
    parser.add_argument("--test_trans_file", type=str, required=True)
    args = parser.parse_args()

    # Read droundtruth and predict data
    train_groundtruth_dict, trainpredic_dict = ReadFile(args.train_ref_file, args.train_trans_file)
    threshold_dict = GetThreshold(train_groundtruth_dict, trainpredic_dict)

    groundtruth_dict, predict_dict = ReadFile(args.test_ref_file, args.test_trans_file)
    TP, FN, FP, TN = ConfusionMatrix(groundtruth_dict, predict_dict, threshold_dict)
    print("TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}".format(TP=TP, FN=FN, FP=FP, TN=TN))
    print( "Accuracy: {acc}".format( acc=(TP+TN)/(TP+FN+FP+TN) ) )
    print( "Canonicals" )
    print( "True Accept Rate: {tar}".format( tar=(TP)/(TP+FN) ) )
    print( "False Reject Rate: {frr}".format( frr=(FN)/(TP+FN) ) )
    print( "Mispronunciations" )
    print( "False Accept Rate: {far}".format( far=(FP)/(TN+FP) ) )
    print( "True Reject Rate: {trr}".format( trr=(TN)/(TN+FP) ) )



if __name__ == "__main__":
    main()