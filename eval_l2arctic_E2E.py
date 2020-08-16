#this code is for dealing recognize results to get which phone has error
import json
import re
import numpy as np
import argparse


class CAPT_detection():
    def  __init__(self):
        self.transcript_filename = '' # trans_origin.txt
        self.recog_filename = '' # data.json
        self.save_path = ''
        self.text_prompt_dict = {} # 前面是id 後面是對應的phone (應該要念的)(文本提示)
        self.recog_result = {} # 0~n-1 best 存辨識結果 -1 真正念甚麼
        self.detection_mode = 1
        self.detection_align_result = {} # detection的對齊結果
        self.TA = 0 # 正確接受
        self.FA = 0 # 錯誤接受
        self.TR = 0 # 正確拒絕
        self.FR = 0 # 錯誤拒絕
        self.Cpre = 0
        self.Crec = 0
        self.Cf1 = 0
        self.Mpre = 0
        self.Mrec = 0
        self.Mf1 = 0

        self.mis_init = [0, 0, 0, 0] # 錯誤聲母 TA FA TR FR
        self.mis_final = [0, 0, 0, 0] # 錯誤韻母 TA FA TR FR
        self.tone_list = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # 紀錄tone的情況 TA FA TR FR

    def get_parser(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--testans_filename",
                            default="/share/nas165/fanjiang/CAPT/E2E/data/should_say_phone_valid/groundtruth_annotation.txt",
                            type=str)

        parser.add_argument("--recog_filename",
                            default="/share/nas165/fanjiang/CAPT/E2E/exp/train_pytorch_train_specaug/decode_test_phone_model.acc.best_decode_lm/data.json",
                            type=str)
        
        parser.add_argument("--detection_mode",
                            default=1,
                            type=int)
        
        parser.add_argument("--save_path",
                            default='capt',
                            type=str)

        args = parser.parse_args()

        self.testans_filename = args.testans_filename 
        self.recog_filename = args.recog_filename
        self.detection_mode = args.detection_mode
        self.save_path = args.save_path

    def wagner_fischer(self, word_1, word_2):
        n = len(word_1) + 1  # counting empty string 
        m = len(word_2) + 1  # counting empty string

        # initialize D matrix
        D = np.zeros(shape=(n, m), dtype=np.int)
        D[:,0] = range(n)
        D[0,:] = range(m)

        # B is the backtrack matrix. At each index, it contains a triple
        # of booleans, used as flags. if B(i,j) = (1, 1, 0) for example,
        # the distance computed in D(i,j) came from a deletion or a
        # substitution. This is used to compute backtracking later.
        B = np.zeros(shape=(n, m), dtype=[("del", 'b'), 
                        ("sub", 'b'),
                        ("ins", 'b')])
        B[1:,0] = (1, 0, 0) 
        B[0,1:] = (0, 0, 1)

        for i, l_1 in enumerate(word_1, start=1):
            for j, l_2 in enumerate(word_2, start=1):
                deletion = D[i-1,j] + 1
                insertion = D[i, j-1] + 1
                substitution = D[i-1,j-1] + (0 if l_1==l_2 else 2)

                mo = np.min([deletion, insertion, substitution])

                B[i,j] = (deletion==mo, substitution==mo, insertion==mo)
                D[i,j] = mo
                
        return D, B
    
    def naive_backtrace(self, B_matrix):
        i, j = B_matrix.shape[0]-1, B_matrix.shape[1]-1
        backtrace_idxs = [(i, j)]

        while (i, j) != (0, 0):
            if B_matrix[i,j][1]:
                i, j = i-1, j-1
            elif B_matrix[i,j][0]:
                i, j = i-1, j
            elif B_matrix[i,j][2]:
                i, j = i, j-1
            backtrace_idxs.append((i,j))

        return backtrace_idxs

    def align(self, word_1, word_2, bt):

        aligned_word_1 = []
        aligned_word_2 = []
        operations = []

        backtrace = bt[::-1]  # make it a forward trace
        for k in range(len(backtrace) - 1): 
            i_0, j_0 = backtrace[k]
            i_1, j_1 = backtrace[k+1]
            w_1_letter = None
            w_2_letter = None
            op = None

            if i_1 > i_0 and j_1 > j_0:  # either substitution or no-op
                if word_1[i_0] == word_2[j_0]:  # no-op, same symbol
                    w_1_letter = word_1[i_0]
                    w_2_letter = word_2[j_0]
                    op = "T"
                else:  # cost increased: substitution
                    w_1_letter = word_1[i_0]
                    w_2_letter = word_2[j_0]
                    op = "F"
            elif i_0 == i_1:  # insertion
                    w_1_letter = " "
                    w_2_letter = word_2[j_0]
                    op = "*"
            else: #  j_0 == j_1,  deletion
                w_1_letter = word_1[i_0]
                w_2_letter = " "
                op = "F"

            aligned_word_1.append(w_1_letter)
            aligned_word_2.append(w_2_letter)
            operations.append(op)

        return aligned_word_1, aligned_word_2, operations

    def read_transcript(self):
        # print('read transcript ans')
        with open(self.testans_filename) as lines:
            for line in lines:
                nline = line.split(',', 1) # 範例 ['uid', 'should_say_phn act_say_phn c/s/a,...,should_say_phn act_say_phn c/s/a\n']
                self.text_prompt_dict[nline[0]] = nline[1] # 前面是id 後面是對應的phone (應該要念的)
        # print(self.text_prompt_dict)

    def read_recog_file(self):
        # print('read data.json ')
        with open(self.recog_filename) as handle: 
            dictdump = json.loads(handle.read())
            for key in dictdump["utts"]:
                pre_ans=[]
                # print(str(key)+" ",end="")
                self.recog_result[key] = []
                for j in range(len(dictdump["utts"][key]["output"])):# --- 5 nbest
                    self.recog_result[key].append(dictdump["utts"][key]["output"][j]["rec_token"])
                self.recog_result[key].append(dictdump["utts"][key]["output"][j]["token"])
                # print(self.recog_result[key]) # {'uid': ['{phn} ... {phn} <eos>', '{a1} <eos>', '{h} <eos>', '{a2} <eos>', '{0} <eos>', '{a1}']}


    def write_file(self):
        print("寫入檔案")
        save_name = '/CAPT_detection'
        if self.detection_mode > 1 :
            save_name +=  '_'+str(self.detection_mode)+'nbest'
        save_name += '.txt'
        with open(self.save_path+save_name, 'w') as out:
            # out.write("%s:%s\n"%(counter, ))
            out.write('TA:'+str(self.TA)+'\n')
            out.write('FA:'+str(self.FA)+'\n')
            out.write('TR:'+str(self.TR)+'\n')
            out.write('FR:'+str(self.FR)+'\n')
            out.write('Cprecision:'+str(self.Cpre)+'\n')
            out.write('Crecall:'+str(self.Crec)+'\n')
            out.write('CF1:'+str(self.Cf1)+'\n')
            out.write('Mprecision:'+str(self.Mpre)+'\n')
            out.write('Mrecall:'+str(self.Mrec)+'\n')
            out.write('MF1:'+str(self.Mf1)+'\n')
        print("檔案位置: %s"%(self.save_path+save_name))
                
    def detection_align(self):
        if self.detection_mode == 1 : # 1 best
            for key in self.recog_result:
                utt = self.recog_result[key][0]
                clear_recog_utt = utt.replace('<eos>','')
                clear_recog_utt = clear_recog_utt.split()
                text_prompt = self.text_prompt_dict[key]
                TF_label = [ text_prompt.split(",")[i].split()[2] for i in range(len(text_prompt.split(","))) if(text_prompt.split(",")[i].split()[0]!="sp") ] # 紀錄correct substitution insertion addition
                text_prompt = [ text_prompt.split(",")[i].split()[0] for i in range(len(text_prompt.split(","))) if(text_prompt.split(",")[i].split()[0]!="sp") ] # 去掉"SIL"以及"SP"
                for idx in range(len(text_prompt)):
                    text_prompt[idx] = '{'+text_prompt[idx]+'}'
                #  資料前處理好了，開始錯誤偵測
                D, B = self.wagner_fischer(text_prompt, clear_recog_utt)
                bt = self.naive_backtrace(B)
                alignment_table = self.align(text_prompt, clear_recog_utt, bt)
                self.detection_align_result[key] = [alignment_table, TF_label] # 對齊結果與正確結果
        elif self.detection_mode > 1 : # n best
            for key in self.recog_result:
                for best in range(self.detection_mode):
                    utt = self.recog_result[key][best]
                    clear_recog_utt = utt.replace('<eos>','')
                    clear_recog_utt = clear_recog_utt.split()

                    rec_ans_token = self.recog_result[key][-1]
                    clear_rec_ans_token = rec_ans_token.replace('<eos>','')
                    clear_rec_ans_token = clear_rec_ans_token.split()
                    if clear_recog_utt == clear_rec_ans_token:
                        break
                text_prompt = self.text_prompt_dict[key]
                prompt_and_res = text_prompt.split()
                text_prompt = prompt_and_res[0]
                text_prompt = text_prompt.replace('-','')
                text_prompt = text_prompt.split(',')
                for idx in range(len(text_prompt)):
                    text_prompt[idx] = '{'+text_prompt[idx]+'}'

                #  資料前處理好了，開始錯誤偵測
                D, B = self.wagner_fischer(text_prompt, clear_recog_utt)
                bt = self.naive_backtrace(B)
                alignment_table = self.align(text_prompt, clear_recog_utt, bt)
                self.detection_align_result[key] = [alignment_table, prompt_and_res[1]] # 對齊結果與正確結果

    def precision(self):
        return  self.TA/(self.TA+self.FA)

    def recall(self):
        return self.TA/(self.TA+self.FR)

    def F1_score(self, pre, rec):
        return 2*pre*rec/(pre+rec)

    def clean_list(self, li): # 去除list中不需要用的值
        del_list = ['*', ' ', '<space>']
        for i in range(len(li)-1,-1,-1): 
            if li[i] in del_list:
                li.pop(i) 

    def get_tone(self, phone): # 這裡不考慮介於兩者之間的 tone 譬如 {a34}
        pattern = re.compile('[0-9]+')
        for char in phone:
            match = pattern.findall(char)
            if match:
                return int(char)
        return 0

    def mispronuncitaion_analysis(self, confusion, phone): # 錯誤分析統計
       # 0 TA, 1 FA, 2 TR, 3 FR
        tone = self.get_tone(phone)
        if confusion == 'TA':
            if tone:
                self.mis_final[0] += 1
                self.tone_list[0][tone-1] += 1
            else:
                self.mis_init[0] += 1
        elif confusion == 'FA':
            if tone:
                self.mis_final[1] += 1
                self.tone_list[1][tone-1] += 1
            else:
                self.mis_init[1] += 1
        elif confusion == 'TR':
            if tone:
                self.mis_final[2] += 1
                self.tone_list[2][tone-1] += 1
            else:
                self.mis_init[2] += 1
        elif confusion == 'FR':
            if tone:
                self.mis_final[3] += 1
                self.tone_list[3][tone-1] += 1
            else:
                self.mis_init[3] += 1

    def detection_score(self):
        for res_key in self.detection_align_result: # Confusion Matrix 統計
            
            md_res = self.detection_align_result[res_key][0][-1] # 對齊結果
            ans = self.detection_align_result[res_key][1] # 正確結果
            # 將ans c,s,a,d 分別轉換成 T,F,*,F
            for i in range(len(ans)):
                if(ans[i]=="c"):
                    ans[i] = "T"
                elif(ans[i]=="s"):
                    ans[i] = "F"
                elif(ans[i]=="a"):
                    ans[i] = "*"
                elif(ans[i]=="d"):
                    ans[i] = "F"
            
            md_res_phone = self.detection_align_result[res_key][0][1]
            ans_phone = self.detection_align_result[res_key][0][0]
            
            ori_rec = md_res.copy()
            ori_ans = ans.copy()
            
            # 消除不要的值
            self.clean_list(md_res)
            self.clean_list(md_res_phone)
            self.clean_list(ans_phone)

            if len(ans) != len(ans_phone):
                print('true label count not match true phone count')
            for i in range(len(ans)): # 計算 TA FA TR FR
                md_pop = md_res.pop()
                ans_pop = ans.pop()
                if md_pop == ans_pop: # TA or FA
                    if ans_pop == 'T':
                        self.TA += 1
                    elif ans_pop == 'F':
                        self.TR += 1
                else: # TR or FR
                    if ans_pop == 'T':
                        self.FR += 1
                    elif ans_pop == 'F':
                        self.FA += 1


        # 正確發音
        self.Cpre = self.precision()
        self.Crec = self.recall()
        self.Cf1 = self.F1_score(self.Cpre, self.Crec)
        print('TA: {TA}, FR: {FR}, FA: {FA}, TR: {TR}'.format(TA=self.TA, FR=self.FR, FA=self.FA, TR=self.TR))
        print('Accuracy: {Acc}'.format( Acc= (self.TA+self.TR)/(self.TA+self.FR+self.FA+self.TR) ))
        print('Canonicals')
        print('True Accept Rate: {TAR}'.format(TAR=self.TA/(self.TA+self.FR)))
        print('False Reject Rate: {FRR}'.format(FRR=self.FR/(self.TA+self.FR)))
        print('Mispronunciations')
        print('False Accept Rate: {FAR}'.format(FAR=self.FA/(self.FA+self.TR)))
        print('True Reject Rate: {TRR}'.format(TRR=self.TR/(self.FA+self.TR)))

        # print('Cprecision:'+str(self.Cpre))
        # print('Crecall:'+str(self.Crec))
        # print('CF1:'+str(self.Cf1))
        # 錯誤發音
        # print('mispronuncitaion')
        # tmp  = self.TA
        # self.TA = self.TR
        # self.TR = tmp

        # self.Mpre = self.precision()
        # self.Mrec = self.recall()
        # self.Mf1 = self.F1_score(self.Mpre, self.Mrec)
        # print('Mprecision:'+str(self.Mpre))
        # print('Mrecall:'+str(self.Mrec))
        # print('MF1:'+str(self.Mf1))
        # # 換回來
        # tmp  = self.TA
        # self.TA = self.TR
        # self.TR = tmp


    def check_process(self):
        # print(self.testans_filename)
        # print(self.recog_filename)
        self.read_transcript()
        self.read_recog_file()
        self.detection_align() # 存到 self.detection_align_result
        self.detection_score()
        # self.write_file()

if __name__ == '__main__': 
    check = CAPT_detection()
    check.get_parser() # 抓參數
    check.check_process() # 執行程序
