import json,sys,os,pdb
dirs = sys.argv[1]
import pdb
avg_small_interval_acc = []
avg_large_interval_acc = []
total_sign_acc = []
for dir in os.listdir(dirs):
    if os.path.isdir(os.path.join(dirs,dir)):
        for dir2 in  os.listdir(os.path.join(dirs,dir)):
            if '.json' in dir2:
                continue
            # pdb.set_trace()
            try:
                file = json.load(open(os.path.join(dirs,dir,dir2,'evaluation_summary.json'),'r'))
            except:
                continue
            total_sign_acc.append(file['metrics']['sign_accuracy'])
            print(file['args']['base_dataset_path'], file['metrics']['sign_accuracy'])
            sc=0
            lc=0
            for key in file['metrics']['interval_breakdown']:
                if int(key.split('-')[1]) < 50 :
                    sc+=1
                    avg_small_interval_acc.append(file['metrics']['interval_breakdown'][key]['sign_accuracy'])
                elif int(key.split('-')[1]) >=50:
                    lc+=1
                    avg_large_interval_acc.append(file['metrics']['interval_breakdown'][key]['sign_accuracy'])
                else:
                    continue
            # print(sum(avg_small_interval_acc[-sc:])/sc)
            # print(sum(avg_large_interval_acc[-lc:])/lc)
            print(max(avg_small_interval_acc[-sc:]))
            print(max(avg_large_interval_acc[-lc:]))

avg_small_interval_acc = sum(avg_small_interval_acc)/len(avg_small_interval_acc)
avg_large_interval_acc = sum(avg_large_interval_acc)/len(avg_large_interval_acc)
total_sign_acc = sum(total_sign_acc)/len(total_sign_acc)
print('avg_small_interval_acc:', avg_small_interval_acc)
print('avg_large_interval_acc:', avg_large_interval_acc)
print('total_sign_acc:', total_sign_acc)