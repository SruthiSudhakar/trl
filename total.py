import json,os,pdb
hun = []
tt = []
root_dir = 'rover-vlm/outputs'
# root_dir = 'outputs/realworld_all_20260223_162917/checkpoint-3800'

for dir in os.listdir(root_dir):
    # if 'eval_realworld_20260301_23' in dir:
    if 'rover_eval_realworld_20260302_18' in dir or 'rover_eval_realworld_20260302_19' in dir or 'rover_eval_realworld_20260302_20' in dir:
        if os.path.exists(root_dir+'/'+dir+'/evaluation_summary.json'):
            file = json.load(open(root_dir+'/'+dir+'/evaluation_summary.json','r'))
            if 'interval_intra-100_sign_accuracy' in file['metrics']:
                hun.append(file['metrics']['interval_intra-100_sign_accuracy'])
            if 'interval_intra-32_sign_accuracy' in file['metrics']:
                tt.append(file['metrics']['interval_intra-32_sign_accuracy'])
            print(file['datasets'])
print(sum(hun)/len(hun))
print(sum(tt)/len(tt))