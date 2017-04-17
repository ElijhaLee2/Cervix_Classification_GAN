import os as os
import os.path as os_path
import shutil as shutil

root_dir = ''
folder = {} # song_name -- puziming_list

# 谱子名列表
f_list = os.listdir(root_dir)
# 每个谱子名
for f in f_list:
    # 歌名
    song_name = f.split('-')[0]
    puziming_list = folder.get(song_name)
    if puziming_list is None:
        folder[song_name]=[]
    folder[song_name].append(f)

# song_name -- puziming_list pair
for (k,v) in folder:
    os.mkdir(os_path.join(root_dir, k))
    for puziming in v:
        shutil.move(os_path.join(root_dir,puziming), os_path.join(root_dir,k,puziming))