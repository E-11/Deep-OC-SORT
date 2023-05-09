import cv2
import os
import json


exp_name = "mggan_17half_wo_img"
dataset = "MOT17"
mode = "val"  # train; test; val
DATA_PATH = "/home/share/datasets/mot"

OUTPUT_PATH = os.path.join("/home/share/exp_result/mggan_motion/", exp_name, "visualize", dataset)
# RESULT_PATH = os.path.join("/home/Deep-OC-SORT/results/trackers/", f"{dataset}-{mode}", exp_name, "data")
RESULT_PATH = "/home/OC_SORT/YOLOX_outputs/yolox_x_mot17_ablation_half_train/track_results/"

gen_video = True
bbox_type = 'tlwh'  # tlwh; xyxy; cx_cy_wh
delimiter = ','


def get_color(idx):
    idx = idx * 3
    color = (int((37 * idx) % 255), int((17 * idx) % 255), int((29 * idx) % 255))
    return color

def get_img_name(frame_id):
    # type(frame_id) = str
    frame_id = str(frame_id)
    if len(frame_id) < 6:
        frame_id = '0'*(6-len(frame_id))+frame_id
    return frame_id + '.jpg'


def draw_bbox(result_txt, img_path, start_frame_num):
    # fp = open('/home/aistudio/work/detection_results/'+filename+'.txt', 'r')
    # video_path = '/home/aistudio/data/mot_images/'+filename+'.mp4'
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    # success, frame = cap.read()
    img = cv2.imread(img_path)
    img_w = img.shape[1]
    img_h = img.shape[0]
    frame_id = int(os.path.split(img_path)[1][:-4])
    print("frame id: ", frame_id, end="\r")

    line = result_txt.readline()
    count = 0
    while line:
        record = line.split(delimiter)
        line = result_txt.readline()
        
        if record[0] != str(frame_id - start_frame_num + 1):
            continue
        
        count += 1
        x1 = float(record[2])
        y1 = float(record[3])
        if bbox_type is 'tlwh':
            x2 = x1 + float(record[4])
            y2 = y1 + float(record[5])
        elif bbox_type is 'xyxy':
            x2 = float(record[4])
            y2 = float(record[5])
        elif bbox_type is 'cx_cy_wh':
            x2 = (x1 + float(record[4])/2)*img_w
            y2 = (y1 + float(record[5])/2)*img_h
            x1 = (x1 - float(record[4])/2)*img_w
            y1 = (y1 - float(record[5])/2)*img_h

        color = get_color(abs(float(record[1])))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 最后一个参数调整边界框粗细
        text_scale = max(1.1, int(img_w)/1400.)  # 调整字体大小
        # put object id
        cv2.putText(img, '{}'.format(int(float(record[1]))), 
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 255),
                    thickness=2)
    # put frame id
    cv2.putText(img, 'frame: %d' % (frame_id), 
                (0, int(15 * text_scale)), 
                cv2.FONT_HERSHEY_PLAIN,
                text_scale, (0, 0, 255),
                thickness=2)
        
    # print('total:', count)
    return img


def tracking_result_visualize():
    # get image
    with open(anno_file, 'r') as fp:
        data_info = json.load(fp)
        seqs = []
        imgs_path = {}
        videos_wh = {}
        for video in data_info['videos']:
            seq = video['file_name']
            seqs.append(seq)
            imgs_path[seq] = []
            if gen_video:
                videos_wh[seq] = [data_info['images'][0]['width'], data_info['images'][0]['height']]

        images = data_info['images']
        for img in images:
            img_file = img['file_name']
            img_seq = img_file.split('/')[0]
            if mode == "test":
                img_path = os.path.join(DATA_PATH, "test", img_file)
            else:
                img_path = os.path.join(DATA_PATH, "train", img_file)
            imgs_path[img_seq].append(img_path)

    # for each sequence
    for seq in seqs:
        start_frame_num = int(os.path.split(imgs_path[seq][0])[1][:-4])
        print("processing【{}】, start from num {}".format(seq, start_frame_num))
        if gen_video:
            frame_rate = 20
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = os.path.join(OUTPUT_PATH, seq+'.mp4')
            print(output_file)
            out = cv2.VideoWriter(output_file, fourcc, frame_rate, (videos_wh[seq][0], videos_wh[seq][1]))
        # draw bbox for each image
        for img_path in imgs_path[seq]:
            result_txt = open(os.path.join(RESULT_PATH, seq+'.txt'), 'r')
            img = draw_bbox(result_txt, img_path, start_frame_num)
            result_txt.close()
            
            if gen_video:
                out.write(img)
            else:
                cv2.imwrite(os.path.join(OUTPUT_PATH, "result_"+os.path.split(img_path)[1]), img)
        out.release()
    

if __name__ == '__main__':
    if mode == "test":
        anno_file = os.path.join(DATA_PATH, "annotations", "test.json")
    elif mode == "val":
        anno_file = os.path.join(DATA_PATH, "annotations", "val_half.json")
    else:
        anno_file = os.path.join(DATA_PATH, "annotations", "train.json")
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    tracking_result_visualize()
