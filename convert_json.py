# # 把anno读取到panads，转成json       test_anno_ours and val_anno_ours.csv csv to pandas to json convert
# 把json单引号转成双引号
import pandas
import json
# anno_path = "/home/lzdjohn/AFSD/adarnn/thumos_annotations/val_Annotation_ours.csv"
# output_path = "/home/lzdjohn/AFSD/adarnn/thumos_annotations/thumos_gt1.json"
anno_path = "/home/lzdjohn/AFSD/adarnn/thumos_annotations/test_Annotation_ours.csv"
output_path = "/home/lzdjohn/AFSD/adarnn/thumos_annotations/thumos_gt2.json"
label_dict = {"run":0, "walk":1, "jump":2, "wave":3, "bend":4, "stand":5, "sit":6}
fps = 50.0
duration = 170.0
def init_container():
    meta_anno = {}
    meta_data_list = []
    data = {}
    return meta_anno, meta_data_list, data

def read_csv(file_path):
    df = pandas.read_csv(file_path)
    pre_video_name = df.iloc[1, 0]
    data_list = []
    output = {}
    meta_anno, meta_data_list, data = init_container()
    for row_index, row in df.iterrows():
        video_name = row["video"]
        start_frame = row["startFrame"]
        start_time = row["start"]*4
        end_frame = row["endFrame"]
        end_time = row["end"]*4
        label = row["type"]

        if video_name == pre_video_name:
            meta_anno = dict()
            meta_anno["segment"] = [(start_time), (end_time)]
            meta_anno["segment(frames)"] = [(start_frame), (end_frame)]
            assert end_frame / end_time == fps
            meta_anno["label"] = label
            meta_anno['label_id'] = label_dict[label]
            meta_data_list.append(meta_anno)
        else:
            # data["subset"]="Validation"
            data["subset"] = "Test"
            data["fps"] = fps
            data["duration"] = duration
            data["annotations"] = meta_data_list
            video_data = {}
            video_data[pre_video_name] = data
            print(video_data)
            data_list.append(video_data)
            pre_video_name = video_name

            meta_anno, meta_data_list, data = init_container()
            meta_anno["segment"] = [(start_time), (end_time)]
            meta_anno["segment(frames)"] = [(start_frame), (end_frame)]
            meta_anno["label"] = label
            meta_anno['label_id'] = label_dict[label]
            meta_data_list.append(meta_anno)

    # data["subset"]="Validation"
    data["subset"] = "Test"
    data["fps"] = fps
    data["duration"] = duration
    data["annotations"] = meta_data_list
    video_data = {}
    video_data[pre_video_name] = data
    # print(video_data)
    data_list.append(video_data)


    meta_anno, meta_data_list, data = init_container()
    meta_anno["segment"] = [(start_time), (end_time)]
    meta_anno["segment(frames)"] = [(start_frame), (end_frame)]
    meta_anno["label"] = label
    meta_anno['label_id'] = label_dict[label]
    meta_data_list.append(meta_anno)

    output["database"] = data_list
    return output

def to_json(df, path):
    # df = json.dumps(df)
    df = str(df)
    df = df.replace("base': [", "base': ")
    df = df.replace("}]}}]}", "}]}}}")
    df = df.replace("}}, {", "}}, ")
    df = df.replace("}]}},", "}]},")
    print(df)

    with open(path, "w") as f:
        json.dump(df, f)
    print("转换json文件完成")


def main():
    df = read_csv(anno_path)
    to_json(df, output_path)


if __name__=="__main__":
    main()
