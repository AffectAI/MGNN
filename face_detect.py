import torch
from tqdm import tqdm, trange
import facer
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=Warning) 


def extract_video(input_dir, model, scale=1.3):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    face_boxes = []
    face_images = []
    face_index = []
    original_frames = OrderedDict()
    halve_frames = OrderedDict()
    index_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()

        if not success:
            continue
        original_frames[i] = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(size=[s // 2 for s in frame.size])
        halve_frames[i] = frame
        index_frames[i] = i

    original_frames = list(original_frames.values())
    halve_frames = list(halve_frames.values())
    index_frames = list(index_frames.values())
    for i in range(0, len(halve_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(halve_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([[0, 0, 0, 0]], dtype=np.int16)
        None_prob = np.array([0], dtype=np.int16)
        None_point = np.zeros([1, 5, 2], dtype=np.int16)
        for index, (bbox, prob, point) in enumerate(zip(batch_boxes, batch_probs, batch_points)):
            if bbox is not None:
                pass
            else:
                batch_boxes[index] = None_array
                batch_probs[index] = None_prob
                batch_points[index] = None_point

        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    halve_frames[i:i + batch_size],
                                                                    method="largest")

        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                # crop = original_frames[i:i + batch_size][index][ymin:ymin+size_bb, xmin:xmin+size_bb]
                face_index.append(index_frames[i:i + batch_size][index])
                face_boxes.append([ymin, ymin + size_bb, xmin, xmin + size_bb])
                crop = original_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                face_images.append(crop)
            else:
                pass

    return face_images, face_boxes, face_index


def extract_video_farl(input_dir, model, scale=1.3):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 1
    face_boxes = []
    face_images = []
    face_index = []
    original_frames = OrderedDict()
    halve_frames = OrderedDict()
    index_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()

        if not success:
            continue
        original_frames[i] = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(size=[s // 2 for s in frame.size])
        halve_frames[i] = frame
        index_frames[i] = i

    original_frames = list(original_frames.values())
    halve_frames = list(halve_frames.values())
    halve_frames = [np.array(frame) for frame in halve_frames]
    index_frames = list(index_frames.values())
    for i in range(0, len(halve_frames), batch_size):
        batch_frames = torch.from_numpy(np.array(halve_frames[i:i + batch_size])).permute(0, 3, 1, 2).cuda()
        faces = model(batch_frames)

        None_array = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16)
        None_prob = torch.tensor([0], dtype=torch.int16)
        None_point = torch.zeros([1, 5, 2], dtype=torch.int16)

        if len(faces) == 0:
            batch_boxes = None_array
            batch_probs = None_prob
            batch_points = None_point
        else:
            batch_boxes, batch_probs, batch_points = faces['rects'], faces['scores'], faces['points']

        batch_boxes = batch_boxes.unsqueeze(1)
        batch_probs = batch_probs.unsqueeze(1)
        batch_points = batch_points.unsqueeze(1)

        if len(batch_boxes) > 1:
            batch_boxes = batch_boxes[0]
            batch_boxes = batch_boxes.unsqueeze(0)
        for index, bbox in enumerate(batch_boxes):

            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                face_index.append(index_frames[i:i + batch_size][index])
                face_boxes.append([ymin, ymin + size_bb, xmin, xmin + size_bb])
                crop = original_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                face_images.append(crop)
            else:
                pass

    return face_images, face_boxes, face_index


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = facer.face_detector('retinaface/resnet50', device=device)

    # preprocess AVEC 2013 和 AVEC 2014
    # def videos():
    #     data_path =\
    #         'AVEC2013/2014 video path'
    #     real_video_path = [data_path + i for i in os.listdir(data_path)]
    #     for single_video in real_video_path:
    #         name = single_video.split('.')[-2].split('/')[-1]
    #         output_path = os.path.join('output path', name)
    #
    #         os.makedirs(output_path, exist_ok=True)
    #         face_images, face_boxes, face_index = extract_video_farl(single_video, model, scale=1.0)
    #
    #         i = 0
    #         for j, index in enumerate(face_index):
    #             if face_images[j].shape[0] == 0:
    #                 i += 1
    #             else:
    #                 cv2.imwrite(os.path.join(output_path, "%04d.jpg" % index), face_images[j])
    #         print(name, '.mp4', 'over', i, '/', len(face_images), 'unpassed!')

    # preprocess First Impression dataset
    def videos():
        data_path =\
            './datasets/First_Impression/test/test_videos/'
        real_video_path_set = [data_path + i for i in os.listdir(data_path)]
        for single_set in real_video_path_set:
            real_video_path = [os.path.join(single_set, i) for i in os.listdir(single_set)]
            for single_video in real_video_path:
                name = os.path.join(single_video.split('/')[-2], single_video.split('.mp4')[0].split('/')[-1])
                output_path = os.path.join('./datasets/First_Impression/processed/test', name)
                os.makedirs(output_path, exist_ok=True)
                face_images, face_boxes, face_index = extract_video_farl(single_video, model, scale=1.0)

                i = 0
                for j, index in enumerate(face_index):
                    if face_images[j].shape[0] == 0:
                        i += 1
                    else:
                        cv2.imwrite(os.path.join(output_path, "%05d.jpg" % index), face_images[j])
                print(name, '.mp4', 'over', i, '/', len(face_images), 'unpassed!')

    videos()



